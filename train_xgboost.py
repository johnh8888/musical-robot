# train_xgboost.py - 使用完整本地历史重训练 XGBoost 模型

import json
import numpy as np
import xgboost as xgb
from collections import defaultdict, Counter
from common import fetch_hk_records_merged, get_zodiac_by_number, ZODIAC_MAP
from strategies_zodiac import _zodiac_omission_map, _row_numbers, _row_special, _compute_zodiac_score

# ---------- 特征提取（与 strategies_zodiac 保持一致）----------
def extract_features_for_zodiac(rows, target_zodiac):
    """
    rows 为历史记录列表（降序，最新在前）
    返回 10 维特征向量
    """
    feats = []
    # 1. 遗漏
    omission = _zodiac_omission_map(rows)
    feats.append(omission.get(target_zodiac, 30))

    # 2. 近10期出现次数
    cnt10 = 0
    for r in rows[:10]:
        if target_zodiac in [get_zodiac_by_number(n) for n in _row_numbers(r)] or \
           target_zodiac == get_zodiac_by_number(_row_special(r)):
            cnt10 += 1
    feats.append(cnt10)

    # 3. 近20期出现次数
    cnt20 = 0
    for r in rows[:20]:
        if target_zodiac in [get_zodiac_by_number(n) for n in _row_numbers(r)] or \
           target_zodiac == get_zodiac_by_number(_row_special(r)):
            cnt20 += 1
    feats.append(cnt20)

    # 4. 最近特别号是否匹配
    latest_sp_zod = get_zodiac_by_number(_row_special(rows[0]))
    feats.append(1 if latest_sp_zod == target_zodiac else 0)

    # 5. 主号频率（近5期）
    main_cnt = 0
    for r in rows[:5]:
        for n in _row_numbers(r):
            if get_zodiac_by_number(n) == target_zodiac:
                main_cnt += 1
    feats.append(main_cnt / 5.0)

    # 6. 冷热变化（近5期出现次数 - 前5期出现次数）
    recent = 0
    for r in rows[:5]:
        if target_zodiac in [get_zodiac_by_number(n) for n in _row_numbers(r)] or \
           target_zodiac == get_zodiac_by_number(_row_special(r)):
            recent += 1
    old = 0
    for r in rows[5:10]:
        if target_zodiac in [get_zodiac_by_number(n) for n in _row_numbers(r)] or \
           target_zodiac == get_zodiac_by_number(_row_special(r)):
            old += 1
    feats.append(recent - old)

    # 7. 特别号出现次数（近10期）
    sp_cnt = 0
    for r in rows[:10]:
        if get_zodiac_by_number(_row_special(r)) == target_zodiac:
            sp_cnt += 1
    feats.append(sp_cnt)

    # 8. 平均间隔
    indices = [i for i, r in enumerate(rows) if target_zodiac in [get_zodiac_by_number(n) for n in _row_numbers(r)] or target_zodiac == get_zodiac_by_number(_row_special(r))]
    if len(indices) >= 2:
        avg_gap = sum(indices[i] - indices[i-1] for i in range(1, len(indices))) / (len(indices) - 1)
    else:
        avg_gap = 30
    feats.append(avg_gap)

    # 9. 规则评分
    rule_scores = _compute_zodiac_score(rows, recent_window=15, special_boost=3.2)
    feats.append(rule_scores.get(target_zodiac, 0.0))

    # 10. 最近特别号与目标生肖号码的最小距离（归一化）
    latest_sp = _row_special(rows[0])
    nums = ZODIAC_MAP.get(target_zodiac, [])
    min_dist = min(abs(n - latest_sp) for n in nums) if nums else 25
    feats.append(min_dist / 50.0)

    return feats

def prepare_dataset(rows_asc, seq_len=20):
    """
    rows_asc: 历史记录按时间升序排列（旧在前，新在后）
    使用滑动窗口构造样本，特征为前 seq_len 期，标签为当期所有生肖（多标签）
    """
    X, y = [], []
    for i in range(seq_len, len(rows_asc)):
        hist = rows_asc[i-seq_len:i]   # 历史窗口（升序，旧->新）
        # 注意：内部特征提取函数期望 rows 是降序（最新在前），所以需要反转
        hist_desc = list(reversed(hist))
        target = rows_asc[i]
        target_zods = set()
        target_zods.add(get_zodiac_by_number(target["special_number"]))
        for n in target["numbers"]:
            target_zods.add(get_zodiac_by_number(n))

        for z in ZODIAC_MAP:
            feats = extract_features_for_zodiac(hist_desc, z)
            X.append(feats)
            y.append(1 if z in target_zods else 0)
    return np.array(X), np.array(y)

def main():
    print("获取本地完整历史数据...")
    records = fetch_hk_records_merged(limit=None, prefer_local=True)
    # 转换为简单格式
    rows = [{"numbers": r["numbers"], "special_number": r["special_number"]} for r in records]
    print(f"总期数: {len(rows)}")

    # 按时间升序排列（旧 -> 新）
    rows_asc = list(reversed(rows))
    # 划分训练集和验证集（前80%训练，后20%验证）
    split_idx = int(len(rows_asc) * 0.8)
    train_rows = rows_asc[:split_idx]
    val_rows = rows_asc[split_idx:]

    print("准备训练数据...")
    X_train, y_train = prepare_dataset(train_rows)
    X_val, y_val = prepare_dataset(val_rows)

    print(f"训练样本数: {X_train.shape[0]}, 特征维度: {X_train.shape[1]}")
    print(f"验证样本数: {X_val.shape[0]}")

    print("训练 XGBoost 模型...")
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    val_acc = model.score(X_val, y_val)
    print(f"验证集准确率: {val_acc:.4f}")

    model.save_model("xgboost_zodiac.json")
    print("模型已保存为 xgboost_zodiac.json")

if __name__ == "__main__":
    main()
