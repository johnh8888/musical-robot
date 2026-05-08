# train_xgboost.py - 训练 XGBoost 模型（香港版）

import json
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from collections import defaultdict, Counter
from common import fetch_hk_records, get_zodiac_by_number, ZODIAC_MAP

# ---------- 特征提取（必须与 strategies_zodiac.py 中的 extract_features_for_zodiac 一致）----------
def extract_features_for_zodiac(rows, target_zodiac):
    from strategies_zodiac import _zodiac_omission_map, _row_numbers, _row_special
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
    # 6. 冷热变化率
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
    from strategies_zodiac import _compute_zodiac_score
    rule_scores = _compute_zodiac_score(rows, recent_window=15, special_boost=3.2)
    feats.append(rule_scores.get(target_zodiac, 0.0))
    # 10. 最近特别号与目标生肖号码的最小距离
    latest_sp = _row_special(rows[0])
    nums = ZODIAC_MAP.get(target_zodiac, [])
    min_dist = min(abs(n - latest_sp) for n in nums) if nums else 25
    feats.append(min_dist / 50.0)
    return feats

def prepare_dataset(rows, start=30, seq_len=20):
    X, y = [], []
    for i in range(start, len(rows)):
        hist = rows[i-seq_len:i]
        target = rows[i]
        target_zods = set()
        target_zods.add(get_zodiac_by_number(target["special_number"]))
        for n in target["numbers"]:
            target_zods.add(get_zodiac_by_number(n))
        for z in ZODIAC_MAP:
            feats = extract_features_for_zodiac(hist, z)
            X.append(feats)
            y.append(1 if z in target_zods else 0)
    return np.array(X), np.array(y)

def main():
    print("获取历史数据...")
    records = fetch_hk_records(limit=600)
    rows = [{"numbers": r["numbers"], "special_number": r["special_number"]} for r in records]
    print(f"获取 {len(rows)} 期")
    X, y = prepare_dataset(rows)
    print(f"样本数: {len(X)}, 特征维度: {X.shape[1]}")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print("训练 XGBoost 模型...")
    model = xgb.XGBClassifier(n_estimators=300, max_depth=7, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    val_acc = model.score(X_val, y_val)
    print(f"验证集准确率: {val_acc:.4f}")
    model.save_model("xgboost_zodiac.json")
    print("模型已保存为 xgboost_zodiac.json")

if __name__ == "__main__":
    main()
