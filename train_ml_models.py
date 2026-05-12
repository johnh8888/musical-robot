#!/usr/bin/env python3
# train_ml_models.py - 训练机器学习模型（LightGBM排序 + XGBoost多分类）

import json
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from collections import Counter
from datetime import datetime
from common import fetch_hk_records_merged, get_zodiac_by_number, ZODIAC_MAP, ZODIAC_PAIR

# ---------- 1. 构建生肖排序特征（每个生肖独立样本）----------
def build_ranking_features(rows):
    """
    rows: 降序排列的历史记录（最新在前）
    返回 X (总样本数 = 期数 * 12), y (0/1), group (每期12)
    """
    X, y, group = [], [], []
    # 使用滑动窗口：用前20期预测下一期
    for idx in range(20, len(rows) - 1):
        train_rows = rows[idx-20:idx]   # 前20期
        target_row = rows[idx]          # 下一期
        # 目标生肖集合
        target_zodiacs = set()
        for n in target_row["numbers"]:
            target_zodiacs.add(get_zodiac_by_number(n))
        target_zodiacs.add(get_zodiac_by_number(target_row["special_number"]))

        # 为每个生肖构造特征
        for z in ZODIAC_MAP.keys():
            # 特征1: 近10期出现次数
            cnt10 = 0
            for r in train_rows[:10]:
                nums = r["numbers"]
                sp = r["special_number"]
                if any(get_zodiac_by_number(n) == z for n in nums) or get_zodiac_by_number(sp) == z:
                    cnt10 += 1
            # 特征2: 近20期出现次数
            cnt20 = 0
            for r in train_rows:
                nums = r["numbers"]
                sp = r["special_number"]
                if any(get_zodiac_by_number(n) == z for n in nums) or get_zodiac_by_number(sp) == z:
                    cnt20 += 1
            # 特征3: 遗漏期数
            omission = 0
            for r in train_rows:
                nums = r["numbers"]
                sp = r["special_number"]
                if any(get_zodiac_by_number(n) == z for n in nums) or get_zodiac_by_number(sp) == z:
                    break
                omission += 1
            # 特征4: 对肖出现次数（近10期）
            pair = ZODIAC_PAIR.get(z)
            pair_cnt = 0
            if pair:
                for r in train_rows[:10]:
                    nums = r["numbers"]
                    sp = r["special_number"]
                    if any(get_zodiac_by_number(n) == pair for n in nums) or get_zodiac_by_number(sp) == pair:
                        pair_cnt += 1
            # 特征5: 最近一期特别号是否是该生肖
            last_sp_zod = get_zodiac_by_number(train_rows[0]["special_number"])
            is_last = 1 if last_sp_zod == z else 0
            X.append([cnt10, cnt20, omission, pair_cnt, is_last])
            y.append(1 if z in target_zodiacs else 0)
        group.append(12)   # 每期对应12个样本
    return np.array(X), np.array(y), np.array(group)

def train_ranking_model(X, y, group):
    """训练LightGBM lambdarank模型"""
    dataset = lgb.Dataset(X, label=y, group=group)
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [2,3],
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'num_threads': 8,
    }
    model = lgb.train(params, dataset, num_boost_round=200)
    return model

# ---------- 2. 特别号数字多分类（XGBoost）----------
def build_special_features(rows, window=30):
    X, y = [], []
    for i in range(window, len(rows)):
        hist = rows[i-window:i]   # 用前30期预测下一期
        target = rows[i]["special_number"]
        # 特征工程
        feats = []
        # 特征1-49: 历史出现频率（近30期）
        freq = Counter(r["special_number"] for r in hist)
        for n in range(1, 50):
            feats.append(freq.get(n, 0))
        # 特征50: 上期特别号
        feats.append(hist[-1]["special_number"])
        # 特征51: 上期正码均值
        feats.append(np.mean(hist[-1]["numbers"]))
        # 特征52: 上期正码中位数
        feats.append(np.median(hist[-1]["numbers"]))
        # 特征53: 星期几
        try:
            wd = datetime.strptime(hist[-1]["draw_date"], "%Y-%m-%d").weekday()
        except:
            wd = 0
        feats.append(wd)
        X.append(feats)
        y.append(target - 1)  # XGBoost 多分类要求 0-48
    return np.array(X), np.array(y)

def train_special_xgb(X, y):
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=49,
        n_estimators=300,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='mlogloss',
        use_label_encoder=False,
        verbosity=0
    )
    model.fit(X, y)
    return model

# ---------- 主训练流程 ----------
def main():
    print("加载历史数据...")
    rows = fetch_hk_records_merged(limit=None, prefer_local=True)
    # 转换为内部格式
    records = []
    for r in rows:
        records.append({
            "numbers": r["numbers"],
            "special_number": r["special_number"],
            "draw_date": r["draw_date"],
            "issue_no": r["issue_no"]
        })
    print(f"总期数: {len(records)}")

    # 1. 训练生肖排序模型
    print("构建生肖排序特征...")
    X_zod, y_zod, group_zod = build_ranking_features(records)
    print(f"样本数: {X_zod.shape[0]}, 期数: {len(group_zod)}")
    print("训练LightGBM排序模型...")
    zod_model = train_ranking_model(X_zod, y_zod, group_zod)
    zod_model.save_model("zodiac_ranker.txt")
    print("✅ 生肖模型已保存为 zodiac_ranker.txt")

    # 2. 训练特别号多分类模型
    print("构建特别号特征...")
    X_spec, y_spec = build_special_features(records, window=30)
    print(f"样本数: {X_spec.shape[0]}")
    print("训练XGBoost多分类模型...")
    spec_model = train_special_xgb(X_spec, y_spec)
    spec_model.save_model("special_xgb.json")
    print("✅ 特别号模型已保存为 special_xgb.json")

if __name__ == "__main__":
    main()