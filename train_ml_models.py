#!/usr/bin/env python3
# train_ml_models.py - 训练机器学习模型（LightGBM排序 + XGBoost多分类）

import json
import pickle
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from collections import Counter
from datetime import datetime
from common import fetch_hk_records_merged, get_zodiac_by_number, ZODIAC_MAP, ZODIAC_PAIR

# ---------- 1. 构建生肖排序特征 ----------
def build_ranking_features(rows):
    X, y, group = [], [], []
    for idx in range(20, len(rows) - 1):
        train_rows = rows[idx-20:idx]
        target_row = rows[idx]
        target_zodiacs = set()
        for n in target_row["numbers"]:
            target_zodiacs.add(get_zodiac_by_number(n))
        target_zodiacs.add(get_zodiac_by_number(target_row["special_number"]))
        for z in ZODIAC_MAP.keys():
            cnt10 = 0
            for r in train_rows[:10]:
                nums = r["numbers"]
                sp = r["special_number"]
                if any(get_zodiac_by_number(n) == z for n in nums) or get_zodiac_by_number(sp) == z:
                    cnt10 += 1
            cnt20 = 0
            for r in train_rows:
                nums = r["numbers"]
                sp = r["special_number"]
                if any(get_zodiac_by_number(n) == z for n in nums) or get_zodiac_by_number(sp) == z:
                    cnt20 += 1
            omission = 0
            for r in train_rows:
                nums = r["numbers"]
                sp = r["special_number"]
                if any(get_zodiac_by_number(n) == z for n in nums) or get_zodiac_by_number(sp) == z:
                    break
                omission += 1
            pair = ZODIAC_PAIR.get(z)
            pair_cnt = 0
            if pair:
                for r in train_rows[:10]:
                    nums = r["numbers"]
                    sp = r["special_number"]
                    if any(get_zodiac_by_number(n) == pair for n in nums) or get_zodiac_by_number(sp) == pair:
                        pair_cnt += 1
            last_sp_zod = get_zodiac_by_number(train_rows[0]["special_number"])
            is_last = 1 if last_sp_zod == z else 0
            X.append([cnt10, cnt20, omission, pair_cnt, is_last])
            y.append(1 if z in target_zodiacs else 0)
        group.append(12)
    return np.array(X), np.array(y), np.array(group)

def train_ranking_model(X, y, group):
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

# ---------- 2. 特别号数字多分类 ----------
def build_special_features(rows, window=30):
    X, y = [], []
    for i in range(window, len(rows)):
        hist = rows[i-window:i]
        target = rows[i]["special_number"]
        freq = Counter(r["special_number"] for r in hist)
        feats = [freq.get(n, 0) for n in range(1, 50)]
        feats.append(hist[-1]["special_number"])
        feats.append(np.mean(hist[-1]["numbers"]))
        feats.append(np.median(hist[-1]["numbers"]))
        try:
            wd = datetime.strptime(hist[-1]["draw_date"], "%Y-%m-%d").weekday()
        except:
            wd = 0
        feats.append(wd)
        X.append(feats)
        y.append(target - 1)
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

# ---------- 主训练 ----------
def main():
    print("加载历史数据...")
    rows = fetch_hk_records_merged(limit=None, prefer_local=True)
    records = []
    for r in rows:
        records.append({
            "numbers": r["numbers"],
            "special_number": r["special_number"],
            "draw_date": r["draw_date"],
            "issue_no": r["issue_no"]
        })
    print(f"总期数: {len(records)}")

    # 生肖排序模型
    print("构建生肖排序特征...")
    X_zod, y_zod, group_zod = build_ranking_features(records)
    print(f"样本数: {X_zod.shape[0]}")
    print("训练LightGBM排序模型...")
    zod_model = train_ranking_model(X_zod, y_zod, group_zod)
    zod_model.save_model("zodiac_ranker.txt")
    print("✅ 生肖模型已保存为 zodiac_ranker.txt")

    # 特别号模型
    print("构建特别号特征...")
    X_spec, y_spec = build_special_features(records, window=30)
    print(f"样本数: {X_spec.shape[0]}")
    print("训练XGBoost多分类模型...")
    spec_model = train_special_xgb(X_spec, y_spec)
    with open("special_xgb.pkl", "wb") as f:
        pickle.dump(spec_model, f)
    print("✅ 特别号模型已保存为 special_xgb.pkl")

if __name__ == "__main__":
    main()