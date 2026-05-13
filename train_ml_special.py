#!/usr/bin/env python3
# train_ml_special.py - 训练特五肖LightGBM排序模型

import json
import numpy as np
import lightgbm as lgb
from collections import Counter
from common import fetch_hk_records_merged, get_zodiac_by_number, ZODIAC_MAP, ZODIAC_PAIR

def build_ranking_features(rows, window=30):
    X, y, group = [], [], []
    for idx in range(window, len(rows)):
        hist = rows[idx-window:idx]
        target = rows[idx]
        target_z = get_zodiac_by_number(target["special_number"])  # 只需特别号生肖
        for z in ZODIAC_MAP:
            # 特征：基于正码和特别号的历史频率等
            cnt10 = sum(1 for r in hist[:10] for n in r["numbers"] if get_zodiac_by_number(n) == z)
            cnt20 = sum(1 for r in hist[:20] for n in r["numbers"] if get_zodiac_by_number(n) == z)
            special_cnt10 = sum(1 for r in hist[:10] if get_zodiac_by_number(r["special_number"]) == z)
            omission = 0
            for r in hist:
                if z in [get_zodiac_by_number(n) for n in r["numbers"]] or z == get_zodiac_by_number(r["special_number"]):
                    break
                omission += 1
            last_sp_z = get_zodiac_by_number(hist[0]["special_number"])
            is_pair = 1 if ZODIAC_PAIR.get(last_sp_z) == z else 0
            # 正码总数
            total_main = sum(1 for r in hist for n in r["numbers"] if get_zodiac_by_number(n) == z)
            X.append([cnt10, cnt20, special_cnt10, omission, is_pair, total_main])
            y.append(1 if z == target_z else 0)
        group.append(12)
    return np.array(X), np.array(y), np.array(group)

def main():
    print("加载历史数据...")
    records = fetch_hk_records_merged(limit=None, prefer_local=True)
    rows = []
    for r in records:
        rows.append({
            "numbers": r["numbers"],
            "special_number": r["special_number"],
            "draw_date": r["draw_date"],
            "issue_no": r["issue_no"]
        })
    rows_asc = list(reversed(rows))
    print(f"总期数: {len(rows_asc)}")

    X, y, group = build_ranking_features(rows_asc, window=30)
    print(f"样本数: {len(X)}")

    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    group_train, group_val = group[:split//12], group[split//12:]

    train_data = lgb.Dataset(X_train, label=y_train, group=group_train)
    val_data = lgb.Dataset(X_val, label=y_val, group=group_val)

    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [5],
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    model = lgb.train(params, train_data, valid_sets=[val_data], num_boost_round=200)
    model.save_model("lgb_special_ranker.txt")
    print("模型已保存为 lgb_special_ranker.txt")

if __name__ == "__main__":
    main()