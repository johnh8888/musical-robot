#!/usr/bin/env python3
# train_ml_special.py - 训练特五肖LightGBM排序模型（修复分组错误）

import json
import numpy as np
import lightgbm as lgb
from common import fetch_hk_records_merged, get_zodiac_by_number, ZODIAC_MAP, ZODIAC_PAIR

def build_ranking_features(rows, window=30):
    X, y, group = [], [], []
    for idx in range(window, len(rows)):
        hist = rows[idx-window:idx]
        target = rows[idx]
        target_z = get_zodiac_by_number(target["special_number"])
        for z in ZODIAC_MAP:
            # 特征1: 近10期正码次数
            cnt10 = sum(1 for r in hist[:10] for n in r["numbers"] if get_zodiac_by_number(n) == z)
            # 特征2: 近20期正码次数
            cnt20 = sum(1 for r in hist[:20] for n in r["numbers"] if get_zodiac_by_number(n) == z)
            # 特征3: 近10期特别号次数
            special_cnt10 = sum(1 for r in hist[:10] if get_zodiac_by_number(r["special_number"]) == z)
            # 特征4: 遗漏期数
            omission = 0
            for r in hist:
                if z in [get_zodiac_by_number(n) for n in r["numbers"]] or z == get_zodiac_by_number(r["special_number"]):
                    break
                omission += 1
            # 特征5: 是否与上期特别号对肖
            last_sp_z = get_zodiac_by_number(hist[0]["special_number"])
            is_pair = 1 if ZODIAC_PAIR.get(last_sp_z) == z else 0
            # 特征6: 正码总次数（30期内）
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
    print(f"特征样本数: {X.shape[0]}, 分组总和: {sum(group)}")  # 应相等
    assert sum(group) == X.shape[0], "分组长度错误"

    # 按样本数划分训练集和验证集（保持分组完整性）
    total_samples = X.shape[0]
    split_ratio = 0.8
    split_idx = int(total_samples * split_ratio)
    # 找到分割点对应的组索引，确保组不被切断
    # 简单方法：按组划分，保留最后一组完整
    cumsum = 0
    group_split = 0
    for i, g in enumerate(group):
        if cumsum + g <= split_idx:
            cumsum += g
            group_split = i + 1
        else:
            break
    # 训练集使用前 group_split 组
    train_end = cumsum
    X_train = X[:train_end]
    y_train = y[:train_end]
    group_train = group[:group_split]
    # 验证集使用剩余
    X_val = X[train_end:]
    y_val = y[train_end:]
    group_val = group[group_split:]

    print(f"训练样本数: {X_train.shape[0]}, 验证样本数: {X_val.shape[0]}")
    print(f"训练组数: {len(group_train)}, 验证组数: {len(group_val)}")

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