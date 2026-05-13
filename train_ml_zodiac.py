#!/usr/bin/env python3
# train_ml_zodiac.py - 训练三生肖XGBoost分类模型

import json
import numpy as np
import xgboost as xgb
from collections import Counter
from common import fetch_hk_records_merged, get_zodiac_by_number, ZODIAC_MAP

def build_features(rows, window=30):
    """
    为每个生肖构建特征，返回 X (样本数*12) 和 y (样本数*12) 的0/1标签
    采用滑动窗口，每行对应一个生肖
    """
    X, y = [], []
    # 初始化特征维度：共10个特征
    for idx in range(window, len(rows)):
        hist = rows[idx-window:idx]  # 历史窗口（降序，最新在前）
        target = rows[idx]
        # 目标生肖集合（包含正码和特别号）
        target_z = set()
        for n in target["numbers"]:
            target_z.add(get_zodiac_by_number(n))
        target_z.add(get_zodiac_by_number(target["special_number"]))

        for z in ZODIAC_MAP:
            # 特征1：近10期出现次数
            cnt10 = 0
            for r in hist[:10]:
                if z in [get_zodiac_by_number(n) for n in r["numbers"]] or z == get_zodiac_by_number(r["special_number"]):
                    cnt10 += 1
            # 特征2：近20期出现次数
            cnt20 = 0
            for r in hist[:20]:
                if z in [get_zodiac_by_number(n) for n in r["numbers"]] or z == get_zodiac_by_number(r["special_number"]):
                    cnt20 += 1
            # 特征3：近30期出现次数
            cnt30 = 0
            for r in hist:
                if z in [get_zodiac_by_number(n) for n in r["numbers"]] or z == get_zodiac_by_number(r["special_number"]):
                    cnt30 += 1
            # 特征4：遗漏期数（距离上次出现间隔）
            omission = 0
            for r in hist:
                if z in [get_zodiac_by_number(n) for n in r["numbers"]] or z == get_zodiac_by_number(r["special_number"]):
                    break
                omission += 1
            # 特征5：是否与上期特别号对肖
            last_sp_z = get_zodiac_by_number(hist[0]["special_number"])
            pair = ZODIAC_PAIR.get(last_sp_z)
            is_pair = 1 if pair == z else 0
            # 特征6：上期特别号是否是该生肖
            is_last_sp = 1 if last_sp_z == z else 0
            # 特征7：近5期特别号出现次数
            sp_cnt5 = sum(1 for r in hist[:5] if get_zodiac_by_number(r["special_number"]) == z)
            # 特征8：近10期特别号出现次数
            sp_cnt10 = sum(1 for r in hist[:10] if get_zodiac_by_number(r["special_number"]) == z)
            # 特征9：最近一次出现的间隔（反向遗漏）与趋势
            # 特征10：正码出现次数加权（近10期）
            main_cnt10 = sum(1 for r in hist[:10] for n in r["numbers"] if get_zodiac_by_number(n) == z)

            X.append([cnt10, cnt20, cnt30, omission, is_pair, is_last_sp, sp_cnt5, sp_cnt10, main_cnt10])
            y.append(1 if z in target_z else 0)

    return np.array(X), np.array(y)

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
    # 按时间升序排列（旧→新）
    rows_asc = list(reversed(rows))
    print(f"总期数: {len(rows_asc)}")

    # 特征构建需要足够窗口，从第30期开始
    X, y = build_features(rows_asc, window=30)
    print(f"样本数: {len(X)}")

    # 划分训练集和验证集（前80%训练，后20%验证）
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print(f"训练样本: {X_train.shape[0]}, 验证样本: {X_val.shape[0]}")

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        use_label_encoder=False,
        verbosity=0
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    acc = model.score(X_val, y_val)
    print(f"验证集准确率（每个生肖0/1）: {acc:.4f}")

    model.save_model("xgboost_zodiac_3.json")
    print("模型已保存为 xgboost_zodiac_3.json")

if __name__ == "__main__":
    main()