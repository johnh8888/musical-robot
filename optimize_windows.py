#!/usr/bin/env python3
# optimize_windows.py - 使用 Optuna 优化窗口及连空参数

import json
import optuna
import numpy as np
from collections import Counter
from copy import deepcopy
from common import fetch_hk_records_merged, get_zodiac_by_number, next_issue
from strategies_zodiac import (
    predict_strong_single, predict_strong_two, predict_strong_three_with_window,
    get_hot_zodiac, get_cold_zodiac
)

# 候选窗口池
CANDIDATE_WINDOWS = [6, 8, 10, 12, 15, 18, 20, 25, 30]

def backtest_with_params(rows, lookback, windows, single_boost, two_boost,
                        miss_two_threshold, three_use_strict):
    """
    回测指定参数下的命中率和最大连空
    rows: 升序排列（旧到新）
    """
    rows_rev = list(reversed(rows))
    total = min(lookback, len(rows_rev) - 20)
    if total <= 0:
        return 0, 0, 0, 0, 0, 0

    hits_single = 0
    hits_two = 0
    hits_three = 0
    miss_single = 0
    max_miss_single = 0
    miss_two = 0
    max_miss_two = 0
    miss_three = 0
    max_miss_three = 0

    for i in range(total):
        train = rows_rev[i+20:]
        if len(train) < 20:
            continue
        actual = rows_rev[i]
        win_main = actual["numbers"]
        win_sp = actual["special_number"]
        win_z = {get_zodiac_by_number(n) for n in win_main}
        win_z.add(get_zodiac_by_number(win_sp))

        votes_single = Counter()
        votes_two = Counter()
        votes_three = Counter()
        for w in windows:
            votes_single[predict_strong_single(train, {"single_recent_window": w, "single_special_boost": single_boost})] += 1
            for p in predict_strong_two(train, {"two_recent_window": w, "two_special_boost": two_boost}):
                votes_two[p] += 1
            for p in predict_strong_three_with_window(train, w):
                votes_three[p] += 1

        pred_single_raw = votes_single.most_common(1)[0][0]
        pred_two_raw = [z for z, _ in votes_two.most_common(2)]
        pred_three_raw = [z for z, _ in votes_three.most_common(3)]

        # 一生肖保护 (固定连空>=3追热，不优化)
        if miss_single >= 3:
            pred_single = get_hot_zodiac(train, window=10)
        else:
            pred_single = pred_single_raw

        # 二生肖保护
        if miss_two >= miss_two_threshold:
            cold = get_cold_zodiac(train, window=30)
            if cold not in pred_two_raw:
                pred_two_raw[-1] = cold
        pred_two = pred_two_raw

        # 三生肖保护（连空>=3时使用二生肖+最热）
        if miss_three >= 3:
            hot = get_hot_zodiac(train, window=10)
            combined = list(dict.fromkeys(pred_two + [hot]))
            pred_three = combined[:3]
        else:
            pred_three = pred_three_raw[:3]

        # 一生肖
        if pred_single in win_z:
            hits_single += 1
            miss_single = 0
        else:
            miss_single += 1
            max_miss_single = max(max_miss_single, miss_single)

        # 二生肖
        if any(z in win_z for z in pred_two):
            hits_two += 1
            miss_two = 0
        else:
            miss_two += 1
            max_miss_two = max(max_miss_two, miss_two)

        # 三生肖命中判定
        hit_cnt = sum(1 for z in pred_three if z in win_z)
        if three_use_strict:
            if hit_cnt >= 2:
                hits_three += 1
                miss_three = 0
            else:
                miss_three += 1
                max_miss_three = max(max_miss_three, miss_three)
        else:
            if hit_cnt >= 1:
                hits_three += 1
                miss_three = 0
            else:
                miss_three += 1
                max_miss_three = max(max_miss_three, miss_three)

    return (hits_single/total, max_miss_single,
            hits_two/total, max_miss_two,
            hits_three/total, max_miss_three)

def objective(trial, train_rows, val_rows):
    # 建议超参数
    # 选择窗口子集
    windows = []
    for w in CANDIDATE_WINDOWS:
        if trial.suggest_categorical(f"use_{w}", [True, False]):
            windows.append(w)
    if len(windows) == 0:
        windows = [12]  # 至少一个窗口
    
    single_boost = trial.suggest_float("single_boost", 1.0, 5.0)
    two_boost = trial.suggest_float("two_boost", 1.0, 5.0)
    miss_two_threshold = trial.suggest_int("miss_two_threshold", 1, 3)
    three_use_strict = trial.suggest_categorical("three_use_strict", [True, False])

    # 在验证集上回测，以二生肖命中率为主要目标，同时惩罚连空过大
    _, _, two_hit, two_max_miss, _, _ = backtest_with_params(
        val_rows, len(val_rows), windows, single_boost, two_boost,
        miss_two_threshold, three_use_strict
    )
    # 目标函数：二生肖命中率，如果连空>5则给予惩罚
    penalty = 0
    if two_max_miss > 5:
        penalty = (two_max_miss - 5) * 0.02
    return two_hit - penalty

def main():
    print("加载全部历史数据...")
    records = fetch_hk_records_merged(limit=None, prefer_local=True)
    # 转换为统一格式（与 zodiac_main.py 一致）
    rows = []
    import json
    for r in records:
        rows.append({
            "numbers": r["numbers"],
            "special_number": r["special_number"],
            "draw_date": r["draw_date"],
            "issue_no": r["issue_no"]
        })
    # 按时间升序（旧->新）
    rows_asc = list(reversed(rows))
    split_idx = int(len(rows_asc) * 0.8)
    train_rows = rows_asc[:split_idx]
    val_rows = rows_asc[split_idx:]

    print(f"训练期数: {len(train_rows)}, 验证期数: {len(val_rows)}")

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, train_rows, val_rows), n_trials=200, n_jobs=1)

    print("最佳参数:")
    best = study.best_params
    print(best)

    # 提取最终使用的窗口列表
    best_windows = [w for w in CANDIDATE_WINDOWS if best.get(f"use_{w}", False)]
    if not best_windows:
        best_windows = [12]
    best_params = {
        "windows": best_windows,
        "single_boost": best["single_boost"],
        "two_boost": best["two_boost"],
        "miss_two_threshold": best["miss_two_threshold"],
        "three_use_strict": best["three_use_strict"]
    }
    with open("best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    print("最佳参数已保存至 best_params.json")

    # 在验证集上输出最终命中率
    _, _, two_hit, two_max_miss, _, _ = backtest_with_params(
        val_rows, len(val_rows), best_windows, best["single_boost"], best["two_boost"],
        best["miss_two_threshold"], best["three_use_strict"]
    )
    print(f"验证集二生肖命中率: {two_hit:.1%}, 最大连空: {two_max_miss}")

if __name__ == "__main__":
    main()