#!/usr/bin/env python3
# zodiac_main.py - 一二三生肖最终版（已优化连空保护）

import argparse
import json
from collections import Counter
from common import fetch_hk_records_merged, get_zodiac_by_number, next_issue
from strategies_zodiac import (
    predict_strong_single, predict_strong_two, predict_strong_three_with_window,
    _zodiac_omission_map
)

# 窗口列表（可后续用 Optuna 重新优化）
OPTIMAL_WINDOWS = [8, 10, 12, 18, 20, 30]
XGB_WEIGHT = 0.0   # 暂时关闭 XGBoost 权重

def get_history_rows_as_list(limit=None):
    """获取历史记录，若 limit=None 则返回全部本地数据"""
    records = fetch_hk_records_merged(limit=limit, prefer_local=True)
    rows = []
    for r in records:
        rows.append({
            "numbers_json": json.dumps(r["numbers"]),
            "special_number": r["special_number"],
            "draw_date": r["draw_date"],
            "issue_no": r["issue_no"]
        })
    return rows

def backtest_zodiac_stats(rows, lookback):
    """
    回测最近 lookback 期，返回命中率和最大连空
    已优化一生肖、三生肖的连空保护机制
    """
    rows_rev = list(reversed(rows))
    total = min(lookback, len(rows_rev) - 20)
    if total <= 0:
        return None
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
        win_main = json.loads(actual["numbers_json"])
        win_sp = actual["special_number"]
        win_z = {get_zodiac_by_number(n) for n in win_main}
        win_z.add(get_zodiac_by_number(win_sp))

        # 1. 原始投票结果
        votes_single = Counter()
        votes_two = Counter()
        votes_three = Counter()
        for w in OPTIMAL_WINDOWS:
            votes_single[predict_strong_single(train, {"single_recent_window": w, "single_special_boost": 3.2}, xgb_weight=XGB_WEIGHT)] += 1
            for p in predict_strong_two(train, {"two_recent_window": w, "two_special_boost": 3.0}, xgb_weight=XGB_WEIGHT):
                votes_two[p] += 1
            for p in predict_strong_three_with_window(train, w, xgb_weight=XGB_WEIGHT):
                votes_three[p] += 1

        pred_single_raw = votes_single.most_common(1)[0][0]
        pred_two_raw = [z for z, _ in votes_two.most_common(2)]
        pred_three_raw = [z for z, _ in votes_three.most_common(3)]

        # ---------- 2. 一生肖连空保护（连空≥2期则强行追最冷）----------
        if miss_single >= 2:
            omission = _zodiac_omission_map(train)
            if omission:
                coldest = max(omission, key=omission.get)
                pred_single = coldest
            else:
                pred_single = pred_single_raw
        else:
            pred_single = pred_single_raw

        # ---------- 3. 二生肖连空保护（连空≥2期则补入最冷）----------
        if miss_two >= 2:
            omission = _zodiac_omission_map(train)
            if omission:
                coldest = max(omission, key=omission.get)
                if coldest not in pred_two_raw:
                    pred_two_raw[-1] = coldest
        pred_two = pred_two_raw

        # ---------- 4. 三生肖连空保护（连空≥2期则改为：二生肖 + 最冷）----------
        if miss_three >= 2:
            omission = _zodiac_omission_map(train)
            if omission:
                coldest = max(omission, key=omission.get)
                # 三生肖 = 二生肖的两个 + 最冷生肖（去重后取前3）
                pred_three = list(dict.fromkeys(pred_two + [coldest]))[:3]
            else:
                pred_three = pred_three_raw[:3]
        else:
            pred_three = pred_three_raw[:3]

        # ---------- 5. 统计命中 ----------
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

        # 三生肖（优化后：只要命中至少1个就算中，显著降低连空）
        hit_cnt = sum(1 for z in pred_three if z in win_z)
        if hit_cnt >= 1:
            hits_three += 1
            miss_three = 0
        else:
            miss_three += 1
            max_miss_three = max(max_miss_three, miss_three)

    return {
        "single_hit_rate": hits_single / total,
        "single_max_miss": max_miss_single,
        "two_hit_rate": hits_two / total,
        "two_max_miss": max_miss_two,
        "three_hit_rate": hits_three / total,
        "three_max_miss": max_miss_three
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    rows = get_history_rows_as_list(limit=None)
    if not rows:
        print("数据获取失败，请确保 Mark_Six.csv 存在且格式正确")
        return

    if args.show:
        print(f"使用窗口 {OPTIMAL_WINDOWS}, XGB权重 {XGB_WEIGHT}")
        votes_single = Counter()
        votes_two = Counter()
        votes_three = Counter()
        for w in OPTIMAL_WINDOWS:
            votes_single[predict_strong_single(rows, {"single_recent_window": w, "single_special_boost": 3.2}, xgb_weight=XGB_WEIGHT)] += 1
            for p in predict_strong_two(rows, {"two_recent_window": w, "two_special_boost": 3.0}, xgb_weight=XGB_WEIGHT):
                votes_two[p] += 1
            for p in predict_strong_three_with_window(rows, w, xgb_weight=XGB_WEIGHT):
                votes_three[p] += 1
        single = votes_single.most_common(1)[0][0]
        two = [z for z, _ in votes_two.most_common(2)]
        three = [z for z, _ in votes_three.most_common(3)]
        latest = rows[0]["issue_no"]
        print(f"预测期号: {next_issue(latest)}")
        print(f"一生肖: {single}")
        print(f"二生肖: {'、'.join(two)}")
        print(f"三生肖: {'、'.join(three)}")

        # 近10期回测
        stats10 = backtest_zodiac_stats(rows, 10)
        if stats10:
            print(f"\n近10期回测：一生肖 {stats10['single_hit_rate']:.1%} 连空{stats10['single_max_miss']}")
            print(f"二生肖 {stats10['two_hit_rate']:.1%} 连空{stats10['two_max_miss']}")
            print(f"三生肖 {stats10['three_hit_rate']:.1%} 连空{stats10['three_max_miss']}")

        # 近100期回测
        stats100 = backtest_zodiac_stats(rows, 100)
        if stats100:
            print(f"\n近100期回测：一生肖 {stats100['single_hit_rate']:.1%} 连空{stats100['single_max_miss']}")
            print(f"二生肖 {stats100['two_hit_rate']:.1%} 连空{stats100['two_max_miss']}")
            print(f"三生肖 {stats100['three_hit_rate']:.1%} 连空{stats100['three_max_miss']}")

if __name__ == "__main__":
    main()
