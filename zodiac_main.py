#!/usr/bin/env python3
# zodiac_main.py - 使用最佳参数（从 best_params.json 读取）

import argparse
import json
from collections import Counter
from common import fetch_hk_records_merged, get_zodiac_by_number, next_issue
from strategies_zodiac import (
    predict_strong_single, predict_strong_two, predict_strong_three_with_window,
    get_hot_zodiac, get_cold_zodiac
)

DEFAULT_PARAMS = {
    "windows": [8, 10, 12, 18, 20, 30],
    "single_boost": 3.2,
    "two_boost": 3.0,
    "miss_two_threshold": 2,
    "three_use_strict": True
}

def load_params():
    try:
        with open("best_params.json", "r") as f:
            params = json.load(f)
            print("✅ 已加载最佳参数")
            return params
    except:
        print("⚠️ 未找到 best_params.json，使用默认参数")
        return DEFAULT_PARAMS

def get_history_rows_as_list(limit=None):
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

def backtest_zodiac_stats(rows, lookback, params):
    windows = params["windows"]
    single_boost = params["single_boost"]
    two_boost = params["two_boost"]
    miss_two_threshold = params["miss_two_threshold"]
    three_use_strict = params["three_use_strict"]

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

        if miss_single >= 3:
            pred_single = get_hot_zodiac(train, window=10)
        else:
            pred_single = pred_single_raw

        if miss_two >= miss_two_threshold:
            cold = get_cold_zodiac(train, window=30)
            if cold not in pred_two_raw:
                pred_two_raw[-1] = cold
        pred_two = pred_two_raw

        if miss_three >= 3:
            hot = get_hot_zodiac(train, window=10)
            combined = list(dict.fromkeys(pred_two + [hot]))
            pred_three = combined[:3]
        else:
            pred_three = pred_three_raw[:3]

        # 统计...
        if pred_single in win_z:
            hits_single += 1
            miss_single = 0
        else:
            miss_single += 1
            max_miss_single = max(max_miss_single, miss_single)

        if any(z in win_z for z in pred_two):
            hits_two += 1
            miss_two = 0
        else:
            miss_two += 1
            max_miss_two = max(max_miss_two, miss_two)

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
        print("数据获取失败")
        return
    params = load_params()
    if args.show:
        windows = params["windows"]
        single_boost = params["single_boost"]
        two_boost = params["two_boost"]
        votes_single = Counter()
        votes_two = Counter()
        votes_three = Counter()
        for w in windows:
            votes_single[predict_strong_single(rows, {"single_recent_window": w, "single_special_boost": single_boost})] += 1
            for p in predict_strong_two(rows, {"two_recent_window": w, "two_special_boost": two_boost}):
                votes_two[p] += 1
            for p in predict_strong_three_with_window(rows, w):
                votes_three[p] += 1
        single = votes_single.most_common(1)[0][0]
        two = [z for z, _ in votes_two.most_common(2)]
        three = [z for z, _ in votes_three.most_common(3)]

        latest = rows[0]["issue_no"]
        print(f"预测期号: {next_issue(latest)}")
        print(f"一生肖: {single}")
        print(f"二生肖: {'、'.join(two)}")
        print(f"三生肖: {'、'.join(three)}")

        stats10 = backtest_zodiac_stats(rows, 10, params)
        if stats10:
            print(f"\n近10期回测：一生肖 {stats10['single_hit_rate']:.1%} 连空{stats10['single_max_miss']}")
            print(f"二生肖 {stats10['two_hit_rate']:.1%} 连空{stats10['two_max_miss']}")
            print(f"三生肖 {stats10['three_hit_rate']:.1%} 连空{stats10['three_max_miss']}")

        stats100 = backtest_zodiac_stats(rows, 100, params)
        if stats100:
            print(f"\n近100期回测：一生肖 {stats100['single_hit_rate']:.1%} 连空{stats100['single_max_miss']}")
            print(f"二生肖 {stats100['two_hit_rate']:.1%} 连空{stats100['two_max_miss']}")
            print(f"三生肖 {stats100['three_hit_rate']:.1%} 连空{stats100['three_max_miss']}")

if __name__ == "__main__":
    main()