#!/usr/bin/env python3
# search_best_windows.py - 穷举搜索使三生肖（3中2）最大连空≤1的窗口组合

import itertools
import json
from collections import Counter
from common import fetch_hk_records_merged, get_zodiac_by_number
from strategies_zodiac import (
    predict_strong_two, predict_strong_three_with_window
)

# 候选窗口池（可根据经验调整）
CANDIDATE_WINDOWS = [6, 8, 10, 12, 15, 18, 20, 25, 30]
# 测试窗口数量范围（2到6个）
MIN_WINDOWS = 2
MAX_WINDOWS = 6

def evaluate_three_zodiac(rows, windows, lookback=100):
    """评估三生肖（3中2）的命中率和最大连空"""
    rows_rev = list(reversed(rows))
    total = min(lookback, len(rows_rev) - 20)
    if total <= 0:
        return 0, 0
    hits = 0
    miss_streak = 0
    max_miss = 0
    for i in range(total):
        train = rows_rev[i+20:]
        if len(train) < 20:
            continue
        actual = rows_rev[i]
        win_main = json.loads(actual["numbers_json"])
        win_sp = actual["special_number"]
        win_z = {get_zodiac_by_number(n) for n in win_main}
        win_z.add(get_zodiac_by_number(win_sp))

        # 多窗口投票
        votes = Counter()
        for w in windows:
            for z in predict_strong_three_with_window(train, w):
                votes[z] += 1
        pred_three = [z for z, _ in votes.most_common(3)]

        # 命中判定（必须中2个）
        hit_cnt = sum(1 for z in pred_three if z in win_z)
        if hit_cnt >= 2:
            hits += 1
            miss_streak = 0
        else:
            miss_streak += 1
            max_miss = max(max_miss, miss_streak)
    return hits / total, max_miss

def evaluate_two_zodiac(rows, windows, lookback=100):
    """评估二生肖（辅助）"""
    rows_rev = list(reversed(rows))
    total = min(lookback, len(rows_rev) - 20)
    hits = 0
    for i in range(total):
        train = rows_rev[i+20:]
        actual = rows_rev[i]
        win_main = json.loads(actual["numbers_json"])
        win_sp = actual["special_number"]
        win_z = {get_zodiac_by_number(n) for n in win_main}
        win_z.add(get_zodiac_by_number(win_sp))

        votes = Counter()
        for w in windows:
            for z in predict_strong_two(train, {"two_recent_window": w, "two_special_boost": 3.0}):
                votes[z] += 1
        pred_two = [z for z, _ in votes.most_common(2)]
        if any(z in win_z for z in pred_two):
            hits += 1
    return hits / total

def main():
    print("加载历史数据...")
    records = fetch_hk_records_merged(limit=None, prefer_local=True)
    rows = []
    for r in records:
        rows.append({
            "numbers_json": json.dumps(r["numbers"]),
            "special_number": r["special_number"],
            "draw_date": r["draw_date"],
            "issue_no": r["issue_no"]
        })
    print(f"总期数: {len(rows)}")

    best_hit = 0
    best_windows = None
    best_max_miss = 999
    good_combs = []  # 存储最大连空≤1的组合

    total_combs = 0
    for k in range(MIN_WINDOWS, MAX_WINDOWS + 1):
        for comb in itertools.combinations(CANDIDATE_WINDOWS, k):
            total_combs += 1
            hit, max_miss = evaluate_three_zodiac(rows, comb, lookback=100)
            if max_miss <= 1:
                good_combs.append((comb, hit, max_miss))
            if hit > best_hit or (hit == best_hit and max_miss < best_max_miss):
                best_hit = hit
                best_max_miss = max_miss
                best_windows = comb
            if total_combs % 50 == 0:
                print(f"已测试 {total_combs} / 126 个组合...")

    print(f"\n穷举完成，共测试 {total_combs} 个组合。")
    if good_combs:
        print(f"\n✅ 找到 {len(good_combs)} 个满足最大连空≤1的组合：")
        for comb, hit, mm in sorted(good_combs, key=lambda x: -x[1]):
            print(f"  窗口 {list(comb):12} : 命中率 {hit:.1%}, 最大连空 {mm}")
    else:
        print("\n❌ 未找到最大连空≤1的组合。最佳结果为：")
        print(f"  窗口 {list(best_windows)} : 三生肖命中率 {best_hit:.1%}, 最大连空 {best_max_miss}")

    # 输出最佳窗口在二生肖上的表现
    two_hit = evaluate_two_zodiac(rows, best_windows, lookback=100)
    print(f"\n在最佳窗口下，二生肖近100期命中率: {two_hit:.1%}")
    print(f"建议将 `OPTIMAL_WINDOWS = {list(best_windows)}` 应用到 zodiac_main.py 中。")

if __name__ == "__main__":
    main()