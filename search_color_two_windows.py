#!/usr/bin/env python3
# search_color_two_windows.py - 搜索特二色（推荐最热两种颜色）的最佳窗口

import itertools
from collections import Counter
from common import fetch_hk_records_merged, get_color_by_number

def get_history_colors(limit=None):
    records = fetch_hk_records_merged(limit=limit, prefer_local=True)
    colors = [get_color_by_number(r["special_number"]) for r in records]
    return colors

def predict_two_colors(colors, windows):
    votes = Counter()
    for w in windows:
        recent = colors[:w]
        cnt = Counter(recent)
        top2 = [c for c, _ in cnt.most_common(2)]
        for c in top2:
            votes[c] += 1
    recommended = [c for c, _ in votes.most_common(2)]
    return recommended

def evaluate(colors, windows, lookback=100):
    total = min(lookback, len(colors) - 20)
    hits = 0
    miss_streak = 0
    max_miss = 0
    for i in range(total):
        train = colors[i+20:]
        actual = colors[i]
        pred = predict_two_colors(train, windows)
        if actual in pred:
            hits += 1
            miss_streak = 0
        else:
            miss_streak += 1
            max_miss = max(max_miss, miss_streak)
    return hits / total, max_miss

def main():
    colors = get_history_colors(limit=None)
    print(f"总期数: {len(colors)}")
    
    CANDIDATE_WINDOWS = [6, 8, 10, 12, 15, 18, 20, 25, 30]
    MIN_WINDOWS = 2
    MAX_WINDOWS = 6
    
    best_hit = 0
    best_windows = None
    best_max_miss = 999
    
    total = 0
    for k in range(MIN_WINDOWS, MAX_WINDOWS+1):
        for comb in itertools.combinations(CANDIDATE_WINDOWS, k):
            total += 1
            hit, max_miss = evaluate(colors, comb, lookback=100)
            if hit > best_hit or (hit == best_hit and max_miss < best_max_miss):
                best_hit = hit
                best_max_miss = max_miss
                best_windows = comb
            if total % 50 == 0:
                print(f"已测试 {total} 个组合...")
    
    print(f"\n最佳窗口: {list(best_windows)}")
    print(f"命中率: {best_hit:.1%}, 最大连空: {best_max_miss}")
    print(f"请将 color_two.py 中的 windows 改为 {list(best_windows)}")

if __name__ == "__main__":
    main()