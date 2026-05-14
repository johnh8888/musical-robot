#!/usr/bin/env python3
# search_color_windows.py - 穷举搜索波色预测的最佳窗口组合

import itertools
from collections import Counter
from common import fetch_hk_records_merged, next_issue

# 波色映射（与 color_only.py 保持一致）
RED_NUMS = [1,2,7,8,12,13,18,19,23,24,29,30,34,35,40,45]
BLUE_NUMS = [3,4,9,10,14,15,20,25,31,36,41,46]
GREEN_NUMS = [5,6,11,16,17,21,22,27,28,32,33,38,39,43,44,49]

def get_color_by_number(n):
    if n in RED_NUMS:
        return "红"
    if n in BLUE_NUMS:
        return "蓝"
    if n in GREEN_NUMS:
        return "绿"
    return "未知"

def get_history_colors(limit=None):
    records = fetch_hk_records_merged(limit=limit, prefer_local=True)
    colors = []
    for r in records:
        colors.append(get_color_by_number(r["special_number"]))
    return colors

def predict_color(colors, windows, miss_streak=0):
    votes = Counter()
    for w in windows:
        recent = colors[:w]
        cnt = Counter(recent)
        if cnt:
            top_color = cnt.most_common(1)[0][0]
            votes[top_color] += 1
    base_pred = votes.most_common(1)[0][0] if votes else "红"
    if miss_streak >= 2:
        hot_counter = Counter(colors[:10])
        if hot_counter:
            hottest = hot_counter.most_common(1)[0][0]
            return hottest
    return base_pred

def evaluate_windows(colors, windows):
    """评估给定窗口组合的性能，返回 (命中率, 最大连空)"""
    total = min(100, len(colors) - 20)  # 测试最近100期
    if total <= 0:
        return 0, 0
    hits = 0
    miss_streak = 0
    max_miss = 0
    for i in range(total):
        train = colors[i+20:]
        actual = colors[i]
        pred = predict_color(train, windows, miss_streak)
        if pred == actual:
            hits += 1
            miss_streak = 0
        else:
            miss_streak += 1
            max_miss = max(max_miss, miss_streak)
    return hits / total, max_miss

def main():
    print("加载历史数据...")
    colors = get_history_colors(limit=None)
    print(f"总期数: {len(colors)}")

    # 候选窗口池
    CANDIDATE_WINDOWS = [6, 8, 10, 12, 15, 18, 20, 25, 30]
    MIN_WINDOWS = 2
    MAX_WINDOWS = 6

    best_hit = 0
    best_windows = None
    best_max_miss = 999
    good_combs = []

    total_combs = 0
    for k in range(MIN_WINDOWS, MAX_WINDOWS + 1):
        for comb in itertools.combinations(CANDIDATE_WINDOWS, k):
            total_combs += 1
            hit, max_miss = evaluate_windows(colors, comb)
            if max_miss <= 1:
                good_combs.append((comb, hit, max_miss))
            if hit > best_hit or (hit == best_hit and max_miss < best_max_miss):
                best_hit = hit
                best_max_miss = max_miss
                best_windows = comb
            if total_combs % 50 == 0:
                print(f"已测试 {total_combs} 个组合...")

    print(f"\n穷举完成，共测试 {total_combs} 个组合。")
    if good_combs:
        print(f"\n✅ 找到 {len(good_combs)} 个满足最大连空≤1的组合：")
        for comb, hit, mm in sorted(good_combs, key=lambda x: -x[1]):
            print(f"  窗口 {list(comb)} : 命中率 {hit:.1%}, 最大连空 {mm}")
    else:
        print("\n❌ 未找到最大连空≤1的组合。最佳结果为：")
        print(f"  窗口 {list(best_windows)} : 命中率 {best_hit:.1%}, 最大连空 {best_max_miss}")
        print("\n建议将以下窗口应用到 color_only.py 中：")
        print(f"  WINDOWS = {list(best_windows)}")

if __name__ == "__main__":
    main()