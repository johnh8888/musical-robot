#!/usr/bin/env python3
# search_color_two_windows.py - 搜索特二色（推荐最热两种颜色）的最佳窗口组合

import itertools
from collections import Counter
from common import fetch_hk_records_merged

# 波色映射（与 color_two.py 保持一致）
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
    colors = [get_color_by_number(r["special_number"]) for r in records]
    return colors

def predict_two_colors(colors, windows):
    """推荐最热的两种颜色"""
    votes = Counter()
    for w in windows:
        recent = colors[:w]
        cnt = Counter(recent)
        top2 = [c for c, _ in cnt.most_common(2)]
        for c in top2:
            votes[c] += 1
    recommended = [c for c, _ in votes.most_common(2)]
    # 确保返回两个颜色
    if len(recommended) < 2:
        all_colors = ["红", "蓝", "绿"]
        for c in all_colors:
            if c not in recommended:
                recommended.append(c)
                if len(recommended) == 2:
                    break
    return recommended

def evaluate(colors, windows, lookback=100):
    total = min(lookback, len(colors) - 20)
    if total <= 0:
        return 0, 0
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
    print("正在加载历史颜色数据...")
    colors = get_history_colors(limit=None)
    print(f"总期数: {len(colors)}")

    CANDIDATE_WINDOWS = [6, 8, 10, 12, 15, 18, 20, 25, 30]
    MIN_WINDOWS = 2
    MAX_WINDOWS = 6

    best_hit = 0
    best_windows = None
    best_max_miss = 999
    total_combs = 0

    print("开始穷举搜索...")
    for k in range(MIN_WINDOWS, MAX_WINDOWS + 1):
        for comb in itertools.combinations(CANDIDATE_WINDOWS, k):
            total_combs += 1
            hit, max_miss = evaluate(colors, comb, lookback=100)
            if hit > best_hit or (hit == best_hit and max_miss < best_max_miss):
                best_hit = hit
                best_max_miss = max_miss
                best_windows = comb
            if total_combs % 50 == 0:
                print(f"已测试 {total_combs} 个组合...")

    print("\n" + "="*50)
    print("搜索完成！")
    print(f"最佳窗口组合: {list(best_windows)}")
    print(f"近100期命中率: {best_hit:.1%}")
    print(f"最大连空: {best_max_miss}")
    print("="*50)
    print("\n请将以下配置更新到 color_two.py 中：")
    print(f"windows = {list(best_windows)}")

if __name__ == "__main__":
    main()