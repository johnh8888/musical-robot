#!/usr/bin/env python3
# color_two.py - 特二色预测（推荐最热两种颜色 + 激进连空保护）

import argparse
from collections import Counter
from common import fetch_hk_records_merged, next_issue

# 波色映射
RED_NUMS = [1,2,7,8,12,13,18,19,23,24,29,30,34,35,40,45]
BLUE_NUMS = [3,4,9,10,14,15,20,25,31,36,41,46]
GREEN_NUMS = [5,6,11,16,17,21,22,27,28,32,33,38,39,43,44,49]

COLORS = ["红", "蓝", "绿"]

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

def predict_two_colors(colors, windows, miss_streak=0):
    votes = Counter()
    for w in windows:
        recent = colors[:w]
        cnt = Counter(recent)
        top2 = [c for c, _ in cnt.most_common(2)]
        for c in top2:
            votes[c] += 1
    recommended = [c for c, _ in votes.most_common(2)]
    if len(recommended) < 2:
        for c in COLORS:
            if c not in recommended:
                recommended.append(c)
                if len(recommended) == 2:
                    break
    # 激进保护：只要上一期未中（miss_streak >= 1），就强制使用最近10期最热两种颜色
    if miss_streak >= 1:
        hot_counter = Counter(colors[:10])
        if hot_counter:
            hottest_two = [c for c, _ in hot_counter.most_common(2)]
            return hottest_two
    return recommended

def backtest_two_colors(colors, lookback, windows):
    total = min(lookback, len(colors) - 20)
    if total <= 0:
        return None, None
    hits = 0
    miss_streak = 0
    max_miss = 0
    for i in range(total):
        train = colors[i+20:]
        actual = colors[i]
        pred = predict_two_colors(train, windows, miss_streak)
        if actual in pred:
            hits += 1
            miss_streak = 0
        else:
            miss_streak += 1
            max_miss = max(max_miss, miss_streak)
    return hits / total, max_miss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    colors = get_history_colors(limit=None)
    if not colors:
        print("数据获取失败")
        return

    # 使用搜索到的最佳窗口
    windows = [10, 18, 20, 25]

    if args.show:
        pred = predict_two_colors(colors, windows, miss_streak=0)
        records = fetch_hk_records_merged(limit=1, prefer_local=True)
        latest_issue = records[0]["issue_no"] if records else ""
        print(f"预测期号: {next_issue(latest_issue)}")
        print(f"特二色推荐: {'、'.join(pred)}")
        print(f"使用窗口: {windows}")

        hit10, miss10 = backtest_two_colors(colors, 10, windows)
        hit100, miss100 = backtest_two_colors(colors, 100, windows)
        if hit10 is not None:
            print(f"\n近10期回测：特二色命中率 {hit10:.1%}，最大连空 {miss10}")
            print(f"近100期回测：特二色命中率 {hit100:.1%}，最大连空 {miss100}")

if __name__ == "__main__":
    main()