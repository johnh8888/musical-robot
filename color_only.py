#!/usr/bin/env python3
# color_only.py - 预测特别号码的颜色（红、蓝、绿），使用最佳窗口 [6,10,30]

import argparse
import json
from collections import Counter
from common import fetch_hk_records_merged, next_issue

# 波色映射表
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

def backtest_color(colors, lookback, windows):
    total = min(lookback, len(colors) - 20)
    if total <= 0:
        return None, None
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    
    colors = get_history_colors(limit=None)
    if not colors:
        print("数据获取失败")
        return
    
    # 应用搜索得到的最佳窗口
    windows = [6, 10, 30]
    
    if args.show:
        pred = predict_color(colors, windows, miss_streak=0)
        records = fetch_hk_records_merged(limit=1, prefer_local=True)
        latest_issue = records[0]["issue_no"] if records else ""
        print(f"预测期号: {next_issue(latest_issue)}")
        print(f"预测特别号颜色: {pred}")
        print(f"使用窗口: {windows}")
        
        hit10, miss10 = backtest_color(colors, 10, windows)
        hit100, miss100 = backtest_color(colors, 100, windows)
        if hit10 is not None:
            print(f"\n近10期回测：颜色命中率 {hit10:.1%}，最大连空 {miss10}")
            print(f"近100期回测：颜色命中率 {hit100:.1%}，最大连空 {miss100}")

if __name__ == "__main__":
    main()