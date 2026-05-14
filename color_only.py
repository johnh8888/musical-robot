#!/usr/bin/env python3
# color_only.py - 预测特别号码的颜色（红、蓝、绿），多窗口投票

import argparse
import json
from collections import Counter
from common import fetch_hk_records_merged

# 波色映射表（根据你提供的参照表整理）
COLOR_MAP = {}
# 红波
red_nums = [1,2,7,8,12,13,18,19,23,24,29,30,34,35,40,45]
# 蓝波
blue_nums = [3,4,9,10,14,15,20,25,31,36,41,46]
# 绿波
green_nums = [5,6,11,16,17,21,22,27,28,32,33,38,39,43,44,49]
for n in red_nums:
    COLOR_MAP[n] = "红"
for n in blue_nums:
    COLOR_MAP[n] = "蓝"
for n in green_nums:
    COLOR_MAP[n] = "绿"

def get_color_by_number(n):
    return COLOR_MAP.get(n, "未知")

def get_history_colors(limit=None):
    records = fetch_hk_records_merged(limit=limit, prefer_local=True)
    colors = []
    for r in records:
        colors.append(get_color_by_number(r["special_number"]))
    return colors

def predict_color(colors, windows, miss_streak=0):
    """
    多窗口投票预测颜色
    colors: 历史颜色列表（降序，最新在前）
    windows: 窗口大小列表
    """
    votes = Counter()
    for w in windows:
        recent = colors[:w]
        cnt = Counter(recent)
        # 预测该窗口下最频繁的颜色
        if cnt:
            top_color = cnt.most_common(1)[0][0]
            votes[top_color] += 1
    base_pred = votes.most_common(1)[0][0] if votes else "红"
    
    # 连空保护：如果连续2期未中，用最近10期最热颜色替换
    if miss_streak >= 2:
        hot_counter = Counter(colors[:10])
        if hot_counter:
            hottest = hot_counter.most_common(1)[0][0]
            return hottest
    return base_pred

def backtest_color(colors, lookback, windows):
    """
    回测颜色预测的命中率和最大连空
    """
    total = min(lookback, len(colors) - 20)
    if total <= 0:
        return None, None
    hits = 0
    miss_streak = 0
    max_miss = 0
    for i in range(total):
        # 训练数据：历史颜色（不包括当前期）
        train = colors[i+20:]   # 因为colors是降序，i+20以后是更早的历史
        # 实际颜色
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
    
    # 窗口配置（与生肖保持一致）
    windows = [8, 12, 20, 30]   # 可调
    
    if args.show:
        # 当前预测
        pred = predict_color(colors, windows, miss_streak=0)
        latest_issue = fetch_hk_records_merged(limit=1)[0]["issue_no"]
        from common import next_issue
        print(f"预测期号: {next_issue(latest_issue)}")
        print(f"预测特别号颜色: {pred}")
        print(f"使用窗口: {windows}")
        
        # 回测
        hit10, miss10 = backtest_color(colors, 10, windows)
        hit100, miss100 = backtest_color(colors, 100, windows)
        if hit10 is not None:
            print(f"\n近10期回测：颜色命中率 {hit10:.1%}，最大连空 {miss10}")
            print(f"近100期回测：颜色命中率 {hit100:.1%}，最大连空 {miss100}")

if __name__ == "__main__":
    main()