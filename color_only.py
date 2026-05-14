#!/usr/bin/env python3
# color_two.py - 预测特别号码的两种颜色（排除一种颜色），高命中率

import argparse
from collections import Counter
from common import fetch_hk_records_merged, next_issue

# 波色映射（与之前一致）
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
    colors = []
    for r in records:
        colors.append(get_color_by_number(r["special_number"]))
    return colors

def predict_two_colors(colors, windows):
    """
    预测两种颜色（排除最不可能的一种颜色）
    策略：多窗口投票，统计每个窗口下出现最少的颜色，综合投票后排除一种颜色
    """
    exclude_votes = Counter()
    for w in windows:
        recent = colors[:w]
        cnt = Counter(recent)
        # 找出出现次数最少的颜色（如果并列，随机选一个）
        if cnt:
            min_count = min(cnt.values())
            least_common = [c for c, v in cnt.items() if v == min_count]
            # 若多个，选择其中一个（这里选第一个）
            exclude = least_common[0]
        else:
            exclude = "红"
        exclude_votes[exclude] += 1
    # 最终排除得票最多的颜色
    exclude_color = exclude_votes.most_common(1)[0][0]
    # 推荐另外两种颜色
    recommended = [c for c in COLORS if c != exclude_color]
    return recommended, exclude_color

def backtest_two_colors(colors, lookback, windows):
    total = min(lookback, len(colors) - 20)
    if total <= 0:
        return None, None, None, None
    hits = 0
    miss_streak = 0
    max_miss = 0
    for i in range(total):
        train = colors[i+20:]
        actual = colors[i]
        recommended, _ = predict_two_colors(train, windows)
        if actual in recommended:
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
    
    # 使用最佳窗口（可调整）
    windows = [6, 10, 30]
    
    if args.show:
        recommended, exclude = predict_two_colors(colors, windows)
        records = fetch_hk_records_merged(limit=1, prefer_local=True)
        latest_issue = records[0]["issue_no"] if records else ""
        print(f"预测期号: {next_issue(latest_issue)}")
        print(f"特二色推荐: {'、'.join(recommended)} (排除: {exclude})")
        print(f"使用窗口: {windows}")
        
        hit10, miss10 = backtest_two_colors(colors, 10, windows)
        hit100, miss100 = backtest_two_colors(colors, 100, windows)
        if hit10 is not None:
            print(f"\n近10期回测：特二色命中率 {hit10:.1%}，最大连空 {miss10}")
            print(f"近100期回测：特二色命中率 {hit100:.1%}，最大连空 {miss100}")

if __name__ == "__main__":
    main()