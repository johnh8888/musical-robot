#!/usr/bin/env python3
# color_two.py - 特二色增强版（遗漏加分 + 窗口加权）

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

def predict_two_colors_advanced(colors, windows, window_weights=None, miss_streak=0):
    """
    增强预测：每个窗口内计算颜色得分 = 频率分 + 遗漏分，然后加权投票。
    窗口权重默认为 1/window（越小权重越高）
    """
    if window_weights is None:
        # 默认权重：窗口越小权重越高，例如窗口6权重1.0，窗口30权重0.2
        max_w = max(windows)
        window_weights = {w: (max_w / w) for w in windows}
    # 归一化权重（可选）
    total_weight = sum(window_weights.values())
    if total_weight > 0:
        window_weights = {w: v / total_weight for w, v in window_weights.items()}

    votes = Counter()
    for w in windows:
        recent = colors[:w]  # 降序，最新在前
        # 频率分：每种颜色在窗口内出现的次数
        freq = Counter(recent)
        # 遗漏分：每种颜色距离上一次出现的间隔（值越大表示越冷）
        omission = {c: 0 for c in COLORS}
        for i, col in enumerate(recent):
            if omission[col] == 0:
                omission[col] = i + 1  # 最近一次出现的位置（1-based）
        # 未出现的颜色遗漏设为窗口长度
        for c in COLORS:
            if omission[c] == 0:
                omission[c] = w
        # 归一化遗漏分（使得遗漏越大得分越高，最大为1）
        max_omit = max(omission.values())
        omit_score = {c: (omit / max_omit) for c, omit in omission.items()}
        # 综合得分 = 频率分（相对值） + 0.5 * 遗漏分
        max_freq = max(freq.values()) if freq else 1
        total_score = {}
        for c in COLORS:
            freq_score = freq.get(c, 0) / max_freq
            total_score[c] = freq_score + 0.5 * omit_score[c]
        # 取该窗口下得分最高的两个颜色
        best_two = sorted(total_score, key=total_score.get, reverse=True)[:2]
        for c in best_two:
            votes[c] += window_weights[w]
    # 综合投票，取总分最高的两个颜色
    recommended = [c for c, _ in votes.most_common(2)]
    if len(recommended) < 2:
        for c in COLORS:
            if c not in recommended:
                recommended.append(c)
                if len(recommended) == 2:
                    break
    # 连空保护（激进版：只要上期未中就干预）
    if miss_streak >= 1:
        hot_counter = Counter(colors[:10])
        if hot_counter:
            hottest_two = [c for c, _ in hot_counter.most_common(2)]
            return hottest_two
    return recommended

def backtest_two_colors(colors, lookback, windows, window_weights=None):
    total = min(lookback, len(colors) - 20)
    if total <= 0:
        return None, None
    hits = 0
    miss_streak = 0
    max_miss = 0
    for i in range(total):
        train = colors[i+20:]
        actual = colors[i]
        pred = predict_two_colors_advanced(train, windows, window_weights, miss_streak)
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

    # 使用之前搜索到的最佳窗口
    windows = [10, 18, 20, 25]
    # 可自定义窗口权重（可选），默认使用逆窗口大小
    # window_weights = {10:1.0, 18:0.6, 20:0.5, 25:0.4}  # 示例
    window_weights = None  # 使用默认权重（窗口越小权重越高）

    if args.show:
        pred = predict_two_colors_advanced(colors, windows, window_weights, miss_streak=0)
        records = fetch_hk_records_merged(limit=1, prefer_local=True)
        latest_issue = records[0]["issue_no"] if records else ""
        print(f"预测期号: {next_issue(latest_issue)}")
        print(f"特二色推荐: {'、'.join(pred)}")
        print(f"使用窗口: {windows}")

        hit10, miss10 = backtest_two_colors(colors, 10, windows, window_weights)
        hit100, miss100 = backtest_two_colors(colors, 100, windows, window_weights)
        if hit10 is not None:
            print(f"\n近10期回测：特二色命中率 {hit10:.1%}，最大连空 {miss10}")
            print(f"近100期回测：特二色命中率 {hit100:.1%}，最大连空 {miss100}")

if __name__ == "__main__":
    main()