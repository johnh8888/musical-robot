#!/usr/bin/env python3
# auto_tune_color_two.py - 随机搜索最佳权重和遗漏因子

import random
import json
from collections import Counter
from common import fetch_hk_records_merged

RED_NUMS = [1,2,7,8,12,13,18,19,23,24,29,30,34,35,40,45]
BLUE_NUMS = [3,4,9,10,14,15,20,25,31,36,41,46]
GREEN_NUMS = [5,6,11,16,17,21,22,27,28,32,33,38,39,43,44,49]
COLORS = ["红", "蓝", "绿"]

def get_color_by_number(n):
    if n in RED_NUMS: return "红"
    if n in BLUE_NUMS: return "蓝"
    if n in GREEN_NUMS: return "绿"
    return "未知"

def get_history_colors(limit=None):
    records = fetch_hk_records_merged(limit=limit, prefer_local=True)
    colors = []
    for r in records:
        c = get_color_by_number(r["special_number"])
        if c != "未知":
            colors.append(c)
    return colors

def evaluate(colors, windows, weights, miss_factor, lookback=100):
    total = min(lookback, len(colors) - 20)
    hits = 0
    miss_streak = 0
    max_miss = 0
    for i in range(total):
        train = colors[i+20:]
        actual = colors[i]
        # 预测
        votes = Counter()
        for w in windows:
            weight = weights.get(w, 1.0)
            recent = train[:w]
            freq = Counter(recent)
            top2 = [c for c, _ in freq.most_common(2)]
            for c in top2:
                votes[c] += weight
        pred = [c for c, _ in votes.most_common(2)]
        if len(pred) < 2:
            for c in COLORS:
                if c not in pred:
                    pred.append(c)
                    if len(pred) == 2:
                        break
        if miss_streak >= 1:
            hot = Counter(train[:10])
            if hot:
                pred = [c for c, _ in hot.most_common(2)]
        if actual in pred:
            hits += 1
            miss_streak = 0
        else:
            miss_streak += 1
            max_miss = max(max_miss, miss_streak)
    return hits / total, max_miss

def main():
    print("加载历史颜色数据...")
    colors = get_history_colors(limit=None)
    print(f"总期数: {len(colors)}")

    windows = [10, 18, 20, 25]   # 固定窗口
    best_hit = 0
    best_max_miss = 999
    best_weights = None
    best_miss_factor = None

    # 随机搜索 500 次
    for _ in range(500):
        # 生成随机权重（归一化）
        raw_weights = [random.uniform(0.1, 1.0) for _ in windows]
        total = sum(raw_weights)
        weights = {w: raw_weights[i]/total for i, w in enumerate(windows)}
        miss_factor = random.uniform(0.0, 1.0)
        hit, max_miss = evaluate(colors, windows, weights, miss_factor, lookback=100)
        if hit > best_hit or (hit == best_hit and max_miss < best_max_miss):
            best_hit = hit
            best_max_miss = max_miss
            best_weights = weights
            best_miss_factor = miss_factor
            print(f"新最佳: 命中率 {best_hit:.1%}, 连空 {best_max_miss}, 权重 {best_weights}, 遗漏因子 {best_miss_factor:.3f}")

    print("\n最终最佳配置:")
    print(f"窗口: {windows}")
    print(f"权重: {best_weights}")
    print(f"遗漏因子: {best_miss_factor:.3f}")
    print(f"近100期命中率: {best_hit:.1%}, 最大连空: {best_max_miss}")

    # 保存配置
    config = {
        "windows": windows,
        "window_weights": {str(k): v for k, v in best_weights.items()},
        "miss_factor": best_miss_factor
    }
    with open("best_color_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("配置已保存到 best_color_config.json")

if __name__ == "__main__":
    main()