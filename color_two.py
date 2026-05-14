#!/usr/bin/env python3
# color_two.py - 特二色预测（同时显示双注和单注）

import argparse
import json
import random
from collections import Counter
from common import fetch_hk_records_merged, next_issue

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

def evaluate(colors, windows, weights, lookback=100):
    total = min(lookback, len(colors) - 20)
    if total <= 0:
        return 0, 0, 0, 0
    hits_two = 0
    hits_single = 0
    miss_streak_two = 0
    miss_streak_single = 0
    max_miss_two = 0
    max_miss_single = 0
    for i in range(total):
        train = colors[i+20:]
        actual = colors[i]
        votes = Counter()
        for w in windows:
            weight = weights.get(w, 1.0)
            recent = train[:w]
            freq = Counter(recent)
            top2 = [c for c, _ in freq.most_common(2)]
            for c in top2:
                votes[c] += weight
        pred_two = [c for c, _ in votes.most_common(2)]
        if len(pred_two) < 2:
            for c in COLORS:
                if c not in pred_two:
                    pred_two.append(c)
                    if len(pred_two) == 2:
                        break
        pred_single = pred_two[0]
        # 连空保护（仅双注使用）
        if miss_streak_two >= 1:
            hot = Counter(train[:10])
            if hot:
                hottest_two = [c for c, _ in hot.most_common(2)]
                pred_two = hottest_two
                pred_single = pred_two[0]
        if actual in pred_two:
            hits_two += 1
            miss_streak_two = 0
        else:
            miss_streak_two += 1
            max_miss_two = max(max_miss_two, miss_streak_two)
        if actual == pred_single:
            hits_single += 1
            miss_streak_single = 0
        else:
            miss_streak_single += 1
            max_miss_single = max(max_miss_single, miss_streak_single)
    return hits_two/total, max_miss_two, hits_single/total, max_miss_single

def auto_tune(colors):
    print("首次运行，正在自动搜索最佳参数（随机搜索200次）...")
    windows = [10, 18, 20, 25]
    best_hit_two = 0
    best_max_miss_two = 999
    best_weights = {w: 0.25 for w in windows}
    for _ in range(200):
        raw_weights = [random.uniform(0.1, 1.0) for _ in windows]
        total = sum(raw_weights)
        weights = {w: raw_weights[i]/total for i, w in enumerate(windows)}
        hit_two, max_miss_two, _, _ = evaluate(colors, windows, weights, lookback=100)
        if hit_two > best_hit_two or (hit_two == best_hit_two and max_miss_two < best_max_miss_two):
            best_hit_two = hit_two
            best_max_miss_two = max_miss_two
            best_weights = weights
    print(f"搜索完成！双注命中率: {best_hit_two:.1%}, 连空 {best_max_miss_two}")
    config = {
        "windows": windows,
        "window_weights": {str(k): v for k, v in best_weights.items()}
    }
    with open("best_color_config.json", "w") as f:
        json.dump(config, f, indent=2)
    return config

def load_config():
    try:
        with open("best_color_config.json", "r") as f:
            cfg = json.load(f)
        cfg["window_weights"] = {int(k): v for k, v in cfg["window_weights"].items()}
        return cfg
    except:
        return None

def predict_two_colors(colors, config, miss_streak=0):
    windows = config["windows"]
    weights = config["window_weights"]
    votes = Counter()
    for w in windows:
        weight = weights.get(w, 1.0)
        recent = colors[:w]
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
        hot = Counter(colors[:10])
        if hot:
            hottest_two = [c for c, _ in hot.most_common(2)]
            return hottest_two
    return pred

def backtest(colors, config, lookback):
    total = min(lookback, len(colors) - 20)
    if total <= 0:
        return 0, 0, 0, 0
    hits_two = 0
    hits_single = 0
    miss_streak_two = 0
    miss_streak_single = 0
    max_miss_two = 0
    max_miss_single = 0
    for i in range(total):
        train = colors[i+20:]
        actual = colors[i]
        pred_two = predict_two_colors(train, config, miss_streak_two)
        pred_single = pred_two[0]
        if actual in pred_two:
            hits_two += 1
            miss_streak_two = 0
        else:
            miss_streak_two += 1
            max_miss_two = max(max_miss_two, miss_streak_two)
        if actual == pred_single:
            hits_single += 1
            miss_streak_single = 0
        else:
            miss_streak_single += 1
            max_miss_single = max(max_miss_single, miss_streak_single)
    return hits_two/total, max_miss_two, hits_single/total, max_miss_single

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    colors = get_history_colors(limit=None)
    if not colors:
        print("数据获取失败")
        return

    config = load_config()
    if config is None:
        config = auto_tune(colors)
    else:
        print("已加载最佳参数配置")

    if args.show:
        # 获取当前预测
        pred_two = predict_two_colors(colors, config, miss_streak=0)
        pred_single = pred_two[0]
        records = fetch_hk_records_merged(limit=1, prefer_local=True)
        latest_issue = records[0]["issue_no"] if records else ""
        print(f"预测期号: {next_issue(latest_issue)}")
        print(f"双注推荐（两个颜色）: {'、'.join(pred_two)}")
        print(f"单注推荐（最热颜色）: {pred_single}")
        print(f"使用窗口: {config['windows']}")
        print(f"窗口权重: {config['window_weights']}")

        # 回测数据
        hit10_two, miss10_two, hit10_single, miss10_single = backtest(colors, config, 10)
        hit100_two, miss100_two, hit100_single, miss100_single = backtest(colors, config, 100)
        if hit10_two is not None:
            print(f"\n近10期回测：双注命中率 {hit10_two:.1%}，最大连空 {miss10_two}")
            print(f"           单注命中率 {hit10_single:.1%}，最大连空 {miss10_single}")
            print(f"近100期回测：双注命中率 {hit100_two:.1%}，最大连空 {miss100_two}")
            print(f"           单注命中率 {hit100_single:.1%}，最大连空 {miss100_single}")

if __name__ == "__main__":
    main()