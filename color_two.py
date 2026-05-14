#!/usr/bin/env python3
# color_two.py - 特二色预测（自适应参数，首次运行自动搜索）

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

def evaluate(colors, windows, weights, miss_factor, lookback=100):
    total = min(lookback, len(colors) - 20)
    if total <= 0:
        return 0, 0
    hits = 0
    miss_streak = 0
    max_miss = 0
    for i in range(total):
        train = colors[i+20:]
        actual = colors[i]
        votes = Counter()
        for w in windows:
            weight = weights.get(w, 1.0)
            recent = train[:w]
            freq = Counter(recent)
            # 简单频率投票，不加遗漏（可简化）
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
        # 连空保护（激进：上一期未中即干预）
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

def auto_tune(colors):
    print("首次运行，正在自动搜索最佳参数（约5秒）...")
    windows = [10, 18, 20, 25]   # 固定窗口
    best_hit = 0
    best_max_miss = 999
    best_weights = {w: 0.25 for w in windows}
    best_miss_factor = 0.5
    for _ in range(200):   # 随机搜索200次
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
    print(f"搜索完成！命中率: {best_hit:.1%}, 最大连空: {best_max_miss}")
    config = {
        "windows": windows,
        "window_weights": {str(k): v for k, v in best_weights.items()},
        "miss_factor": best_miss_factor
    }
    with open("best_color_config.json", "w") as f:
        json.dump(config, f, indent=2)
    return config

def load_config():
    try:
        with open("best_color_config.json", "r") as f:
            cfg = json.load(f)
        # 转换键为整数
        cfg["window_weights"] = {int(k): v for k, v in cfg["window_weights"].items()}
        return cfg
    except:
        return None

def predict_two_colors(colors, config, miss_streak=0):
    windows = config["windows"]
    weights = config["window_weights"]
    miss_factor = config["miss_factor"]
    votes = Counter()
    for w in windows:
        weight = weights.get(w, 1.0)
        recent = colors[:w]
        freq = Counter(recent)
        # 可加入遗漏加分，但为简化，不使用 miss_factor
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
            return [c for c, _ in hot.most_common(2)]
    return pred

def backtest_two_colors(colors, config, lookback):
    total = min(lookback, len(colors) - 20)
    if total <= 0:
        return 0, 0
    hits = 0
    miss_streak = 0
    max_miss = 0
    for i in range(total):
        train = colors[i+20:]
        actual = colors[i]
        pred = predict_two_colors(train, config, miss_streak)
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

    config = load_config()
    if config is None:
        config = auto_tune(colors)
    else:
        print("已加载最佳参数配置")

    if args.show:
        pred = predict_two_colors(colors, config, miss_streak=0)
        records = fetch_hk_records_merged(limit=1, prefer_local=True)
        latest_issue = records[0]["issue_no"] if records else ""
        print(f"预测期号: {next_issue(latest_issue)}")
        print(f"特二色推荐: {'、'.join(pred)}")
        print(f"使用窗口: {config['windows']}")
        print(f"窗口权重: {config['window_weights']}")
        print(f"遗漏因子: {config['miss_factor']}")

        hit10, miss10 = backtest_two_colors(colors, config, 10)
        hit100, miss100 = backtest_two_colors(colors, config, 100)
        if hit10 is not None:
            print(f"\n近10期回测：特二色命中率 {hit10:.1%}，最大连空 {miss10}")
            print(f"近100期回测：特二色命中率 {hit100:.1%}，最大连空 {miss100}")

if __name__ == "__main__":
    main()