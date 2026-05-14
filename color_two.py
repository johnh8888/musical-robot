#!/usr/bin/env python3
# color_two.py - 特二色预测（双注与单注分离窗口）

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

# ---------- 双注模式（4窗口） ----------
def evaluate_two(colors, windows, weights, lookback=100):
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
        # 连空保护
        if miss_streak >= 1:
            hot = Counter(train[:10])
            if hot:
                hottest_two = [c for c, _ in hot.most_common(2)]
                pred = hottest_two
        if actual in pred:
            hits += 1
            miss_streak = 0
        else:
            miss_streak += 1
            max_miss = max(max_miss, miss_streak)
    return hits / total, max_miss

def auto_tune_two(colors):
    print("双注模式：自动搜索最佳参数（随机搜索200次）...")
    windows = [10, 18, 20, 25]
    best_hit = 0
    best_max_miss = 999
    best_weights = {w: 0.25 for w in windows}
    for _ in range(200):
        raw_weights = [random.uniform(0.1, 1.0) for _ in windows]
        total = sum(raw_weights)
        weights = {w: raw_weights[i]/total for i, w in enumerate(windows)}
        hit, max_miss = evaluate_two(colors, windows, weights, lookback=100)
        if hit > best_hit or (hit == best_hit and max_miss < best_max_miss):
            best_hit = hit
            best_max_miss = max_miss
            best_weights = weights
    print(f"双注搜索完成！命中率: {best_hit:.1%}, 连空 {best_max_miss}")
    config = {
        "windows": windows,
        "window_weights": {str(k): v for k, v in best_weights.items()}
    }
    with open("best_color_config.json", "w") as f:
        json.dump(config, f, indent=2)
    return config

def load_config_two():
    try:
        with open("best_color_config.json", "r") as f:
            cfg = json.load(f)
        cfg["window_weights"] = {int(k): v for k, v in cfg["window_weights"].items()}
        return cfg
    except:
        return None

def predict_two(colors, config, miss_streak=0):
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

def backtest_two(colors, config, lookback):
    total = min(lookback, len(colors) - 20)
    if total <= 0:
        return 0, 0
    hits = 0
    miss_streak = 0
    max_miss = 0
    for i in range(total):
        train = colors[i+20:]
        actual = colors[i]
        pred = predict_two(train, config, miss_streak)
        if actual in pred:
            hits += 1
            miss_streak = 0
        else:
            miss_streak += 1
            max_miss = max(max_miss, miss_streak)
    return hits / total, max_miss

# ---------- 单注模式（2窗口） ----------
def evaluate_single(colors, windows, weights, lookback=100):
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
            top1 = [c for c, _ in freq.most_common(1)]  # 只取最热一个
            for c in top1:
                votes[c] += weight
        if votes:
            pred = votes.most_common(1)[0][0]
        else:
            pred = "红"
        # 单注可不用连空保护（或可选，这里不加保护以评估真实命中率）
        if actual == pred:
            hits += 1
            miss_streak = 0
        else:
            miss_streak += 1
            max_miss = max(max_miss, miss_streak)
    return hits / total, max_miss

def auto_tune_single(colors):
    print("单注模式：自动搜索最佳参数（随机搜索200次）...")
    windows = [10, 30]  # 2窗口，可调整候选
    best_hit = 0
    best_max_miss = 999
    best_weights = {w: 0.5 for w in windows}
    for _ in range(200):
        raw_weights = [random.uniform(0.1, 1.0) for _ in windows]
        total = sum(raw_weights)
        weights = {w: raw_weights[i]/total for i, w in enumerate(windows)}
        hit, max_miss = evaluate_single(colors, windows, weights, lookback=100)
        if hit > best_hit or (hit == best_hit and max_miss < best_max_miss):
            best_hit = hit
            best_max_miss = max_miss
            best_weights = weights
    print(f"单注搜索完成！命中率: {best_hit:.1%}, 连空 {best_max_miss}")
    config = {
        "windows": windows,
        "window_weights": {str(k): v for k, v in best_weights.items()}
    }
    with open("best_color_single_config.json", "w") as f:
        json.dump(config, f, indent=2)
    return config

def load_config_single():
    try:
        with open("best_color_single_config.json", "r") as f:
            cfg = json.load(f)
        cfg["window_weights"] = {int(k): v for k, v in cfg["window_weights"].items()}
        return cfg
    except:
        return None

def predict_single(colors, config, miss_streak=0):
    windows = config["windows"]
    weights = config["window_weights"]
    votes = Counter()
    for w in windows:
        weight = weights.get(w, 1.0)
        recent = colors[:w]
        freq = Counter(recent)
        top1 = [c for c, _ in freq.most_common(1)]
        for c in top1:
            votes[c] += weight
    if votes:
        pred = votes.most_common(1)[0][0]
    else:
        pred = "红"
    # 不加连空保护，直接返回
    return pred

def backtest_single(colors, config, lookback):
    total = min(lookback, len(colors) - 20)
    if total <= 0:
        return 0, 0
    hits = 0
    miss_streak = 0
    max_miss = 0
    for i in range(total):
        train = colors[i+20:]
        actual = colors[i]
        pred = predict_single(train, config)
        if actual == pred:
            hits += 1
            miss_streak = 0
        else:
            miss_streak += 1
            max_miss = max(max_miss, miss_streak)
    return hits / total, max_miss

# ---------- 主程序 ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--single", action="store_true", help="单注模式（使用2窗口）")
    args = parser.parse_args()
    colors = get_history_colors(limit=None)
    if not colors:
        print("数据获取失败")
        return

    if args.single:
        # 单注模式
        config = load_config_single()
        if config is None:
            config = auto_tune_single(colors)
        else:
            print("已加载单注最佳参数配置")
        pred = predict_single(colors, config)
        records = fetch_hk_records_merged(limit=1, prefer_local=True)
        latest_issue = records[0]["issue_no"] if records else ""
        print(f"预测期号: {next_issue(latest_issue)}")
        print(f"单注推荐（最热颜色）: {pred}")
        print(f"使用窗口: {config['windows']}")
        print(f"窗口权重: {config['window_weights']}")
        hit10, miss10 = backtest_single(colors, config, 10)
        hit100, miss100 = backtest_single(colors, config, 100)
        if hit10 is not None:
            print(f"\n近10期回测（单注）: 命中率 {hit10:.1%}，最大连空 {miss10}")
            print(f"近100期回测（单注）: 命中率 {hit100:.1%}，最大连空 {miss100}")
    else:
        # 双注模式（默认）
        config = load_config_two()
        if config is None:
            config = auto_tune_two(colors)
        else:
            print("已加载双注最佳参数配置")
        pred_two = predict_two(colors, config, miss_streak=0)
        pred_single = pred_two[0]  # 顺便显示单注
        records = fetch_hk_records_merged(limit=1, prefer_local=True)
        latest_issue = records[0]["issue_no"] if records else ""
        print(f"预测期号: {next_issue(latest_issue)}")
        print(f"双注推荐（两个颜色）: {'、'.join(pred_two)}")
        print(f"（单注参考）最热颜色: {pred_single}")
        print(f"使用窗口: {config['windows']}")
        print(f"窗口权重: {config['window_weights']}")
        hit10_two, miss10_two = backtest_two(colors, config, 10)
        hit100_two, miss100_two = backtest_two(colors, config, 100)
        if hit10_two is not None:
            print(f"\n近10期回测（双注）: 命中率 {hit10_two:.1%}，最大连空 {miss10_two}")
            print(f"近100期回测（双注）: 命中率 {hit100_two:.1%}，最大连空 {miss100_two}")

if __name__ == "__main__":
    main()