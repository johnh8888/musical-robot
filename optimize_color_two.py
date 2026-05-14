#!/usr/bin/env python3
# optimize_color_two.py - 使用 Optuna 自动搜索特二色预测的最佳超参数

import optuna
import json
from collections import Counter
from common import fetch_hk_records_merged

# 波色映射
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
    return [get_color_by_number(r["special_number"]) for r in records]

def predict(colors, windows, weights, miss_factor, miss_streak):
    """
    预测两个颜色
    weights: 每个窗口的权重列表（与windows等长）
    miss_factor: 遗漏分在综合得分中的系数（0~1）
    """
    scores = {c: 0.0 for c in COLORS}
    for w, weight in zip(windows, weights):
        recent = colors[:w]
        freq = Counter(recent)
        omission = {c: 0 for c in COLORS}
        for i, col in enumerate(recent):
            if omission[col] == 0:
                omission[col] = i + 1
        for c in COLORS:
            if omission[c] == 0:
                omission[c] = w
        max_omit = max(omission.values())
        omit_norm = {c: omission[c] / max_omit for c in COLORS}
        max_freq = max(freq.values()) if freq else 1
        for c in COLORS:
            freq_score = freq.get(c, 0) / max_freq
            total_score = freq_score + miss_factor * omit_norm[c]
            scores[c] += weight * total_score
    best_two = sorted(scores, key=scores.get, reverse=True)[:2]
    if miss_streak >= 1:
        hot = Counter(colors[:10])
        if hot:
            return [c for c, _ in hot.most_common(2)]
    return best_two

def evaluate(colors, windows, weights, miss_factor, lookback=100):
    total = min(lookback, len(colors) - 20)
    hits = 0
    miss_streak = 0
    max_miss = 0
    for i in range(total):
        train = colors[i+20:]
        actual = colors[i]
        pred = predict(train, windows, weights, miss_factor, miss_streak)
        if actual in pred:
            hits += 1
            miss_streak = 0
        else:
            miss_streak += 1
            max_miss = max(max_miss, miss_streak)
    return hits / total, max_miss

def objective(trial, colors):
    # 窗口数量（2~6个）
    n_windows = trial.suggest_int("n_windows", 2, 6)
    candidate_windows = [6,8,10,12,15,18,20,25,30]
    # 选择 n_windows 个不重复的窗口
    windows = trial.suggest_categorical("windows", 
        [list(comb) for comb in __import__('itertools').combinations(candidate_windows, n_windows)])
    # 每个窗口的权重（软权重，和为1）
    weights = []
    for i in range(n_windows):
        weights.append(trial.suggest_float(f"weight_{i}", 0.1, 1.0))
    total = sum(weights)
    weights = [w/total for w in weights]
    # 遗漏因子
    miss_factor = trial.suggest_float("miss_factor", 0.0, 1.0)
    # 评估
    hit, max_miss = evaluate(colors, windows, weights, miss_factor, lookback=100)
    # 目标：最大化命中率，同时惩罚连空（例如连空每多1减0.01）
    return hit - 0.01 * max_miss

def main():
    print("加载历史颜色数据...")
    colors = get_history_colors(limit=None)
    print(f"总期数: {len(colors)}")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, colors), n_trials=200, n_jobs=1)
    best = study.best_params
    best_windows = best["windows"]
    # 提取权重
    n = len(best_windows)
    best_weights = [best[f"weight_{i}"] for i in range(n)]
    # 归一化
    total = sum(best_weights)
    best_weights = [w/total for w in best_weights]
    best_miss_factor = best["miss_factor"]
    print("\n最佳超参数:")
    print(f"窗口: {best_windows}")
    print(f"权重: {best_weights}")
    print(f"遗漏因子: {best_miss_factor:.3f}")
    # 评估最终命中率和连空
    hit, max_miss = evaluate(colors, best_windows, best_weights, best_miss_factor, lookback=100)
    print(f"近100期命中率: {hit:.1%}, 最大连空: {max_miss}")
    # 生成 color_two.py 的配置代码段
    config = f"""
# 自动优化生成的最佳配置
windows = {best_windows}
window_weights = {dict(zip(best_windows, best_weights))}
miss_factor = {best_miss_factor:.3f}
"""
    print(config)
    # 保存到文件，供工作流使用
    with open("best_color_config.json", "w") as f:
        json.dump({
            "windows": best_windows,
            "window_weights": {str(k): v for k, v in zip(best_windows, best_weights)},
            "miss_factor": best_miss_factor
        }, f, indent=2)

if __name__ == "__main__":
    main()