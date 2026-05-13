#!/usr/bin/env python3
# special_only.py - 特五肖预测（香港版 · 自适应正码权重 + 连空保护）

import argparse
import json
from collections import Counter
from itertools import combinations
from typing import List, Dict, Tuple
from common import (
    fetch_hk_records_merged, get_zodiac_by_number, next_issue, ZODIAC_MAP
)

# ===================== 权重配置（与澳门版相同） =====================
SPECIAL_WEIGHT = 1.0           # 特码生肖固定权重
COLD_BONUS = 0.8               # 最冷2个生肖额外加票
MISS_PENALTY = 0.1             # 窗口优化时的连空惩罚系数
ADAPTIVE_LOOKBACK = 10         # 自适应权重参考期数
BASE_NORMAL_WEIGHT = 0.3       # 正码基础权重
MIN_NORMAL_WEIGHT = 0.1        # 正码权重下限
MAX_NORMAL_WEIGHT = 0.6        # 正码权重上限

# 候选窗口池（可根据香港数据量调整，此处沿用澳门设置）
CANDIDATE_WINDOWS = [4, 6, 8, 10, 12, 15, 18, 20, 24, 30, 36, 42, 48, 54, 60]
OPTIMAL_WINDOWS = [4, 6, 8, 10, 12]   # 默认值，首次运行后自动优化

# ===================== 工具函数 =====================
def window_weight(w: int, base: int = 60) -> float:
    return round(base / w, 2)

def get_history_rows_as_list(limit=None):
    """与旧版保持一致，返回包含 numbers_json 的列表"""
    records = fetch_hk_records_merged(limit=limit, prefer_local=True)
    rows = []
    for r in records:
        rows.append({
            "numbers_json": json.dumps(r["numbers"]),
            "special_number": r["special_number"],
            "draw_date": r["draw_date"],
            "issue_no": r["issue_no"]
        })
    return rows

# ===================== 特码生肖遗漏与自适应正码权重 =====================
def special_zodiac_omission_map(rows):
    """统计特码生肖连续遗漏期数"""
    zodiacs = list(ZODIAC_MAP.keys())
    if not rows:
        return {z: 0 for z in zodiacs}
    omission = {z: 0 for z in zodiacs}
    for row in reversed(rows):
        sp_zodiac = get_zodiac_by_number(row["special_number"])
        for z in zodiacs:
            if z == sp_zodiac:
                omission[z] = 0
            else:
                omission[z] += 1
    return omission

def compute_adaptive_normal_weight(history):
    """
    基于最近 ADAPTIVE_LOOKBACK 期，计算正码与下期特码的关联度。
    返回实际使用的正码权重（0.1~0.6）。
    """
    if len(history) < ADAPTIVE_LOOKBACK + 1:
        return BASE_NORMAL_WEIGHT

    recent = history[-(ADAPTIVE_LOOKBACK + 1):]
    hit_count = 0
    total = 0
    for i in range(len(recent) - 1):
        # 当前期正码生肖集合
        cur_numbers = json.loads(recent[i]["numbers_json"])
        cur_zodiacs = set(get_zodiac_by_number(n) for n in cur_numbers)
        # 下一期特码生肖
        next_sp = get_zodiac_by_number(recent[i + 1]["special_number"])
        if next_sp in cur_zodiacs:
            hit_count += 1
        total += 1

    if total == 0:
        return BASE_NORMAL_WEIGHT
    hit_rate = hit_count / total
    factor = 0.5 + hit_rate   # 范围 0.5~1.5
    weight = BASE_NORMAL_WEIGHT * factor
    return max(MIN_NORMAL_WEIGHT, min(MAX_NORMAL_WEIGHT, weight))

# ===================== 核心预测函数（替代旧版 predict_five_zodiac） =====================
def predict_five_zodiac(rows, windows=None, force_cold=False):
    """
    自适应正码权重 + 多窗口投票 + 冷号加票 + 连空保护。
    windows: 使用的窗口列表，None 则用全局 OPTIMAL_WINDOWS。
    force_cold: 是否强制用最冷2个替换得票末2位（连空保护时启用）。
    返回推荐5生肖列表。
    """
    if windows is None:
        windows = OPTIMAL_WINDOWS
    if not rows:
        return list(ZODIAC_MAP.keys())[:5]

    # 1. 自适应正码权重
    normal_w = compute_adaptive_normal_weight(rows)

    # 2. 遗漏与冷号
    omission = special_zodiac_omission_map(rows)
    sorted_cold = sorted(omission, key=omission.get, reverse=True)

    votes = Counter()
    wgt = {w: window_weight(w) for w in windows}

    for w in windows:
        recent = rows[-w:] if len(rows) >= w else rows
        cnt = Counter()
        for r in recent:
            # 特码生肖
            sp_z = get_zodiac_by_number(r["special_number"])
            cnt[sp_z] += SPECIAL_WEIGHT
            # 正码生肖（自适应权重）
            numbers = json.loads(r["numbers_json"])
            for n in numbers:
                cnt[get_zodiac_by_number(n)] += normal_w
        # 该窗口得分前5名加权投票
        for z, _ in cnt.most_common(5):
            votes[z] += wgt[w]

    # 3. 冷号额外加票（固定值）
    for z in sorted_cold[:2]:
        votes[z] += COLD_BONUS

    # 4. 取前5
    preds = [z for z, _ in votes.most_common(5)]

    # 5. 连空保护（如果启用，保留前3，后2换最冷2）
    if force_cold and len(preds) >= 5:
        keep = preds[:3]
        new_cold = [z for z in sorted_cold[:2] if z not in keep]
        preds = keep + new_cold
        # 补足5个
        while len(preds) < 5:
            for z, _ in votes.most_common():
                if z not in preds:
                    preds.append(z)
                    break
            else:
                preds.append(list(ZODIAC_MAP.keys())[0])
    return preds[:5]

# ===================== 回测函数（使用连空保护） =====================
def backtest_five_zodiac(rows, lookback, windows=None):
    if windows is None:
        windows = OPTIMAL_WINDOWS
    rows_rev = list(reversed(rows))
    total = min(lookback, len(rows_rev) - 20)
    if total <= 0:
        return None, None

    hits = 0
    cur_miss = 0
    max_miss = 0

    for i in range(total):
        train = rows_rev[i+20:]  # 历史数据，不包含当前期
        if len(train) < 20:
            continue
        actual_sp = rows_rev[i]["special_number"]
        actual_z = get_zodiac_by_number(actual_sp)

        # 如果当前连空，启动保护
        use_protection = (cur_miss >= 1)
        preds = predict_five_zodiac(train, windows=windows, force_cold=use_protection)

        if actual_z in preds:
            hits += 1
            cur_miss = 0
        else:
            cur_miss += 1
            max_miss = max(max_miss, cur_miss)

    return hits / total, max_miss

# ===================== 窗口优化函数 =====================
def optimize_windows(rows, opt_lookback=40):
    """在历史数据上搜索最优5窗口组合"""
    best_combo = None
    best_score = -float("inf")
    min_required = opt_lookback + 20
    if len(rows) < min_required:
        opt_lookback = max(10, len(rows) - 20)

    for combo in combinations(CANDIDATE_WINDOWS, 5):
        hr, max_miss = backtest_five_zodiac(rows, opt_lookback, windows=list(combo))
        if hr is None:
            continue
        score = hr - max_miss * MISS_PENALTY
        if score > best_score:
            best_score = score
            best_combo = list(combo)
    return best_combo if best_combo else [4, 6, 8, 10, 12]

# ===================== 主程序 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    rows = get_history_rows_as_list(limit=None)
    if not rows:
        print("数据获取失败")
        return

    # 自动选择最优窗口（首次运行稍慢，之后可保存结果）
    global OPTIMAL_WINDOWS
    OPTIMAL_WINDOWS = optimize_windows(rows, opt_lookback=40)
    print(f"自动选择最优窗口: {OPTIMAL_WINDOWS}")

    if args.show:
        # 预测下一期
        latest = rows[0]["issue_no"]
        pred_issue = next_issue(latest)
        print(f"预测期号: {pred_issue}")

        current_weight = compute_adaptive_normal_weight(rows)
        print(f"当前自适应正码权重: {current_weight:.2f}")

        zodiac5 = predict_five_zodiac(rows, windows=OPTIMAL_WINDOWS, force_cold=False)
        print(f"\n【特五肖推荐】: {'、'.join(zodiac5)}")

        # 回测
        hit10, miss10 = backtest_five_zodiac(rows, 10, windows=OPTIMAL_WINDOWS)
        hit100, miss100 = backtest_five_zodiac(rows, 100, windows=OPTIMAL_WINDOWS)
        if hit10 is not None:
            print(f"\n近10期回测：特五肖命中率 {hit10:.1%}，最大连空 {miss10}")
            print(f"近100期回测：特五肖命中率 {hit100:.1%}，最大连空 {miss100}")

if __name__ == "__main__":
    main()
