#!/usr/bin/env python3
# special_only.py - 特五肖4窗口投票 + 强制融合二生肖

import argparse
import json
from collections import Counter
from common import fetch_hk_records_merged, get_zodiac_by_number, next_issue, ZODIAC_MAP, ZODIAC_PAIR
from strategies_zodiac import predict_strong_two

# 特五肖投票窗口（4个经典窗口）
WINDOWS = [8, 12, 20, 30]

def get_history_rows_as_list(limit=None):
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

def compute_zodiac_score(rows, window, special_boost=2.0):
    scores = {z: 0 for z in ZODIAC_MAP}
    for r in rows[:window]:
        for n in json.loads(r["numbers_json"]):
            z = get_zodiac_by_number(n)
            scores[z] += 1
        sp_z = get_zodiac_by_number(r["special_number"])
        scores[sp_z] += special_boost
    return scores

def predict_five_zodiac_fused(rows, windows):
    # 1. 窗口投票得到基础5个生肖
    votes = Counter()
    for w in windows:
        scores = compute_zodiac_score(rows, w, special_boost=2.0)
        top5 = sorted(scores, key=scores.get, reverse=True)[:5]
        for z in top5:
            votes[z] += 1
    base_top5 = [z for z, _ in votes.most_common(5)]
    
    # 2. 获取二生肖（使用与 zodiac_main.py 一致的参数）
    two_zodiac = predict_strong_two(rows, {"two_recent_window": 20, "two_special_boost": 3.0})
    
    # 3. 融合：二生肖优先，然后补充 base_top5 中不重复的，取前5
    fused = []
    for z in two_zodiac:
        if z not in fused:
            fused.append(z)
    for z in base_top5:
        if len(fused) >= 5:
            break
        if z not in fused:
            fused.append(z)
    return fused[:5]

def backtest_five_zodiac(rows, lookback, windows):
    rows_rev = list(reversed(rows))
    total = min(lookback, len(rows_rev) - 20)
    if total <= 0:
        return None, None
    hits = 0
    miss_streak = 0
    max_miss = 0
    for i in range(total):
        train = rows_rev[i+20:]
        actual = rows_rev[i]["special_number"]
        actual_zod = get_zodiac_by_number(actual)
        pred = predict_five_zodiac_fused(train, windows)
        if actual_zod in pred:
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
    rows = get_history_rows_as_list(limit=None)
    if not rows:
        print("数据获取失败")
        return
    if args.show:
        zodiac5 = predict_five_zodiac_fused(rows, WINDOWS)
        latest = rows[0]["issue_no"]
        pred_issue = next_issue(latest)
        print(f"预测期号: {pred_issue}")
        print(f"\n【特五肖推荐 (4窗口+二生肖融合)】: {'、'.join(zodiac5)}")
        print(f"使用窗口: {WINDOWS}")

        hit10, miss10 = backtest_five_zodiac(rows, 10, WINDOWS)
        hit100, miss100 = backtest_five_zodiac(rows, 100, WINDOWS)
        if hit10 is not None:
            print(f"\n近10期回测：特五肖命中率 {hit10:.1%}，最大连空 {miss10}")
            print(f"近100期回测：特五肖命中率 {hit100:.1%}，最大连空 {miss100}")

if __name__ == "__main__":
    main()