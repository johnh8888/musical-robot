#!/usr/bin/env python3
# special_only.py - 特五肖3窗口投票版（8,12,20）

import argparse
import json
from collections import Counter
from common import fetch_hk_records_merged, get_zodiac_by_number, next_issue, ZODIAC_MAP, ZODIAC_PAIR

# 使用3窗口（与三生肖优化窗口一致）
WINDOWS = [8, 12, 20]

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

def predict_five_zodiac(rows, windows, miss_streak=0):
    votes = Counter()
    for w in windows:
        scores = compute_zodiac_score(rows, w, special_boost=2.0)
        top5 = sorted(scores, key=scores.get, reverse=True)[:5]
        for z in top5:
            votes[z] += 1
    base_top5 = [z for z, _ in votes.most_common(5)]

    if miss_streak >= 2:
        hot_counter = Counter()
        for r in rows[:10]:
            for n in json.loads(r["numbers_json"]):
                hot_counter[get_zodiac_by_number(n)] += 1
            hot_counter[get_zodiac_by_number(r["special_number"])] += 1
        if hot_counter:
            hottest = hot_counter.most_common(1)[0][0]
            if hottest not in base_top5:
                base_top5[-1] = hottest
    return base_top5[:5]

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
        pred = predict_five_zodiac(train, windows, miss_streak)
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
        zodiac5 = predict_five_zodiac(rows, WINDOWS, miss_streak=0)
        latest = rows[0]["issue_no"]
        pred_issue = next_issue(latest)
        print(f"预测期号: {pred_issue}")
        print(f"\n【特五肖推荐 (3窗口投票)】: {'、'.join(zodiac5)}")
        print(f"使用窗口: {WINDOWS}")

        hit10, miss10 = backtest_five_zodiac(rows, 10, WINDOWS)
        hit100, miss100 = backtest_five_zodiac(rows, 100, WINDOWS)
        if hit10 is not None:
            print(f"\n近10期回测：特五肖命中率 {hit10:.1%}，最大连空 {miss10}")
            print(f"近100期回测：特五肖命中率 {hit100:.1%}，最大连空 {miss100}")

if __name__ == "__main__":
    main()