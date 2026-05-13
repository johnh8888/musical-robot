#!/usr/bin/env python3
# special_only.py - 特五肖（正码+特别号频率+遗漏加分）

import argparse
import json
from collections import Counter
from common import fetch_hk_records_merged, get_zodiac_by_number, next_issue, ZODIAC_MAP

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

def predict_five_zodiac(rows):
    cnt = Counter()
    for r in rows[:100]:
        for n in json.loads(r["numbers_json"]):
            cnt[get_zodiac_by_number(n)] += 0.7
        cnt[get_zodiac_by_number(r["special_number"])] += 0.3
    # 遗漏加分：最近50期从未出现的生肖加0.5
    appeared = set()
    for r in rows[:50]:
        for n in json.loads(r["numbers_json"]):
            appeared.add(get_zodiac_by_number(n))
        appeared.add(get_zodiac_by_number(r["special_number"]))
    for z in ZODIAC_MAP:
        if z not in appeared:
            cnt[z] += 0.5
    return [z for z, _ in cnt.most_common(5)]

def backtest_five_zodiac(rows, lookback):
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
        pred = predict_five_zodiac(train)
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
        zodiac5 = predict_five_zodiac(rows)
        latest = rows[0]["issue_no"]
        pred_issue = next_issue(latest)
        print(f"预测期号: {pred_issue}")
        print(f"\n【特五肖推荐】: {'、'.join(zodiac5)}")
        hit10, miss10 = backtest_five_zodiac(rows, 10)
        hit100, miss100 = backtest_five_zodiac(rows, 100)
        if hit10 is not None:
            print(f"\n近10期回测：特五肖命中率 {hit10:.1%}，最大连空 {miss10}")
            print(f"近100期回测：特五肖命中率 {hit100:.1%}，最大连空 {miss100}")

if __name__ == "__main__":
    main() 