#!/usr/bin/env python3
# special_only.py - 增强版特五肖（正码0.8+冷门+邻肖+对肖）

import argparse
import json
from collections import Counter
from common import fetch_hk_records_merged, get_zodiac_by_number, next_issue, ZODIAC_MAP, ZODIAC_PAIR

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
    # 正码权重0.8，特别号权重0.2
    for r in rows[:100]:
        for n in json.loads(r["numbers_json"]):
            cnt[get_zodiac_by_number(n)] += 0.8
        cnt[get_zodiac_by_number(r["special_number"])] += 0.2
    
    # 冷门补充：最近50期从未出现的生肖给予额外0.6分
    appeared = set()
    for r in rows[:50]:
        for n in json.loads(r["numbers_json"]):
            appeared.add(get_zodiac_by_number(n))
        appeared.add(get_zodiac_by_number(r["special_number"]))
    for z in ZODIAC_MAP:
        if z not in appeared:
            cnt[z] += 0.6
    
    # 邻肖加成：上期特别号的左右相邻生肖
    last_zod = get_zodiac_by_number(rows[0]["special_number"])
    zodiac_order = ["鼠","牛","虎","兔","龙","蛇","马","羊","猴","鸡","狗","猪"]
    try:
        idx = zodiac_order.index(last_zod)
        neighbors = [zodiac_order[(idx-1)%12], zodiac_order[(idx+1)%12]]
        for nz in neighbors:
            cnt[nz] += 0.2
    except:
        pass
    
    # 对肖加成
    pair = ZODIAC_PAIR.get(last_zod)
    if pair:
        cnt[pair] += 0.3
    
    # 最终取top5
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