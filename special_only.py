#!/usr/bin/env python3
# special_only.py - 特五肖 + 特别号数字（频率+热冷+邻号）

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
    """特五肖：正码生肖频率（近100期）"""
    cnt = Counter()
    for r in rows[:100]:
        nums = json.loads(r["numbers_json"])
        for n in nums:
            cnt[get_zodiac_by_number(n)] += 1
    # 如果正码样本不足，补充特别号
    if len(cnt) < 5:
        for r in rows[:100]:
            cnt[get_zodiac_by_number(r["special_number"])] += 1
    return [z for z, _ in cnt.most_common(5)]

def predict_special_number(rows, top_k=5):
    """特别号数字：热号（近20期） + 冷号（遗漏最大） + 邻号（上期±1,±2）"""
    recent_sp = [r["special_number"] for r in rows[:20]]
    hot = Counter(recent_sp).most_common(5)
    hot_nums = [n for n, _ in hot]
    
    # 冷号：遗漏最大的号码（近50期）
    omission = {n: 0 for n in range(1, 50)}
    for i, r in enumerate(rows[:50]):
        sp = r["special_number"]
        for n in range(1, 50):
            if n == sp:
                omission[n] = 0
            else:
                omission[n] += 1
    coldest = max(omission, key=omission.get)
    
    # 邻号：上期特别号 ±1, ±2
    last_sp = rows[0]["special_number"]
    neighbors = set([last_sp-2, last_sp-1, last_sp+1, last_sp+2]) & set(range(1,50))
    
    # 合并去重
    combined = []
    for n in hot_nums:
        if n not in combined:
            combined.append(n)
    for n in neighbors:
        if n not in combined:
            combined.append(n)
    if coldest not in combined:
        combined.append(coldest)
    return combined[:top_k]

def backtest_five_zodiac(rows, lookback):
    rows_rev = list(reversed(rows))
    total = min(lookback, len(rows_rev) - 20)
    hits = 0
    miss = 0
    max_miss = 0
    for i in range(total):
        train = rows_rev[i+20:]
        actual = rows_rev[i]["special_number"]
        actual_zod = get_zodiac_by_number(actual)
        pred = predict_five_zodiac(train)
        if actual_zod in pred:
            hits += 1
            miss = 0
        else:
            miss += 1
            max_miss = max(max_miss, miss)
    return hits/total, max_miss

def backtest_special_number(rows, lookback):
    rows_rev = list(reversed(rows))
    total = min(lookback, len(rows_rev) - 20)
    hits = 0
    for i in range(total):
        train = rows_rev[i+20:]
        actual = rows_rev[i]["special_number"]
        pred = predict_special_number(train, top_k=1)
        if actual == pred[0]:
            hits += 1
    return hits/total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    rows = get_history_rows_as_list(limit=None)
    if not rows:
        print("数据获取失败")
        return
    if args.show:
        top5 = predict_special_number(rows, top_k=5)
        zodiac5 = predict_five_zodiac(rows)
        latest = rows[0]["issue_no"]
        pred_issue = next_issue(latest)
        print(f"预测期号: {pred_issue}")
        print(f"\n【特别号数字】")
        print(f"主推: {top5[0]:02d}")
        print(f"防守(5码): {' '.join(f'{n:02d}' for n in top5[1:5])}")
        print(f"\n【特五肖】: {'、'.join(zodiac5)}")
        
        # 回测
        hit_num10 = backtest_special_number(rows, 10)
        hit_num100 = backtest_special_number(rows, 100)
        print(f"\n近10期回测（特别号主推）: 命中率 {hit_num10:.1%}")
        print(f"近100期回测（特别号主推）: 命中率 {hit_num100:.1%}")
        
        hit_zod10, miss_zod10 = backtest_five_zodiac(rows, 10)
        hit_zod100, miss_zod100 = backtest_five_zodiac(rows, 100)
        print(f"\n近10期回测（特五肖）: 命中率 {hit_zod10:.1%}, 最大连空 {miss_zod10}")
        print(f"近100期回测（特五肖）: 命中率 {hit_zod100:.1%}, 最大连空 {miss_zod100}")

if __name__ == "__main__":
    main()