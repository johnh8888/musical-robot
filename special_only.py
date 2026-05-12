#!/usr/bin/env python3
# special_only.py - 特别号数字 + 特五肖（完整改进版）

import argparse
import json
from collections import Counter
from common import fetch_hk_records_merged, get_zodiac_by_number, next_issue

# ---------- 特别号数字：热号+冷号+邻号 ----------
def predict_special_number_hybrid(rows, top_k=5):
    # 热号：近20期特别号频率
    recent_specials = [r["special_number"] for r in rows[:20]]
    hot_counter = Counter(recent_specials)
    hot_rank = [n for n, _ in hot_counter.most_common(5)]
    
    # 冷号：近50期遗漏最大的号码
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
    
    # 合并推荐
    combined = []
    for n in hot_rank:
        if n not in combined:
            combined.append(n)
    for n in neighbors:
        if n not in combined:
            combined.append(n)
    if coldest not in combined:
        combined.append(coldest)
    return combined[:top_k]

# ---------- 特五肖：基于正码生肖频率 ----------
def predict_five_zodiac(rows):
    zodiac_cnt = Counter()
    for r in rows[:100]:
        for n in json.loads(r["numbers_json"]):
            zodiac_cnt[get_zodiac_by_number(n)] += 1
    # 如果正码样本不足，补上特别号
    if len(zodiac_cnt) < 5:
        for r in rows[:100]:
            zodiac_cnt[get_zodiac_by_number(r["special_number"])] += 1
    return [z for z, _ in zodiac_cnt.most_common(5)]

# ---------- 回测函数 ----------
def backtest_special_number(rows, lookback):
    rows_rev = list(reversed(rows))
    total = min(lookback, len(rows_rev) - 20)
    hits = 0
    for i in range(total):
        train = rows_rev[i+20:]
        actual = rows_rev[i]["special_number"]
        preds = predict_special_number_hybrid(train, top_k=1)
        if actual == preds[0]:
            hits += 1
    return hits / total if total else 0

def backtest_five_zodiac(rows, lookback):
    rows_rev = list(reversed(rows))
    total = min(lookback, len(rows_rev) - 20)
    hits = 0
    miss_streak = 0
    max_miss = 0
    for i in range(total):
        train = rows_rev[i+20:]
        actual = rows_rev[i]["special_number"]
        actual_zod = get_zodiac_by_number(actual)
        preds = predict_five_zodiac(train)
        if actual_zod in preds:
            hits += 1
            miss_streak = 0
        else:
            miss_streak += 1
            max_miss = max(max_miss, miss_streak)
    return hits/total, max_miss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    records = fetch_hk_records_merged(limit=None, prefer_local=True)
    rows = []
    for r in records:
        rows.append({
            "numbers_json": json.dumps(r["numbers"]),
            "special_number": r["special_number"],
            "draw_date": r["draw_date"],
            "issue_no": r["issue_no"]
        })
    if not rows:
        print("数据获取失败")
        return

    if args.show:
        # 特别号数字
        top5 = predict_special_number_hybrid(rows, top_k=5)
        main_num = top5[0]
        defenses = top5[1:5]
        # 特五肖
        zodiac5 = predict_five_zodiac(rows)

        latest = rows[0]["issue_no"]
        pred_issue = next_issue(latest)
        print(f"预测期号: {pred_issue}")
        print(f"\n【特别号数字】")
        print(f"主推: {main_num:02d} (热号+冷号+邻号)")
        print(f"防守(5码): {' '.join(f'{n:02d}' for n in defenses[:5])}")
        print(f"\n【特五肖推荐】: {'、'.join(zodiac5)}")

        # 回测
        hit_num = backtest_special_number(rows, 100)
        print(f"\n特别号数字（主推）近100期命中率: {hit_num:.1%}")
        hit_zod, max_miss = backtest_five_zodiac(rows, 100)
        print(f"特五肖近100期命中率: {hit_zod:.1%}，最大连空: {max_miss}")

if __name__ == "__main__":
    main()