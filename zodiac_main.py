#!/usr/bin/env python3
# zodiac_main.py - 一二三生肖最终版

import argparse
import json
from collections import Counter
from common import fetch_hk_records_merged, get_zodiac_by_number, next_issue
from strategies_zodiac import (
    predict_strong_single, predict_strong_two, predict_strong_three_with_window,
    _zodiac_omission_map
)

OPTIMAL_WINDOWS = [8,10,12,18,20,30]
XGB_WEIGHT = 0.0

def get_history_rows_as_list(limit=1000):
    records = fetch_hk_records_merged(limit=limit)
    rows = []
    for r in records:
        rows.append({
            "numbers_json": json.dumps(r["numbers"]),
            "special_number": r["special_number"],
            "draw_date": r["draw_date"],
            "issue_no": r["issue_no"]
        })
    return rows

def backtest_zodiac_stats(rows, lookback):
    rows_rev = list(reversed(rows))
    total = min(lookback, len(rows_rev)-20)
    if total<=0: return None
    hits_single = hits_two = hits_three = 0
    miss_single = max_miss_single = 0
    miss_two = max_miss_two = 0
    miss_three = max_miss_three = 0
    for i in range(total):
        train = rows_rev[i+20:]
        if len(train)<20: continue
        actual = rows_rev[i]
        win_main = json.loads(actual["numbers_json"])
        win_sp = actual["special_number"]
        win_z = {get_zodiac_by_number(n) for n in win_main}
        win_z.add(get_zodiac_by_number(win_sp))
        votes_s = Counter(); votes_t = Counter(); votes_th = Counter()
        for w in OPTIMAL_WINDOWS:
            votes_s[predict_strong_single(train, {"single_recent_window":w,"single_special_boost":3.2}, xgb_weight=XGB_WEIGHT)] += 1
            for p in predict_strong_two(train, {"two_recent_window":w,"two_special_boost":3.0}, xgb_weight=XGB_WEIGHT):
                votes_t[p] += 1
            for p in predict_strong_three_with_window(train, w, xgb_weight=XGB_WEIGHT):
                votes_th[p] += 1
        pred_s = votes_s.most_common(1)[0][0]
        pred_t = [z for z,_ in votes_t.most_common(2)]
        pred_th = [z for z,_ in votes_th.most_common(3)]
        # 连空保护（略）
        if miss_three>=2:
            omission = _zodiac_omission_map(train)
            if omission:
                coldest2 = sorted(omission, key=omission.get, reverse=True)[:2]
                pred_th = [pred_th[0]] + [c for c in coldest2 if c!=pred_th[0]]
        if miss_two>=2:
            omission = _zodiac_omission_map(train)
            if omission:
                coldest = max(omission, key=omission.get)
                if coldest not in pred_t: pred_t[-1]=coldest
        # 统计
        if pred_s in win_z: hits_s+=1; miss_s=0
        else: miss_s+=1; max_miss_s = max(max_miss_s, miss_s)
        if any(z in win_z for z in pred_t): hits_t+=1; miss_t=0
        else: miss_t+=1; max_miss_t = max(max_miss_t, miss_t)
        hit_cnt = sum(1 for z in pred_th if z in win_z)
        if hit_cnt>=2: hits_th+=1; miss_th=0
        else: miss_th+=1; max_miss_th = max(max_miss_th, miss_th)
    return {
        "single_hit_rate": hits_s/total, "single_max_miss": max_miss_s,
        "two_hit_rate": hits_t/total, "two_max_miss": max_miss_t,
        "three_hit_rate": hits_th/total, "three_max_miss": max_miss_th
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    rows = get_history_rows_as_list(limit=1000)
    if not rows: print("数据获取失败"); return
    if args.show:
        print(f"使用窗口 {OPTIMAL_WINDOWS}, XGB权重 {XGB_WEIGHT}")
        votes_s = Counter(); votes_t = Counter(); votes_th = Counter()
        for w in OPTIMAL_WINDOWS:
            votes_s[predict_strong_single(rows, {"single_recent_window":w,"single_special_boost":3.2}, xgb_weight=XGB_WEIGHT)] += 1
            for p in predict_strong_two(rows, {"two_recent_window":w,"two_special_boost":3.0}, xgb_weight=XGB_WEIGHT):
                votes_t[p] += 1
            for p in predict_strong_three_with_window(rows, w, xgb_weight=XGB_WEIGHT):
                votes_th[p] += 1
        single = votes_s.most_common(1)[0][0]
        two = [z for z,_ in votes_t.most_common(2)]
        three = [z for z,_ in votes_th.most_common(3)]
        latest = rows[0]["issue_no"]
        print(f"预测期号: {next_issue(latest)}")
        print(f"一生肖: {single}")
        print(f"二生肖: {'、'.join(two)}")
        print(f"三生肖: {'、'.join(three)}")
        stats10 = backtest_zodiac_stats(rows, 10)
        if stats10:
            print(f"\n近10期回测：一生肖 {stats10['single_hit_rate']:.1%} 连空{stats10['single_max_miss']}")
            print(f"二生肖 {stats10['two_hit_rate']:.1%} 连空{stats10['two_max_miss']}")
            print(f"三生肖 {stats10['three_hit_rate']:.1%} 连空{stats10['three_max_miss']}")

if __name__ == "__main__":
    main()
