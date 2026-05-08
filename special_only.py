#!/usr/bin/env python3
# special_only.py - 特五肖自适应预测

import argparse
import json
from collections import Counter
from common import fetch_hk_records, get_zodiac_by_number, next_issue
from strategies_special import predict_strong_five, get_special_number_recommendation

def get_history_rows_as_list(limit=600):
    records = fetch_hk_records(limit=limit)
    rows = []
    for r in records:
        rows.append({
            "numbers_json": json.dumps(r["numbers"]),
            "special_number": r["special_number"],
            "draw_date": r["draw_date"],
            "issue_no": r["issue_no"]
        })
    return rows

def backtest_special_zodiac(rows, lookback):
    rows_rev = list(reversed(rows))
    total = min(lookback, len(rows_rev) - 20)
    if total <= 0:
        return None
    hits = 0
    miss_streak = 0
    max_miss = 0
    # 不再需要历史命中率，此处只计算最终命中率
    for i in range(total):
        train = rows_rev[i+20:]
        if len(train) < 20:
            continue
        actual = rows_rev[i]
        actual_zod = get_zodiac_by_number(actual["special_number"])
        # 预测时传入最近10期命中率（此处无法获取，回测中为None）
        picks = predict_strong_five(train, {}, miss_streak, recent_hit_rate=None)
        if actual_zod in picks:
            hits += 1
            miss_streak = 0
        else:
            miss_streak += 1
            max_miss = max(max_miss, miss_streak)
    return {"hit_rate": hits / total, "max_miss": max_miss}

def get_recent_hit_rate(rows):
    """计算最近10期（使用当前数据）特五肖命中率，用于自适应"""
    rows_rev = list(reversed(rows))
    total = min(10, len(rows_rev) - 20)
    if total <= 0:
        return 0.5
    hits = 0
    miss_streak = 0
    for i in range(total):
        train = rows_rev[i+20:]
        if len(train) < 20:
            continue
        actual = rows_rev[i]
        actual_zod = get_zodiac_by_number(actual["special_number"])
        picks = predict_strong_five(train, {}, miss_streak, recent_hit_rate=None)  # 注意：递归调用可能有问题，这里简化
        # 为简化，直接使用最近10期的规则评分预测（不传入recent_hit_rate）
        picks = predict_strong_five(train, {}, miss_streak, recent_hit_rate=None)
        if actual_zod in picks:
            hits += 1
            miss_streak = 0
        else:
            miss_streak += 1
    return hits / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    rows = get_history_rows_as_list(limit=600)
    if not rows:
        print("数据获取失败")
        return

    if args.show:
        # 计算最近10期命中率（用于自适应切换）
        recent_rate = get_recent_hit_rate(rows)
        print(f"最近10期特五肖命中率: {recent_rate*100:.1f}%")
        # 预测时传入recent_rate
        miss_streak = 0
        picks = predict_strong_five(rows, {}, miss_streak, recent_hit_rate=recent_rate)
        sp, defenses = get_special_number_recommendation(rows, top_n=3, recent_window=30)
        latest_issue = rows[0]["issue_no"]
        pred_issue = next_issue(latest_issue)
        print(f"预测期号: {pred_issue}")
        print(f"主推特别号: {sp:02d}")
        print(f"防守特别号: {' '.join(f'{n:02d}' for n in defenses[:2])}")
        print(f"特别生肖推荐(特五肖): {'、'.join(picks)}")
        print("\n近10期回测统计：")
        stats10 = backtest_special_zodiac(rows, 10)
        if stats10:
            print(f"特五肖: 命中率 {stats10['hit_rate']:.1%}, 最大连空 {stats10['max_miss']}")

if __name__ == "__main__":
    main()
