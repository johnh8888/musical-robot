#!/usr/bin/env python3
# special_only_ml.py - 机器学习版特别号预测（含回测）

import argparse
import json
from common import fetch_hk_records_merged, next_issue, get_zodiac_by_number
from ml_predict import predict_special_number_ml, predict_five_zodiac_ml

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

def backtest_special_number(rows, lookback):
    """回测特别号主推命中率"""
    rows_rev = list(reversed(rows))
    total = min(lookback, len(rows_rev) - 20)
    hits = 0
    for i in range(total):
        train = rows_rev[i+20:]
        actual = rows_rev[i]["special_number"]
        pred = predict_special_number_ml(train, top_k=1)
        if actual == pred[0]:
            hits += 1
    return hits / total if total > 0 else 0

def backtest_five_zodiac(rows, lookback):
    """回测特五肖命中率和最大连空"""
    rows_rev = list(reversed(rows))
    total = min(lookback, len(rows_rev) - 20)
    hits = 0
    miss_streak = 0
    max_miss = 0
    for i in range(total):
        train = rows_rev[i+20:]
        actual = rows_rev[i]["special_number"]
        actual_zod = get_zodiac_by_number(actual)
        pred = predict_five_zodiac_ml(train)
        if actual_zod in pred:
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
    rows = get_history_rows_as_list(limit=None)
    if not rows:
        print("数据获取失败")
        return
    if args.show:
        # 当前预测
        top5 = predict_special_number_ml(rows, top_k=5)
        zodiac5 = predict_five_zodiac_ml(rows)
        latest = rows[0]["issue_no"]
        pred_issue = next_issue(latest)
        print(f"预测期号: {pred_issue}")
        print(f"\n【特别号数字 (XGBoost)】")
        print(f"主推: {top5[0]:02d}")
        print(f"防守(5码): {' '.join(f'{n:02d}' for n in top5[1:5])}")
        print(f"\n【特五肖 (正码频率)】: {'、'.join(zodiac5)}")

        # 回测特别号主推
        hit_num10 = backtest_special_number(rows, 10)
        hit_num100 = backtest_special_number(rows, 100)
        print(f"\n近10期回测（特别号主推）: 命中率 {hit_num10:.1%}")
        print(f"近100期回测（特别号主推）: 命中率 {hit_num100:.1%}")

        # 回测特五肖
        hit_five10, miss_five10 = backtest_five_zodiac(rows, 10)
        hit_five100, miss_five100 = backtest_five_zodiac(rows, 100)
        print(f"\n近10期回测（特五肖）: 命中率 {hit_five10:.1%}, 最大连空 {miss_five10}")
        print(f"近100期回测（特五肖）: 命中率 {hit_five100:.1%}, 最大连空 {miss_five100}")

if __name__ == "__main__":
    main()