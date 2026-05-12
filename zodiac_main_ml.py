#!/usr/bin/env python3
# zodiac_main_ml.py - 机器学习版生肖预测（带二生肖回测）

import argparse
import json
from common import fetch_hk_records_merged, next_issue, get_zodiac_by_number
from ml_predict import predict_top_k_zodiac

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

def backtest_ml(rows, lookback, k=2):
    """回测机器学习模型命中率和最大连空（注意：模型固定，有轻微未来信息）"""
    rows_rev = list(reversed(rows))
    total = min(lookback, len(rows_rev) - 20)
    if total <= 0:
        return None, None
    hits = 0
    miss_streak = 0
    max_miss = 0
    for i in range(total):
        train = rows_rev[i+20:]  # 使用 i+20 之后的所有历史（固定模型）
        actual = rows_rev[i]
        win_main = json.loads(actual["numbers_json"])
        win_sp = actual["special_number"]
        win_z = {get_zodiac_by_number(n) for n in win_main}
        win_z.add(get_zodiac_by_number(win_sp))
        pred_z = predict_top_k_zodiac(train, k=k)
        if any(z in win_z for z in pred_z):
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
        two = predict_top_k_zodiac(rows, k=2)
        three = predict_top_k_zodiac(rows, k=3)
        single = three[0]
        latest = rows[0]["issue_no"]
        print(f"预测期号: {next_issue(latest)}")
        print(f"一生肖 (ML): {single}")
        print(f"二生肖 (ML): {'、'.join(two)}")
        print(f"三生肖 (ML): {'、'.join(three)}")

        # 二生肖回测
        hit10, miss10 = backtest_ml(rows, lookback=10, k=2)
        hit100, miss100 = backtest_ml(rows, lookback=100, k=2)
        if hit10 is not None:
            print(f"\n近10期回测（二生肖）: 命中率 {hit10:.1%}, 最大连空 {miss10}")
            print(f"近100期回测（二生肖）: 命中率 {hit100:.1%}, 最大连空 {miss100}")

if __name__ == "__main__":
    main()