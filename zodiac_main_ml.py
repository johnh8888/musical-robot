#!/usr/bin/env python3
# zodiac_main_ml.py - 机器学习版生肖预测（含全部模式回测）

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

def backtest_zodiac(rows, lookback, k, strict_three=False):
    """
    回测生肖模式
    k: 预测个数 (1,2,3)
    strict_three: 仅当 k==3 时生效，True表示需要命中至少2个才算中，False表示命中1个就算中
    """
    rows_rev = list(reversed(rows))
    total = min(lookback, len(rows_rev) - 20)
    if total <= 0:
        return None, None
    hits = 0
    miss_streak = 0
    max_miss = 0
    for i in range(total):
        train = rows_rev[i+20:]
        actual = rows_rev[i]
        win_main = json.loads(actual["numbers_json"])
        win_sp = actual["special_number"]
        win_z = {get_zodiac_by_number(n) for n in win_main}
        win_z.add(get_zodiac_by_number(win_sp))
        pred = predict_top_k_zodiac(train, k=k)
        if k == 3 and strict_three:
            # 三生肖必须命中至少2个
            hit_cnt = sum(1 for z in pred if z in win_z)
            if hit_cnt >= 2:
                hits += 1
                miss_streak = 0
            else:
                miss_streak += 1
                max_miss = max(max_miss, miss_streak)
        else:
            # 一生肖或二生肖：至少命中1个
            if any(z in win_z for z in pred):
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
        two = predict_top_k_zodiac(rows, k=2)
        three = predict_top_k_zodiac(rows, k=3)
        single = three[0]
        latest = rows[0]["issue_no"]
        print(f"预测期号: {next_issue(latest)}")
        print(f"一生肖 (ML): {single}")
        print(f"二生肖 (ML): {'、'.join(two)}")
        print(f"三生肖 (ML): {'、'.join(three)}")

        # 回测一生肖
        hit10_s, miss10_s = backtest_zodiac(rows, 10, k=1)
        hit100_s, miss100_s = backtest_zodiac(rows, 100, k=1)
        print(f"\n近10期回测（一生肖）: 命中率 {hit10_s:.1%}, 最大连空 {miss10_s}")
        print(f"近100期回测（一生肖）: 命中率 {hit100_s:.1%}, 最大连空 {miss100_s}")

        # 回测二生肖
        hit10_t, miss10_t = backtest_zodiac(rows, 10, k=2)
        hit100_t, miss100_t = backtest_zodiac(rows, 100, k=2)
        print(f"\n近10期回测（二生肖）: 命中率 {hit10_t:.1%}, 最大连空 {miss10_t}")
        print(f"近100期回测（二生肖）: 命中率 {hit100_t:.1%}, 最大连空 {miss100_t}")

        # 回测三生肖（严格：中2个）
        hit10_th, miss10_th = backtest_zodiac(rows, 10, k=3, strict_three=True)
        hit100_th, miss100_th = backtest_zodiac(rows, 100, k=3, strict_three=True)
        print(f"\n近10期回测（三生肖，需中2个）: 命中率 {hit10_th:.1%}, 最大连空 {miss10_th}")
        print(f"近100期回测（三生肖，需中2个）: 命中率 {hit100_th:.1%}, 最大连空 {miss100_th}")

if __name__ == "__main__":
    main()