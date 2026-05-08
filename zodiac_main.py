#!/usr/bin/env python3
# zodiac_main.py - 每日自动诊断并选择最优窗口

import argparse
import json
from collections import Counter
from common import fetch_hk_records, get_zodiac_by_number, next_issue
from strategies_zodiac import (
    predict_strong_single,
    predict_strong_two,
    predict_strong_three_with_window,
    _zodiac_omission_map
)

ALL_WINDOWS = list(range(8, 33, 2))   # 候选窗口 8,10,12,...,32

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

def diagnose_best_windows(rows, lookback=40, top_k=6):
    """基于最近 lookback 期数据诊断各窗口性能，返回最优 top_k 个窗口"""
    rows_rev = list(reversed(rows))
    total = min(lookback, len(rows_rev) - 20)
    if total <= 0:
        return ALL_WINDOWS[:top_k]
    window_stats = {}
    for w in ALL_WINDOWS:
        hits = 0
        miss_streak = 0
        max_miss = 0
        for i in range(total):
            train = rows_rev[i+20:]
            if len(train) < 20:
                continue
            actual = rows_rev[i]
            win_main = json.loads(actual["numbers_json"])
            win_sp = actual["special_number"]
            win_z = {get_zodiac_by_number(n) for n in win_main}
            win_z.add(get_zodiac_by_number(win_sp))
            picks = predict_strong_two(train, {"two_recent_window": w, "two_special_boost": 3.0}, xgb_weight=0.6)
            hit = any(z in win_z for z in picks)
            if hit:
                hits += 1
                miss_streak = 0
            else:
                miss_streak += 1
                max_miss = max(max_miss, miss_streak)
        hit_rate = hits / total if total > 0 else 0
        window_stats[w] = {"hit_rate": hit_rate, "max_miss": max_miss}
    sorted_windows = sorted(window_stats.items(), key=lambda x: (-x[1]["hit_rate"], x[1]["max_miss"]))
    best_windows = [w for w, _ in sorted_windows[:top_k]]
    best_windows.sort()
    # 打印诊断结果（便于观察）
    print("=== 自动诊断各窗口性能（二生肖） ===")
    for w, stats in window_stats.items():
        print(f"窗口 {w:2d}: 命中率 {stats['hit_rate']*100:.1f}%，最大连空 {stats['max_miss']}")
    print(f"今日最优 {top_k} 个窗口: {best_windows}")
    return best_windows

def backtest_zodiac_stats(rows, lookback, windows):
    rows_rev = list(reversed(rows))
    total = min(lookback, len(rows_rev) - 20)
    if total <= 0:
        return None
    hits_single = 0
    hits_two = 0
    hits_three = 0
    miss_single = 0
    max_miss_single = 0
    miss_two = 0
    max_miss_two = 0
    miss_three = 0
    max_miss_three = 0
    for i in range(total):
        train = rows_rev[i+20:]
        if len(train) < 20:
            continue
        actual = rows_rev[i]
        win_main = json.loads(actual["numbers_json"])
        win_sp = actual["special_number"]
        win_z = {get_zodiac_by_number(n) for n in win_main}
        win_z.add(get_zodiac_by_number(win_sp))
        votes_single = Counter()
        votes_two = Counter()
        votes_three = Counter()
        for w in windows:
            pred_s = predict_strong_single(train, {"single_recent_window": w, "single_special_boost": 3.2}, xgb_weight=0.6)
            votes_single[pred_s] += 1
            picks_t = predict_strong_two(train, {"two_recent_window": w, "two_special_boost": 3.0}, xgb_weight=0.6)
            votes_two.update(picks_t)
            picks_th = predict_strong_three_with_window(train, w, xgb_weight=0.6)
            votes_three.update(picks_th)
        pred_single = votes_single.most_common(1)[0][0]
        pred_two = [z for z, _ in votes_two.most_common(2)]
        pred_three = [z for z, _ in votes_three.most_common(3)]
        # 连空保护
        if miss_three >= 2:
            omission = _zodiac_omission_map(train)
            if omission:
                coldest_two = sorted(omission, key=omission.get, reverse=True)[:2]
                new_three = [pred_three[0]] + [c for c in coldest_two if c != pred_three[0]]
                pred_three = new_three[:3]
        if miss_two >= 2:
            omission = _zodiac_omission_map(train)
            if omission:
                coldest = max(omission, key=omission.get)
                if coldest not in pred_two:
                    pred_two[-1] = coldest
        # 统计
        if pred_single in win_z:
            hits_single += 1
            miss_single = 0
        else:
            miss_single += 1
            max_miss_single = max(max_miss_single, miss_single)
        if any(z in win_z for z in pred_two):
            hits_two += 1
            miss_two = 0
        else:
            miss_two += 1
            max_miss_two = max(max_miss_two, miss_two)
        hit_cnt = sum(1 for z in pred_three if z in win_z)
        if hit_cnt >= 2:
            hits_three += 1
            miss_three = 0
        else:
            miss_three += 1
            max_miss_three = max(max_miss_three, miss_three)
    return {
        "single_hit_rate": hits_single / total,
        "single_max_miss": max_miss_single,
        "two_hit_rate": hits_two / total,
        "two_max_miss": max_miss_two,
        "three_hit_rate": hits_three / total,
        "three_max_miss": max_miss_three
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    rows = get_history_rows_as_list(limit=600)
    if not rows:
        print("数据获取失败")
        return

    if args.show:
        # 每天自动诊断并选择最优窗口
        best_windows = diagnose_best_windows(rows, lookback=40, top_k=6)
        votes_single = Counter()
        votes_two = Counter()
        votes_three = Counter()
        for w in best_windows:
            pred_s = predict_strong_single(rows, {"single_recent_window": w, "single_special_boost": 3.2}, xgb_weight=0.6)
            votes_single[pred_s] += 1
            picks_t = predict_strong_two(rows, {"two_recent_window": w, "two_special_boost": 3.0}, xgb_weight=0.6)
            votes_two.update(picks_t)
            picks_th = predict_strong_three_with_window(rows, w, xgb_weight=0.6)
            votes_three.update(picks_th)
        single = votes_single.most_common(1)[0][0]
        two = [z for z, _ in votes_two.most_common(2)]
        three = [z for z, _ in votes_three.most_common(3)]
        latest_issue = rows[0]["issue_no"]
        pred_issue = next_issue(latest_issue)
        print(f"预测期号: {pred_issue}")
        print(f"一生肖: {single}")
        print(f"二生肖: {'、'.join(two)}")
        print(f"三生肖: {'、'.join(three)}")
        print("\n近10期回测统计：")
        stats10 = backtest_zodiac_stats(rows, 10, best_windows)
        if stats10:
            print(f"一生肖: 命中率 {stats10['single_hit_rate']:.1%}, 最大连空 {stats10['single_max_miss']}")
            print(f"二生肖: 命中率 {stats10['two_hit_rate']:.1%}, 最大连空 {stats10['two_max_miss']}")
            print(f"三生肖: 命中率 {stats10['three_hit_rate']:.1%}, 最大连空 {stats10['three_max_miss']}")

if __name__ == "__main__":
    main()
