#!/usr/bin/env python3
# zodiac_main.py - 带诊断模式的一二三生肖预测

import argparse
import json
from collections import Counter, defaultdict
from common import fetch_hk_records, get_zodiac_by_number, next_issue
from strategies_zodiac import predict_strong_single, predict_strong_two, predict_strong_three_with_window, _zodiac_omission_map

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

def diagnose_windows(rows, lookback=40):
    """输出每个窗口对二生肖预测的独立命中率和最大连空"""
    rows_rev = list(reversed(rows))
    total = min(lookback, len(rows_rev) - 20)
    if total <= 0:
        print("数据不足，无法诊断")
        return
    windows = [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    window_stats = {}
    for w in windows:
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
            picks = predict_strong_two(train, {"two_recent_window": w, "two_special_boost": 3.0})
            if any(z in win_z for z in picks):
                hits += 1
                miss_streak = 0
            else:
                miss_streak += 1
                max_miss = max(max_miss, miss_streak)
        hit_rate = hits / total if total > 0 else 0
        window_stats[w] = {"hit_rate": hit_rate, "max_miss": max_miss}
    # 输出诊断信息
    print("\n=== 各窗口诊断（二生肖） ===")
    for w, stats in window_stats.items():
        print(f"窗口 {w:2d}: 命中率 {stats['hit_rate']*100:.1f}%，最大连空 {stats['max_miss']}")
    # 找出最佳窗口
    best = max(window_stats.items(), key=lambda x: x[1]["hit_rate"])
    print(f"\n最佳窗口: {best[0]} (命中率 {best[1]['hit_rate']*100:.1f}%，连空 {best[1]['max_miss']})")
    worst = min(window_stats.items(), key=lambda x: x[1]["hit_rate"])
    print(f"最差窗口: {worst[0]} (命中率 {worst[1]['hit_rate']*100:.1f}%，连空 {worst[1]['max_miss']})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true", help="显示预测")
    parser.add_argument("--diagnose", action="store_true", help="诊断各窗口性能")
    args = parser.parse_args()

    rows = get_history_rows_as_list(limit=600)
    if not rows:
        print("数据获取失败")
        return

    if args.diagnose:
        diagnose_windows(rows)
        return

    if args.show:
        windows = [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
        votes_single = Counter()
        votes_two = Counter()
        votes_three = Counter()
        for w in windows:
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
        # 复用之前的回测逻辑（略，可保留原回测代码）
        from zodiac_main_original import backtest_zodiac_stats  # 注意：这里需要合并，实际应将原回测函数复制到此处
        stats10 = backtest_zodiac_stats(rows, 10)
        if stats10:
            print(f"一生肖: 命中率 {stats10['single_hit_rate']:.1%}, 最大连空 {stats10['single_max_miss']}")
            print(f"二生肖: 命中率 {stats10['two_hit_rate']:.1%}, 最大连空 {stats10['two_max_miss']}")
            print(f"三生肖: 命中率 {stats10['three_hit_rate']:.1%}, 最大连空 {stats10['three_max_miss']}")

if __name__ == "__main__":
    main()
