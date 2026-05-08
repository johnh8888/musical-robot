#!/usr/bin/env python3
# special_only.py - 带诊断模式的特五肖预测

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

def diagnose_windows(rows, lookback=40):
    """输出每个窗口对特五肖预测的独立命中率和最大连空"""
    rows_rev = list(reversed(rows))
    total = min(lookback, len(rows_rev) - 20)
    if total <= 0:
        print("数据不足，无法诊断")
        return
    windows = [12, 16, 20, 24, 28, 32]   # 当前使用的6窗口
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
            actual_zod = get_zodiac_by_number(actual["special_number"])
            # 直接使用单个窗口预测（不投票）
            scores = _compute_special_five_score_single_window(train, w)   # 需要实现单窗口评分函数
            ranked = sorted(scores.items(), key=lambda x: -x[1])
            picks = [ranked[i][0] for i in range(5)]
            if actual_zod in picks:
                hits += 1
                miss_streak = 0
            else:
                miss_streak += 1
                max_miss = max(max_miss, miss_streak)
        hit_rate = hits / total if total > 0 else 0
        window_stats[w] = {"hit_rate": hit_rate, "max_miss": max_miss}
    print("\n=== 各窗口诊断（特五肖） ===")
    for w, stats in window_stats.items():
        print(f"窗口 {w:2d}: 命中率 {stats['hit_rate']*100:.1f}%，最大连空 {stats['max_miss']}")
    best = max(window_stats.items(), key=lambda x: x[1]["hit_rate"])
    print(f"\n最佳窗口: {best[0]} (命中率 {best[1]['hit_rate']*100:.1f}%，连空 {best[1]['max_miss']})")
    worst = min(window_stats.items(), key=lambda x: x[1]["hit_rate"])
    print(f"最差窗口: {worst[0]} (命中率 {worst[1]['hit_rate']*100:.1f}%，连空 {worst[1]['max_miss']})")

def _compute_special_five_score_single_window(rows, recent_window):
    """单个窗口的特五肖评分（不投票）"""
    from strategies_special import _compute_special_five_score
    return _compute_special_five_score(rows, recent_window)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
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
        miss_streak = 0
        picks = predict_strong_five(rows, {"four_recent_special_window": 20}, miss_streak)
        sp, defenses = get_special_number_recommendation(rows, top_n=3, recent_window=30)
        latest_issue = rows[0]["issue_no"]
        pred_issue = next_issue(latest_issue)
        print(f"预测期号: {pred_issue}")
        print(f"主推特别号: {sp:02d}")
        print(f"防守特别号: {' '.join(f'{n:02d}' for n in defenses[:2])}")
        print(f"特别生肖推荐(特五肖): {'、'.join(picks)}")
        print("\n近10期回测统计：")
        # 回测逻辑（略，可复用之前的 backtest_special_zodiac）
        stats10 = backtest_special_zodiac(rows, 10)
        if stats10:
            print(f"特五肖: 命中率 {stats10['hit_rate']:.1%}, 最大连空 {stats10['max_miss']}")

if __name__ == "__main__":
    main()
