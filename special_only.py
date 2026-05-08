#!/usr/bin/env python3
# special_only.py - 全自动版（自动诊断并缓存最优窗口）

import argparse
import json
from collections import Counter
from common import fetch_hk_records, get_zodiac_by_number, next_issue
from strategies_special import get_special_number_recommendation, _compute_special_five_score

ALL_WINDOWS = [12, 16, 20, 24, 28, 32]
DEFAULT_WINDOWS = [12, 16, 20, 32]

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
    rows_rev = list(reversed(rows))
    total = min(lookback, len(rows_rev) - 20)
    if total <= 0:
        return DEFAULT_WINDOWS
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
            actual_zod = get_zodiac_by_number(actual["special_number"])
            # 单窗口评分
            scores = _compute_special_five_score(train, w)
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
    sorted_windows = sorted(window_stats.items(), key=lambda x: (-x[1]["hit_rate"], x[1]["max_miss"]))
    best_windows = [w for w, _ in sorted_windows[:4]]
    best_windows.sort()
    return best_windows

def load_or_diagnose_windows(rows):
    best_windows_file = "best_special_windows.json"
    try:
        with open(best_windows_file, "r") as f:
            windows = json.load(f)
            print(f"已加载最优窗口: {windows}")
            return windows
    except FileNotFoundError:
        print("未找到窗口缓存文件，正在自动诊断...")
        windows = diagnose_windows(rows)
        with open(best_windows_file, "w") as f:
            json.dump(windows, f)
        print(f"已诊断并保存最优窗口: {windows}")
        return windows

def backtest_special_zodiac(rows, lookback, windows):
    rows_rev = list(reversed(rows))
    total = min(lookback, len(rows_rev) - 20)
    if total <= 0:
        return None
    hits = 0
    miss_streak = 0
    max_miss = 0
    for i in range(total):
        train = rows_rev[i+20:]
        if len(train) < 20:
            continue
        actual = rows_rev[i]
        actual_zod = get_zodiac_by_number(actual["special_number"])
        votes = Counter()
        for w in windows:
            scores = _compute_special_five_score(train, w)
            ranked = sorted(scores.items(), key=lambda x: -x[1])
            picks = [ranked[i][0] for i in range(5)]
            votes.update(picks)
        final_picks = [z for z, _ in votes.most_common(5)]
        if actual_zod in final_picks:
            hits += 1
            miss_streak = 0
        else:
            miss_streak += 1
            max_miss = max(max_miss, miss_streak)
    return {"hit_rate": hits / total, "max_miss": max_miss}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    rows = get_history_rows_as_list(limit=600)
    if not rows:
        print("数据获取失败")
        return

    if args.show:
        windows = load_or_diagnose_windows(rows)
        votes = Counter()
        for w in windows:
            scores = _compute_special_five_score(rows, w)
            ranked = sorted(scores.items(), key=lambda x: -x[1])
            picks = [ranked[i][0] for i in range(5)]
            votes.update(picks)
        final_picks = [z for z, _ in votes.most_common(5)]
        sp, defenses = get_special_number_recommendation(rows, top_n=3, recent_window=30)
        latest_issue = rows[0]["issue_no"]
        pred_issue = next_issue(latest_issue)
        print(f"预测期号: {pred_issue}")
        print(f"主推特别号: {sp:02d}")
        print(f"防守特别号: {' '.join(f'{n:02d}' for n in defenses[:2])}")
        print(f"特别生肖推荐(特五肖): {'、'.join(final_picks)}")
        print("\n近10期回测统计：")
        stats10 = backtest_special_zodiac(rows, 10, windows)
        if stats10:
            print(f"特五肖: 命中率 {stats10['hit_rate']:.1%}, 最大连空 {stats10['max_miss']}")

if __name__ == "__main__":
    main()
