#!/usr/bin/env python3
# special_only.py - 自适应选择最佳窗口（回测命中率优先）

import argparse
import json
from collections import Counter
from common import fetch_hk_records, get_zodiac_by_number, next_issue
from strategies_special import compute_special_five_score, get_special_number_recommendation

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

def evaluate_window(rows, window):
    """评估某个窗口在最近20期上的命中率（单窗口，不投票）"""
    rows_rev = list(reversed(rows))
    test_len = min(20, len(rows_rev) - 20)
    if test_len <= 0:
        return 0.0
    hits = 0
    for i in range(test_len):
        train = rows_rev[i+20:]
        actual = rows_rev[i]
        actual_zod = get_zodiac_by_number(actual["special_number"])
        scores = compute_special_five_score(train, window)
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        picks = [ranked[i][0] for i in range(5)]
        if actual_zod in picks:
            hits += 1
    return hits / test_len

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    rows = get_history_rows_as_list(limit=600)
    if not rows:
        print("数据获取失败")
        return

    if args.show:
        candidate_windows = [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
        best_window = None
        best_score = -1
        for w in candidate_windows:
            score = evaluate_window(rows, w)
            if score > best_score:
                best_score = score
                best_window = w
        # 使用最佳窗口进行预测
        scores = compute_special_five_score(rows, best_window)
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        picks = [ranked[i][0] for i in range(5)]
        sp, defenses = get_special_number_recommendation(rows, top_n=3, recent_window=30)
        latest_issue = rows[0]["issue_no"]
        pred_issue = next_issue(latest_issue)
        print(f"自适应选择窗口 {best_window} (该窗口最近20期回测命中率 {best_score*100:.1f}%)")
        print(f"预测期号: {pred_issue}")
        print(f"主推特别号: {sp:02d}")
        print(f"防守特别号: {' '.join(f'{n:02d}' for n in defenses[:2])}")
        print(f"特别生肖推荐(特五肖): {'、'.join(picks)}")
        # 重新计算近10期命中率（使用最佳窗口）
        rows_rev = list(reversed(rows))
        total = min(10, len(rows_rev) - 20)
        if total > 0:
            hits = 0
            miss_streak = 0
            max_miss = 0
            for i in range(total):
                train = rows_rev[i+20:]
                actual = rows_rev[i]
                actual_zod = get_zodiac_by_number(actual["special_number"])
                sc = compute_special_five_score(train, best_window)
                rkd = sorted(sc.items(), key=lambda x: -x[1])
                pks = [rkd[j][0] for j in range(5)]
                if actual_zod in pks:
                    hits += 1
                    miss_streak = 0
                else:
                    miss_streak += 1
                    max_miss = max(max_miss, miss_streak)
            print(f"\n近10期回测统计（使用窗口 {best_window}）:")
            print(f"特五肖: 命中率 {hits/total:.1%}, 最大连空 {max_miss}")

if __name__ == "__main__":
    main()