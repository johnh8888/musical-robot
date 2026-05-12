#!/usr/bin/env python3
# special_only.py - 特别号数字 + 特五肖（频率升级版）

import argparse
import json
from collections import Counter
from common import fetch_hk_records_merged, get_zodiac_by_number, next_issue, ZODIAC_MAP

# ---------- 特别号数字预测（基于频率+遗漏）----------
def predict_special_number_freq(rows, top_k=5):
    specials = [r["special_number"] for r in rows]
    # 统计近100期频率
    recent = specials[:100]
    counter = Counter(recent)
    # 加入遗漏加权：遗漏越大越有可能出（冷号回补）
    omission_weights = {}
    for n in range(1, 50):
        omission = 0
        for sp in specials:
            if sp != n:
                omission += 1
            else:
                break
        omission_weights[n] = min(omission / 30, 1.0)  # 归一化
    # 综合得分 = 频率分(0-1) + 0.3*遗漏分
    scores = {}
    for n in range(1, 50):
        freq_score = counter.get(n, 0) / max(counter.values()) if counter else 0
        scores[n] = freq_score + 0.3 * omission_weights[n]
    sorted_nums = sorted(scores.items(), key=lambda x: -x[1])
    return [n for n, _ in sorted_nums[:top_k]]

# ---------- 特五肖预测（基于频率+趋势）----------
def predict_five_zodiac(rows):
    # 取近50期特别号生肖分布
    special_zodiacs = []
    for r in rows[:50]:
        special_zodiacs.append(get_zodiac_by_number(r["special_number"]))
    counter = Counter(special_zodiacs)
    # 权重：出现次数
    total = sum(counter.values())
    probs = {z: count/total for z, count in counter.items()}
    sorted_zod = sorted(probs.items(), key=lambda x: -x[1])
    return [z for z, _ in sorted_zod[:5]]

def backtest_special_zodiac(rows, lookback):
    rows_rev = list(reversed(rows))
    total = min(lookback, len(rows_rev) - 20)
    hits = 0
    miss_streak = 0
    max_miss = 0
    for i in range(total):
        train = rows_rev[i+20:]
        actual = rows_rev[i]
        actual_zod = get_zodiac_by_number(actual["special_number"])
        picks = predict_five_zodiac(train)
        if actual_zod in picks:
            hits += 1
            miss_streak = 0
        else:
            miss_streak += 1
            max_miss = max(max_miss, miss_streak)
    return hits/total, max_miss

def backtest_special_number(rows, lookback):
    rows_rev = list(reversed(rows))
    total = min(lookback, len(rows_rev) - 20)
    hits = 0
    for i in range(total):
        train = rows_rev[i+20:]
        actual = rows_rev[i]["special_number"]
        preds = predict_special_number_freq(train, top_k=1)
        if actual == preds[0]:
            hits += 1
    return hits/total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    rows = []
    records = fetch_hk_records_merged(limit=None, prefer_local=True)
    for r in records:
        rows.append({
            "numbers_json": json.dumps(r["numbers"]),
            "special_number": r["special_number"],
            "draw_date": r["draw_date"],
            "issue_no": r["issue_no"]
        })
    if not rows:
        print("❌ 数据获取失败")
        return

    if args.show:
        # 特别号数字
        top5 = predict_special_number_freq(rows, top_k=5)
        main_num = top5[0]
        defenses = top5[1:5]
        # 特五肖
        zodiac5 = predict_five_zodiac(rows)

        latest = rows[0]["issue_no"]
        pred_issue = next_issue(latest)
        print(f"预测期号: {pred_issue}")
        print(f"\n【特别号数字】")
        print(f"主推: {main_num:02d} (基于频率+遗漏)")
        print(f"防守(5码): {' '.join(f'{n:02d}' for n in defenses[:5])}")
        print(f"\n【特五肖推荐】: {'、'.join(zodiac5)}")

        # 回测
        hit_rate_num = backtest_special_number(rows, 100)
        print(f"\n特别号数字（主推）近100期命中率: {hit_rate_num:.1%}")

        hit_rate_zod, max_miss_zod = backtest_special_zodiac(rows, 100)
        print(f"特五肖近100期命中率: {hit_rate_zod:.1%}，最大连空: {max_miss_zod}")

if __name__ == "__main__":
    main()
