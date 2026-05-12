#!/usr/bin/env python3
# strategies_zodiac.py - 兼容两种数据格式

import json
from collections import Counter
from datetime import datetime
from common import get_zodiac_by_number, ZODIAC_MAP, ZODIAC_PAIR

def _row_numbers(row):
    """兼容两种字段：优先使用 numbers_json，否则使用 numbers"""
    if "numbers_json" in row:
        return json.loads(row["numbers_json"])
    elif "numbers" in row:
        return row["numbers"]
    else:
        raise KeyError("row must contain 'numbers_json' or 'numbers'")

def _row_special(row):
    if "special_number" in row:
        return row["special_number"]
    else:
        raise KeyError("row must contain 'special_number'")

def _zodiac_omission_map(rows):
    omission = {z: 0 for z in ZODIAC_MAP}
    found = {z: False for z in ZODIAC_MAP}
    for i, row in enumerate(rows):
        for n in _row_numbers(row):
            z = get_zodiac_by_number(n)
            if not found[z]:
                found[z] = True
                omission[z] = i
        z_sp = get_zodiac_by_number(_row_special(row))
        if not found[z_sp]:
            found[z_sp] = True
            omission[z_sp] = i
        if all(found.values()):
            break
    for z in ZODIAC_MAP:
        if not found[z]:
            omission[z] = len(rows)
    return omission

def _compute_zodiac_score_advanced(rows, recent_window=20, special_boost=2.0,
                                   weight_omission=0.3, weight_pair=0.5, weight_weekday=0.1):
    scores = {}
    # 星期几分布
    weekday_zodiac_count = {i: Counter() for i in range(7)}
    for r in rows:
        try:
            wd = datetime.strptime(r["draw_date"], "%Y-%m-%d").weekday()
        except:
            continue
        for n in _row_numbers(r):
            z = get_zodiac_by_number(n)
            weekday_zodiac_count[wd][z] += 1
        z_sp = get_zodiac_by_number(_row_special(r))
        weekday_zodiac_count[wd][z_sp] += 1
    today_wd = -1
    if rows:
        try:
            today_wd = datetime.strptime(rows[0]["draw_date"], "%Y-%m-%d").weekday()
        except:
            pass

    for z in ZODIAC_MAP:
        nums = ZODIAC_MAP[z]
        base = 0
        for r in rows[:recent_window]:
            for n in _row_numbers(r):
                if n in nums:
                    base += 1
        special_bonus = 0
        for r in rows[:recent_window]:
            if _row_special(r) in nums:
                special_bonus += special_boost
        omission = _zodiac_omission_map(rows).get(z, 0)
        omission_score = omission / (recent_window + 1)
        pair_bonus = 0
        if rows:
            last_sp = _row_special(rows[0])
            last_zod = get_zodiac_by_number(last_sp)
            pair = ZODIAC_PAIR.get(last_zod)
            if pair == z:
                pair_bonus = weight_pair
        weekday_bonus = 0
        if today_wd != -1 and weekday_zodiac_count[today_wd].total() > 0:
            prob = weekday_zodiac_count[today_wd].get(z, 0) / weekday_zodiac_count[today_wd].total()
            weekday_bonus = prob * weight_weekday * 10
        scores[z] = base + special_bonus + weight_omission * omission_score + pair_bonus + weekday_bonus
    return scores

# 以下预测函数保持不变，但使用 _row_numbers etc.
def predict_strong_single(rows, params, xgb_weight=0.0, xgb_model=None):
    window = params.get("single_recent_window", 20)
    boost = params.get("single_special_boost", 2.0)
    scores = _compute_zodiac_score_advanced(rows, recent_window=window, special_boost=boost)
    return max(scores, key=scores.get)

def predict_strong_two(rows, params, xgb_weight=0.0, xgb_model=None):
    window = params.get("two_recent_window", 20)
    boost = params.get("two_special_boost", 2.0)
    scores = _compute_zodiac_score_advanced(rows, recent_window=window, special_boost=boost)
    sorted_z = sorted(scores.items(), key=lambda x: -x[1])
    return [z for z, _ in sorted_z[:2]]

def predict_strong_three_with_window(rows, window, xgb_weight=0.0, xgb_model=None):
    scores = _compute_zodiac_score_advanced(rows, recent_window=window, special_boost=2.0)
    sorted_z = sorted(scores.items(), key=lambda x: -x[1])
    return [z for z, _ in sorted_z[:3]]

def get_hot_zodiac(rows, window=10):
    zodiacs = []
    for r in rows[:window]:
        for n in _row_numbers(r):
            zodiacs.append(get_zodiac_by_number(n))
        zodiacs.append(get_zodiac_by_number(_row_special(r)))
    cnt = Counter(zodiacs)
    return cnt.most_common(1)[0][0] if cnt else "鼠"

def get_cold_zodiac(rows, window=30):
    omission = {z: 0 for z in ZODIAC_MAP}
    found = {z: False for z in ZODIAC_MAP}
    for i, r in enumerate(rows[:window]):
        for n in _row_numbers(r):
            z = get_zodiac_by_number(n)
            if not found[z]:
                found[z] = True
                omission[z] = i
        z_sp = get_zodiac_by_number(_row_special(r))
        if not found[z_sp]:
            found[z_sp] = True
            omission[z_sp] = i
        if all(found.values()):
            break
    for z in ZODIAC_MAP:
        if not found[z]:
            omission[z] = len(rows[:window])
    return max(omission, key=omission.get)