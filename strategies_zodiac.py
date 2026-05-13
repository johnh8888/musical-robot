#!/usr/bin/env python3
# strategies_zodiac.py - 增强规则策略（含 get_trend_zodiac）

import json
from collections import Counter
from datetime import datetime
from common import get_zodiac_by_number, ZODIAC_MAP, ZODIAC_PAIR

def _row_numbers(row):
    return json.loads(row["numbers_json"])

def _row_special(row):
    return row["special_number"]

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

def _compute_zodiac_score(rows, recent_window=20, special_boost=2.0):
    scores = {}
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
        pair = ZODIAC_PAIR.get(z)
        if pair:
            pair_bonus = 0
            pair_nums = ZODIAC_MAP.get(pair, [])
            for r in rows[:recent_window]:
                for n in _row_numbers(r):
                    if n in pair_nums:
                        pair_bonus += 0.5
                if _row_special(r) in pair_nums:
                    pair_bonus += 0.5
            base += pair_bonus
        scores[z] = base + special_bonus
    return scores

def predict_strong_single(rows, params, xgb_weight=0.0, xgb_model=None):
    window = params.get("single_recent_window", 20)
    boost = params.get("single_special_boost", 2.0)
    scores = _compute_zodiac_score(rows, recent_window=window, special_boost=boost)
    return max(scores, key=scores.get)

def predict_strong_two(rows, params, xgb_weight=0.0, xgb_model=None):
    window = params.get("two_recent_window", 20)
    boost = params.get("two_special_boost", 2.0)
    scores = _compute_zodiac_score(rows, recent_window=window, special_boost=boost)
    sorted_z = sorted(scores.items(), key=lambda x: -x[1])
    return [z for z, _ in sorted_z[:2]]

def predict_strong_three_with_window(rows, window, xgb_weight=0.0, xgb_model=None):
    scores = _compute_zodiac_score(rows, recent_window=window, special_boost=2.0)
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

def get_trend_zodiac(rows, window=5):
    """根据最近window期的特别号生肖，取出现次数最多的作为趋势推荐"""
    zodiacs = []
    for r in rows[:window]:
        zodiacs.append(get_zodiac_by_number(_row_special(r)))
    cnt = Counter(zodiacs)
    if not cnt:
        return "鼠"
    return cnt.most_common(1)[0][0]