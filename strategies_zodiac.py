#!/usr/bin/env python3
# strategies_zodiac.py - 增强规则策略（包含多特征评分）

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

def _compute_zodiac_score_advanced(rows, recent_window=20, special_boost=2.0, 
                                   weight_omission=0.3, weight_pair=0.5, weight_weekday=0.1):
    """
    增强评分：正码频率 + 特别号加权 + 遗漏加权 + 对肖奖励 + 星期几倾向
    """
    scores = {}
    # 预计算星期几的生肖分布（基于所有历史，静态）
    weekday_zodiac_count = {i: Counter() for i in range(7)}
    for r in rows:
        try:
            wd = datetime.strptime(r["draw_date"], "%Y-%m-%d").weekday()
        except:
            continue
        for n in r["numbers"]:
            z = get_zodiac_by_number(n)
            weekday_zodiac_count[wd][z] += 1
        z_sp = get_zodiac_by_number(r["special_number"])
        weekday_zodiac_count[wd][z_sp] += 1
    # 当前星期几
    today_wd = -1
    if rows:
        try:
            today_wd = datetime.strptime(rows[0]["draw_date"], "%Y-%m-%d").weekday()
        except:
            pass

    for z in ZODIAC_MAP:
        nums = ZODIAC_MAP[z]
        # 基础分：正码出现次数（近 recent_window 期）
        base = 0
        for r in rows[:recent_window]:
            for n in _row_numbers(r):
                if n in nums:
                    base += 1
        # 特别号加权分
        special_bonus = 0
        for r in rows[:recent_window]:
            if _row_special(r) in nums:
                special_bonus += special_boost
        
        # 遗漏分（值越大越冷，给予正向分数）
        omission = _zodiac_omission_map(rows).get(z, 0)
        omission_score = omission / recent_window
        
        # 对肖分：上期特别号的对肖
        pair_bonus = 0
        if rows:
            last_sp = _row_special(rows[0])
            last_zod = get_zodiac_by_number(last_sp)
            pair = ZODIAC_PAIR.get(last_zod)
            if pair == z:
                pair_bonus = weight_pair
        
        # 星期几倾向分：该生肖在今天这个星期几的历史出现比例（归一化）
        weekday_bonus = 0
        if today_wd != -1 and weekday_zodiac_count[today_wd].total() > 0:
            prob = weekday_zodiac_count[today_wd].get(z, 0) / weekday_zodiac_count[today_wd].total()
            weekday_bonus = prob * weight_weekday * 10  # 放大系数
        
        scores[z] = base + special_bonus + weight_omission * omission_score + pair_bonus + weekday_bonus
    return scores

def predict_strong_single(rows, params, xgb_weight=0.0, xgb_model=None):
    window = params.get("single_recent_window", 20)
    boost = params.get("single_special_boost", 2.0)
    scores = _compute_zodiac_score_advanced(rows, recent_window=window, special_boost=boost,
                                            weight_omission=0.3, weight_pair=0.5, weight_weekday=0.1)
    return max(scores, key=scores.get)

def predict_strong_two(rows, params, xgb_weight=0.0, xgb_model=None):
    window = params.get("two_recent_window", 20)
    boost = params.get("two_special_boost", 2.0)
    scores = _compute_zodiac_score_advanced(rows, recent_window=window, special_boost=boost,
                                            weight_omission=0.3, weight_pair=0.5, weight_weekday=0.1)
    sorted_z = sorted(scores.items(), key=lambda x: -x[1])
    return [z for z, _ in sorted_z[:2]]

def predict_strong_three_with_window(rows, window, xgb_weight=0.0, xgb_model=None):
    scores = _compute_zodiac_score_advanced(rows, recent_window=window, special_boost=2.0,
                                            weight_omission=0.3, weight_pair=0.5, weight_weekday=0.1)
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