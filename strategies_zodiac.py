#!/usr/bin/env python3
# strategies_zodiac.py - 生肖预测策略（完整版，含连空保护辅助函数）

import json
from collections import Counter
from common import get_zodiac_by_number, ZODIAC_MAP, ZODIAC_PAIR

# ---------- 辅助函数 ----------
def _row_numbers(row):
    """从 row 中解析出正码列表"""
    return json.loads(row["numbers_json"])

def _row_special(row):
    return row["special_number"]

def _zodiac_omission_map(rows):
    """计算每个生肖的遗漏期数（rows 降序，最新在前）"""
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

def _compute_zodiac_score(rows, recent_window=15, special_boost=3.0):
    """计算每个生肖的综合得分（规则评分）"""
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

# ---------- 单一生肖预测 ----------
def predict_strong_single(rows, params, xgb_weight=0.0, xgb_model=None):
    window = params.get("single_recent_window", 15)
    boost = params.get("single_special_boost", 3.0)
    scores = _compute_zodiac_score(rows, recent_window=window, special_boost=boost)
    rule_top = max(scores, key=scores.get)
    if xgb_weight <= 0 or xgb_model is None:
        return rule_top
    return rule_top

# ---------- 二生肖预测 ----------
def predict_strong_two(rows, params, xgb_weight=0.0, xgb_model=None):
    window = params.get("two_recent_window", 15)
    boost = params.get("two_special_boost", 3.0)
    scores = _compute_zodiac_score(rows, recent_window=window, special_boost=boost)
    sorted_z = sorted(scores.items(), key=lambda x: -x[1])
    rule_two = [z for z, _ in sorted_z[:2]]
    if xgb_weight <= 0 or xgb_model is None:
        return rule_two
    return rule_two

# ---------- 三生肖预测 ----------
def predict_strong_three_with_window(rows, window, xgb_weight=0.0, xgb_model=None):
    scores = _compute_zodiac_score(rows, recent_window=window, special_boost=3.0)
    sorted_z = sorted(scores.items(), key=lambda x: -x[1])
    return [z for z, _ in sorted_z[:3]]

# ---------- 新增：追热/追冷辅助（用于连空保护） ----------
def get_hot_zodiac(rows, window=10):
    """返回最近 window 期内出现次数最多的生肖（用于追热）"""
    zodiacs = []
    for r in rows[:window]:
        for n in _row_numbers(r):
            zodiacs.append(get_zodiac_by_number(n))
        zodiacs.append(get_zodiac_by_number(_row_special(r)))
    cnt = Counter(zodiacs)
    if not cnt:
        return "鼠"
    return cnt.most_common(1)[0][0]

def get_cold_zodiac(rows, window=30):
    """返回遗漏最大的生肖（用于追冷）"""
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
    coldest = max(omission, key=omission.get)
    return coldest