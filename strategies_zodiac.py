# strategies_zodiac.py - 香港六合彩一二三生肖策略（完整版）

import json
import math
import random
from collections import Counter
from typing import List, Dict, Tuple, Optional, Sequence

from common import ZODIAC_MAP, get_zodiac_by_number, ALL_NUMBERS

# ---------- 辅助函数 ----------
def _row_numbers(r):
    """安全提取主号列表，返回 List[int]"""
    try:
        if isinstance(r, dict):
            return json.loads(r["numbers_json"])
        if hasattr(r, "keys") and "numbers_json" in r.keys():
            return json.loads(r["numbers_json"])
        return json.loads(r[0]) if isinstance(r[0], str) else r[0]
    except:
        return []

def _row_special(r):
    """安全提取特别号，返回 int"""
    try:
        if isinstance(r, dict):
            return int(r["special_number"])
        if hasattr(r, "keys") and "special_number" in r.keys():
            return int(r["special_number"])
        return int(r[1])
    except:
        return 0

def _zodiac_omission_map(rows: Sequence) -> Dict[str, int]:
    """计算每个生肖的遗漏期数（rows 按时间倒序，最近的在前面）"""
    omission = {z: len(rows) + 1 for z in ZODIAC_MAP}
    for i, row in enumerate(rows):
        numbers = _row_numbers(row)
        special = _row_special(row)
        if not numbers:
            continue
        appeared = set()
        for n in numbers:
            appeared.add(get_zodiac_by_number(n))
        if special != 0:
            appeared.add(get_zodiac_by_number(special))
        for z in appeared:
            if omission[z] > i + 1:
                omission[z] = i + 1
    return omission

# ---------- 核心评分函数 ----------
def _compute_zodiac_score(rows, recent_window=15, special_boost=3.2, main_boost=1.0):
    """计算所有生肖的综合得分（热度 + 遗漏 + 转移概率简化版）"""
    scores = {z: 0.0 for z in ZODIAC_MAP}
    # 1. 热度（指数衰减）
    for idx, row in enumerate(rows[:recent_window]):
        w = math.exp(-idx / 12.0)
        nums = _row_numbers(row)
        sp = _row_special(row)
        for n in nums:
            scores[get_zodiac_by_number(n)] += w * main_boost
        if sp != 0:
            scores[get_zodiac_by_number(sp)] += special_boost * w
    # 2. 遗漏（对数）
    omission = _zodiac_omission_map(rows)
    for z, omit in omission.items():
        scores[z] += math.log(omit + 1) * 1.5
    # 3. 特别号→主号转移（简化）
    if len(rows) >= 2:
        latest_sp_zod = get_zodiac_by_number(_row_special(rows[0]))
        scores[latest_sp_zod] += 1.0
    return scores

# ---------- 预测函数 ----------
def predict_strong_single(rows, params, xgb_weight=0.6):
    """预测一生肖（返回一个生肖）"""
    scores = _compute_zodiac_score(
        rows,
        recent_window=params.get("single_recent_window", 15),
        special_boost=params.get("single_special_boost", 3.2),
        main_boost=params.get("main_boost", 1.0)
    )
    if not scores:
        return "马"
    sorted_items = sorted(scores.items(), key=lambda x: -x[1])
    top = sorted_items[:3]
    raw_scores = [s for _, s in top]
    if max(raw_scores) == min(raw_scores):
        return top[0][0]
    temp = 0.5
    calibrated = [math.exp(s / temp) for s in raw_scores]
    total = sum(calibrated)
    probs = [c / total for c in calibrated]
    zodiacs = [z for z, _ in top]
    seed = params.get("seed", 42)
    random.seed(seed)
    return random.choices(zodiacs, weights=probs, k=1)[0]

def predict_strong_two(rows, params, xgb_weight=0.6):
    """预测二生肖（返回两个生肖）"""
    scores = _compute_zodiac_score(
        rows,
        recent_window=params.get("two_recent_window", 15),
        special_boost=params.get("two_special_boost", 3.0),
        main_boost=params.get("main_boost", 1.0)
    )
    if not scores:
        return ["马", "蛇"]
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    picks = [ranked[0][0], ranked[1][0]] if len(ranked) >= 2 else ["马", "蛇"]
    # 冷号保护：如果前两名得分差距很小，用最冷的生肖替换第二个
    if len(ranked) >= 2 and (ranked[0][1] - ranked[1][1]) < 0.15:
        omission = _zodiac_omission_map(rows)
        coldest = max(omission, key=omission.get)
        if coldest not in picks:
            picks[1] = coldest
    return picks

def predict_strong_three(rows, params, xgb_weight=0.6):
    """预测三生肖（返回三个生肖）"""
    scores = _compute_zodiac_score(rows, recent_window=18, special_boost=3.2, main_boost=1.0)
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [ranked[i][0] for i in range(3)]

def predict_strong_three_with_window(rows, window, xgb_weight=0.6):
    """支持指定窗口的三生肖预测（用于投票）"""
    scores = _compute_zodiac_score(rows, recent_window=window, special_boost=3.2, main_boost=1.0)
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [ranked[i][0] for i in range(3)]
