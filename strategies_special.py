# strategies_special.py - 特五肖60%稳定版（3窗口投票，轻度连空保护）

import json
import math
from collections import Counter
from typing import List, Dict, Tuple, Optional, Sequence

from common import ZODIAC_MAP, get_zodiac_by_number, ALL_NUMBERS

def _row_numbers(r):
    try:
        if isinstance(r, dict):
            return json.loads(r["numbers_json"])
        if hasattr(r, "keys") and "numbers_json" in r.keys():
            return json.loads(r["numbers_json"])
        return json.loads(r[0]) if isinstance(r[0], str) else r[0]
    except:
        return []

def _row_special(r):
    try:
        if isinstance(r, dict):
            return int(r["special_number"])
        if hasattr(r, "keys") and "special_number" in r.keys():
            return int(r["special_number"])
        return int(r[1])
    except:
        return 0

def _zodiac_omission_map(rows: Sequence) -> Dict[str, int]:
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

def get_special_number_recommendation(rows, top_n=3, main_pool=None, recent_window=30):
    if len(rows) < 5:
        return 1, [2, 3]
    recent_specials = [_row_special(r) for r in rows[:recent_window]]
    omission = {}
    for i, sp in enumerate(recent_specials):
        if sp not in omission:
            omission[sp] = i + 1
    recent_3_set = set(recent_specials[:3])
    main_set = set(main_pool) if main_pool else set()
    tail_counts = Counter([sp % 10 for sp in recent_specials[:40]])
    coldest_tail = min(tail_counts, key=lambda t: tail_counts[t]) if tail_counts else 0
    scores = {}
    for n in ALL_NUMBERS:
        omit = omission.get(n, 30)
        score = math.log(omit + 1) * 2.0
        neighbors = {n-1, n-2, n-3, n+1, n+2, n+3} & set(ALL_NUMBERS)
        if neighbors:
            min_omit = min(omission.get(m, 30) for m in neighbors)
            neighbor_score = 1.0 - (min_omit - 1) * 0.03
            neighbor_score = max(0.0, min(1.0, neighbor_score))
            score += neighbor_score * 2.5
        if n in recent_3_set:
            score *= 0.6
        if main_set and n in main_set:
            score *= 0.7
        if n % 10 == coldest_tail:
            score += 3.0
        scores[n] = max(0.0, score)
    sorted_nums = sorted(scores.items(), key=lambda x: -x[1])
    primary = sorted_nums[0][0]
    defenses = [n for n, _ in sorted_nums[1:4]]
    return primary, defenses[:top_n-1]

def _compute_special_five_score(rows, recent_window=20):
    scores = {z: 0.0 for z in ZODIAC_MAP}
    specials = [_row_special(r) for r in rows]
    seq = [get_zodiac_by_number(sp) for sp in specials]
    n = len(seq)
    if n == 0:
        return {z: 1.0 for z in ZODIAC_MAP}
    for i, z in enumerate(seq[:recent_window]):
        w = math.exp(-i / 15.0)
        scores[z] += 5.0 * w
    omission = {z: 0 for z in ZODIAC_MAP}
    for i, z in enumerate(seq):
        if omission[z] == 0:
            omission[z] = i + 1
    for z, omit in omission.items():
        scores[z] += math.log(omit + 1) * 2.0
    return scores

def predict_strong_five(rows, params, miss_streak=0):
    # 3窗口投票（12,20,28）—— 历史达到60%的配置
    windows = [12, 20, 28]
    votes = Counter()
    for w in windows:
        scores = _compute_special_five_score(rows, w)
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        picks = [ranked[i][0] for i in range(5)]
        votes.update(picks)
    final_picks = [z for z, _ in votes.most_common(5)]
    # 轻度连空保护：仅当连续2期错误时，强制加入遗漏最长的1个生肖
    if miss_streak >= 2 and rows:
        omission = _zodiac_omission_map(rows)
        if omission:
            coldest = max(omission, key=omission.get)
            if coldest not in final_picks:
                final_picks[-1] = coldest
    return final_picks[:5]