# strategies_zodiac.py - 升级版（支持 XGBoost 融合）

import json
import math
import random
import xgboost as xgb
from collections import Counter
from typing import List, Dict, Tuple, Optional, Sequence

from common import ZODIAC_MAP, get_zodiac_by_number, ALL_NUMBERS

# ---------- 辅助函数 ----------
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

# ---------- 特征提取（用于 XGBoost） ----------
def extract_features_for_zodiac(rows, target_zodiac):
    """提取 12 个特征（与 train_xgboost.py 保持一致）"""
    feats = []
    # 1. 遗漏
    omission = _zodiac_omission_map(rows)
    feats.append(omission.get(target_zodiac, 30))
    # 2. 近10期出现次数
    cnt10 = 0
    for r in rows[:10]:
        if target_zodiac in [get_zodiac_by_number(n) for n in _row_numbers(r)] or \
           target_zodiac == get_zodiac_by_number(_row_special(r)):
            cnt10 += 1
    feats.append(cnt10)
    # 3. 近20期出现次数
    cnt20 = 0
    for r in rows[:20]:
        if target_zodiac in [get_zodiac_by_number(n) for n in _row_numbers(r)] or \
           target_zodiac == get_zodiac_by_number(_row_special(r)):
            cnt20 += 1
    feats.append(cnt20)
    # 4. 最近特别号是否匹配
    latest_sp_zod = get_zodiac_by_number(_row_special(rows[0]))
    feats.append(1 if latest_sp_zod == target_zodiac else 0)
    # 5. 主号频率（近5期）
    main_cnt = 0
    for r in rows[:5]:
        for n in _row_numbers(r):
            if get_zodiac_by_number(n) == target_zodiac:
                main_cnt += 1
    feats.append(main_cnt / 5.0)
    # 6. 冷热变化率
    recent = 0
    for r in rows[:5]:
        if target_zodiac in [get_zodiac_by_number(n) for n in _row_numbers(r)] or \
           target_zodiac == get_zodiac_by_number(_row_special(r)):
            recent += 1
    old = 0
    for r in rows[5:10]:
        if target_zodiac in [get_zodiac_by_number(n) for n in _row_numbers(r)] or \
           target_zodiac == get_zodiac_by_number(_row_special(r)):
            old += 1
    feats.append(recent - old)
    # 7. 特别号出现次数（近10期）
    sp_cnt = 0
    for r in rows[:10]:
        if get_zodiac_by_number(_row_special(r)) == target_zodiac:
            sp_cnt += 1
    feats.append(sp_cnt)
    # 8. 平均间隔
    indices = [i for i, r in enumerate(rows) if target_zodiac in [get_zodiac_by_number(n) for n in _row_numbers(r)] or target_zodiac == get_zodiac_by_number(_row_special(r))]
    if len(indices) >= 2:
        avg_gap = sum(indices[i] - indices[i-1] for i in range(1, len(indices))) / (len(indices) - 1)
    else:
        avg_gap = 30
    feats.append(avg_gap)
    # 9. 规则评分（使用当前得分函数）
    rule_scores = _compute_zodiac_score(rows, recent_window=15, special_boost=3.2)
    feats.append(rule_scores.get(target_zodiac, 0.0))
    # 10. 最近特别号与目标生肖号码的最小距离
    latest_sp = _row_special(rows[0])
    nums = ZODIAC_MAP.get(target_zodiac, [])
    min_dist = min(abs(n - latest_sp) for n in nums) if nums else 25
    feats.append(min_dist / 50.0)
    return feats

# ---------- XGBoost 模型加载 ----------
_xgb_model = None
def load_xgb_model():
    global _xgb_model
    if _xgb_model is None:
        try:
            _xgb_model = xgb.Booster()
            _xgb_model.load_model("xgboost_zodiac.json")
            print("XGBoost 模型已加载")
        except:
            print("XGBoost 模型加载失败，仅使用规则")
            _xgb_model = None
    return _xgb_model

def predict_xgb_proba(rows, target_zodiac):
    model = load_xgb_model()
    if model is None:
        return 0.5
    feats = extract_features_for_zodiac(rows, target_zodiac)
    dmat = xgb.DMatrix([feats])
    prob = model.predict(dmat)[0]
    return prob

# ---------- 核心评分函数（规则部分） ----------
def _compute_zodiac_score(rows, recent_window=15, special_boost=3.2, main_boost=1.0):
    scores = {z: 0.0 for z in ZODIAC_MAP}
    for idx, row in enumerate(rows[:recent_window]):
        w = math.exp(-idx / 12.0)
        nums = _row_numbers(row)
        sp = _row_special(row)
        for n in nums:
            scores[get_zodiac_by_number(n)] += w * main_boost
        if sp != 0:
            scores[get_zodiac_by_number(sp)] += special_boost * w
    omission = _zodiac_omission_map(rows)
    for z, omit in omission.items():
        scores[z] += math.log(omit + 1) * 1.5
    if len(rows) >= 2:
        latest_sp_zod = get_zodiac_by_number(_row_special(rows[0]))
        scores[latest_sp_zod] += 1.0
    return scores

# ---------- 融合 XGBoost 的预测函数 ----------
def predict_strong_single(rows, params, xgb_weight=0.6):
    rule_scores = _compute_zodiac_score(
        rows,
        recent_window=params.get("single_recent_window", 15),
        special_boost=params.get("single_special_boost", 3.2),
        main_boost=params.get("main_boost", 1.0)
    )
    if load_xgb_model() is not None:
        for z in ZODIAC_MAP:
            xgb_prob = predict_xgb_proba(rows, z)
            rule_scores[z] = rule_scores.get(z, 0) * (1 - xgb_weight) + xgb_prob * xgb_weight
    if not rule_scores:
        return "马"
    sorted_items = sorted(rule_scores.items(), key=lambda x: -x[1])
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
    rule_scores = _compute_zodiac_score(
        rows,
        recent_window=params.get("two_recent_window", 15),
        special_boost=params.get("two_special_boost", 3.0),
        main_boost=params.get("main_boost", 1.0)
    )
    if load_xgb_model() is not None:
        for z in ZODIAC_MAP:
            xgb_prob = predict_xgb_proba(rows, z)
            rule_scores[z] = rule_scores.get(z, 0) * (1 - xgb_weight) + xgb_prob * xgb_weight
    if not rule_scores:
        return ["马", "蛇"]
    ranked = sorted(rule_scores.items(), key=lambda x: -x[1])
    picks = [ranked[0][0], ranked[1][0]] if len(ranked) >= 2 else ["马", "蛇"]
    if len(ranked) >= 2 and (ranked[0][1] - ranked[1][1]) < 0.15:
        omission = _zodiac_omission_map(rows)
        coldest = max(omission, key=omission.get)
        if coldest not in picks:
            picks[1] = coldest
    return picks

def predict_strong_three(rows, params, xgb_weight=0.6):
    rule_scores = _compute_zodiac_score(rows, recent_window=18, special_boost=3.2, main_boost=1.0)
    if load_xgb_model() is not None:
        for z in ZODIAC_MAP:
            xgb_prob = predict_xgb_proba(rows, z)
            rule_scores[z] = rule_scores.get(z, 0) * (1 - xgb_weight) + xgb_prob * xgb_weight
    ranked = sorted(rule_scores.items(), key=lambda x: -x[1])
    return [ranked[i][0] for i in range(3)]

def predict_strong_three_with_window(rows, window, xgb_weight=0.6):
    rule_scores = _compute_zodiac_score(rows, recent_window=window, special_boost=3.2, main_boost=1.0)
    if load_xgb_model() is not None:
        for z in ZODIAC_MAP:
            xgb_prob = predict_xgb_proba(rows, z)
            rule_scores[z] = rule_scores.get(z, 0) * (1 - xgb_weight) + xgb_prob * xgb_weight
    ranked = sorted(rule_scores.items(), key=lambda x: -x[1])
    return [ranked[i][0] for i in range(3)]
