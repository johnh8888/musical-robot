#!/usr/bin/env python3
# ml_predict.py - 加载模型并提供预测函数

import json
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from collections import Counter
from datetime import datetime
from common import get_zodiac_by_number, ZODIAC_MAP, ZODIAC_PAIR

# 全局模型（延迟加载）
_zodiac_model = None
_special_model = None

def load_zodiac_model():
    global _zodiac_model
    if _zodiac_model is None:
        _zodiac_model = lgb.Booster(model_file="zodiac_ranker.txt")
    return _zodiac_model

def load_special_model():
    global _special_model
    if _special_model is None:
        _special_model = xgb.Booster()
        _special_model.load_model("special_xgb.json")
    return _special_model

def predict_top_k_zodiac(rows, k=2):
    """使用LightGBM排序模型预测Top K个生肖"""
    model = load_zodiac_model()
    # 用最近20期数据构造特征
    train_rows = rows[:20]
    features = []
    for z in ZODIAC_MAP.keys():
        # 特征1: 近10期出现次数
        cnt10 = 0
        for r in train_rows[:10]:
            nums = json.loads(r["numbers_json"])
            sp = r["special_number"]
            if any(get_zodiac_by_number(n) == z for n in nums) or get_zodiac_by_number(sp) == z:
                cnt10 += 1
        # 特征2: 近20期出现次数
        cnt20 = 0
        for r in train_rows:
            nums = json.loads(r["numbers_json"])
            sp = r["special_number"]
            if any(get_zodiac_by_number(n) == z for n in nums) or get_zodiac_by_number(sp) == z:
                cnt20 += 1
        # 特征3: 遗漏
        omission = 0
        for r in train_rows:
            nums = json.loads(r["numbers_json"])
            sp = r["special_number"]
            if any(get_zodiac_by_number(n) == z for n in nums) or get_zodiac_by_number(sp) == z:
                break
            omission += 1
        # 特征4: 对肖出现次数（近10期）
        pair = ZODIAC_PAIR.get(z)
        pair_cnt = 0
        if pair:
            for r in train_rows[:10]:
                nums = json.loads(r["numbers_json"])
                sp = r["special_number"]
                if any(get_zodiac_by_number(n) == pair for n in nums) or get_zodiac_by_number(sp) == pair:
                    pair_cnt += 1
        # 特征5: 最近一期特别号是否是该生肖
        last_sp_zod = get_zodiac_by_number(train_rows[0]["special_number"])
        is_last = 1 if last_sp_zod == z else 0
        features.extend([cnt10, cnt20, omission, pair_cnt, is_last])
    X_pred = np.array([features])
    scores = model.predict(X_pred).reshape(12)  # 12个生肖得分
    scored = [(z, scores[i]) for i, z in enumerate(ZODIAC_MAP.keys())]
    scored.sort(key=lambda x: -x[1])
    return [z for z, _ in scored[:k]]

def predict_special_number_ml(rows, top_k=5):
    """使用XGBoost多分类预测特别号数字，返回top_k号码"""
    model = load_special_model()
    # 用最近30期构造特征
    hist = rows[:30]
    # 特征1-49: 频率
    freq = Counter(r["special_number"] for r in hist)
    feats = [freq.get(n, 0) for n in range(1, 50)]
    # 特征50: 上期特别号
    feats.append(hist[0]["special_number"])
    # 特征51: 上期正码均值
    numbers = json.loads(hist[0]["numbers_json"])
    feats.append(np.mean(numbers))
    # 特征52: 上期正码中位数
    feats.append(np.median(numbers))
    # 特征53: 星期
    try:
        wd = datetime.strptime(hist[0]["draw_date"], "%Y-%m-%d").weekday()
    except:
        wd = 0
    feats.append(wd)
    X_pred = np.array([feats])
    proba = model.predict_proba(X_pred)[0]  # 49个类别的概率
    top_indices = np.argsort(proba)[::-1][:top_k]
    return [i+1 for i in top_indices]

def predict_five_zodiac_ml(rows):
    """特五肖: 使用正码生肖频率（不依赖模型，稳定）"""
    zodiac_cnt = Counter()
    for r in rows[:100]:
        nums = json.loads(r["numbers_json"])
        for n in nums:
            zodiac_cnt[get_zodiac_by_number(n)] += 1
    if len(zodiac_cnt) < 5:
        for r in rows[:100]:
            zodiac_cnt[get_zodiac_by_number(r["special_number"])] += 1
    return [z for z, _ in zodiac_cnt.most_common(5)]