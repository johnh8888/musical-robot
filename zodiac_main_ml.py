#!/usr/bin/env python3
# zodiac_main_ml.py - 三生肖（3中2）规则+模型融合版

import argparse
import json
import numpy as np
import xgboost as xgb
from collections import Counter
from common import fetch_hk_records_merged, get_zodiac_by_number, next_issue, ZODIAC_MAP, ZODIAC_PAIR
from strategies_zodiac import (
    predict_strong_single, predict_strong_two, predict_strong_three_with_window,
    get_hot_zodiac, get_cold_zodiac
)

WINDOW_WEIGHTS = [(8, 1.0), (10, 0.9), (12, 0.8), (18, 0.6), (20, 0.5), (30, 0.4)]
MODEL_WEIGHT = 0.6
RULE_WEIGHT = 0.4

def load_xgb_model():
    try:
        model = xgb.Booster()
        model.load_model("xgboost_zodiac_3.json")
        return model
    except:
        print("⚠️ XGBoost模型不存在，将仅使用规则")
        return None

def get_model_probs(rows, model):
    """用XGBoost为每个生肖输出概率"""
    # 构造当前30期的特征
    window = 30
    hist = rows[:window]
    probs = []
    for z in ZODIAC_MAP:
        cnt10 = 0
        for r in hist[:10]:
            if z in [get_zodiac_by_number(n) for n in json.loads(r["numbers_json"])] or z == get_zodiac_by_number(r["special_number"]):
                cnt10 += 1
        cnt20 = 0
        for r in hist[:20]:
            if z in [get_zodiac_by_number(n) for n in json.loads(r["numbers_json"])] or z == get_zodiac_by_number(r["special_number"]):
                cnt20 += 1
        cnt30 = 0
        for r in hist:
            if z in [get_zodiac_by_number(n) for n in json.loads(r["numbers_json"])] or z == get_zodiac_by_number(r["special_number"]):
                cnt30 += 1
        omission = 0
        for r in hist:
            if z in [get_zodiac_by_number(n) for n in json.loads(r["numbers_json"])] or z == get_zodiac_by_number(r["special_number"]):
                break
            omission += 1
        last_sp_z = get_zodiac_by_number(hist[0]["special_number"])
        is_pair = 1 if ZODIAC_PAIR.get(last_sp_z) == z else 0
        is_last_sp = 1 if last_sp_z == z else 0
        sp_cnt5 = sum(1 for r in hist[:5] if get_zodiac_by_number(r["special_number"]) == z)
        sp_cnt10 = sum(1 for r in hist[:10] if get_zodiac_by_number(r["special_number"]) == z)
        main_cnt10 = sum(1 for r in hist[:10] for n in json.loads(r["numbers_json"]) if get_zodiac_by_number(n) == z)
        feats = [cnt10, cnt20, cnt30, omission, is_pair, is_last_sp, sp_cnt5, sp_cnt10, main_cnt10]
        prob = model.predict(xgb.DMatrix([feats]))[0]
        probs.append(prob)
    # 软最大化，转为概率
    probs = np.exp(probs) / np.sum(np.exp(probs))
    return {z: probs[i] for i, z in enumerate(ZODIAC_MAP.keys())}

def get_history_rows_as_list(limit=None):
    records = fetch_hk_records_merged(limit=limit, prefer_local=True)
    rows = []
    for r in records:
        rows.append({
            "numbers_json": json.dumps(r["numbers"]),
            "special_number": r["special_number"],
            "draw_date": r["draw_date"],
            "issue_no": r["issue_no"]
        })
    return rows

def backtest_zodiac_stats(rows, lookback, model):
    rows_rev = list(reversed(rows))
    total = min(lookback, len(rows_rev) - 20)
    if total <= 0:
        return None
    hits_three = 0
    miss_three = 0
    max_miss_three = 0

    for i in range(total):
        train = rows_rev[i+20:]
        if len(train) < 20:
            continue
        actual = rows_rev[i]
        win_main = json.loads(actual["numbers_json"])
        win_sp = actual["special_number"]
        win_z = {get_zodiac_by_number(n) for n in win_main}
        win_z.add(get_zodiac_by_number(win_sp))

        # 规则投票（加权）
        votes_three = Counter()
        for w, weight in WINDOW_WEIGHTS:
            for z in predict_strong_three_with_window(train, w):
                votes_three[z] += weight
        rule_top3 = [z for z, _ in votes_three.most_common(3)]

        # 模型概率
        if model:
            model_probs = get_model_probs(train, model)
            # 融合：规则分数（归一化后）与模型概率加权
            # 此处简化：直接用模型概率 top3 与规则 top3 混合
            model_top3 = sorted(model_probs, key=model_probs.get, reverse=True)[:3]
            # 融合：取并集，按规则票数+模型概率加权排序
            combined_scores = {}
            for z in set(rule_top3 + model_top3):
                rule_score = votes_three.get(z, 0)
                model_score = model_probs.get(z, 0)
                combined_scores[z] = RULE_WEIGHT * rule_score + MODEL_WEIGHT * model_score
            pred_three = sorted(combined_scores, key=combined_scores.get, reverse=True)[:3]
        else:
            pred_three = rule_top3

        # 命中判定
        hit_cnt = sum(1 for z in pred_three if z in win_z)
        if hit_cnt >= 2:
            hits_three += 1
            miss_three = 0
        else:
            miss_three += 1
            max_miss_three = max(max_miss_three, miss_three)

    return hits_three/total, max_miss_three

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    rows = get_history_rows_as_list(limit=None)
    if not rows:
        print("数据获取失败")
        return
    model = load_xgb_model()
    if args.show:
        # 规则投票
        votes_three = Counter()
        for w, weight in WINDOW_WEIGHTS:
            for z in predict_strong_three_with_window(rows, w):
                votes_three[z] += weight
        rule_top3 = [z for z, _ in votes_three.most_common(3)]
        if model:
            model_probs = get_model_probs(rows, model)
            model_top3 = sorted(model_probs, key=model_probs.get, reverse=True)[:3]
            combined_scores = {}
            for z in set(rule_top3 + model_top3):
                rule_score = votes_three.get(z, 0)
                model_score = model_probs.get(z, 0)
                combined_scores[z] = RULE_WEIGHT * rule_score + MODEL_WEIGHT * model_score
            pred_three = sorted(combined_scores, key=combined_scores.get, reverse=True)[:3]
        else:
            pred_three = rule_top3

        latest = rows[0]["issue_no"]
        print(f"预测期号: {next_issue(latest)}")
        print(f"三生肖（3中2）: {'、'.join(pred_three)}")

        # 回测近100期
        hit_rate, max_miss = backtest_zodiac_stats(rows, 100, model)
        if hit_rate:
            print(f"\n近100期回测（三生肖）: 命中率 {hit_rate:.1%}，最大连空 {max_miss}")

if __name__ == "__main__":
    main()