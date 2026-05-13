#!/usr/bin/env python3
# special_only_ml.py - 特五肖LightGBM排序预测

import argparse
import json
import lightgbm as lgb
import numpy as np
from common import fetch_hk_records_merged, get_zodiac_by_number, next_issue, ZODIAC_MAP, ZODIAC_PAIR

def load_lgb_model():
    try:
        model = lgb.Booster(model_file="lgb_special_ranker.txt")
        return model
    except:
        print("⚠️ LightGBM模型不存在，请先运行 train_ml_special.py")
        return None

def predict_five_zodiac(rows, model):
    # 构造当前30期特征
    window = 30
    hist = rows[:window]
    features = []
    for z in ZODIAC_MAP:
        cnt10 = sum(1 for r in hist[:10] for n in json.loads(r["numbers_json"]) if get_zodiac_by_number(n) == z)
        cnt20 = sum(1 for r in hist[:20] for n in json.loads(r["numbers_json"]) if get_zodiac_by_number(n) == z)
        special_cnt10 = sum(1 for r in hist[:10] if get_zodiac_by_number(r["special_number"]) == z)
        omission = 0
        for r in hist:
            if z in [get_zodiac_by_number(n) for n in json.loads(r["numbers_json"])] or z == get_zodiac_by_number(r["special_number"]):
                break
            omission += 1
        last_sp_z = get_zodiac_by_number(hist[0]["special_number"])
        is_pair = 1 if ZODIAC_PAIR.get(last_sp_z) == z else 0
        total_main = sum(1 for r in hist for n in json.loads(r["numbers_json"]) if get_zodiac_by_number(n) == z)
        features.append([cnt10, cnt20, special_cnt10, omission, is_pair, total_main])
    X_pred = np.array(features)
    scores = model.predict(X_pred)
    scored = [(z, scores[i]) for i, z in enumerate(ZODIAC_MAP.keys())]
    scored.sort(key=lambda x: -x[1])
    return [z for z, _ in scored[:5]]

def backtest_five_zodiac(rows, lookback, model):
    rows_rev = list(reversed(rows))
    total = min(lookback, len(rows_rev) - 20)
    if total <= 0:
        return None, None
    hits = 0
    miss_streak = 0
    max_miss = 0
    for i in range(total):
        train = rows_rev[i+20:]
        actual = rows_rev[i]["special_number"]
        actual_zod = get_zodiac_by_number(actual)
        pred = predict_five_zodiac(train, model)
        if actual_zod in pred:
            hits += 1
            miss_streak = 0
        else:
            miss_streak += 1
            max_miss = max(max_miss, miss_streak)
    return hits / total, max_miss

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    rows = get_history_rows_as_list(limit=None)
    if not rows:
        print("数据获取失败")
        return
    model = load_lgb_model()
    if not model:
        return
    if args.show:
        zodiac5 = predict_five_zodiac(rows, model)
        latest = rows[0]["issue_no"]
        pred_issue = next_issue(latest)
        print(f"预测期号: {pred_issue}")
        print(f"\n【特五肖推荐】: {'、'.join(zodiac5)}")
        hit10, miss10 = backtest_five_zodiac(rows, 10, model)
        hit100, miss100 = backtest_five_zodiac(rows, 100, model)
        if hit10 is not None:
            print(f"\n近10期回测：特五肖命中率 {hit10:.1%}，最大连空 {miss10}")
            print(f"近100期回测：特五肖命中率 {hit100:.1%}，最大连空 {miss100}")

if __name__ == "__main__":
    main()