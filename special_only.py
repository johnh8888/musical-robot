#!/usr/bin/env python3
# special_only.py - 特别号数字预测（XGBoost 模型 + 方差分析 + 分级推荐）

import argparse
import json
import pickle
from collections import Counter
from datetime import datetime
import xgboost as xgb
import numpy as np
from common import fetch_hk_records_merged, get_zodiac_by_number, next_issue, ALL_NUMBERS
from strategies_special import get_special_number_recommendation, compute_special_five_score

# ---------- 特征工程 ----------
def extract_features_for_special(rows, target_num):
    specials = [r["special_number"] for r in rows]
    # 1. 遗漏
    omission = 0
    for sp in specials[::-1]:
        if sp != target_num:
            omission += 1
        else:
            break
    # 2. 近10/20期出现次数
    cnt10 = sum(1 for sp in specials[:10] if sp == target_num)
    cnt20 = sum(1 for sp in specials[:20] if sp == target_num)
    # 3. 与上期差值归一化
    latest_sp = specials[0] if specials else 0
    diff = abs(target_num - latest_sp) if latest_sp else 99
    diff_feat = min(diff, 9) / 9.0
    # 4. 尾数
    tail = target_num % 10
    # 5. 星期（从最新日期）
    wd = -1
    if rows and "draw_date" in rows[0]:
        try:
            wd = datetime.strptime(rows[0]["draw_date"], "%Y-%m-%d").weekday()
        except:
            pass
    # 6. 冷热变化率（近5期 vs 前5期）
    recent = sum(1 for sp in specials[:5] if sp == target_num)
    old = sum(1 for sp in specials[5:10] if sp == target_num)
    change = recent - old
    # 7. 是否与最近特别号同尾
    same_tail = 1 if (latest_sp % 10) == (target_num % 10) else 0
    return [omission, cnt10, cnt20, diff_feat, tail, wd, change, same_tail]

def train_special_model(rows, model_path="special_model.pkl"):
    X, y = [], []
    for i in range(30, len(rows)):
        hist = rows[i-20:i]
        target = rows[i]["special_number"]
        for n in range(1, 50):
            X.append(extract_features_for_special(hist, n))
            y.append(1 if n == target else 0)
    model = xgb.XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, use_label_encoder=False, eval_metric='logloss')
    model.fit(np.array(X), np.array(y))
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"模型已保存至 {model_path}")
    return model

def load_special_model(model_path="special_model.pkl"):
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except:
        return None

def predict_special_number(rows, model, top_k=5):
    """返回 (主推, 概率, 防守列表)"""
    probs = []
    for n in range(1, 50):
        feats = extract_features_for_special(rows, n)
        prob = model.predict_proba([feats])[0][1]
        probs.append((n, prob))
    probs.sort(key=lambda x: -x[1])
    main = probs[0][0]
    main_prob = probs[0][1]
    defenses = [n for n, _ in probs[1:1+top_k]]
    return main, main_prob, defenses

def get_confidence_label(prob):
    if prob >= 0.8:
        return "高置信度"
    elif prob >= 0.6:
        return "中置信度"
    else:
        return "低置信度"

# ---------- 分析函数 ----------
def analyze_special_distribution(rows):
    specials = [r["special_number"] for r in rows]
    total = len(specials)
    freq = Counter(specials)
    hot_numbers = [n for n, _ in freq.most_common(10)]
    cold_numbers = [n for n in ALL_NUMBERS if freq.get(n, 0) <= total/49*0.5][:10]
    # 间隔分析
    intervals = {}
    for n in range(1, 50):
        last = -1
        gaps = []
        for i, sp in enumerate(specials):
            if sp == n:
                if last != -1:
                    gaps.append(i - last)
                last = i
        if gaps:
            intervals[n] = {"avg": sum(gaps)/len(gaps), "max": max(gaps)}
    print("\n=== 特别号分布分析 ===")
    print(f"总期数: {total}")
    print(f"最热号码 (前10): {hot_numbers}")
    print(f"最冷号码 (后10): {cold_numbers}")
    print(f"平均间隔最长的5个号码: {sorted(intervals.items(), key=lambda x: -x[1]['avg'])[:5]}")
    print(f"最大间隔 >20 期的号码: {[n for n,d in intervals.items() if d['max']>20]}")

def diagnose_prediction_errors(rows, model, lookback=50):
    rows_rev = list(reversed(rows))
    correct = 0
    wrong = []
    for i in range(min(lookback, len(rows_rev)-20)):
        train = rows_rev[i+20:]
        actual = rows_rev[i]["special_number"]
        pred, _, _ = predict_special_number(train, model, top_k=3)
        if actual == pred:
            correct += 1
        else:
            wrong.append((actual, pred))
    total = len(wrong)+correct
    print(f"\n=== 最近{total}期预测诊断 ===")
    print(f"正确: {correct}, 错误: {len(wrong)}")
    print(f"错误率: {len(wrong)/total:.1%}")
    if wrong:
        print(f"常见错误实际号: {Counter([w[0] for w in wrong]).most_common(5)}")
        print(f"常见错误推荐号: {Counter([w[1] for w in wrong]).most_common(5)}")

# ---------- 特五肖投票预测（保留原有窗口投票）----------
def predict_strong_five(rows, params, miss_streak=0):
    windows = [12,20,28]
    votes = Counter()
    for w in windows:
        scores = compute_special_five_score(rows, w)
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        for z in [ranked[i][0] for i in range(5)]:
            votes[z] += 1
    final = [z for z,_ in votes.most_common(5)]
    if miss_streak >= 2:
        omission = {z: len(rows)+1 for z in ZODIAC_MAP}
        for i,row in enumerate(rows):
            for n in json.loads(row["numbers_json"]):
                omission[get_zodiac_by_number(n)] = 0
            omission[get_zodiac_by_number(row["special_number"])] = 0
        coldest = max(omission, key=omission.get)
        if coldest not in final:
            final[-1] = coldest
    return final[:5]

# ---------- 主程序 ----------
def get_history_rows_as_list(limit=1000):
    records = fetch_hk_records_merged(limit=limit)
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
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    rows = get_history_rows_as_list(limit=1000)
    if not rows:
        print("数据获取失败")
        return

    if args.analyze:
        analyze_special_distribution(rows)
        return

    if args.train:
        train_special_model(rows)
        return

    model = load_special_model()
    if args.diagnose:
        if model is None:
            print("请先运行 --train 训练模型")
            return
        diagnose_prediction_errors(rows, model)
        return

    if args.show:
        if model is None:
            print("模型未训练，正在自动训练...")
            model = train_special_model(rows)
        # 特别号数字预测
        main_num, prob, defenses = predict_special_number(rows, model, top_k=5)
        conf_label = get_confidence_label(prob)
        # 特五肖预测
        zodiacs = predict_strong_five(rows, {}, miss_streak=0)
        # 规则对照
        sp_rule, def_rule = get_special_number_recommendation(rows, top_n=3, recent_window=30)

        latest_issue = rows[0]["issue_no"]
        pred_issue = next_issue(latest_issue)
        print(f"预测期号: {pred_issue}")
        print(f"\n【特别号数字】")
        print(f"主推: {main_num:02d} (置信度: {conf_label}, 概率:{prob:.2%})")
        print(f"防守(5码): {' '.join(f'{n:02d}' for n in defenses[:5])}")
        print(f"规则主推: {sp_rule:02d} (防守: {' '.join(f'{n:02d}' for n in def_rule[:2])})")
        print(f"\n【特五肖推荐】: {'、'.join(zodiacs)}")

if __name__ == "__main__":
    main()
