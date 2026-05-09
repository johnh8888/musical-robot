#!/usr/bin/env python3
# special_only.py - 改进版（特别号建模、分级推荐、诊断）

import argparse
import json
from collections import Counter
from datetime import datetime
import xgboost as xgb
import numpy as np
from common import fetch_hk_records, get_zodiac_by_number, next_issue, ALL_NUMBERS
from strategies_special import get_special_number_recommendation

# ---------- 特别号专用模型 ----------
def extract_features_for_special(rows, target_num):
    specials = [r["special_number"] for r in rows]
    # 遗漏
    omission = 0
    for sp in specials[::-1]:
        if sp != target_num:
            omission += 1
        else:
            break
    # 近10/20期出现次数
    cnt10 = sum(1 for sp in specials[:10] if sp == target_num)
    cnt20 = sum(1 for sp in specials[:20] if sp == target_num)
    # 与上期差值
    latest_sp = specials[0] if specials else 0
    diff = abs(target_num - latest_sp) if latest_sp else 99
    diff_feat = min(diff, 9) / 9.0
    # 尾数
    tail = target_num % 10
    # 星期（从最新日期获取）
    wd = -1
    if rows and "draw_date" in rows[0]:
        try:
            wd = datetime.strptime(rows[0]["draw_date"], "%Y-%m-%d").weekday()
        except:
            pass
    # 冷热变化率
    recent = sum(1 for sp in specials[:5] if sp == target_num)
    old = sum(1 for sp in specials[5:10] if sp == target_num)
    change = recent - old
    return [omission, cnt10, cnt20, diff_feat, tail, wd, change]

def train_special_model(rows):
    X, y = [], []
    for i in range(30, len(rows)):
        hist = rows[i-20:i]
        target = rows[i]["special_number"]
        for n in range(1, 50):
            feats = extract_features_for_special(hist, n)
            X.append(feats)
            y.append(1 if n == target else 0)
    model = xgb.XGBClassifier(n_estimators=100, max_depth=5, use_label_encoder=False, eval_metric='logloss')
    model.fit(np.array(X), np.array(y))
    return model

def predict_special_number_model(model, rows, top_k=5):
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
    print("\n=== 特别号分布分析 ===")
    print(f"总期数: {total}")
    print(f"最热号码: {hot_numbers}")
    print(f"最冷号码: {cold_numbers}")

def diagnose_prediction_errors(rows, model, lookback=50):
    rows_rev = list(reversed(rows))
    correct = 0
    wrong = []
    for i in range(min(lookback, len(rows_rev)-20)):
        train = rows_rev[i+20:]
        actual = rows_rev[i]["special_number"]
        pred, _, _ = predict_special_number_model(model, train, top_k=3)
        if actual == pred:
            correct += 1
        else:
            wrong.append((actual, pred))
    total = len(wrong) + correct
    print(f"\n=== 最近{total}期预测诊断 ===")
    print(f"正确: {correct}, 错误: {len(wrong)}")
    print(f"错误率: {len(wrong)/total:.1%}")
    if wrong:
        print(f"常见错误实际号: {Counter([w[0] for w in wrong]).most_common(3)}")
        print(f"常见错误推荐号: {Counter([w[1] for w in wrong]).most_common(3)}")

# ---------- 主程序 ----------
def get_history_rows_as_list(limit=600):
    records = fetch_hk_records(limit=limit)
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
    parser.add_argument("--analyze", action="store_true", help="分析特别号分布")
    parser.add_argument("--diagnose", action="store_true", help="诊断预测错误")
    parser.add_argument("--train", action="store_true", help="训练特别号模型")
    args = parser.parse_args()

    rows = get_history_rows_as_list(limit=600)
    if not rows:
        print("数据获取失败")
        return

    if args.analyze:
        analyze_special_distribution(rows)
        return

    # 训练或加载模型
    model = None
    if args.train or not hasattr(main, 'model'):
        print("正在训练特别号模型...")
        model = train_special_model(rows)
        print("模型训练完成")
    else:
        # 尝试加载预训练模型
        try:
            import pickle
            with open("special_model.pkl", "rb") as f:
                model = pickle.load(f)
        except:
            print("未找到预训练模型，请先运行 --train")
            return

    if args.diagnose:
        if model:
            diagnose_prediction_errors(rows, model)
        return

    if args.show:
        if model:
            primary, prob, defenses = predict_special_number_model(rows, model, top_k=5)
            confidence_label = get_confidence_label(prob)
            print(f"预测期号: {next_issue(rows[0]['issue_no'])}")
            print(f"主推特别号: {primary:02d} (置信度: {confidence_label}, 概率: {prob:.2%})")
            print(f"防守特别号(5个): {' '.join(f'{n:02d}' for n in defenses[:5])}")
            # 同时保留原规则推荐作为对照
            sp_rule, def_rule = get_special_number_recommendation(rows, top_n=3, recent_window=30)
            print(f"规则推荐主推: {sp_rule:02d} | 防守: {' '.join(f'{n:02d}' for n in def_rule[:2])}")
        else:
            print("模型未训练，请运行 --train")

if __name__ == "__main__":
    main()
