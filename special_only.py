#!/usr/bin/env python3
# special_only.py - 特别号预测（使用本地完整历史，频率模型）

import argparse
import pickle
from pathlib import Path
from collections import Counter
from common import fetch_hk_records_merged

MODEL_PATH = "special_model.pkl"

def train_special_model():
    """基于本地全部历史数据训练特别号频率模型"""
    records = fetch_hk_records_merged(limit=None, prefer_local=True)
    specials = [r["special_number"] for r in records]
    counter = Counter(specials)
    total = len(specials)
    probs = {num: count/total for num, count in counter.items()}
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(probs, f)
    print(f"✅ 特别号模型训练完成，基于 {total} 期数据，共 {len(probs)} 个不同号码")
    return probs

def predict_special(top_n=5):
    """预测下一期特别号（返回 top_n 个最可能号码）"""
    if not Path(MODEL_PATH).exists():
        probs = train_special_model()
    else:
        with open(MODEL_PATH, "rb") as f:
            probs = pickle.load(f)
    sorted_nums = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    return [num for num, _ in sorted_nums[:top_n]]

def backtest_special(rows, lookback=100):
    """回测特别号预测命中率（使用历史频率作为预测）"""
    rows_rev = list(reversed(rows))
    total = min(lookback, len(rows_rev) - 20)
    hits = 0
    for i in range(total):
        train = rows_rev[i+20:]
        specials = [r["special_number"] for r in train]
        counter = Counter(specials)
        pred = counter.most_common(1)[0][0]
        actual = rows_rev[i]["special_number"]
        if pred == actual:
            hits += 1
    return hits / total if total > 0 else 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="训练频率模型")
    parser.add_argument("--show", action="store_true", help="显示预测结果和回测")
    args = parser.parse_args()

    if args.train:
        train_special_model()
        return

    if args.show:
        records = fetch_hk_records_merged(limit=None, prefer_local=True)
        if not records:
            print("数据获取失败，请确保 Mark_Six.csv 存在")
            return
        top5 = predict_special(5)
        print(f"预测特别号 TOP5: {', '.join(str(n) for n in top5)}")
        hit_rate_10 = backtest_special(records, 10)
        print(f"近10期回测命中率（最高频）: {hit_rate_10:.1%}")
        hit_rate_100 = backtest_special(records, 100)
        print(f"近100期回测命中率（最高频）: {hit_rate_100:.1%}")

if __name__ == "__main__":
    main()
