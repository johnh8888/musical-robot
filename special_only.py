#!/usr/bin/env python3
# special_only.py - 特五肖 v5（高稳定性版）

import gzip
import json
import re
import time
import urllib.request
from collections import Counter, defaultdict

API_URL = "https://marksix6.net/index.php?api=1"

ZODIAC_MAP = {
    "鼠": [7,19,31,43], "牛": [8,20,32,44], "虎": [9,21,33,45],
    "兔": [10,22,34,46], "龙": [11,23,35,47], "蛇": [12,24,36,48],
    "马": [1,13,25,37,49], "羊": [2,14,26,38], "猴": [3,15,27,39],
    "鸡": [4,16,28,40], "狗": [5,17,29,41], "猪": [6,18,30,42],
}

ZODIAC_LIST = list(ZODIAC_MAP.keys())

COLOR_MAP = { ... }  # 保持你之前的波色定义

DECAY_ALPHA = 0.85

def get_zodiac(n): ...  # 保持不变
def get_color(n): ... 
def is_big(n): return n >= 25
def is_odd(n): return n % 2 == 1
def decay(idx): return DECAY_ALPHA ** idx

def omission_map(rows):
    om = {z: 0 for z in ZODIAC_LIST}
    for r in reversed(rows):
        appeared = {get_zodiac(n) for n in r.get("numbers", []) + [r["special_number"]]}
        for z in ZODIAC_LIST:
            om[z] = 0 if z in appeared else om[z] + 1
    return om

def parse_nums(value):
    return [int(t) for t in re.split(r'[，,]', value) if t.strip().isdigit() and 1 <= int(t) <= 49]

def fetch_hk_online(limit=250):   # 增加数据量
    # ... 保持你的 fetch 函数，limit 改成250
    ...

def stable_score(history, miss_count=0):
    score = Counter()
    recent = history[-80:]   # 更长窗口

    # 1. 短期强趋势
    for idx, r in enumerate(reversed(recent[-10:])):
        z = get_zodiac(r["special_number"])
        score[z] += 0.65 * decay(idx)

    # 2. 中期趋势
    for idx, r in enumerate(reversed(recent[-30:])):
        z = get_zodiac(r["special_number"])
        score[z] += 0.22 * decay(idx)

    # 3. 综合特征（波色 + 大小 + 单双 + 热力）
    sp_list = [r["special_number"] for r in recent[-20:]]
    dom_color = Counter(get_color(n) for n in sp_list).most_common(1)[0][0]
    big_ratio = sum(is_big(n) for n in sp_list) / len(sp_list) if sp_list else 0.5
    odd_ratio = sum(is_odd(n) for n in sp_list) / len(sp_list) if sp_list else 0.5

    # 频率加成
    freq = Counter(get_zodiac(r["special_number"]) for r in recent)
    max_freq = max(freq.values()) if freq else 1

    for z in ZODIAC_LIST:
        sample = ZODIAC_MAP[z][0]
        base = 0.0
        if get_color(sample) == dom_color:
            base += 0.32
        if (is_big(sample) and big_ratio > 0.5) or (not is_big(sample) and big_ratio <= 0.5):
            base += 0.18
        if (is_odd(sample) and odd_ratio > 0.5) or (not is_odd(sample) and odd_ratio <= 0.5):
            base += 0.14
        # 频率奖励
        score[z] += base + (freq[z] / max_freq) * 0.25

    # 4. 强遗漏补偿（降低连空关键）
    om = omission_map(history)
    for z in ZODIAC_LIST:
        omission_bonus = min(om[z] * 0.085, 1.2)
        score[z] += omission_bonus * 0.22

    # 5. 连续miss强冷号保护
    if miss_count >= 3:
        cold = sorted(om.items(), key=lambda x: x[1], reverse=True)[:4]
        for z, _ in cold:
            score[z] += 0.35

    return score

def recommend(history, miss_count=0):
    score = stable_score(history, miss_count)
    ranked = [z for z, _ in score.most_common()]
    # 热 + 冷平衡
    final = list(dict.fromkeys(ranked[:4] + ranked[-5:]))
    return final[:5]

# backtest 函数也同步增加窗口和更严格测试
def backtest(rows):
    # ... (类似之前，但 start 从 100 开始，train 长度更长)
    ...

def main():
    rows = fetch_hk_online(250)
    if not rows:
        print("❌ 获取失败")
        return
    preds = recommend(rows)
    print("\n【稳定版特五肖 v5 - 加强冷号保护】")
    print("、".join(preds))
    backtest(rows)

if __name__ == "__main__":
    main()