#!/usr/bin/env python3
# special_only.py - 香港特五肖预测（增强版）
import argparse, json, re, time, urllib.request, gzip
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import List, Dict, Tuple
import csv

# ===================== 生肖映射（与 common.py 一致） =====================
ZODIAC_MAP = {
    "马": [1, 13, 25, 37, 49], "蛇": [2, 14, 26, 38], "龙": [3, 15, 27, 39],
    "兔": [4, 16, 28, 40], "虎": [5, 17, 29, 41], "牛": [6, 18, 30, 42],
    "鼠": [7, 19, 31, 43], "猪": [8, 20, 32, 44], "狗": [9, 21, 33, 45],
    "鸡": [10, 22, 34, 46], "猴": [11, 23, 35, 47], "羊": [12, 24, 36, 48],
}
ZODIAC_LIST = list(ZODIAC_MAP.keys())

# ===================== 数据获取（保留本地CSV） =====================
def load_local_csv(path="Mark_Six.csv"):
    records = []
    if not Path(path).exists():
        print(f"⚠️ CSV不存在: {path}")
        return records
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                nums = [int(row[c].strip()) for c in ["中獎號碼 1","中獎號碼 2","中獎號碼 3","中獎號碼 4","中獎號碼 5","中獎號碼 6"]]
                sp = int(row["特別號碼"].strip())
                records.append({
                    "issue_no": row["期數"].strip(),
                    "draw_date": row["日期"].strip(),
                    "numbers": nums,
                    "special_number": sp
                })
            except:
                continue
    records.sort(key=lambda x: x["issue_no"], reverse=True)
    print(f"✅ 从本地CSV加载 {len(records)} 期")
    return records

def get_history_rows():
    return load_local_csv()

def next_issue(issue_no: str) -> str:
    try:
        if '/' in issue_no:
            y, s = issue_no.split('/')
        else:
            y = issue_no[:4]
            s = issue_no[4:].lstrip('0') or '0'
        return f"{y}/{str(int(s)+1).zfill(3)}"
    except:
        return issue_no

def get_zodiac_by_number(n):
    for z, nums in ZODIAC_MAP.items():
        if n in nums: return z
    return "马"

# ===================== 配置参数 =====================
CANDIDATE_WINDOWS = [4,6,8,10,12,15,18,20,24,30,36,42,48,54,60,72,84]
OPTIMAL_WINDOWS = [4,6,8,10,12]   # 默认值

SPECIAL_WEIGHT = 1.0
COLD_BASE = 0.8
COLD_STEP = 0.3
COLD_MAX = 1.5
MISS_PENALTY = 0.1
ADAPTIVE_LOOKBACK = 10
BASE_NORMAL_WEIGHT = 0.5
SIGNAL_THRESHOLD = 0.5
RECOMMEND_COUNT = 5

TREND_LOOKBACK = 12
HOT_OVERHEAT_THRESHOLD = 0.7

def window_weight(w, base=84):
    return round(base / w, 2)

# ===================== 遗漏与信号检测 =====================
def special_omission(rows):
    if not rows: return {z:0 for z in ZODIAC_LIST}
    om = {z:0 for z in ZODIAC_LIST}
    for r in reversed(rows):
        spz = get_zodiac_by_number(r["special_number"])
        for z in ZODIAC_LIST:
            om[z] = 0 if z==spz else om[z]+1
    return om

def normal_signal(history):
    """返回(是否启用正码, 正码权重)"""
    if len(history) < ADAPTIVE_LOOKBACK+1:
        return False, 0.0
    recent = history[-(ADAPTIVE_LOOKBACK+1):]
    hits = 0
    for i in range(len(recent)-1):
        cur_set = set(get_zodiac_by_number(n) for n in recent[i]["numbers"])
        if get_zodiac_by_number(recent[i+1]["special_number"]) in cur_set:
            hits += 1
    rate = hits/(len(recent)-1)
    if rate >= SIGNAL_THRESHOLD:
        # 自适应计算权重
        factor = 0.5 + rate
        w = BASE_NORMAL_WEIGHT * factor
        w = max(0.1, min(0.6, w))
        return True, w
    return False, 0.0

def trend_factor(history):
    """返回 (热号信任乘数, 冷号加票乘数)"""
    if len(history) < TREND_LOOKBACK+1:
        return 1.0, 1.0
    recent = history[-TREND_LOOKBACK:]
    hot_hit = 0
    total = len(recent)-1
    if total <= 0: return 1.0, 1.0
    for i in range(total):
        prev = recent[i]
        cur_sp = get_zodiac_by_number(recent[i+1]["special_number"])
        # 简单估算热号：上期前30期特码频率前5
        sub = history[:history.index(prev)+1][-30:]
        cnt = Counter()
        for r in sub:
            cnt[get_zodiac_by_number(r["special_number"])] += 1
        top5 = [z for z,_ in cnt.most_common(5)]
        if cur_sp in top5:
            hot_hit += 1
    ratio = hot_hit/total
    if ratio > HOT_OVERHEAT_THRESHOLD:
        return 0.9, 1.5
    elif ratio < 0.3:
        return 1.0, 0.8
    return 1.0, 1.0

def dynamic_cold_bonus(omission_val):
    if omission_val <= 0: return 0.0
    return min(COLD_BASE + (omission_val//10)*COLD_STEP, COLD_MAX)

# ===================== 推荐核心 =====================
def recommend(history, windows, force_cold=False):
    if not history:
        return ZODIAC_LIST[:RECOMMEND_COUNT]

    use_normal, normal_w = normal_signal(history)
    hot_boost, cold_mult = trend_factor(history)

    omission = special_omission(history)
    sorted_cold = sorted(omission, key=omission.get, reverse=True)
    votes = Counter()
    wgt = {w: window_weight(w) for w in windows}

    for w in windows:
        recent = history[-w:] if len(history)>=w else history
        cnt = Counter()
        for r in recent:
            spz = get_zodiac_by_number(r["special_number"])
            cnt[spz] += SPECIAL_WEIGHT * hot_boost
            if use_normal:
                for n in r["numbers"]:
                    cnt[get_zodiac_by_number(n)] += normal_w
        for z,_ in cnt.most_common(RECOMMEND_COUNT):
            votes[z] += wgt[w]

    # 动态冷号加票
    for z in sorted_cold:
        base_bonus = dynamic_cold_bonus(omission[z])
        if base_bonus > 0:
            votes[z] += base_bonus * cold_mult

    preds = [z for z,_ in votes.most_common(RECOMMEND_COUNT)]

    # 连空保护
    if force_cold and len(preds)>=RECOMMEND_COUNT:
        keep = preds[:RECOMMEND_COUNT-2]
        new_cold = [z for z in sorted_cold[:2] if z not in keep]
        preds = keep + new_cold
        while len(preds) < RECOMMEND_COUNT:
            for z,_ in votes.most_common():
                if z not in preds:
                    preds.append(z)
                    break
            else:
                preds.append(ZODIAC_LIST[0])
    return preds[:RECOMMEND_COUNT]

# ===================== 回测与优化 =====================
def backtest(rows, lookback, windows):
    rows_rev = list(reversed(rows))
    total = min(lookback, len(rows_rev)-20)
    if total <= 0: return None, None
    hits, cur_miss, max_miss = 0,0,0
    for i in range(total):
        train = rows_rev[i+20:]
        if len(train) < 20: continue
        actual_z = get_zodiac_by_number(rows_rev[i]["special_number"])
        preds = recommend(train, windows, force_cold=(cur_miss>=1))
        if actual_z in preds:
            hits += 1
            cur_miss = 0
        else:
            cur_miss += 1
            max_miss = max(max_miss, cur_miss)
    return hits/total, max_miss

def optimize_windows(rows, opt_lookback=40):
    best_combo = None
    best_score = -float("inf")
    min_req = opt_lookback+20
    if len(rows) < min_req:
        opt_lookback = max(10, len(rows)-20)
    for combo in combinations(CANDIDATE_WINDOWS, 5):
        hr, max_miss = backtest(rows, opt_lookback, list(combo))
        if hr is None: continue
        score = hr - max_miss * MISS_PENALTY
        if score > best_score:
            best_score = score
            best_combo = list(combo)
    return best_combo if best_combo else [4,6,8,10,12]

# ===================== 主程序 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    rows = get_history_rows()
    if not rows:
        print("数据获取失败")
        return

    global OPTIMAL_WINDOWS
    OPTIMAL_WINDOWS = optimize_windows(rows)
    print(f"自动选择最优窗口: {OPTIMAL_WINDOWS}")

    if args.show:
        latest = rows[0]["issue_no"]
        pred = next_issue(latest)
        print(f"预测期号: {pred}")

        use_n, nw = normal_signal(rows)
        hot_b, cold_m = trend_factor(rows)
        print(f"正码增强: {'开启' if use_n else '关闭'} | 热号乘数:{hot_b:.2f} 冷号乘数:{cold_m:.2f}")

        preds = recommend(rows, OPTIMAL_WINDOWS, force_cold=False)
        print(f"\n【特五肖推荐】: {'、'.join(preds)}")

        hr10, miss10 = backtest(rows, 10, OPTIMAL_WINDOWS)
        hr100, miss100 = backtest(rows, 100, OPTIMAL_WINDOWS)
        if hr10 is not None:
            print(f"\n近10期回测：特五肖命中率 {hr10:.1%}，最大连空 {miss10}")
            print(f"近100期回测：特五肖命中率 {hr100:.1%}，最大连空 {miss100}")

if __name__ == "__main__":
    main()
