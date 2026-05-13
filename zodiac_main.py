#!/usr/bin/env python3
# zodiac_main.py - 香港一二三生肖预测（适配 common.py）

import argparse
from collections import Counter
from itertools import combinations
from common import fetch_hk_records_merged, get_zodiac_by_number, ZODIAC_MAP

# ===================== 期号工具（兼容纯数字格式） =====================
def next_issue(issue_no: str) -> str:
    """兼容 '年/期号' 和纯数字（如 '2024133'）"""
    try:
        if '/' in issue_no:
            year, seq = issue_no.split('/')
        else:
            # 假设前4位是年份，后面是期数（至少3位）
            year = issue_no[:4]
            seq = issue_no[4:].lstrip('0') or '0'
        return f"{year}/{str(int(seq)+1).zfill(3)}"
    except:
        return issue_no

# ===================== 可调参数 =====================
CANDIDATE_SINGLE = [2, 3, 4, 6, 8, 10, 12, 15, 18, 20, 24, 30, 36, 42]
CANDIDATE_TWO    = [4, 6, 8, 10, 12, 15, 18, 20, 24, 30, 36, 42]
CANDIDATE_THREE  = [4, 6, 8, 10, 12, 15, 18, 20, 24, 30, 36, 42, 48]

OPTIMAL_SINGLE = [6, 10, 12, 18]
OPTIMAL_TWO    = [4, 6, 8, 10, 12]
OPTIMAL_THREE  = [8, 10, 12, 15, 30]

SINGLE_SPECIAL_BOOST = 3.2
TWO_SPECIAL_BOOST    = 3.0
MISS_PROTECTION      = 1        # 连空保护阈值

def window_weight(w, base=42):
    return round(base / w, 2)

# ===================== 数据准备 =====================
def get_history_rows(limit=None):
    """返回时间倒序的历史记录（最新在前），与 common.py 字段一致"""
    records = fetch_hk_records_merged(limit=limit, prefer_local=True)
    rows = []
    for r in records:
        rows.append({
            "numbers": r["numbers"],          # list[int]
            "special_number": r["special_number"],
            "issue_no": r["issue_no"],
            "draw_date": r["draw_date"]
        })
    return rows

# ===================== 策略核心 =====================
def zodiac_omission_map(rows):
    all_zodiacs = list(ZODIAC_MAP.keys())
    if not rows:
        return {z: 0 for z in all_zodiacs}
    om = {z: 0 for z in all_zodiacs}
    for row in reversed(rows):
        appeared = set(get_zodiac_by_number(n) for n in row["numbers"])
        appeared.add(get_zodiac_by_number(row["special_number"]))
        for z in all_zodiacs:
            om[z] = 0 if z in appeared else om[z] + 1
    return om

def predict_single(train, window, boost=SINGLE_SPECIAL_BOOST):
    if not train:
        return list(ZODIAC_MAP.keys())[0]
    recent = train[-window:] if len(train) >= window else train
    cnt = Counter()
    for r in recent:
        for n in r["numbers"]:
            cnt[get_zodiac_by_number(n)] += 1
        cnt[get_zodiac_by_number(r["special_number"])] += boost
    return cnt.most_common(1)[0][0] if cnt else list(ZODIAC_MAP.keys())[0]

def predict_two(train, window, boost=TWO_SPECIAL_BOOST):
    if not train:
        return list(ZODIAC_MAP.keys())[:2]
    recent = train[-window:] if len(train) >= window else train
    cnt = Counter()
    for r in recent:
        for n in r["numbers"]:
            cnt[get_zodiac_by_number(n)] += 1
        cnt[get_zodiac_by_number(r["special_number"])] += boost
    return [z for z, _ in cnt.most_common(2)]

def predict_three(train, window):
    if not train:
        return list(ZODIAC_MAP.keys())[:3]
    recent = train[-window:] if len(train) >= window else train
    cnt = Counter()
    for r in recent:
        for n in r["numbers"]:
            cnt[get_zodiac_by_number(n)] += 1
        cnt[get_zodiac_by_number(r["special_number"])] += 1
    return [z for z, _ in cnt.most_common(3)]

def predict_all(history, win_single, win_two, win_three):
    votes_s = Counter()
    wgt_s = {w: window_weight(w) for w in win_single}
    for w in win_single:
        s = predict_single(history, w)
        votes_s[s] += wgt_s[w]
    single = votes_s.most_common(1)[0][0]

    votes_t = Counter()
    wgt_t = {w: window_weight(w) for w in win_two}
    for w in win_two:
        for z in predict_two(history, w):
            votes_t[z] += wgt_t[w]
    two = [z for z, _ in votes_t.most_common(2)]

    votes_th = Counter()
    wgt_th = {w: window_weight(w, base=48) for w in win_three}
    for w in win_three:
        for z in predict_three(history, w):
            votes_th[z] += wgt_th[w]
    three = [z for z, _ in votes_th.most_common(3)]

    return single, two, three

# ===================== 回测（含连空保护） =====================
def backtest(rows, lookback, win_single, win_two, win_three):
    rows_rev = list(reversed(rows))
    total = min(lookback, len(rows_rev) - 20)
    if total <= 0:
        return None

    hits_s = hits_t = hits_th = 0
    miss_s = miss_t = miss_th = 0
    max_s = max_t = max_th = 0

    wgt_s = {w: window_weight(w) for w in win_single}
    wgt_t = {w: window_weight(w) for w in win_two}
    wgt_th = {w: window_weight(w, base=48) for w in win_three}

    for i in range(total):
        train = rows_rev[i+20:]
        if len(train) < 20:
            continue
        actual = rows_rev[i]
        win_z = set(get_zodiac_by_number(n) for n in actual["numbers"])
        win_z.add(get_zodiac_by_number(actual["special_number"]))
        omission = zodiac_omission_map(train)

        # 一肖
        votes_s = Counter()
        for w in win_single:
            s = predict_single(train, w)
            votes_s[s] += wgt_s[w]
        pred_s = votes_s.most_common(1)[0][0]
        if miss_s >= MISS_PROTECTION and omission:
            pred_s = max(omission, key=omission.get)

        # 二肖
        votes_t = Counter()
        for w in win_two:
            for z in predict_two(train, w):
                votes_t[z] += wgt_t[w]
        pred_t = [z for z, _ in votes_t.most_common(2)]
        if miss_t >= MISS_PROTECTION and omission:
            coldest = max(omission, key=omission.get)
            if coldest not in pred_t:
                pred_t[-1] = coldest

        # 三肖
        votes_th = Counter()
        for w in win_three:
            for z in predict_three(train, w):
                votes_th[z] += wgt_th[w]
        pred_th = [z for z, _ in votes_th.most_common(3)]
        if miss_th >= MISS_PROTECTION and omission:
            coldest2 = sorted(omission, key=omission.get, reverse=True)[:2]
            pred_th = [pred_th[0]] + [c for c in coldest2 if c != pred_th[0]]
            while len(pred_th) < 3:
                for z, _ in votes_th.most_common():
                    if z not in pred_th:
                        pred_th.append(z)
                        break
                else:
                    pred_th.append(list(ZODIAC_MAP.keys())[0])

        # 统计
        if pred_s in win_z:
            hits_s += 1; miss_s = 0
        else:
            miss_s += 1; max_s = max(max_s, miss_s)
        if any(z in win_z for z in pred_t):
            hits_t += 1; miss_t = 0
        else:
            miss_t += 1; max_t = max(max_t, miss_t)
        if sum(1 for z in pred_th if z in win_z) >= 2:
            hits_th += 1; miss_th = 0
        else:
            miss_th += 1; max_th = max(max_th, miss_th)

    return {
        "single_hit": hits_s / total, "single_miss": max_s,
        "two_hit": hits_t / total, "two_miss": max_t,
        "three_hit": hits_th / total, "three_miss": max_th,
    }

# ===================== 窗口优化 =====================
def optimize_single(rows, opt_lookback=40):
    best_combo = [6,10,12,18]
    best_score = -1
    for combo in combinations(CANDIDATE_SINGLE, 4):
        stats = backtest(rows, opt_lookback, list(combo), OPTIMAL_TWO, OPTIMAL_THREE)
        if stats is None: continue
        score = stats["single_hit"] - stats["single_miss"] * 0.1
        if score > best_score:
            best_score = score
            best_combo = list(combo)
    return best_combo

def optimize_two(rows, opt_lookback=40):
    best_combo = [4,6,8,10,12]
    best_score = -1
    for combo in combinations(CANDIDATE_TWO, 5):
        stats = backtest(rows, opt_lookback, OPTIMAL_SINGLE, list(combo), OPTIMAL_THREE)
        if stats is None: continue
        score = stats["two_hit"] - stats["two_miss"] * 0.1
        if score > best_score:
            best_score = score
            best_combo = list(combo)
    return best_combo

def optimize_three(rows, opt_lookback=40):
    best_combo = [8,10,12,15,30]
    best_score = -1
    for combo in combinations(CANDIDATE_THREE, 5):
        stats = backtest(rows, opt_lookback, OPTIMAL_SINGLE, OPTIMAL_TWO, list(combo))
        if stats is None: continue
        score = stats["three_hit"] - stats["three_miss"] * 0.1
        if score > best_score:
            best_score = score
            best_combo = list(combo)
    return best_combo

# ===================== 主程序 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    rows = get_history_rows()
    if not rows:
        print("数据获取失败")
        return

    global OPTIMAL_SINGLE, OPTIMAL_TWO, OPTIMAL_THREE
    print("正在优化一肖窗口...", end=" ")
    OPTIMAL_SINGLE = optimize_single(rows, 40)
    print(OPTIMAL_SINGLE)
    print("正在优化二肖窗口...", end=" ")
    OPTIMAL_TWO = optimize_two(rows, 40)
    print(OPTIMAL_TWO)
    print("正在优化三肖窗口...", end=" ")
    OPTIMAL_THREE = optimize_three(rows, 40)
    print(OPTIMAL_THREE)

    if args.show:
        latest_issue = rows[0]["issue_no"]
        pred_issue = next_issue(latest_issue)
        print(f"\n预测期号: {pred_issue}")
        print(f"使用历史数据截止至: {latest_issue}")

        single, two, three = predict_all(rows, OPTIMAL_SINGLE, OPTIMAL_TWO, OPTIMAL_THREE)
        print(f"一生肖: {single}")
        print(f"二生肖: {'、'.join(two)}")
        print(f"三生肖: {'、'.join(three)}")

        stats = backtest(rows, 10, OPTIMAL_SINGLE, OPTIMAL_TWO, OPTIMAL_THREE)
        if stats:
            print(f"\n近10期回测：一生肖 {stats['single_hit']:.1%} 连空{stats['single_miss']}")
            print(f"二生肖 {stats['two_hit']:.1%} 连空{stats['two_miss']}")
            print(f"三生肖 {stats['three_hit']:.1%} 连空{stats['three_miss']}")

if __name__ == "__main__":
    main()
