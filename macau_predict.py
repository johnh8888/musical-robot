#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新澳门六合彩 - 内部分析强大 · 输出精简投注
用法:
    python macau_bet.py sync   # 同步全年数据
    python macau_bet.py show   # 显示投注推荐（默认）
    python macau_bet.py auto   # 同步并显示
"""

import argparse
import json
import math
import random
import requests
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Set

# ---------- 配置 ----------
ZODIAC_MAP = {
    "马": [1,13,25,37,49], "羊": [12,24,36,48], "猴": [11,23,35,47],
    "鸡": [10,22,34,46], "狗": [9,21,33,45], "猪": [8,20,32,44],
    "鼠": [7,19,31,43], "牛": [6,18,30,42], "虎": [5,17,29,41],
    "兔": [4,16,28,40], "龙": [3,15,27,39], "蛇": [2,14,26,38]
}

COLOR_MAP = {
    "红": [1,2,7,8,12,13,18,19,23,24,29,30,34,35,40,45,46],
    "蓝": [3,4,9,10,14,15,20,21,25,26,31,32,36,37,41,42,47,48],
    "绿": [5,6,11,16,17,22,27,28,33,38,39,43,44,49]
}

ALL_NUMBERS = list(range(1, 50))
DATA_FILE = Path("macau_history.json")
MAX_HISTORY = 300
history = []


# ---------- 数据管理 ----------
def load_history() -> None:
    global history
    if DATA_FILE.exists():
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except:
            history = []


def save_history() -> None:
    global history
    history.sort(key=lambda x: x["issue"])
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def fetch_new_macau_only() -> bool:
    """从线上历史接口批量获取当前年份的新澳门数据"""
    year = datetime.now().year
    url = f"https://history.macaumarksix.com/history/macaujc2/y/{year}"
    print(f"🌐 正在从线上获取 {year} 年新澳门历史数据...")

    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            print(f"❌ 请求失败: HTTP {r.status_code}")
            return False

        data = r.json()
        if data.get("code") != 200 or not data.get("data"):
            print("⚠️ 接口返回数据为空")
            return False

        new_count = 0
        existing = {d.get("issue") for d in history}

        for item in data["data"]:
            issue = str(item.get("expect", "")).strip()
            if not issue.startswith(str(year)):
                continue

            open_code = item.get("openCode", "")
            if not open_code:
                continue

            try:
                nums = [int(x.strip()) for x in open_code.split(",") if x.strip().isdigit()]
            except:
                continue

            if len(nums) >= 6 and issue not in existing:
                history.append({
                    "issue": issue,
                    "numbers": nums[:6],
                    "special": nums[6] if len(nums) > 6 else nums[-1]
                })
                new_count += 1
                print(f"   ✅ 新增第 {issue} 期")

        if new_count > 0:
            save_history()
            print(f"🎉 本次新增 {new_count} 期，历史共 {len(history)} 期")
            return True
        else:
            print("ℹ️ 暂无新数据")
            return False

    except Exception as e:
        print(f"❌ 获取失败: {e}")
        return False


# ---------- 内部强大分析模块（冷热号、波色、大小单双、遗漏等） ----------
def get_recent_draws(limit: int) -> List[List[int]]:
    recent = history[-limit:] if len(history) >= limit else history
    return [d["numbers"] for d in recent]


def get_recent_specials(limit: int) -> List[int]:
    recent = history[-limit:] if len(history) >= limit else history
    return [d["special"] for d in recent]


def calculate_frequency(draws: List[List[int]]) -> Dict[int, float]:
    freq = {n: 0.0 for n in ALL_NUMBERS}
    for draw in draws:
        for n in draw:
            freq[n] += 1.0
    return freq


def calculate_momentum(draws: List[List[int]]) -> Dict[int, float]:
    mom = {n: 0.0 for n in ALL_NUMBERS}
    for i, draw in enumerate(draws[-20:]):
        weight = math.exp(-i / 5)
        for n in draw:
            mom[n] += weight
    return mom


def calculate_omission(draws: List[List[int]]) -> Dict[int, int]:
    om = {}
    for n in ALL_NUMBERS:
        for i, draw in enumerate(draws):
            if n in draw:
                om[n] = i
                break
        else:
            om[n] = len(draws)
    return om


def get_color_trend(draws: List[List[int]], limit: int = 10) -> Counter:
    recent = draws[-limit:] if len(draws) >= limit else draws
    color_count = Counter()
    for draw in recent:
        for n in draw:
            for c, nums in COLOR_MAP.items():
                if n in nums:
                    color_count[c] += 1
    return color_count


def get_odd_even_big_small(draws: List[List[int]]) -> Dict:
    flat = [n for draw in draws[-10:] for n in draw]
    odd = sum(1 for n in flat if n % 2 == 1)
    big = sum(1 for n in flat if n >= 25)
    return {
        "odd": odd, "even": len(flat) - odd,
        "big": big, "small": len(flat) - big
    }


def get_hot_pairs(draws: List[List[int]], top_n: int = 15) -> List[Tuple[Tuple[int, int], int]]:
    pair_count = Counter()
    for draw in draws[-50:]:
        for a, b in combinations(sorted(draw), 2):
            score = 1
            if abs(a - b) == 1:
                score = 4
            elif a % 10 == b % 10:
                score = 3
            for z, nums in ZODIAC_MAP.items():
                if a in nums and b in nums:
                    score = max(score, 2)
                    break
            pair_count[(a, b)] += score
    return pair_count.most_common(top_n)


def optimize_weights(draws: List[List[int]], specials: List[int], test_window: int = 40) -> Tuple[Dict[str, float], float]:
    if len(draws) < test_window + 10:
        return {"hot": 0.4, "momentum": 0.4, "cold": 0.2}, 0.0

    best_weights = {"hot": 0.4, "momentum": 0.4, "cold": 0.2}
    best_score = 0.0

    for w_hot in [0.2, 0.3, 0.4, 0.5, 0.6]:
        for w_mom in [0.2, 0.3, 0.4, 0.5]:
            w_cold = 1.0 - w_hot - w_mom
            if w_cold < 0.1 or w_cold > 0.4:
                continue

            total_hits = 0
            count = 0
            for i in range(test_window, len(draws)):
                past_draws = draws[:i]
                past_specials = specials[:i]
                pred_nums, _ = ensemble_vote_core(past_draws, past_specials,
                                                  weights=(w_hot, w_mom, w_cold),
                                                  use_monte_carlo=False)
                actual = set(draws[i])
                total_hits += len([n for n in pred_nums if n in actual])
                count += 1

            avg_hits = total_hits / count if count else 0
            if avg_hits > best_score:
                best_score = avg_hits
                best_weights = {"hot": w_hot, "momentum": w_mom, "cold": w_cold}

    return best_weights, best_score


def smart_filter(nums: List[int]) -> bool:
    if len(nums) != 6:
        return False
    s = sorted(nums)
    total = sum(s)
    odd = sum(1 for n in s if n % 2 == 1)
    big = sum(1 for n in s if n >= 25)

    if total < 90 or total > 210:
        return False
    if odd == 0 or odd == 6:
        return False
    if big == 0 or big == 6:
        return False

    consec = 1
    max_consec = 1
    for i in range(1, 6):
        if s[i] - s[i-1] == 1:
            consec += 1
            max_consec = max(max_consec, consec)
        else:
            consec = 1
    if max_consec > 3:
        return False

    tails = [n % 10 for n in s]
    if max(Counter(tails).values()) > 2:
        return False

    zones = [(n - 1) // 10 for n in s]
    if max(Counter(zones).values()) > 3:
        return False

    return True


def monte_carlo_pick(scores: Dict[int, float],
                     pair_bonus: Dict[Tuple[int, int], float],
                     omission: Dict[int, int],
                     trials: int = 2000) -> List[int]:
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:35]
    candidates = [n for n, _ in ranked]

    best_combo = []
    best_total = -1e9

    for _ in range(trials):
        combo = sorted(random.sample(candidates, 6))
        if not smart_filter(combo):
            continue

        total = sum(scores[n] for n in combo)

        for a, b in combinations(combo, 2):
            total += pair_bonus.get((a, b), 0)
            total += pair_bonus.get((b, a), 0)

        for n in combo:
            omit = omission.get(n, 0)
            if 15 <= omit <= 25:
                total += 0.15

        if total > best_total:
            best_total = total
            best_combo = combo

    if not best_combo:
        picked = []
        for n, _ in ranked:
            if len(picked) == 6:
                break
            if n not in picked:
                picked.append(n)
        best_combo = sorted(picked)

    return best_combo


def ensemble_vote_core(draws: List[List[int]],
                       specials: List[int],
                       weights: Tuple[float, float, float] = (0.4, 0.4, 0.2),
                       use_monte_carlo: bool = True) -> Tuple[List[int], int]:
    w_hot, w_mom, w_cold = weights

    hot_scores = calculate_frequency(draws[-40:]) if len(draws) >= 40 else calculate_frequency(draws)
    mom_scores = calculate_momentum(draws)
    omission = calculate_omission(draws)
    cold_scores = {n: omission[n] / max(1, len(draws)) for n in ALL_NUMBERS}

    def normalize(d: Dict[int, float]) -> Dict[int, float]:
        vals = list(d.values())
        mn, mx = min(vals), max(vals)
        if mx == mn:
            return {k: 0.0 for k in d}
        return {k: (v - mn) / (mx - mn) for k, v in d.items()}

    hot_norm = normalize(hot_scores)
    mom_norm = normalize(mom_scores)
    cold_norm = normalize(cold_scores)

    final_scores = {}
    for n in ALL_NUMBERS:
        final_scores[n] = (hot_norm[n] * w_hot +
                           mom_norm[n] * w_mom +
                           cold_norm[n] * w_cold)

    hot_pairs = get_hot_pairs(draws, top_n=20)
    pair_bonus = {pair: score / 100.0 for pair, score in hot_pairs}

    if use_monte_carlo and len(draws) >= 30:
        picked = monte_carlo_pick(final_scores, pair_bonus, omission, trials=1500)
    else:
        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        picked = []
        for n, _ in ranked:
            if len(picked) == 6:
                break
            if n not in picked:
                picked.append(n)
        picked = sorted(picked)

    special_scores = {n: 0.0 for n in ALL_NUMBERS}
    for i in range(min(40, len(draws))):
        weight = math.exp(-i / 8)
        for n in draws[-(i+1)]:
            special_scores[n] += weight * 1.5
        if i < len(specials):
            special_scores[specials[-(i+1)]] += weight * 3.0

    for n in picked:
        special_scores[n] = -1e9
    special = max(special_scores, key=lambda n: special_scores[n])

    picked = [n for n in picked if n != special]
    while len(picked) < 6:
        for n, _ in sorted(final_scores.items(), key=lambda x: x[1], reverse=True):
            if n not in picked and n != special:
                picked.append(n)
                break

    return picked, special


# ---------- 精简投注输出 ----------
def show_prediction() -> None:
    load_history()

    if len(history) < 30:
        print("⚠️ 历史数据不足（至少30期），请先运行 sync 获取数据。")
        return

    draws = [d["numbers"] for d in history]
    specials = [d["special"] for d in history]
    latest = history[-1]

    # 内部优化权重
    best_w, avg_hit = optimize_weights(draws, specials)
    weights_tuple = (best_w['hot'], best_w['momentum'], best_w['cold'])
    picked_6, picked_special = ensemble_vote_core(draws, specials, weights=weights_tuple, use_monte_carlo=True)

    # 精选5个正码（从6个中去掉一个相对最弱的，保留前5）
    # 这里使用内部得分排序，选出得分最高的5个
    hot_scores = calculate_frequency(draws[-40:])
    mom_scores = calculate_momentum(draws)
    omission = calculate_omission(draws)
    cold_scores = {n: omission[n] / max(1, len(draws)) for n in ALL_NUMBERS}
    def normalize(d): 
        vals = list(d.values()); mn, mx = min(vals), max(vals); return {k: (v-mn)/(mx-mn) if mx!=mn else 0.0 for k,v in d.items()}
    hot_n = normalize(hot_scores); mom_n = normalize(mom_scores); cold_n = normalize(cold_scores)
    final_scores = {n: hot_n[n]*best_w['hot'] + mom_n[n]*best_w['momentum'] + cold_n[n]*best_w['cold'] for n in ALL_NUMBERS}
    sorted_6 = sorted(picked_6, key=lambda n: final_scores[n], reverse=True)
    hot5 = sorted_6[:5]

    # 最强/次强生肖（基于近期正码加权）
    zodiac_score = Counter()
    for i, draw in enumerate(draws[-8:]):
        weight = math.exp(-i / 3)
        for n in draw:
            for z, nums in ZODIAC_MAP.items():
                if n in nums:
                    zodiac_score[z] += weight
    top_zod = zodiac_score.most_common(2)
    top1 = top_zod[0][0] if top_zod else "龙"
    top2 = top_zod[1][0] if len(top_zod) > 1 else "马"

    # 特别号生肖
    special_zod = "龙"
    for z, nums in ZODIAC_MAP.items():
        if picked_special in nums:
            special_zod = z
            break

    # 生肖近期命中率（近8期）
    def zodiac_hit_rate(zod, draws, limit=8):
        hits = 0
        for draw in draws[-limit:]:
            if any(n in ZODIAC_MAP[zod] for n in draw):
                hits += 1
        return hits / limit * 100

    rate1 = zodiac_hit_rate(top1, draws)
    rate2 = zodiac_hit_rate(top2, draws)

    # 生成三中三组合（从5个号码中选3个，共10组）
    three_combos = list(combinations(sorted(hot5), 3))

    # 输出精简投注单
    print("\n" + "=" * 50)
    print("🎯 新澳门六合彩 · 投注推荐单")
    print("=" * 50)
    print(f"📅 参考期号: {latest['issue']}")
    print("-" * 50)
    print(f"🐉 最强生肖: {top1}  (近8期命中率 {rate1:.0f}%)")
    print(f"🐉 次强生肖: {top2}  (近8期命中率 {rate2:.0f}%)")
    print(f"🎲 正码5个: {' '.join(f'{n:02d}' for n in hot5)}")
    print(f"🔮 特别号生肖: {special_zod}")
    print("-" * 50)
    print("🎰 三中三组合 (共10组):")
    for i, combo in enumerate(three_combos, 1):
        print(f"   {i:2d}. {' '.join(f'{n:02d}' for n in combo)}")
    print("=" * 50)
    print("⚠️ 数据仅供参考，理性投注。\n")


# ---------- 主程序 ----------
def main():
    parser = argparse.ArgumentParser(description="新澳门六合彩-内部分析强大/输出精简")
    parser.add_argument("cmd", choices=["sync", "show", "auto"], nargs="?", default="show")
    args = parser.parse_args()

    if args.cmd == "sync":
        fetch_new_macau_only()
    elif args.cmd == "auto":
        fetch_new_macau_only()
        show_prediction()
    else:
        show_prediction()


if __name__ == "__main__":
    main()
