#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新澳门六合彩 - 最终完整版（正码5个命中分析基于近6期）
用法:
    python macau_bet.py sync
    python macau_bet.py show
    python macau_bet.py auto
"""

import argparse
import json
import math
import requests
from collections import Counter
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

# ---------- 配置 ----------
ZODIAC_MAP = {
    "马": [1,13,25,37,49], "羊": [12,24,36,48], "猴": [11,23,35,47],
    "鸡": [10,22,34,46], "狗": [9,21,33,45], "猪": [8,20,32,44],
    "鼠": [7,19,31,43], "牛": [6,18,30,42], "虎": [5,17,29,41],
    "兔": [4,16,28,40], "龙": [3,15,27,39], "蛇": [2,14,26,38]
}

ALL_NUMBERS = list(range(1, 50))
DATA_FILE = Path("macau_history.json")
MAX_HISTORY = 300
history = []


def get_zodiac(num: int) -> str:
    for z, nums in ZODIAC_MAP.items():
        if num in nums:
            return z
    return ""


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


# ---------- 核心算法 ----------
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
    for i, draw in enumerate(draws[-15:]):
        weight = math.exp(-i / 4)
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


def optimize_weights(draws: List[List[int]], specials: List[int], test_window: int = 30) -> Tuple[Dict[str, float], float]:
    if len(draws) < test_window + 10:
        return {"hot": 0.5, "momentum": 0.3, "cold": 0.2}, 0.0

    best_weights = {"hot": 0.5, "momentum": 0.3, "cold": 0.2}
    best_score = 0.0

    for w_hot in [0.3, 0.4, 0.5, 0.6]:
        for w_mom in [0.2, 0.3, 0.4]:
            w_cold = 1.0 - w_hot - w_mom
            if w_cold < 0.1 or w_cold > 0.4:
                continue

            total_hits = 0
            count = 0
            for i in range(test_window, len(draws)):
                past_draws = draws[:i]
                past_specials = specials[:i]
                pred_nums, _ = ensemble_vote_core(past_draws, past_specials,
                                                  weights=(w_hot, w_mom, w_cold))
                actual = set(draws[i])
                total_hits += len([n for n in pred_nums if n in actual])
                count += 1

            avg_hits = total_hits / count if count else 0
            if avg_hits > best_score:
                best_score = avg_hits
                best_weights = {"hot": w_hot, "momentum": w_mom, "cold": w_cold}

    return best_weights, best_score


def ensemble_vote_core(draws: List[List[int]],
                       specials: List[int],
                       weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)) -> Tuple[List[int], int]:
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

    ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    picked = []
    for n, _ in ranked:
        if len(picked) == 6:
            break
        if n not in picked:
            picked.append(n)

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


def analyze_hit_expectation(hot5: List[int], draws: List[List[int]], backtest_games: int = 6) -> Dict:
    """
    基于最近 backtest_games 期数据，统计如果每期都买这5个号码，平均能中几个。
    """
    if len(draws) < backtest_games:
        backtest_games = len(draws)

    total_hits = 0
    hit_counts = []
    for draw in draws[-backtest_games:]:
        hits = len(set(hot5) & set(draw))
        hit_counts.append(hits)
        total_hits += hits

    avg_hits = total_hits / backtest_games if backtest_games > 0 else 0
    at_least_one = sum(1 for h in hit_counts if h >= 1) / backtest_games * 100 if backtest_games > 0 else 0

    single_probs = {}
    for n in hot5:
        hits = sum(1 for draw in draws[-backtest_games:] if n in draw)
        single_probs[n] = hits / backtest_games * 100 if backtest_games > 0 else 0

    return {
        "avg_hits": avg_hits,
        "at_least_one": at_least_one,
        "single_probs": single_probs
    }


# ---------- 输出展示 ----------
def show_prediction() -> None:
    load_history()

    if len(history) < 30:
        print("⚠️ 历史数据不足（至少30期），请先运行 sync 获取数据。")
        return

    draws = [d["numbers"] for d in history]
    specials = [d["special"] for d in history]
    latest = history[-1]

    # 上期开奖记录
    print("\n" + "=" * 55)
    print("📋 上期开奖记录")
    print("-" * 55)
    print(f"期号: {latest['issue']}")
    nums_str = " ".join(f"{n:02d}({get_zodiac(n)})" for n in latest["numbers"])
    print(f"正码: {nums_str}")
    print(f"特别号: {latest['special']:02d}({get_zodiac(latest['special'])})")
    print("=" * 55)

    # 优化权重
    best_w, _ = optimize_weights(draws, specials)
    weights_tuple = (best_w['hot'], best_w['momentum'], best_w['cold'])
    picked_6, picked_special = ensemble_vote_core(draws, specials, weights=weights_tuple)

    # 计算最终得分，选出前5个正码
    hot_scores = calculate_frequency(draws[-40:])
    mom_scores = calculate_momentum(draws)
    omission = calculate_omission(draws)
    cold_scores = {n: omission[n] / max(1, len(draws)) for n in ALL_NUMBERS}
    def normalize(d): 
        vals = list(d.values())
        mn, mx = min(vals), max(vals)
        if mx == mn: return {k: 0.0 for k in d}
        return {k: (v-mn)/(mx-mn) for k,v in d.items()}
    hot_n = normalize(hot_scores); mom_n = normalize(mom_scores); cold_n = normalize(cold_scores)
    final_scores = {n: hot_n[n]*best_w['hot'] + mom_n[n]*best_w['momentum'] + cold_n[n]*best_w['cold'] for n in ALL_NUMBERS}
    sorted_6 = sorted(picked_6, key=lambda n: final_scores[n], reverse=True)
    hot5 = sorted_6[:5]

    # 生肖分析（近5期）
    zodiac_score = Counter()
    for draw in draws[-5:]:
        for n in draw:
            for z, nums in ZODIAC_MAP.items():
                if n in nums:
                    zodiac_score[z] += 1
    top_zod = zodiac_score.most_common(2)
    top1 = top_zod[0][0] if top_zod else "龙"
    top2 = top_zod[1][0] if len(top_zod) > 1 else "马"

    special_zod = get_zodiac(picked_special)

    def zodiac_hit_rate(zod, draws, limit=5):
        hits = 0
        for draw in draws[-limit:]:
            if any(n in ZODIAC_MAP[zod] for n in draw):
                hits += 1
        return hits / limit * 100

    rate1 = zodiac_hit_rate(top1, draws)
    rate2 = zodiac_hit_rate(top2, draws)

    # 命中分析（近6期）
    hit_analysis = analyze_hit_expectation(hot5, draws, backtest_games=6)

    # 输出
    print("\n🎯 本期投注推荐")
    print("-" * 55)
    print(f"🐉 最强生肖: {top1}  (近5期命中率 {rate1:.0f}%)")
    print(f"🐉 次强生肖: {top2}  (近5期命中率 {rate2:.0f}%)")
    print(f"🎲 正码5个:")
    for n in hot5:
        prob = hit_analysis['single_probs'].get(n, 0)
        print(f"      {n:02d} ({get_zodiac(n)})  ─ 近6期命中率 {prob:.0f}%")
    print(f"🔮 特别号: {picked_special:02d} ({special_zod})")
    print("-" * 55)
    print(f"📊 正码5个预期表现 (基于近6期回测):")
    print(f"      • 平均每期命中: {hit_analysis['avg_hits']:.2f} 个")
    print(f"      • 至少命中1个的概率: {hit_analysis['at_least_one']:.0f}%")
    print("=" * 55)
    print("⚠️ 数据仅供参考，理性投注。\n")


# ---------- 主程序 ----------
def main():
    parser = argparse.ArgumentParser(description="新澳门六合彩-最终完整版")
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
