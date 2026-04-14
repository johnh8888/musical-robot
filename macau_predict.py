#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新澳门六合彩 - 终极增强版（全自动在线数据 + 动态权重 + 蒙特卡洛）
用法:
    python macau_predict.py sync   # 从线上同步全年历史数据
    python macau_predict.py show   # 显示本期智能推荐（默认）
    python macau_predict.py auto   # 同步后显示推荐
"""

import argparse
import json
import math
import random
import requests
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import List, Dict, Tuple, Set
from datetime import datetime

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
MAX_HISTORY = 300  # 保留最近300期用于分析
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
    # 按期号排序并截断
    history.sort(key=lambda x: x["issue"])
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def fetch_new_macau_only() -> bool:
    """从线上历史接口批量获取当前年份的新澳门数据"""
    year = datetime.now().year  # 自动使用当前年份（如2026）
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


# ---------- 特征计算 ----------
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


def get_hot_pairs(draws: List[List[int]], top_n: int = 15) -> List[Tuple[Tuple[int, int], int]]:
    """挖掘高频关联对（连号、同尾、同生肖）"""
    pair_count = Counter()
    for draw in draws[-50:]:
        for a, b in combinations(sorted(draw), 2):
            score = 1
            if abs(a - b) == 1:
                score = 4  # 连号
            elif a % 10 == b % 10:
                score = 3  # 同尾
            # 同生肖
            for z, nums in ZODIAC_MAP.items():
                if a in nums and b in nums:
                    score = max(score, 2)
                    break
            pair_count[(a, b)] += score
    return pair_count.most_common(top_n)


# ---------- 动态权重优化 ----------
def optimize_weights(draws: List[List[int]], specials: List[int], test_window: int = 40) -> Tuple[Dict[str, float], float]:
    """网格搜索最优权重组合"""
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


# ---------- 核心选号引擎 ----------
def smart_filter(nums: List[int]) -> bool:
    """过滤不合理组合"""
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

    # 连号不超过3连
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

    # 同尾不超过2个
    tails = [n % 10 for n in s]
    if max(Counter(tails).values()) > 2:
        return False

    # 同区间不超过3个
    zones = [(n - 1) // 10 for n in s]
    if max(Counter(zones).values()) > 3:
        return False

    return True


def monte_carlo_pick(scores: Dict[int, float],
                     pair_bonus: Dict[Tuple[int, int], float],
                     omission: Dict[int, int],
                     trials: int = 2000) -> List[int]:
    """蒙特卡洛模拟选最优6码组合"""
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:35]
    candidates = [n for n, _ in ranked]

    best_combo = []
    best_total = -1e9

    for _ in range(trials):
        combo = sorted(random.sample(candidates, 6))
        if not smart_filter(combo):
            continue

        total = sum(scores[n] for n in combo)

        # 关联对加分
        for a, b in combinations(combo, 2):
            total += pair_bonus.get((a, b), 0)
            total += pair_bonus.get((b, a), 0)

        # 遗漏加分：15-25期额外加
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
    """核心集成投票"""
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

    # 特别号
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


# ---------- 预测展示 ----------
def show_prediction() -> None:
    load_history()
    print("\n" + "=" * 85)
    print("🎯 新澳门六合彩 - 终极增强版 (动态权重 + 蒙特卡洛)")
    print("=" * 85)

    if len(history) < 30:
        print("⚠️ 历史数据不足（至少30期），请先运行 sync 获取数据。")
        return

    draws = [d["numbers"] for d in history]
    specials = [d["special"] for d in history]
    latest = history[-1]

    print("⚙️ 正在优化权重...")
    best_w, avg_hit = optimize_weights(draws, specials)
    print(f"📈 最优权重: 热号{best_w['hot']:.2f} 动量{best_w['momentum']:.2f} 冷号{best_w['cold']:.2f}")
    print(f"   近40期回测平均命中: {avg_hit:.2f} 码")

    weights_tuple = (best_w['hot'], best_w['momentum'], best_w['cold'])
    picked_nums, picked_special = ensemble_vote_core(draws, specials, weights=weights_tuple, use_monte_carlo=True)

    print("\n" + "=" * 85)
    print("🎲 本期智能推荐")
    print("=" * 85)
    print(f"📅 最新开奖参考: 第 {latest['issue']} 期")
    print(f"   正码: {' '.join(f'{n:02d}' for n in latest['numbers'])}  特别号: {latest['special']:02d}\n")

    print(f"🌟 正码 6 码: {' '.join(f'{n:02d}' for n in picked_nums)}")
    print(f"🔮 特别号: {picked_special:02d}")

    hot_counter = calculate_frequency(draws[-30:])
    hot6 = [n for n, _ in sorted(hot_counter.items(), key=lambda x: x[1], reverse=True)[:6]]
    omission = calculate_omission(draws)
    cold5 = [n for n, omit in sorted(omission.items(), key=lambda x: x[1], reverse=True)[:5]]

    print("\n📊 冷热号参考")
    print(f"   🔥 热号(近30期): {' '.join(f'{n:02d}' for n in hot6)}")
    print(f"   ❄️ 冷号(遗漏最长): {' '.join(f'{n:02d}' for n in cold5)}")

    zodiac_counter = Counter()
    for draw in draws[-5:]:
        for n in draw:
            for z, nums in ZODIAC_MAP.items():
                if n in nums:
                    zodiac_counter[z] += 1
    top_zod = zodiac_counter.most_common(3)
    print(f"\n🐉 热门生肖: {' > '.join([z for z, _ in top_zod])}")

    hot_pairs = get_hot_pairs(draws, top_n=5)
    print("\n🔗 近期高频关联对 (供连码参考):")
    for (a, b), score in hot_pairs:
        print(f"   {a:02d}-{b:02d} (热度: {score})")

    print("\n" + "=" * 85)
    print("⚠️ 理性提醒：以上分析基于历史统计，仅供娱乐参考。")
    print("=" * 85)


# ---------- 主程序 ----------
def main():
    parser = argparse.ArgumentParser(description="新澳门六合彩预测工具（终极增强版）")
    parser.add_argument("cmd", choices=["sync", "show", "auto"], nargs="?", default="show",
                        help="sync:同步数据, show:显示推荐, auto:同步后显示")
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
