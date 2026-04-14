#!/usr/bin/env python3
# ==================== 新澳门六合彩 - 修正加强版 ====================
import argparse
import json
import math
import requests
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple

# ---------- 配置 ----------
ZODIAC_MAP = {
    "马": [1,13,25,37,49], "羊": [12,24,36,48], "猴": [11,23,35,47],
    "鸡": [10,22,34,46], "狗": [9,21,33,45], "猪": [8,20,32,44],
    "鼠": [7,19,31,43], "牛": [6,18,30,42], "虎": [5,17,29,41],
    "兔": [4,16,28,40], "龙": [3,15,27,39], "蛇": [2,14,26,38]
}

# 修正后的波色（蓝波补全 21,26,32）
COLOR_MAP = {
    "红": [1,2,7,8,12,13,18,19,23,24,29,30,34,35,40,45,46],
    "蓝": [3,4,9,10,14,15,20,21,25,26,31,32,36,37,41,42,47,48],
    "绿": [5,6,11,16,17,22,27,28,33,38,39,43,44,49]
}

ALL_NUMBERS = list(range(1, 50))
DATA_FILE = Path("macau_history.json")
MAX_HISTORY = 200  # 最多保存期数
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
    url = "https://marksix6.net/index.php?api=1"
    print("🌐 正在获取新澳门最新开奖...")

    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            print(f"❌ 请求失败: HTTP {r.status_code}")
            return False

        data = r.json()
        lottery_list = data.get("lottery_data", [])
        if not lottery_list:
            print("⚠️ 接口返回数据为空")
            return False

        new_count = 0
        existing = {d.get("issue") for d in history}

        for lottery in lottery_list:
            name = str(lottery.get("name", ""))
            # 识别新澳门数据（排除老澳门）
            if any(k in name for k in ["新澳门", "澳门", "澳彩", "Macau", "macau"]) and "老" not in name:
                issue = str(lottery.get("expect") or lottery.get("issue")).strip()
                open_code = lottery.get("openCode") or lottery.get("numbers")
                if not issue or not open_code:
                    continue

                try:
                    nums = [int(x.strip()) for x in str(open_code).split(",") if x.strip()]
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

    except requests.exceptions.RequestException as e:
        print(f"❌ 网络异常: {e}")
        return False
    except Exception as e:
        print(f"❌ 解析失败: {e}")
        return False


# ---------- 分析核心 ----------
def get_recent_draws(limit: int) -> List[List[int]]:
    """获取最近 limit 期正码列表"""
    recent = history[-limit:] if len(history) >= limit else history
    return [d["numbers"] for d in recent]


def calculate_zodiac_score(draws: List[List[int]], method: str = "recent_3") -> Counter:
    """计算生肖得分"""
    zodiac_counter = Counter()
    if method == "recent_3":
        source = draws[-3:] if len(draws) >= 3 else draws
    else:
        source = draws

    for draw in source:
        for n in draw:
            for z, nums in ZODIAC_MAP.items():
                if n in nums:
                    zodiac_counter[z] += 1
                    break
    return zodiac_counter


def calculate_number_heat(draws: List[List[int]], window: int = 30) -> Counter:
    """号码热度（可指定窗口）"""
    flat = [n for draw in draws[-window:] for n in draw]
    return Counter(flat)


def calculate_special_score(draws: List[List[int]], specials: List[int]) -> Dict[int, float]:
    """特别号加权评分（指数衰减）"""
    scores = {n: 0.0 for n in ALL_NUMBERS}
    total_draws = min(40, len(draws))
    for i in range(total_draws):
        weight = math.exp(-i / 8)
        # 正码贡献
        for n in draws[-(i+1)]:
            scores[n] += weight * 1.5
        # 特别号额外权重
        if i < len(specials):
            scores[specials[-(i+1)]] += weight * 3.0
    return scores


def get_cold_numbers(draws: List[List[int]], window: int = 30) -> List[int]:
    """获取冷号（window期内未出现的号码）"""
    appeared = set(n for draw in draws[-window:] for n in draw)
    return [n for n in ALL_NUMBERS if n not in appeared]


def ensemble_vote(draws: List[List[int]], specials: List[int]) -> Tuple[List[int], int]:
    """
    简单集成投票：
    - 热号策略：频率最高
    - 动量策略：近期指数加权
    - 冷号策略：长期遗漏
    综合得出6个正码和1个特别号
    """
    # 策略1：热号 (30期频次)
    hot_counter = calculate_number_heat(draws, 30)
    hot_scores = {n: hot_counter[n] for n in ALL_NUMBERS}

    # 策略2：动量 (10期指数加权)
    momentum_scores = {n: 0.0 for n in ALL_NUMBERS}
    for i, draw in enumerate(draws[-10:]):
        weight = math.exp(-i / 3)
        for n in draw:
            momentum_scores[n] += weight

    # 策略3：冷号回补 (50期内未出次数)
    cold_window = min(50, len(draws))
    cold_counter = calculate_number_heat(draws, cold_window)
    cold_scores = {n: (cold_window - cold_counter[n]) for n in ALL_NUMBERS}

    # 归一化并合并
    def normalize(d: Dict[int, float]) -> Dict[int, float]:
        vals = list(d.values())
        mn, mx = min(vals), max(vals)
        if mx == mn:
            return {k: 0.0 for k in d}
        return {k: (v - mn) / (mx - mn) for k, v in d.items()}

    hot_norm = normalize(hot_scores)
    mom_norm = normalize(momentum_scores)
    cold_norm = normalize(cold_scores)

    final_scores = {}
    for n in ALL_NUMBERS:
        final_scores[n] = hot_norm[n] * 0.4 + mom_norm[n] * 0.4 + cold_norm[n] * 0.2

    # 选6个正码（避免全单全双全大全小）
    sorted_nums = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    picked = []
    for n, _ in sorted_nums:
        if len(picked) == 6:
            break
        test = picked + [n]
        odd_cnt = sum(1 for x in test if x % 2 == 1)
        big_cnt = sum(1 for x in test if x >= 25)
        if len(test) >= 4 and (odd_cnt == 0 or odd_cnt == len(test)):
            continue
        if len(test) >= 4 and (big_cnt == 0 or big_cnt == len(test)):
            continue
        picked.append(n)

    # 补足
    while len(picked) < 6:
        for n, _ in sorted_nums:
            if n not in picked:
                picked.append(n)
                break

    # 特别号
    special_scores = calculate_special_score(draws, specials)
    special = max(special_scores, key=lambda n: special_scores[n] if n not in picked else -1)

    return picked, special


# ---------- 预测展示 ----------
def show_prediction() -> None:
    load_history()
    print("\n" + "=" * 85)
    print("🎯 新澳门六合彩 - 修正加强版 (多策略集成)")
    print("=" * 85)

    if not history:
        print("⚠️ 暂无数据，请先运行 sync 获取数据。")
        return

    latest = history[-1]
    nums_str = " ".join(f"{n:02d}" for n in latest["numbers"])
    print(f"📅 最新开奖 → 第 {latest['issue']} 期")
    print(f"   正码: {nums_str}   特别号: {latest['special']:02d}\n")

    draws = [d["numbers"] for d in history]
    specials = [d["special"] for d in history]

    # 1. 一肖推荐（修正：使用最近3期，概率基于3期）
    zodiac_counter = calculate_zodiac_score(draws, method="recent_3")
    top_zodiacs = zodiac_counter.most_common(3)
    print("1️⃣ 一肖推荐（近3期热度）")
    for i, (zod, cnt) in enumerate(top_zodiacs, 1):
        prob = round(cnt / 3 * 100, 1)
        star = "⭐" if i == 1 else "  "
        print(f"   {star} {zod}: 出现 {cnt}/3 期 ({prob}%)")
    if top_zodiacs:
        print(f"   💡 建议: 重点关注 {top_zodiacs[0][0]}，可搭配 {top_zodiacs[1][0] if len(top_zodiacs)>1 else ''}")

    # 2. 冷热号分析
    hot_counter = calculate_number_heat(draws, 30)
    hot6 = [n for n, _ in hot_counter.most_common(6)]
    cold5 = get_cold_numbers(draws, 30)[:5]

    print("\n2️⃣ 冷热号参考")
    print(f"   🔥 热号(近30期高频): {' '.join(f'{n:02d}' for n in hot6)}")
    print(f"   ❄️ 冷号(30期未出): {' '.join(f'{n:02d}' for n in cold5) if cold5 else '无'}")

    # 3. 集成投票推荐
    ensemble_nums, ensemble_special = ensemble_vote(draws, specials)
    print("\n3️⃣ 集成投票推荐 (热号+动量+冷号)")
    print(f"   🎲 正码6码: {' '.join(f'{n:02d}' for n in ensemble_nums)}")
    print(f"   🔮 特别号: {ensemble_special:02d}")

    # 计算该推荐的历史平均命中率（简单回测）
    if len(history) >= 20:
        hits = []
        for i in range(20, len(history)):
            past_draws = draws[:i]
            past_specials = specials[:i]
            pred_nums, pred_special = ensemble_vote(past_draws, past_specials)
            actual_nums = set(draws[i])
            hit_cnt = len([n for n in pred_nums if n in actual_nums])
            hits.append(hit_cnt)
        avg_hit = sum(hits) / len(hits) if hits else 0
        print(f"   📊 近{len(hits)}期回测平均命中: {avg_hit:.2f} 码")

    # 4. 趋势分析
    latest_nums = latest["numbers"]
    odd = sum(1 for n in latest_nums if n % 2 == 1)
    big = sum(1 for n in latest_nums if n >= 25)
    red = sum(1 for n in latest_nums if n in COLOR_MAP["红"])
    blue = sum(1 for n in latest_nums if n in COLOR_MAP["蓝"])
    green = 6 - red - blue

    print("\n4️⃣ 上期形态")
    print(f"   单/双: {odd}/{6-odd}  |  大/小: {big}/{6-big}")
    print(f"   波色: 红{red} 蓝{blue} 绿{green}")

    print("\n" + "=" * 85)
    print("⚠️ 理性提醒：以上分析基于历史统计，仅供娱乐参考。")
    print("=" * 85)


# ---------- 主程序 ----------
def main():
    parser = argparse.ArgumentParser(description="新澳门六合彩预测工具（修正加强版）")
    parser.add_argument("cmd", choices=["sync", "show", "auto"], nargs="?", default="show",
                        help="sync:同步数据, show:显示推荐, auto:同步后显示")
    args = parser.parse_args()

    if args.cmd == "sync":
        fetch_new_macau_only()
    elif args.cmd == "auto":
        fetch_new_macau_only()
        show_prediction()
    else:  # show
        show_prediction()


if __name__ == "__main__":
    main()
