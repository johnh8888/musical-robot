#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# special_only.py
# 香港六合彩 · 特五肖 V8 职业量化版
#
# 特性：
# 1. 真正时间序回测
# 2. 转移链模型
# 3. 周期识别
# 4. 状态机
# 5. 熔断器
# 6. 波动率切换
# 7. 动态冷热融合
# 8. 特五肖：
#    2热 + 2趋势 + 1极冷
#
# GitHub Actions 兼容
#

import gzip
import json
import re
import time
import urllib.request

from collections import Counter, defaultdict

API_URL = "https://marksix6.net/index.php?api=1"

# =========================
# 正确生肖映射（2026）
# =========================

ZODIAC_MAP = {
    "马": [1,13,25,37,49],
    "蛇": [2,14,26,38],
    "龙": [3,15,27,39],
    "兔": [4,16,28,40],
    "虎": [5,17,29,41],
    "牛": [6,18,30,42],
    "鼠": [7,19,31,43],
    "猪": [8,20,32,44],
    "狗": [9,21,33,45],
    "鸡": [10,22,34,46],
    "猴": [11,23,35,47],
    "羊": [12,24,36,48],
}

ZODIAC_LIST = list(ZODIAC_MAP.keys())

# =========================
# 工具
# =========================

def get_zodiac(n):

    for z, nums in ZODIAC_MAP.items():

        if n in nums:
            return z

    return "马"

def next_issue(issue_no):

    try:

        if "/" in issue_no:

            y, s = issue_no.split("/")

        else:

            y = issue_no[:4]
            s = issue_no[4:]

        return f"{y}/{str(int(s)+1).zfill(3)}"

    except:

        return issue_no

# =========================
# 数据获取
# =========================

def parse_nums(value):

    out = []

    for t in re.split(r"[，,]", value):

        t = t.strip()

        if not t:
            continue

        try:

            n = int(t)

            if 1 <= n <= 49:
                out.append(n)

        except:
            pass

    return out

def fetch_hk(limit=300):

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    for _ in range(5):

        try:

            req = urllib.request.Request(
                API_URL,
                headers=headers
            )

            with urllib.request.urlopen(req, timeout=20) as resp:

                raw = resp.read()

                if "gzip" in resp.headers.get(
                    "Content-Encoding",
                    ""
                ).lower():

                    raw = gzip.decompress(raw)

                data = json.loads(raw.decode("utf-8"))

                rows = []

                for item in data.get("lottery_data", []):

                    name = item.get("name","")

                    if (
                        "香港" not in name and
                        "六合彩" not in name
                    ):
                        continue

                    for line in item.get("history", []):

                        m = re.match(
                            r"(\d{7})\s*期[：:]\s*([\d,，]+)",
                            line
                        )

                        if not m:
                            continue

                        nums = parse_nums(m.group(2))

                        if len(nums) < 7:
                            continue

                        raw_issue = m.group(1)

                        issue_no = (
                            f"{raw_issue[2:4]}/"
                            f"{int(raw_issue[4:]):03d}"
                        )

                        rows.append({
                            "issue_no": issue_no,
                            "numbers": nums[:6],
                            "special_number": nums[6]
                        })

                rows = sorted(
                    rows,
                    key=lambda x: x["issue_no"],
                    reverse=True
                )

                return rows[:limit]

        except:
            time.sleep(2)

    return []

# =========================
# 遗漏值
# =========================

def omission_map(history):

    om = {}

    specials = [
        get_zodiac(r["special_number"])
        for r in history
    ]

    for z in ZODIAC_LIST:

        miss = 0

        for x in specials:

            if x == z:
                break

            miss += 1

        om[z] = miss

    return om

# =========================
# 热度模型
# =========================

def hot_score(history):

    score = Counter()

    recent_short = history[:12]
    recent_mid = history[:30]
    recent_long = history[:60]

    # 短周期
    for idx, r in enumerate(recent_short):

        weight = 4.2 - idx * 0.18

        z = get_zodiac(
            r["special_number"]
        )

        score[z] += weight

    # 中周期
    for idx, r in enumerate(recent_mid):

        weight = 2.0 - idx * 0.04

        z = get_zodiac(
            r["special_number"]
        )

        score[z] += weight

    # 长周期
    for r in recent_long:

        z = get_zodiac(
            r["special_number"]
        )

        score[z] += 0.55

    return score

# =========================
# 转移链
# =========================

def transition_model(history):

    trans = defaultdict(Counter)

    specials = [
        get_zodiac(r["special_number"])
        for r in history
    ]

    for i in range(len(specials)-1):

        cur = specials[i+1]
        nxt = specials[i]

        trans[cur][nxt] += 1

    return trans

# =========================
# 周期识别
# =========================

def cycle_boost(history):

    score = Counter()

    specials = [
        get_zodiac(r["special_number"])
        for r in history[:48]
    ]

    for z in ZODIAC_LIST:

        pos = []

        for i, x in enumerate(specials):

            if x == z:
                pos.append(i)

        if len(pos) >= 2:

            gaps = []

            for i in range(len(pos)-1):
                gaps.append(pos[i+1]-pos[i])

            avg_gap = sum(gaps)/len(gaps)

            last_gap = pos[0]

            if abs(last_gap-avg_gap) <= 2:

                score[z] += 3.2

    return score

# =========================
# 状态机
# =========================

def detect_state(history):

    specials = [
        get_zodiac(r["special_number"])
        for r in history[:15]
    ]

    freq = Counter(specials)

    top = freq.most_common(1)[0][1]

    ratio = top / len(specials)

    if ratio >= 0.45:
        return "HOT"

    if len(set(specials[:6])) >= 5:
        return "CHAOS"

    return "NORMAL"

# =========================
# 波动率
# =========================

def volatility_mode(history):

    specials = [
        get_zodiac(r["special_number"])
        for r in history[:12]
    ]

    uniq = len(set(specials))

    if uniq >= 8:
        return "HIGH"

    if uniq <= 4:
        return "LOW"

    return "MID"

# =========================
# 职业预测核心
# =========================

def professional_predict(history, miss=0):

    hot = hot_score(history)

    om = omission_map(history)

    trans = transition_model(history)

    cyc = cycle_boost(history)

    state = detect_state(history)

    vol = volatility_mode(history)

    score = Counter()

    # =====================
    # 热度
    # =====================

    for z in ZODIAC_LIST:

        score[z] += hot[z]

    # =====================
    # 周期
    # =====================

    for z in cyc:

        score[z] += cyc[z]

    # =====================
    # 转移链
    # =====================

    last_sp = get_zodiac(
        history[0]["special_number"]
    )

    if last_sp in trans:

        total = sum(
            trans[last_sp].values()
        )

        if total > 0:

            for z, v in trans[last_sp].items():

                p = v / total

                score[z] += p * 10.0

    # =====================
    # 状态机
    # =====================

    if state == "HOT":

        hottest = hot.most_common(3)

        for z, _ in hottest:

            score[z] += 4.5

    elif state == "CHAOS":

        for z, _ in sorted(
            om.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]:

            score[z] += 4.0

    # =====================
    # 波动率
    # =====================

    if vol == "HIGH":

        for z, _ in sorted(
            om.items(),
            key=lambda x: x[1],
            reverse=True
        )[:6]:

            score[z] += 2.5

    elif vol == "LOW":

        hottest = hot.most_common(5)

        for z, _ in hottest:

            score[z] += 2.8

    # =====================
    # 熔断器
    # =====================

    if miss >= 1:

        for z, _ in sorted(
            om.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]:

            score[z] += 6.5

    ranked = [
        z for z, _
        in score.most_common()
    ]

    # =====================
    # 特五肖核心
    # 2热 + 2趋势 + 1极冷
    # =====================

    final = []

    # 热
    hot_part = ranked[:2]

    # 趋势
    trend_part = []

    if last_sp in trans:

        trend_sorted = sorted(
            trans[last_sp].items(),
            key=lambda x: x[1],
            reverse=True
        )

        for z, _ in trend_sorted:

            if z not in trend_part:
                trend_part.append(z)

    # 极冷
    cold_part = sorted(
        om.items(),
        key=lambda x: x[1],
        reverse=True
    )

    cold_one = cold_part[0][0]

    for z in hot_part:

        if z not in final:
            final.append(z)

    for z in trend_part:

        if z not in final:
            final.append(z)

        if len(final) >= 4:
            break

    if cold_one not in final:
        final.append(cold_one)

    for z in ranked:

        if z not in final:
            final.append(z)

    return final[:5]

# =========================
# 回测
# =========================

def backtest(rows):

    rev = list(reversed(rows))

    if len(rev) < 80:

        print("\n数据不足")
        return

    total = min(
        100,
        len(rev) - 40
    )

    if total <= 0:

        print("\n回测不足")
        return

    hits = 0

    miss = 0

    max_miss = 0

    for i in range(40, 40 + total):

        train = list(
            reversed(rev[:i])
        )

        actual = get_zodiac(
            rev[i]["special_number"]
        )

        pred = professional_predict(
            train,
            miss
        )

        if actual in pred:

            hits += 1

            miss = 0

        else:

            miss += 1

            max_miss = max(
                max_miss,
                miss
            )

    print("\n===== 特五肖 V8 回测 =====")

    print(f"测试期数: {total}")

    print(
        f"命中率: {hits/total:.1%}"
    )

    print(
        f"最大连空: {max_miss}"
    )

# =========================
# 主程序
# =========================

def main():

    print("正在获取最新数据...")

    rows = fetch_hk(300)

    if not rows:

        print("数据获取失败")
        return

    latest = rows[0]["issue_no"]

    pred = professional_predict(rows)

    print("\n【特五肖 V8 职业量化模型】")

    print(f"预测期号: {next_issue(latest)}")

    print(
        f"特五肖: {'、'.join(pred)}"
    )

    backtest(rows)

# =========================

if __name__ == "__main__":
    main()