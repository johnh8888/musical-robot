#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# zodiac_main.py
# 香港六合彩 · 一二三肖 V8 职业量化版
#
# 特性：
# 1. 真正时间序回测（无未来函数）
# 2. 转移链模型
# 3. 状态机
# 4. 周期识别
# 5. 熔断器
# 6. 波动率切换
# 7. 三肖强制：
#    1热 + 1趋势 + 1极冷
#
# GitHub Actions 直接运行兼容
#

import gzip
import json
import re
import time
import urllib.request

from collections import Counter, defaultdict

API_URL = "https://marksix6.net/index.php?api=1"

# =========================
# 正确生肖映射（2026版）
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
# 获取数据
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

    for z in ZODIAC_LIST:

        miss = 0

        for r in history:

            cur = set(
                get_zodiac(n)
                for n in (
                    r["numbers"] +
                    [r["special_number"]]
                )
            )

            if z in cur:
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

        weight = 3.5 - idx * 0.15

        for n in r["numbers"] + [r["special_number"]]:

            score[get_zodiac(n)] += weight

    # 中周期
    for idx, r in enumerate(recent_mid):

        weight = 1.8 - idx * 0.03

        for n in r["numbers"] + [r["special_number"]]:

            score[get_zodiac(n)] += weight

    # 长周期
    for r in recent_long:

        for n in r["numbers"] + [r["special_number"]]:

            score[get_zodiac(n)] += 0.45

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

    # 单边
    if ratio >= 0.45:
        return "HOT"

    # 混乱
    if len(set(specials[:6])) >= 5:
        return "CHAOS"

    return "NORMAL"

# =========================
# 周期识别
# =========================

def cycle_boost(history):

    score = Counter()

    specials = [
        get_zodiac(r["special_number"])
        for r in history[:36]
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

            # 接近周期
            if abs(last_gap-avg_gap) <= 1.5:

                score[z] += 2.2

    return score

# =========================
# 波动率识别
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
# 核心预测
# =========================

def professional_predict(history, miss3=0):

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

                score[z] += p * 8.0

    # =====================
    # 状态机
    # =====================

    if state == "HOT":

        hottest = hot.most_common(1)[0][0]

        score[hottest] += 5.0

    elif state == "CHAOS":

        for z, v in sorted(
            om.items(),
            key=lambda x: x[1],
            reverse=True
        )[:4]:

            score[z] += 3.5

    # =====================
    # 波动率
    # =====================

    if vol == "HIGH":

        for z, v in sorted(
            om.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]:

            score[z] += 2.2

    elif vol == "LOW":

        hottest = hot.most_common(3)

        for z, _ in hottest:

            score[z] += 2.5

    # =====================
    # 熔断器
    # =====================

    if miss3 >= 1:

        for z, v in sorted(
            om.items(),
            key=lambda x: x[1],
            reverse=True
        )[:4]:

            score[z] += 5.5

    ranked = [
        z for z, _
        in score.most_common()
    ]

    # =====================
    # 一肖
    # =====================

    one = [ranked[0]]

    # =====================
    # 二肖
    # =====================

    two = ranked[:2]

    # =====================
    # 三肖核心
    # 1热 + 1趋势 + 1极冷
    # =====================

    hot_one = ranked[0]

    trend_one = ranked[1]

    cold_one = sorted(
        om.items(),
        key=lambda x: x[1],
        reverse=True
    )[0][0]

    three = []

    for z in [hot_one, trend_one, cold_one]:

        if z not in three:
            three.append(z)

    for z in ranked:

        if z not in three:
            three.append(z)

    three = three[:3]

    return one, two, three

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

    h1 = h2 = h3 = 0

    miss1 = miss2 = miss3 = 0

    max1 = max2 = max3 = 0

    for i in range(40, 40 + total):

        train = list(
            reversed(rev[:i])
        )

        actual = set(
            get_zodiac(n)
            for n in (
                rev[i]["numbers"] +
                [rev[i]["special_number"]]
            )
        )

        s, t, th = professional_predict(
            train,
            miss3
        )

        # 一肖
        if s[0] in actual:

            h1 += 1
            miss1 = 0

        else:

            miss1 += 1
            max1 = max(max1, miss1)

        # 二肖
        if any(z in actual for z in t):

            h2 += 1
            miss2 = 0

        else:

            miss2 += 1
            max2 = max(max2, miss2)

        # 三肖（必须中2）
        hit3 = sum(
            1 for z in th
            if z in actual
        )

        if hit3 >= 2:

            h3 += 1
            miss3 = 0

        else:

            miss3 += 1
            max3 = max(max3, miss3)

    print("\n===== V8 职业量化回测 =====")

    print(f"测试期数: {total}")

    print(
        f"一肖: {h1/total:.1%} | 最大连空 {max1}"
    )

    print(
        f"二肖: {h2/total:.1%} | 最大连空 {max2}"
    )

    print(
        f"三肖: {h3/total:.1%} | 最大连空 {max3}"
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

    s, t, th = professional_predict(rows)

    print("\n【V8 职业量化模型】")

    print(f"预测期号: {next_issue(latest)}")

    print(f"一生肖: {'、'.join(s)}")

    print(f"二生肖: {'、'.join(t)}")

    print(f"三生肖: {'、'.join(th)}")

    backtest(rows)

# =========================

if __name__ == "__main__":
    main()