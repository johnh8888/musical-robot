#!/usr/bin/env python3
# zodiac_main_v7_fixed.py

import gzip
import json
import re
import urllib.request
from collections import Counter

API_URL = "https://marksix6.net/index.php?api=1"

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

ZLIST = list(ZODIAC_MAP.keys())

def zodiac(n):
    for z, nums in ZODIAC_MAP.items():
        if n in nums:
            return z
    return "马"

def fetch(limit=300):
    req = urllib.request.Request(
        API_URL,
        headers={"User-Agent":"Mozilla/5.0"}
    )

    with urllib.request.urlopen(req, timeout=20) as r:
        raw = r.read()

        if "gzip" in r.headers.get("Content-Encoding","").lower():
            raw = gzip.decompress(raw)

        data = json.loads(raw.decode())

    rows = []

    for item in data.get("lottery_data",[]):

        if "香港" not in item.get("name","") and "六合彩" not in item.get("name",""):
            continue

        for line in item.get("history",[]):

            m = re.match(r"(\d{7})\s*期[：:]\s*([\d,，]+)", line)

            if not m:
                continue

            nums = [
                int(x)
                for x in re.split(r"[，,]", m.group(2))
                if x.strip()
            ]

            if len(nums) < 7:
                continue

            rows.append({
                "numbers": nums[:6],
                "special": nums[6]
            })

    return rows[:limit]

def omission(rows):

    om = {z:0 for z in ZLIST}

    for r in reversed(rows):

        hit = {
            zodiac(x)
            for x in r["numbers"] + [r["special"]]
        }

        for z in ZLIST:
            om[z] = 0 if z in hit else om[z] + 1

    return om

def transition(rows):

    trans = Counter()

    recent = rows[-80:]

    for i in range(len(recent)-1):

        a = zodiac(recent[i]["special"])
        b = zodiac(recent[i+1]["special"])

        trans[(a,b)] += 1

    return trans

def volatility(rows):

    recent = rows[-12:]

    seq = [zodiac(r["special"]) for r in recent]

    return len(set(seq))/12

def score(rows, miss=0):

    sc = Counter()

    recent = rows[-60:]

    om = omission(rows)

    trans = transition(rows)

    vol = volatility(rows)

    for idx, r in enumerate(reversed(recent[-12:])):

        z = zodiac(r["special"])

        sc[z] += 1.2 * (0.88 ** idx)

    for idx, r in enumerate(reversed(recent[-30:])):

        z = zodiac(r["special"])

        sc[z] += 0.35 * (0.93 ** idx)

    if recent:

        last = zodiac(recent[-1]["special"])

        for z in ZLIST:
            sc[z] += trans[(last,z)] * 0.18

    for z in ZLIST:

        if vol > 0.72:
            sc[z] += om[z] * 0.26
        else:
            sc[z] += om[z] * 0.14

    hot = [z for z,_ in sc.most_common(4)]

    if miss >= 1:
        for z in hot:
            sc[z] *= 0.82

    if miss >= 2:

        cold = sorted(
            om,
            key=om.get,
            reverse=True
        )[:4]

        for z in cold:
            sc[z] += 2.2

    return sc

def predict(rows):
    sc = score(rows)                     # miss 默认为 0，不做连空惩罚
    ranked = [z for z, _ in sc.most_common()]

    om = omission(rows)
    trans = transition(rows)

    if not rows:
        return ["马"], ["马", "蛇"], ["马", "蛇", "龙"]

    last_z = zodiac(rows[-1]["special"])

    # 1热：评分最高
    hot_z = ranked[0]

    # 1趋势：从 last_z 出发转移次数最多的目标生肖
    targets = [b for (a, b) in trans if a == last_z]
    if targets:
        trend_z = max(targets, key=lambda b: trans[(last_z, b)])
    else:
        # 无转移记录时，用评分第二高的作为趋势
        trend_z = ranked[1] if len(ranked) > 1 else ranked[0]

    selected = {hot_z, trend_z}

    # 1极冷：遗漏值最大且不重复的生肖
    cold_sorted = sorted(om.items(), key=lambda x: x[1], reverse=True)
    cold_z = None
    for z, _ in cold_sorted:
        if z not in selected:
            cold_z = z
            break
    if cold_z is None:
        for z in ranked:
            if z not in selected:
                cold_z = z
                break
        if cold_z is None:
            cold_z = ranked[0]  # 极端兜底

    one = [hot_z]
    two = [hot_z, trend_z]
    three = [hot_z, trend_z, cold_z]

    return one, two, three

def backtest(rows):
    rev = list(reversed(rows))

    if len(rev) < 60:
        print("\n数据不足，无法回测")
        return

    total = min(100, len(rev) - 40)
    if total <= 0:
        print("\n回测样本不足")
        return

    h1 = h2 = h3 = 0
    miss1 = miss2 = miss3 = 0
    max1 = max2 = max3 = 0

    for i in range(40, 40 + total):
        # 用第 i 期之前的历史训练（严格无未来信息）
        train = list(reversed(rev[:i]))
        actual = {
            zodiac(n)
            for n in (rev[i]["numbers"] + [rev[i]["special"]])
        }

        one, two, three = predict(train)

        # 一肖
        if one[0] in actual:
            h1 += 1
            miss1 = 0
        else:
            miss1 += 1
            max1 = max(max1, miss1)

        # 二肖
        if any(z in actual for z in two):
            h2 += 1
            miss2 = 0
        else:
            miss2 += 1
            max2 = max(max2, miss2)

        # 三肖（至少中2个）
        hit3 = sum(1 for z in three if z in actual)
        if hit3 >= 2:
            h3 += 1
            miss3 = 0
        else:
            miss3 += 1
            max3 = max(max3, miss3)

    print("\n===== V7 回测（修复版） =====")
    print(f"测试期数: {total}")
    print(f"一肖: {h1/total:.1%}  连空{max1}")
    print(f"二肖: {h2/total:.1%}  连空{max2}")
    print(f"三肖: {h3/total:.1%}  连空{max3}")

if __name__ == "__main__":
    rows = fetch()
    one, two, three = predict(rows)

    print("\n【V7 自动模型】")
    print("一肖:", "、".join(one))
    print("二肖:", "、".join(two))
    print("三肖:", "、".join(three))

    backtest(rows)