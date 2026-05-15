#!/usr/bin/env python3
# zodiac_main_v7.py

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

            m = re.match(r"(\\d{7})\\s*期[：:]\\s*([\\d,，]+)", line)

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

def predict(rows, miss=0):

    sc = score(rows, miss)

    rank = [z for z,_ in sc.most_common()]

    one = rank[:1]

    om = omission(rows)

    cold = sorted(
        om,
        key=om.get,
        reverse=True
    )

    two = [rank[0]]

    for z in cold:
        if z not in two:
            two.append(z)
            break

    three = [rank[0]]

    for z in cold:

        if z not in three:
            three.append(z)

        if len(three) >= 3:
            break

    return one, two, three

def backtest(rows):

    rev = list(reversed(rows))

    miss1 = miss2 = miss3 = 0
    max1 = max2 = max3 = 0

    h1 = h2 = h3 = 0

    total = 0

    for i in range(120, len(rev)-1):

        train = rev[i:]

        actual = {
            zodiac(x)
            for x in rev[i-1]["numbers"] + [rev[i-1]["special"]]
        }

        one, two, three = predict(train, miss3)

        total += 1

        if one[0] in actual:
            h1 += 1
            miss1 = 0
        else:
            miss1 += 1
            max1 = max(max1, miss1)

        if any(z in actual for z in two):
            h2 += 1
            miss2 = 0
        else:
            miss2 += 1
            max2 = max(max2, miss2)

        cnt = sum(1 for z in three if z in actual)

        if cnt >= 2:
            h3 += 1
            miss3 = 0
        else:
            miss3 += 1
            max3 = max(max3, miss3)

    print("\\n===== 回测 =====")

    print(f"一肖: {h1/total:.1%} 连空{max1}")
    print(f"二肖: {h2/total:.1%} 连空{max2}")
    print(f"三肖(中2): {h3/total:.1%} 连空{max3}")

rows = fetch()

one, two, three = predict(rows)

print("\\n【V7 自动模型】")

print("一肖:", "、".join(one))
print("二肖:", "、".join(two))
print("三肖:", "、".join(three))

backtest(rows)