#!/usr/bin/env python3
# special_only_v7.py

import gzip
import json
import re
import urllib.request
from collections import Counter

API_URL = "https://marksix6.net/index.php?api=1"

ZMAP = {
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

ZLIST = list(ZMAP.keys())

def zodiac(n):

    for z, nums in ZMAP.items():

        if n in nums:
            return z

    return "马"

def fetch():

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

            if len(nums) >= 7:
                rows.append({
                    "special": nums[6]
                })

    return rows[:300]

def omission(rows):

    om = {z:0 for z in ZLIST}

    for r in reversed(rows):

        hit = zodiac(r["special"])

        for z in ZLIST:
            om[z] = 0 if z == hit else om[z] + 1

    return om

def score(rows, miss=0):

    sc = Counter()

    om = omission(rows)

    recent = rows[-80:]

    for idx, r in enumerate(reversed(recent[-15:])):

        sc[zodiac(r["special"])] += 1.3 * (0.87 ** idx)

    for idx, r in enumerate(reversed(recent[-40:])):

        sc[zodiac(r["special"])] += 0.32 * (0.94 ** idx)

    for z in ZLIST:
        sc[z] += om[z] * 0.22

    if miss >= 1:

        hot = [z for z,_ in sc.most_common(4)]

        for z in hot:
            sc[z] *= 0.80

    if miss >= 2:

        cold = sorted(
            om,
            key=om.get,
            reverse=True
        )[:5]

        for z in cold:
            sc[z] += 2.5

    return sc

def recommend(rows, miss=0):

    sc = score(rows, miss)

    rank = [z for z,_ in sc.most_common()]

    hot = rank[:3]

    cold = sorted(
        omission(rows),
        key=omission(rows).get,
        reverse=True
    )

    final = hot[:]

    for z in cold:

        if z not in final:
            final.append(z)

        if len(final) >= 5:
            break

    return final

def backtest(rows):

    rev = list(reversed(rows))

    hit = 0
    miss = 0
    maxmiss = 0
    total = 0

    for i in range(120, len(rev)-1):

        train = rev[i:]

        actual = zodiac(rev[i-1]["special"])

        pred = recommend(train, miss)

        total += 1

        if actual in pred:

            hit += 1
            miss = 0

        else:

            miss += 1
            maxmiss = max(maxmiss, miss)

    print("\\n===== 特五肖回测 =====")

    print(f"命中率: {hit/total:.1%}")
    print(f"最大连空: {maxmiss}")

rows = fetch()

pred = recommend(rows)

print("\\n【特五肖 V7】")

print("、".join(pred))

backtest(rows)