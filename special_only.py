#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gzip
import json
import re
import time
import urllib.request
from collections import Counter

API_URL = "https://marksix6.net/index.php?api=1"

ZODIAC_MAP = {
    "马":[1,13,25,37,49],
    "蛇":[2,14,26,38],
    "龙":[3,15,27,39],
    "兔":[4,16,28,40],
    "虎":[5,17,29,41],
    "牛":[6,18,30,42],
    "鼠":[7,19,31,43],
    "猪":[8,20,32,44],
    "狗":[9,21,33,45],
    "鸡":[10,22,34,46],
    "猴":[11,23,35,47],
    "羊":[12,24,36,48],
}

ZODIAC_LIST = list(ZODIAC_MAP.keys())

def get_zodiac(n):

    for z, nums in ZODIAC_MAP.items():

        if n in nums:
            return z

    return "马"

def parse_nums(s):

    out = []

    for x in re.split(r"[，,]", s):

        x = x.strip()

        if x.isdigit():

            n = int(x)

            if 1 <= n <= 49:
                out.append(n)

    return out

def fetch_data(limit=300):

    headers = {
        "User-Agent":"Mozilla/5.0"
    }

    for _ in range(5):

        try:

            req = urllib.request.Request(API_URL, headers=headers)

            with urllib.request.urlopen(req, timeout=20) as resp:

                raw = resp.read()

                if "gzip" in resp.headers.get("Content-Encoding",""):
                    raw = gzip.decompress(raw)

                data = json.loads(raw.decode("utf-8"))

                rows = []

                for item in data.get("lottery_data", []):

                    if "香港" not in item.get("name",""):
                        continue

                    for line in item.get("history", []):

                        m = re.match(r"(\d{7})\s*期[：:]\s*([\d,，]+)", line)

                        if not m:
                            continue

                        nums = parse_nums(m.group(2))

                        if len(nums) < 7:
                            continue

                        rows.append({
                            "numbers": nums[:6],
                            "special_number": nums[6]
                        })

                if rows:
                    return rows

        except:
            time.sleep(2)

    return []

def omission_map(history):

    om = {z:0 for z in ZODIAC_LIST}

    for r in reversed(history):

        z = get_zodiac(r["special_number"])

        for k in om:

            om[k] = 0 if k == z else om[k]+1

    return om

def recommend(history, miss=0):

    score = Counter()

    om = omission_map(history)

    recent = history[-100:]

    for idx, r in enumerate(reversed(recent)):

        z = get_zodiac(r["special_number"])

        score[z] += (0.985 ** idx)

    ranked = [z for z,_ in score.most_common()]

    hot = ranked[:3]

    mid = ranked[3:8]

    cold = sorted(om.items(), key=lambda x:x[1], reverse=True)

    final = []

    # 2热
    final.extend(hot[:2])

    # 2温
    for z in mid:

        if z not in final:
            final.append(z)

        if len(final) >= 4:
            break

    # 1冷
    for z,_ in cold:

        if z not in final:
            final.append(z)
            break

    # 连空保护
    if miss >= 2:

        for z,_ in cold:

            if z not in final:
                final[-1] = z
                break

    return final[:5]

def backtest(rows, lookback):

    total = 0

    hit = 0

    miss = 0

    max_miss = 0

    for i in range(100, len(rows)-1):

        train = rows[:i]

        if len(train) < 100:
            continue

        total += 1

        actual = get_zodiac(rows[i]["special_number"])

        pred = recommend(train, miss)

        if actual in pred:

            hit += 1
            miss = 0

        else:

            miss += 1
            max_miss = max(max_miss, miss)

    if total == 0:
        total = 1

    print(f"\n===== 最近{lookback}期 =====")
    print(f"命中率: {hit/total:.1%}")
    print(f"最大连空: {max_miss}")

def main():

    print("正在获取最新数据...")

    rows = fetch_data()

    if not rows:
        print("获取失败")
        return

    pred = recommend(rows)

    print("\n【特五肖 V12 组合版】")
    print("、".join(pred))

    backtest(rows[-110:],10)

    backtest(rows[-200:],100)

if __name__ == "__main__":
    main()