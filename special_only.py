#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import gzip
import json
import re
import time
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

ZODIAC_LIST = list(ZODIAC_MAP.keys())

DECAY_ALPHA = 0.84

def decay(i):
    return DECAY_ALPHA ** i

def get_zodiac(n):

    for z, nums in ZODIAC_MAP.items():

        if n in nums:
            return z

    return "马"

def parse_nums(value):

    return [
        int(x)
        for x in re.split(r"[，,]", value)
        if x.strip().isdigit()
    ]

def fetch_hk_online(limit=300):

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    for _ in range(5):

        try:

            req = urllib.request.Request(API_URL, headers=headers)

            with urllib.request.urlopen(req, timeout=20) as resp:

                raw = resp.read()

                if "gzip" in resp.headers.get("Content-Encoding", ""):
                    raw = gzip.decompress(raw)

                data = json.loads(raw.decode("utf-8"))

                rows = []

                for item in data.get("lottery_data", []):

                    if "香港" not in item.get("name", ""):
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

                        issue = m.group(1)

                        issue_no = f"{issue[2:4]}/{int(issue[4:]):03d}"

                        rows.append({
                            "issue_no": issue_no,
                            "numbers": nums[:6],
                            "special_number": nums[6]
                        })

                rows = sorted(
                    rows,
                    key=lambda x:x["issue_no"],
                    reverse=True
                )

                return rows[:limit]

        except:
            time.sleep(2)

    return []

def omission_map(rows):

    om = {z:0 for z in ZODIAC_LIST}

    for r in reversed(rows):

        z = get_zodiac(r["special_number"])

        for k in ZODIAC_LIST:
            om[k] = 0 if k == z else om[k] + 1

    return om

def stable_score(history):

    score = Counter()

    recent = history[-90:]

    for idx, r in enumerate(reversed(recent[-15:])):

        z = get_zodiac(r["special_number"])

        score[z] += 0.72 * decay(idx)

    for idx, r in enumerate(reversed(recent[-45:])):

        z = get_zodiac(r["special_number"])

        score[z] += 0.21 * decay(idx)

    om = omission_map(history)

    for z in ZODIAC_LIST:

        score[z] += min(
            om[z] * 0.14,
            2.6
        ) * 0.36

    return score

def recommend(history):

    score = stable_score(history)

    om = omission_map(history)

    ranked = [z for z,_ in score.most_common()]

    cold = sorted(
        om.items(),
        key=lambda x:x[1],
        reverse=True
    )

    cold = [z for z,_ in cold[:5]]

    final = ranked[:3]

    for z in cold:

        if z not in final:

            final.append(z)

        if len(final) >= 5:
            break

    return final[:5]

def backtest(rows):

    rev = list(reversed(rows))

    total = 0

    hit = 0

    miss = 0

    max_miss = 0

    total10 = 0

    hit10 = 0

    for i in range(100, len(rev)-1):

        train = rev[i:]

        if len(train) < 60:
            continue

        total += 1

        actual = get_zodiac(
            rev[i-1]["special_number"]
        )

        pred = recommend(train)

        if actual in pred:

            hit += 1

            miss = 0

        else:

            miss += 1

            max_miss = max(max_miss, miss)

        if total <= 10:

            total10 += 1

            if actual in pred:
                hit10 += 1

    print("\n===== 特五肖 V10 =====")

    if total > 0:

        print(f"100期命中率: {hit/total:.1%}")
        print(f"100期最大连空: {max_miss}")

    if total10 > 0:

        print("\n===== 最近10期 =====")

        print(f"10期命中率: {hit10/total10:.1%}")

def main():

    print("正在获取最新数据...")

    rows = fetch_hk_online(300)

    if not rows:

        print("数据获取失败")

        return

    pred = recommend(rows)

    print("\n【特五肖 V10 职业版】")

    print("、".join(pred))

    backtest(rows)

if __name__ == "__main__":
    main()