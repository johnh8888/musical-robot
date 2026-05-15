#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gzip
import json
import re
import time
import urllib.request
from collections import Counter
from itertools import combinations

API_URL = "https://marksix6.net/index.php?api=1"

# 2026生肖映射
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

def next_issue(issue):
    try:
        y, s = issue.split("/")
        return f"{y}/{int(s)+1:03d}"
    except:
        return issue

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

                        issue = m.group(1)

                        issue_no = f"{issue[2:4]}/{int(issue[4:]):03d}"

                        rows.append({
                            "issue_no": issue_no,
                            "numbers": nums[:6],
                            "special_number": nums[6]
                        })

                if rows:
                    return sorted(rows, key=lambda x:x["issue_no"])

        except:
            time.sleep(2)

    return []

def omission_map(history):

    om = {z:0 for z in ZODIAC_LIST}

    for r in reversed(history):

        s = set(get_zodiac(n) for n in r["numbers"]+[r["special_number"]])

        for z in ZODIAC_LIST:
            om[z] = 0 if z in s else om[z]+1

    return om

def build_combo_scores(history):

    combo_score = Counter()

    recent = history[-120:]

    for idx, r in enumerate(reversed(recent)):

        weight = 0.985 ** idx

        zset = set(get_zodiac(n) for n in r["numbers"]+[r["special_number"]])

        for combo in combinations(sorted(zset), 3):

            combo_score[combo] += weight

    return combo_score

def predict(history):

    combo_score = build_combo_scores(history)

    om = omission_map(history)

    best_combo = None
    best_score = -1

    for combo, score in combo_score.items():

        vals = [om[z] for z in combo]

        extra = 0

        if max(vals) >= 8:
            extra += 0.6

        if min(vals) <= 1:
            extra += 0.2

        extra += 0.3

        final_score = score + extra

        if final_score > best_score:
            best_score = final_score
            best_combo = combo

    th = list(best_combo)

    pair_score = Counter()

    for combo, s in combo_score.items():

        for p in combinations(combo, 2):
            pair_score[p] += s

    best_pair = list(pair_score.most_common(1)[0][0])

    single = Counter()

    for combo, s in combo_score.items():
        for z in combo:
            single[z] += s

    best_single = [single.most_common(1)[0][0]]

    return best_single, best_pair, th

def backtest(rows, lookback):

    total = 0

    h1=h2=h3=0

    m1=m2=m3=0

    max1=max2=max3=0

    for i in range(120, len(rows)-1):

        train = rows[:i]

        if len(train) < 120:
            continue

        actual = set(get_zodiac(n) for n in rows[i]["numbers"]+[rows[i]["special_number"]])

        s,b,t = predict(train)

        total += 1

        # 一肖
        if s[0] in actual:
            h1 += 1
            m1 = 0
        else:
            m1 += 1
            max1 = max(max1,m1)

        # 二肖
        if any(z in actual for z in b):
            h2 += 1
            m2 = 0
        else:
            m2 += 1
            max2 = max(max2,m2)

        # 三肖中二
        cnt = sum(1 for z in t if z in actual)

        if cnt >= 2:
            h3 += 1
            m3 = 0
        else:
            m3 += 1
            max3 = max(max3,m3)

    if total == 0:
        total = 1

    print(f"\n===== 最近{lookback}期 =====")

    print(f"一肖: {h1/total:.1%} | 最大连空 {max1}")
    print(f"二肖: {h2/total:.1%} | 最大连空 {max2}")
    print(f"三肖(中二): {h3/total:.1%} | 最大连空 {max3}")

def main():

    print("正在获取最新数据...")

    rows = fetch_data()

    if not rows:
        print("获取失败")
        return

    latest = rows[-1]["issue_no"]

    s,b,t = predict(rows)

    print("\n【V12 组合职业版】")
    print(f"预测期号: {next_issue(latest)}")
    print(f"一肖: {'、'.join(s)}")
    print(f"二肖: {'、'.join(b)}")
    print(f"三肖: {'、'.join(t)}")

    backtest(rows[-110:],10)

    backtest(rows[-200:],100)

if __name__ == "__main__":
    main()