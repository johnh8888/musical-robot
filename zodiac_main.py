#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import gzip
import json
import re
import time
import urllib.request
from collections import Counter

API_URL = "https://marksix6.net/index.php?api=1"

# 正确生肖映射
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
                    key=lambda x: x["issue_no"],
                    reverse=True
                )

                return rows[:limit]

        except:
            time.sleep(2)

    return []

def omission_map(rows):

    om = {z:0 for z in ZODIAC_LIST}

    for r in reversed(rows):

        appeared = set(
            get_zodiac(n)
            for n in r["numbers"] + [r["special_number"]]
        )

        for z in ZODIAC_LIST:
            om[z] = 0 if z in appeared else om[z] + 1

    return om

def stable_score(history):

    score = Counter()

    recent = history[-90:]

    # 近12期强化
    for idx, r in enumerate(reversed(recent[-12:])):

        for n in r["numbers"] + [r["special_number"]]:

            z = get_zodiac(n)

            score[z] += 0.62 * decay(idx)

    # 中期趋势
    for idx, r in enumerate(reversed(recent[-45:])):

        for n in r["numbers"] + [r["special_number"]]:

            z = get_zodiac(n)

            score[z] += 0.18 * decay(idx)

    # 特码热度
    sp_freq = Counter(
        get_zodiac(r["special_number"])
        for r in recent[-35:]
    )

    for z in ZODIAC_LIST:
        score[z] += sp_freq.get(z, 0) * 0.11

    # 冷号补偿
    om = omission_map(history)

    for z in ZODIAC_LIST:

        score[z] += min(
            om[z] * 0.12,
            2.2
        ) * 0.32

    return score

def predict(history):

    score = stable_score(history)

    om = omission_map(history)

    ranked = [z for z,_ in score.most_common()]

    hot = ranked[:4]

    middle = ranked[4:8]

    cold = sorted(
        om.items(),
        key=lambda x:x[1],
        reverse=True
    )

    cold = [z for z,_ in cold[:4]]

    # 一肖
    s = [hot[0]]

    # 二肖
    t = []

    t.append(hot[0])

    for z in middle:
        if z not in t:
            t.append(z)
            break

    # 三肖中二模型
    th = []

    # 热
    th.append(hot[0])

    # 温
    for z in middle:
        if z not in th:
            th.append(z)
            break

    # 冷
    for z in cold:
        if z not in th:
            th.append(z)
            break

    # 防止过热
    recent_special = [
        get_zodiac(r["special_number"])
        for r in history[-6:]
    ]

    recent_freq = Counter(recent_special)

    overheat = [
        z for z,c in recent_freq.items()
        if c >= 3
    ]

    th = [z for z in th if z not in overheat]

    while len(th) < 3:

        for z in ranked:

            if z not in th:

                th.append(z)

                break

    return s, t[:2], th[:3]

def backtest(rows):

    rev = list(reversed(rows))

    total = 0

    h1 = h2 = h3 = 0

    miss1 = miss2 = miss3 = 0

    max1 = max2 = max3 = 0

    total10 = 0

    h1_10 = h2_10 = h3_10 = 0

    miss1_10 = miss2_10 = miss3_10 = 0

    max1_10 = max2_10 = max3_10 = 0

    for i in range(100, len(rev)-1):

        train = rev[i:]

        if len(train) < 60:
            continue

        total += 1

        actual = set(
            get_zodiac(n)
            for n in rev[i-1]["numbers"] + [rev[i-1]["special_number"]]
        )

        s, t, th = predict(train)

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

        # 三肖中二
        th_hit = sum(1 for z in th if z in actual)

        if th_hit >= 2:
            h3 += 1
            miss3 = 0
        else:
            miss3 += 1
            max3 = max(max3, miss3)

        # 最近10期
        if total <= 10:

            total10 += 1

            if s[0] in actual:
                h1_10 += 1

            if any(z in actual for z in t):
                h2_10 += 1

            if th_hit >= 2:
                h3_10 += 1

    print("\n===== V10 职业终极版 =====")

    if total > 0:

        print(f"测试期数: {total}")

        print(f"一肖: {h1/total:.1%} | 最大连空 {max1}")
        print(f"二肖: {h2/total:.1%} | 最大连空 {max2}")
        print(f"三肖: {h3/total:.1%} | 最大连空 {max3}")

    if total10 > 0:

        print("\n===== 最近10期 =====")

        print(f"一肖: {h1_10/total10:.1%}")
        print(f"二肖: {h2_10/total10:.1%}")
        print(f"三肖: {h3_10/total10:.1%}")

def main():

    print("正在获取最新数据...")

    rows = fetch_hk_online(300)

    if not rows:

        print("数据获取失败")

        return

    s, t, th = predict(rows)

    print("\n【V10 职业终极模型】")

    print(f"预测期号: {rows[0]['issue_no']}")

    print(f"一肖: {'、'.join(s)}")
    print(f"二肖: {'、'.join(t)}")
    print(f"三肖: {'、'.join(th)}")

    backtest(rows)

if __name__ == "__main__":
    main()