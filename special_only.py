#!/usr/bin/env python3
# special_only.py
# 稳定版特五肖

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

DECAY_ALPHA = 0.94

def get_zodiac(n):

    for z, nums in ZODIAC_MAP.items():

        if n in nums:
            return z

    return "马"

def decay(idx):
    return DECAY_ALPHA ** idx

def omission_map(rows):

    om = {z:0 for z in ZODIAC_LIST}

    for r in reversed(rows):

        sp = get_zodiac(r["special_number"])

        for z in ZODIAC_LIST:

            if z == sp:
                om[z] = 0
            else:
                om[z] += 1

    return om

def parse_nums(value):

    out = []

    for t in value.replace("，", ",").split(","):

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

def fetch_hk_online(limit=120):

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Referer": "https://marksix6.net/"
    }

    req = urllib.request.Request(API_URL, headers=headers)

    for _ in range(3):

        try:

            with urllib.request.urlopen(req, timeout=15) as resp:

                raw = resp.read()

                if "gzip" in resp.headers.get("Content-Encoding","").lower():
                    raw = gzip.decompress(raw)

                data = json.loads(raw.decode("utf-8"))

                rows = []

                for item in data.get("lottery_data", []):

                    name = item.get("name","")

                    if "香港" not in name and "六合彩" not in name:
                        continue

                    for line in item.get("history", []):

                        m = re.match(r"(\d{7})\s*期[：:]\s*([\d,]+)", line)

                        if not m:
                            continue

                        nums = parse_nums(m.group(2))

                        if len(nums) < 7:
                            continue

                        raw_issue = m.group(1)

                        y = raw_issue[2:4]
                        s = str(int(raw_issue[4:]))

                        issue_no = f"{y}/{s.zfill(3)}"

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

def stable_score(history, miss_count=0):

    score = Counter()

    recent = history[-30:]

    for idx, r in enumerate(reversed(recent[-5:])):

        z = get_zodiac(r["special_number"])

        score[z] += 0.45 * decay(idx)

    for idx, r in enumerate(reversed(recent[-10:])):

        z = get_zodiac(r["special_number"])

        score[z] += 0.30 * decay(idx)

    for idx, r in enumerate(reversed(recent[-20:])):

        z = get_zodiac(r["special_number"])

        score[z] += 0.15 * decay(idx)

    om = omission_map(history)

    for z in ZODIAC_LIST:

        score[z] += min(om[z] * 0.05, 0.45) * 0.10

    if miss_count >= 2:

        cold_rank = sorted(
            om.items(),
            key=lambda x:x[1],
            reverse=True
        )

        for z, _ in cold_rank[:2]:

            score[z] += 0.18

    return score

def recommend(history, miss_count=0):

    score = stable_score(history, miss_count)

    ranked = [z for z,_ in score.most_common()]

    hot = ranked[:3]

    warm = ranked[3:4]

    cold = ranked[-1:]

    final = []

    for z in hot + warm + cold:

        if z not in final:
            final.append(z)

    return final[:5]

def backtest(rows, lookback=80):

    rev = list(reversed(rows))

    total = min(lookback, len(rev)-30)

    hit = 0

    cur_miss = 0
    max_miss = 0

    for i in range(total):

        train = rev[i+30:]

        if len(train) < 30:
            continue

        actual = get_zodiac(
            rev[i]["special_number"]
        )

        preds = recommend(
            train,
            miss_count=cur_miss
        )

        if actual in preds:

            hit += 1
            cur_miss = 0

        else:

            cur_miss += 1

            max_miss = max(
                max_miss,
                cur_miss
            )

    print("\n===== 特五肖回测 =====")

    print(f"命中率: {hit/total:.1%}")
    print(f"最大连空: {max_miss}")

def main():

    rows = fetch_hk_online()

    if not rows:
        print("获取数据失败")
        return

    preds = recommend(rows)

    print("\n【稳定版特五肖】")

    print("、".join(preds))

    backtest(rows)

if __name__ == "__main__":
    main()