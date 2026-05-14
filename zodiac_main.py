
#!/usr/bin/env python3
# zodiac_main.py
# 稳定版 一二三肖

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

        appeared = set()

        for n in r["numbers"]:
            appeared.add(get_zodiac(n))

        appeared.add(get_zodiac(r["special_number"]))

        for z in ZODIAC_LIST:

            if z in appeared:
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

def stable_score(history):

    score = Counter()

    recent = history[-30:]

    for idx, r in enumerate(reversed(recent[-5:])):

        sp = get_zodiac(r["special_number"])

        score[sp] += 0.45 * decay(idx)

    for idx, r in enumerate(reversed(recent[-10:])):

        sp = get_zodiac(r["special_number"])

        score[sp] += 0.30 * decay(idx)

    for idx, r in enumerate(reversed(recent[-20:])):

        sp = get_zodiac(r["special_number"])

        score[sp] += 0.15 * decay(idx)

    om = omission_map(history)

    for z in ZODIAC_LIST:

        score[z] += min(om[z] * 0.05, 0.45) * 0.10

    return score

def predict(history):

    score = stable_score(history)

    ranked = [z for z,_ in score.most_common()]

    single = ranked[:1]

    two = ranked[:2]

    three = ranked[:3]

    return single, two, three

def backtest(rows, lookback=80):

    rev = list(reversed(rows))

    total = min(lookback, len(rev)-30)

    s_hit = 0
    t_hit = 0
    th_hit = 0

    s_miss = 0
    t_miss = 0
    th_miss = 0

    s_max = 0
    t_max = 0
    th_max = 0

    for i in range(total):

        train = rev[i+30:]

        if len(train) < 30:
            continue

        actual = set()

        for n in rev[i]["numbers"]:
            actual.add(get_zodiac(n))

        actual.add(get_zodiac(rev[i]["special_number"]))

        s, t, th = predict(train)

        if any(z in actual for z in s):
            s_hit += 1
            s_miss = 0
        else:
            s_miss += 1
            s_max = max(s_max, s_miss)

        if any(z in actual for z in t):
            t_hit += 1
            t_miss = 0
        else:
            t_miss += 1
            t_max = max(t_max, t_miss)

        th_count = sum(1 for z in th if z in actual)

        if th_count >= 2:
            th_hit += 1
            th_miss = 0
        else:
            th_miss += 1
            th_max = max(th_max, th_miss)

    print("\n===== 回测 =====")

    print(f"一生肖: {s_hit/total:.1%} | 最大连空 {s_max}")
    print(f"二生肖: {t_hit/total:.1%} | 最大连空 {t_max}")
    print(f"三生肖: {th_hit/total:.1%} | 最大连空 {th_max}")

def main():

    rows = fetch_hk_online()

    if not rows:
        print("获取数据失败")
        return

    s, t, th = predict(rows)

    print("\n【稳定版一二三肖】")

    print(f"一生肖: {'、'.join(s)}")
    print(f"二生肖: {'、'.join(t)}")
    print(f"三生肖: {'、'.join(th)}")

    backtest(rows)

if __name__ == "__main__":
    main()
