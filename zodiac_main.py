#!/usr/bin/env python3
# zodiac_main.py - 优化版 一二三肖 (v3)

import gzip
import json
import re
import time
import urllib.request
from collections import Counter

API_URL = "https://marksix6.net/index.php?api=1"

ZODIAC_MAP = {
    "鼠": [7,19,31,43], "牛": [8,20,32,44], "虎": [9,21,33,45],
    "兔": [10,22,34,46], "龙": [11,23,35,47], "蛇": [12,24,36,48],
    "马": [1,13,25,37,49], "羊": [2,14,26,38], "猴": [3,15,27,39],
    "鸡": [4,16,28,40], "狗": [5,17,29,41], "猪": [6,18,30,42],
}

ZODIAC_LIST = list(ZODIAC_MAP.keys())

COLOR_MAP = {
    "红": [1,2,7,8,12,13,18,19,23,24,29,30,34,35,40,41,45,46],
    "蓝": [3,4,9,10,14,15,20,21,25,26,31,32,36,37,42,43,47,48],
    "绿": [5,6,11,16,17,22,27,28,33,38,39,44,49]
}

DECAY_ALPHA = 0.90

def get_zodiac(n):
    for z, nums in ZODIAC_MAP.items():
        if n in nums:
            return z
    return "马"

def get_color(n):
    for c, nums in COLOR_MAP.items():
        if n in nums:
            return c
    return "红"

def decay(idx):
    return DECAY_ALPHA ** idx

def omission_map(rows):
    om = {z: 0 for z in ZODIAC_LIST}
    for r in reversed(rows):
        appeared = {get_zodiac(n) for n in r["numbers"] + [r["special_number"]]}
        for z in ZODIAC_LIST:
            om[z] = 0 if z in appeared else om[z] + 1
    return om

def parse_nums(value):
    out = []
    for t in re.split(r'[，,]', value):
        t = t.strip()
        if t.isdigit():
            n = int(t)
            if 1 <= n <= 49:
                out.append(n)
    return out

def fetch_hk_online(limit=160):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    req = urllib.request.Request(API_URL, headers=headers)
    
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read()
                if "gzip" in resp.headers.get("Content-Encoding", "").lower():
                    raw = gzip.decompress(raw)
                data = json.loads(raw.decode("utf-8"))

                rows = []
                for item in data.get("lottery_data", []):
                    if not any(x in item.get("name", "") for x in ["香港", "六合彩"]):
                        continue
                    for line in item.get("history", []):
                        m = re.match(r"(\d{7})\s*期[：:]\s*([\d,，]+)", line)
                        if not m:
                            continue
                        nums = parse_nums(m.group(2))
                        if len(nums) < 7:
                            continue
                        raw_issue = m.group(1)
                        issue_no = f"{raw_issue[2:4]}/{int(raw_issue[4:]):03d}"
                        rows.append({
                            "issue_no": issue_no,
                            "numbers": nums[:6],
                            "special_number": nums[6]
                        })
                return sorted(rows, key=lambda x: x["issue_no"], reverse=True)[:limit]
        except:
            time.sleep(2)
    print("⚠️ 数据获取失败")
    return []

def stable_score(history):
    score = Counter()
    recent = history[-45:]

    # 更强的近期权重
    weights = [(5, 0.48), (12, 0.26), (25, 0.14)]
    for window, base_weight in weights:
        for idx, r in enumerate(reversed(recent[-window:])):
            for n in r["numbers"] + [r["special_number"]]:
                z = get_zodiac(n)
                score[z] += base_weight * decay(idx)

    # 波色趋势
    recent_colors = [get_color(r["special_number"]) for r in recent[-12:]]
    dominant = Counter(recent_colors).most_common(1)[0][0]
    for z in ZODIAC_LIST:
        if get_color(ZODIAC_MAP[z][0]) == dominant:
            score[z] += 0.28

    # 遗漏
    om = omission_map(history)
    for z in ZODIAC_LIST:
        score[z] += min(om[z] * 0.055, 0.65) * 0.14

    return score

def predict(history):
    score = stable_score(history)
    ranked = [z for z, _ in score.most_common()]
    return ranked[:1], ranked[:2], ranked[:3]

def backtest(rows, lookback=140):
    rev = list(reversed(rows))
    total = 0
    s_hit = t_hit = th_hit = 0
    s_miss = t_miss = th_miss = 0
    s_max = t_max = th_max = 0

    for i in range(50, len(rev) - 5):
        train = rev[i:]
        if len(train) < 45:
            continue
        total += 1

        actual = {get_zodiac(n) for n in rev[i-1]["numbers"] + [rev[i-1]["special_number"]]}

        s, t, th = predict(train)

        # 一肖
        if s[0] in actual:
            s_hit += 1
            s_miss = 0
        else:
            s_miss += 1
            s_max = max(s_max, s_miss)

        # 二肖
        if any(z in actual for z in t):
            t_hit += 1
            t_miss = 0
        else:
            t_miss += 1
            t_max = max(t_max, t_miss)

        # 三肖（至少中2个）
        th_count = sum(1 for z in th if z in actual)
        if th_count >= 2:
            th_hit += 1
            th_miss = 0
        else:
            th_miss += 1
            th_max = max(th_max, th_miss)

    print("\n===== 优化后一二三肖回测 (v3) =====")
    print(f"测试期数: {total}")
    print(f"一肖: {s_hit/total:.1%} | 最大连空 {s_max}")
    print(f"二肖: {t_hit/total:.1%} | 最大连空 {t_max}")
    print(f"三肖: {th_hit/total:.1%} | 最大连空 {th_max}")

def main():
    print("正在获取最新六合彩数据...")
    rows = fetch_hk_online(limit=160)
    
    if not rows:
        print("❌ 获取数据失败")
        return

    s, t, th = predict(rows)

    print("\n【稳定版一二三肖 v3】")
    print(f"一生肖: {'、'.join(s)}")
    print(f"二生肖: {'、'.join(t)}")
    print(f"三生肖: {'、'.join(th)}")

    backtest(rows)

if __name__ == "__main__":
    main()