#!/usr/bin/env python3
# zodiac_main.py - 一二三肖 v5（加强版）

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

DECAY_ALPHA = 0.86

def get_zodiac(n):
    for z, nums in ZODIAC_MAP.items():
        if n in nums: return z
    return "马"

def get_color(n):
    for c, nums in COLOR_MAP.items():
        if n in nums: return c
    return "红"

def is_big(n): return n >= 25
def is_odd(n): return n % 2 == 1
def decay(idx): return DECAY_ALPHA ** idx

def omission_map(rows):
    om = {z: 0 for z in ZODIAC_LIST}
    for r in reversed(rows):
        appeared = {get_zodiac(n) for n in r.get("numbers", []) + [r["special_number"]]}
        for z in ZODIAC_LIST:
            om[z] = 0 if z in appeared else om[z] + 1
    return om

def parse_nums(value):
    return [int(t) for t in re.split(r'[，,]', value) if t.strip().isdigit() and 1 <= int(t) <= 49]

def fetch_hk_online(limit=250):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    req = urllib.request.Request(API_URL, headers=headers)
    for _ in range(4):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read()
                if "gzip" in resp.headers.get("Content-Encoding", "").lower():
                    raw = gzip.decompress(raw)
                data = json.loads(raw.decode("utf-8"))
                rows = []
                for item in data.get("lottery_data", []):
                    if not any(x in item.get("name","") for x in ["香港","六合彩"]): continue
                    for line in item.get("history", []):
                        m = re.match(r"(\d{7})\s*期[：:]\s*([\d,，]+)", line)
                        if not m: continue
                        nums = parse_nums(m.group(2))
                        if len(nums) < 7: continue
                        issue = m.group(1)
                        issue_no = f"{issue[2:4]}/{int(issue[4:]):03d}"
                        rows.append({"issue_no": issue_no, "numbers": nums[:6], "special_number": nums[6]})
                return sorted(rows, key=lambda x: x["issue_no"], reverse=True)[:limit]
        except:
            time.sleep(2)
    print("⚠️ 数据获取失败")
    return []

def stable_score(history):
    score = Counter()
    recent = history[-70:]

    # 近期 + 中期权重
    for idx, r in enumerate(reversed(recent[-10:])):
        for n in r["numbers"] + [r["special_number"]]:
            z = get_zodiac(n)
            score[z] += 0.52 * decay(idx)

    for idx, r in enumerate(reversed(recent[-35:])):
        for n in r["numbers"] + [r["special_number"]]:
            z = get_zodiac(n)
            score[z] += 0.19 * decay(idx)

    # 趋势特征
    sp_list = [r["special_number"] for r in recent[-20:]]
    dom_color = Counter(get_color(n) for n in sp_list).most_common(1)[0][0]
    big_ratio = sum(is_big(n) for n in sp_list) / len(sp_list) if sp_list else 0.5
    odd_ratio = sum(is_odd(n) for n in sp_list) / len(sp_list) if sp_list else 0.5

    freq = Counter(get_zodiac(r["special_number"]) for r in recent)

    for z in ZODIAC_LIST:
        sample = ZODIAC_MAP[z][0]
        bonus = 0.0
        if get_color(sample) == dom_color:
            bonus += 0.31
        if (is_big(sample) and big_ratio > 0.5) or (not is_big(sample) and big_ratio <= 0.5):
            bonus += 0.17
        if (is_odd(sample) and odd_ratio > 0.5) or (not is_odd(sample) and odd_ratio <= 0.5):
            bonus += 0.13
        score[z] += bonus + (freq[z] * 0.08)

    # 强遗漏补偿
    om = omission_map(history)
    for z in ZODIAC_LIST:
        score[z] += min(om[z] * 0.09, 1.4) * 0.24

    return score

def predict(history):
    score = stable_score(history)
    ranked = [z for z, _ in score.most_common()]
    return ranked[:1], ranked[:2], ranked[:3]

def backtest(rows):
    rev = list(reversed(rows))
    total = s_hit = t_hit = th_hit = 0
    s_miss = t_miss = th_miss = 0
    s_max = t_max = th_max = 0

    for i in range(90, len(rev)-5):
        train = rev[i:]
        if len(train) < 70: continue
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

    print("\n===== 一二三肖 v5 回测 =====")
    print(f"测试期数: {total}")
    print(f"一肖: {s_hit/total:.1%} | 最大连空 {s_max}")
    print(f"二肖: {t_hit/total:.1%} | 最大连空 {t_max}")
    print(f"三肖: {th_hit/total:.1%} | 最大连空 {th_max}")

def main():
    print("正在获取最新数据...")
    rows = fetch_hk_online(250)
    if not rows:
        print("❌ 获取失败")
        return

    s, t, th = predict(rows)
    print("\n【稳定版一二三肖 v5】")
    print(f"一生肖: {'、'.join(s)}")
    print(f"二生肖: {'、'.join(t)}")
    print(f"三生肖: {'、'.join(th)}")

    backtest(rows)

if __name__ == "__main__":
    main()