#!/usr/bin/env python3
# special_only.py - 特五肖 v3（优化版）

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

DECAY_ALPHA = 0.89

def get_zodiac(n):
    for z, nums in ZODIAC_MAP.items():
        if n in nums: return z
    return "马"

def get_color(n):
    for c, nums in COLOR_MAP.items():
        if n in nums: return c
    return "红"

def is_big(n): return n > 24
def is_odd(n): return n % 2 == 1

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
    return [int(t) for t in re.split(r'[，,]', value) if t.strip().isdigit() and 1 <= int(t.strip()) <= 49]

def fetch_hk_online(limit=180):
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
                    if not any(x in item.get("name","") for x in ["香港","六合彩"]): continue
                    for line in item.get("history", []):
                        m = re.match(r"(\d{7})\s*期[：:]\s*([\d,，]+)", line)
                        if not m: continue
                        nums = parse_nums(m.group(2))
                        if len(nums) < 7: continue
                        issue_no = f"{m.group(1)[2:4]}/{int(m.group(1)[4:]):03d}"
                        rows.append({"issue_no": issue_no, "numbers": nums[:6], "special_number": nums[6]})
                return sorted(rows, key=lambda x: x["issue_no"], reverse=True)[:limit]
        except:
            time.sleep(2)
    print("⚠️ 数据获取失败")
    return []

def stable_score(history, miss_count=0):
    score = Counter()
    recent = history[-50:]

    # 多窗口加权
    for idx, r in enumerate(reversed(recent[-6:])):
        z = get_zodiac(r["special_number"])
        score[z] += 0.52 * decay(idx)

    for idx, r in enumerate(reversed(recent[-15:])):
        z = get_zodiac(r["special_number"])
        score[z] += 0.25 * decay(idx)

    # 波色 + 大小单双趋势
    recent_sp = [r["special_number"] for r in recent[-12:]]
    dom_color = Counter(get_color(n) for n in recent_sp).most_common(1)[0][0]
    big_count = sum(1 for n in recent_sp if is_big(n))
    odd_count = sum(1 for n in recent_sp if is_odd(n))

    for z in ZODIAC_LIST:
        sample = ZODIAC_MAP[z][0]
        if get_color(sample) == dom_color:
            score[z] += 0.26
        if (is_big(sample) and big_count > 6) or (not is_big(sample) and big_count <= 6):
            score[z] += 0.12
        if (is_odd(sample) and odd_count > 6) or (not is_odd(sample) and odd_count <= 6):
            score[z] += 0.10

    # 遗漏 + 冷号
    om = omission_map(history)
    for z in ZODIAC_LIST:
        score[z] += min(om[z] * 0.06, 0.72) * 0.15

    if miss_count >= 3:
        for z, _ in sorted(om.items(), key=lambda x: x[1], reverse=True)[:3]:
            score[z] += 0.22

    return score

def recommend(history, miss_count=0):
    score = stable_score(history, miss_count)
    ranked = [z for z,_ in score.most_common()]
    final = ranked[:4] + ranked[-3:]
    seen = set()
    return [z for z in final if not (z in seen or seen.add(z))][:5]

def backtest(rows, lookback=160):
    rev = list(reversed(rows))
    total = hit = 0
    cur_miss = max_miss = 0
    for i in range(60, len(rev)-5):
        train = rev[i:]
        if len(train) < 50: continue
        total += 1
        actual = get_zodiac(rev[i-1]["special_number"])
        preds = recommend(train, cur_miss)
        if actual in preds:
            hit += 1
            cur_miss = 0
        else:
            cur_miss += 1
            max_miss = max(max_miss, cur_miss)
    print("\n===== 特五肖 v3 回测 =====")
    print(f"测试期数: {total}")
    print(f"命中率: {hit/total:.1%}" if total else "N/A")
    print(f"最大连空: {max_miss}")

def main():
    print("正在获取最新数据...")
    rows = fetch_hk_online(180)
    if not rows:
        print("❌ 获取失败")
        return
    preds = recommend(rows)
    print("\n【稳定版特五肖 v3】")
    print("、".join(preds))
    backtest(rows)

if __name__ == "__main__":
    main()