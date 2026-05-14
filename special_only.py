#!/usr/bin/env python3
# special_only.py - 优化稳定版特五肖 (v2)

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

# 波色定义
COLOR_MAP = {
    "红": [1,2,7,8,12,13,18,19,23,24,29,30,34,35,40,41,45,46],
    "蓝": [3,4,9,10,14,15,20,21,25,26,31,32,36,37,42,43,47,48],
    "绿": [5,6,11,16,17,22,27,28,33,38,39,44,49]
}

DECAY_ALPHA = 0.92   # 略微调低，增强近期影响

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
        if t and t.isdigit():
            n = int(t)
            if 1 <= n <= 49:
                out.append(n)
    return out

def fetch_hk_online(limit=150):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Referer": "https://marksix6.net/"
    }
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
                    if "香港" not in item.get("name", "") and "六合彩" not in item.get("name", ""):
                        continue
                    for line in item.get("history", []):
                        m = re.match(r"(\d{7})\s*期[：:]\s*([\d,，]+)", line)
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
                rows = sorted(rows, key=lambda x: x["issue_no"], reverse=True)
                return rows[:limit]
        except Exception as e:
            time.sleep(2)
    print("⚠️ 数据获取失败，使用备用方案或稍后重试")
    return []

def stable_score(history, miss_count=0):
    score = Counter()
    recent = history[-40:]   # 扩大历史参考

    # 1. 近期加强权重
    for idx, r in enumerate(reversed(recent[-5:])):
        z = get_zodiac(r["special_number"])
        score[z] += 0.55 * decay(idx)

    for idx, r in enumerate(reversed(recent[-10:])):
        z = get_zodiac(r["special_number"])
        score[z] += 0.28 * decay(idx)

    for idx, r in enumerate(reversed(recent[-20:])):
        z = get_zodiac(r["special_number"])
        score[z] += 0.12 * decay(idx)

    # 2. 波色趋势奖励
    colors = [get_color(r["special_number"]) for r in recent[-8:]]
    color_count = Counter(colors)
    dominant_color = color_count.most_common(1)[0][0] if color_count else "红"
    
    for z in ZODIAC_LIST:
        # 预测生肖对应的波色若为主流色则加分
        sample_num = ZODIAC_MAP[z][0]
        if get_color(sample_num) == dominant_color:
            score[z] += 0.22

    # 3. 遗漏 + 冷号
    om = omission_map(history)
    for z in ZODIAC_LIST:
        score[z] += min(om[z] * 0.045, 0.55) * 0.11

    # 冷号额外加成
    if miss_count >= 2:
        cold = sorted(om.items(), key=lambda x: x[1], reverse=True)[:3]
        for z, _ in cold:
            score[z] += 0.20

    return score

def recommend(history, miss_count=0):
    score = stable_score(history, miss_count)
    ranked = [z for z, _ in score.most_common()]
    
    # 优先取前3热 + 1暖 + 1冷
    final = []
    for z in ranked[:4] + ranked[-2:]:
        if z not in final:
            final.append(z)
    return final[:5]

def backtest(rows, lookback=120):
    rev = list(reversed(rows))
    total = 0
    hit = 0
    cur_miss = 0
    max_miss = 0

    for i in range(30, len(rev) - 5):
        train = rev[i:]
        if len(train) < 40:
            continue
        total += 1
        actual = get_zodiac(rev[i-1]["special_number"])   # 注意索引
        preds = recommend(train, miss_count=cur_miss)

        if actual in preds:
            hit += 1
            cur_miss = 0
        else:
            cur_miss += 1
            max_miss = max(max_miss, cur_miss)

    print("\n===== 优化后特五肖回测 =====")
    print(f"测试期数: {total}")
    print(f"命中率: {hit/total:.1%}" if total > 0 else "命中率: N/A")
    print(f"最大连空: {max_miss}")

def main():
    print("正在获取最新六合彩数据...")
    rows = fetch_hk_online(limit=150)
    
    if not rows:
        print("❌ 获取数据失败")
        return

    preds = recommend(rows)
    print("\n【稳定版特五肖 v2】")
    print("、".join(preds))

    backtest(rows)

if __name__ == "__main__":
    main()