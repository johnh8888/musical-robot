#!/usr/bin/env python3
# special_only_v7_fixed.py

import gzip
import json
import re
import urllib.request
from collections import Counter

API_URL = "https://marksix6.net/index.php?api=1"

ZMAP = {
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

ZLIST = list(ZMAP.keys())

def zodiac(n):
    for z, nums in ZMAP.items():
        if n in nums:
            return z
    return "马"

def fetch():
    req = urllib.request.Request(
        API_URL,
        headers={"User-Agent":"Mozilla/5.0"}
    )

    with urllib.request.urlopen(req, timeout=20) as r:
        raw = r.read()

        if "gzip" in r.headers.get("Content-Encoding","").lower():
            raw = gzip.decompress(raw)

        data = json.loads(raw.decode())

    rows = []

    for item in data.get("lottery_data",[]):
        if "香港" not in item.get("name","") and "六合彩" not in item.get("name",""):
            continue

        for line in item.get("history",[]):
            # 正则修正，更清晰
            m = re.match(r"(\d{7})\s*期[：:]\s*([\d,，]+)", line)

            if not m:
                continue

            nums = [
                int(x)
                for x in re.split(r"[，,]", m.group(2))
                if x.strip()
            ]

            if len(nums) >= 7:
                rows.append({
                    "special": nums[6],
                    "issue": m.group(1)   # 保存期号，便于输出
                })

    return rows[:300]

def omission(rows):
    om = {z:0 for z in ZLIST}

    for r in reversed(rows):
        hit = zodiac(r["special"])
        for z in ZLIST:
            om[z] = 0 if z == hit else om[z] + 1

    return om

def score(rows, miss=0):
    sc = Counter()
    om = omission(rows)
    recent = rows[-80:]

    # 近期特码加权
    for idx, r in enumerate(reversed(recent[-15:])):
        sc[zodiac(r["special"])] += 1.3 * (0.87 ** idx)

    for idx, r in enumerate(reversed(recent[-40:])):
        sc[zodiac(r["special"])] += 0.32 * (0.94 ** idx)

    # 遗漏加分
    for z in ZLIST:
        sc[z] += om[z] * 0.22

    # 连空惩罚/冷门奖励
    if miss >= 1:
        hot = [z for z,_ in sc.most_common(4)]
        for z in hot:
            sc[z] *= 0.80

    if miss >= 2:
        cold = sorted(om, key=om.get, reverse=True)[:5]
        for z in cold:
            sc[z] += 2.5

    return sc

def recommend(rows, miss=0):
    sc = score(rows, miss)
    rank = [z for z,_ in sc.most_common()]

    hot = rank[:3]

    cold = sorted(omission(rows).items(), key=lambda x: x[1], reverse=True)

    final = hot[:]
    for z, _ in cold:
        if z not in final:
            final.append(z)
        if len(final) >= 5:
            break

    return final

def backtest(rows):
    # rows 假定为升序（旧→新），反转后 rev[0] 为最新
    rev = list(reversed(rows))

    # 至少需要 121 期：100 期测试 + 至少 20 期训练（保守取 40）
    if len(rev) < 141:
        print("\n数据不足，无法回测（至少需要141期）")
        return

    total = min(100, len(rev) - 40)   # 测试期数
    if total <= 0:
        print("\n回测样本不足")
        return

    h1 = 0
    miss = 0
    maxmiss = 0

    details = []   # 存储最近10期详情

    for i in range(total):
        # 用比 rev[i] 更旧的数据训练（严格避免未来信息）
        train = rev[i+1:]

        # 预测目标期 rev[i] 的实际特肖
        actual_z = zodiac(rev[i]["special"])
        pred = recommend(train, miss)   # 用上一轮累积的 miss

        hit = actual_z in pred
        details.append({
            "offset": i,
            "issue": rev[i].get("issue", "?"),
            "pred": pred,
            "actual": actual_z,
            "hit": hit
        })

        if hit:
            h1 += 1
            miss = 0
        else:
            miss += 1
            maxmiss = max(maxmiss, miss)

    # 输出整体统计
    print("\n===== 特五肖回测（修复版）=====")
    print(f"测试期数: {total}")
    print(f"命中率: {h1/total:.1%}")
    print(f"最大连空: {maxmiss}")

    # 输出最近10期详情
    print("\n===== 最近10期预测详情（0=最新）=====")
    print(f"{'距今天数':<8} {'期号':<10} {'预测五肖':<24} {'实际特肖':<6} {'结果':<5}")
    print("-" * 65)
    for d in details[:10]:
        offset_day = d["offset"] + 1   # 第1期就是上一期
        print(f"{offset_day:<8} {d['issue']:<10} {','.join(d['pred']):<24} {d['actual']:<6} {'✓' if d['hit'] else '✗':<5}")
    print()

if __name__ == "__main__":
    rows = fetch()
    pred = recommend(rows)

    print("\n【特五肖 V7 修复版】")
    print("、".join(pred))

    backtest(rows)