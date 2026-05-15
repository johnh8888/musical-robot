#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# color_two_pro.py  (增强版：支持 --show，最近10期详情)
# 香港六合彩 · 特二色职业稳定版

import argparse
import gzip
import json
import re
import time
import urllib.request
from collections import Counter, defaultdict

API_URL = "https://marksix6.net/index.php?api=1"

# =========================
# 波色映射
# =========================
RED = {1,2,7,8,12,13,18,19,23,24,29,30,34,35,40,45,46}
BLUE = {3,4,9,10,14,15,20,25,26,31,36,37,41,42,47,48}
GREEN = {5,6,11,16,17,21,22,27,28,32,33,38,39,43,44,49}
COLORS = ["红", "蓝", "绿"]

def get_color(n):
    if n in RED:
        return "红"
    if n in BLUE:
        return "蓝"
    if n in GREEN:
        return "绿"
    return "红"

def next_issue(issue_no):
    try:
        if '/' in issue_no:
            y, s = issue_no.split('/')
        else:
            y = issue_no[:4]
            s = issue_no[4:]
        return f"{y}/{str(int(s)+1).zfill(3)}"
    except:
        return issue_no

def parse_nums(value):
    out = []
    for t in re.split(r"[，,]", value):
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

def fetch_hk(limit=300):
    headers = {"User-Agent": "Mozilla/5.0"}
    for _ in range(5):
        try:
            req = urllib.request.Request(API_URL, headers=headers)
            with urllib.request.urlopen(req, timeout=20) as resp:
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
                rows = sorted(rows, key=lambda x: x["issue_no"], reverse=True)
                return rows[:limit]
        except:
            time.sleep(2)
    return []

def get_color_history(rows):
    colors = []
    for r in rows:
        colors.append(get_color(r["special_number"]))
    return colors

# =========================
# 冷热分析
# =========================
def calc_hot_cold(train):
    recent_short = train[:12]
    recent_mid = train[:30]
    recent_long = train[:60]
    score = Counter()
    for c in recent_short:
        score[c] += 2.4
    for c in recent_mid:
        score[c] += 1.2
    for c in recent_long:
        score[c] += 0.6
    om = {}
    for c in COLORS:
        miss = 0
        for x in train:
            if x == c:
                break
            miss += 1
        om[c] = miss
    for c in COLORS:
        score[c] += min(om[c] * 0.35, 3.5)
    return score, om

# =========================
# 转移矩阵
# =========================
def calc_transition(train):
    trans = defaultdict(Counter)
    for i in range(len(train)-1):
        cur = train[i]
        nxt = train[i+1]
        trans[cur][nxt] += 1
    return trans

# =========================
# 状态识别
# =========================
def detect_state(train):
    recent = train[:15]
    freq = Counter(recent)
    top = freq.most_common(1)[0][1]
    ratio = top / len(recent)
    if ratio >= 0.60:
        return "单边"
    if len(set(recent[:6])) >= 3:
        return "混乱"
    return "正常"

# =========================
# 职业预测核心
# =========================
def professional_predict(train, miss_streak=0, dual=True):
    score, om = calc_hot_cold(train)
    trans = calc_transition(train)
    state = detect_state(train)
    last = train[0]

    # 转移概率增强
    if last in trans:
        total = sum(trans[last].values())
        if total > 0:
            for c, v in trans[last].items():
                p = v / total
                score[c] += p * 3.2

    # 状态增强
    if state == "单边":
        hottest = score.most_common(1)[0][0]
        score[hottest] += 2.5
    elif state == "混乱":
        coldest = sorted(om.items(), key=lambda x: x[1], reverse=True)[0][0]
        score[coldest] += 2.2

    # 连空熔断
    if miss_streak >= 1:
        cold_rank = sorted(om.items(), key=lambda x: x[1], reverse=True)
        for c, _ in cold_rank[:2]:
            score[c] += 2.8

    ranked = [c for c, _ in score.most_common()]

    if not dual:
        return ranked[0]

    # 双色：热冷混合
    hot = ranked[0]
    cold = sorted(om.items(), key=lambda x: x[1], reverse=True)[0][0]
    result = [hot]
    if cold not in result:
        result.append(cold)
    for c in ranked:
        if c not in result:
            result.append(c)
    return result[:2]

# =========================
# 回测（真正时间序，返回详情）
# =========================
def backtest(colors, dual=True, lookback=100):
    rev = list(reversed(colors))
    total = min(lookback, len(rev)-80)
    if total <= 0:
        return 0, 0, []
    hits = 0
    miss_streak = 0
    max_miss = 0
    details = []   # 存储最近10期详情

    for i in range(80, 80+total):
        train = list(reversed(rev[:i]))
        actual = rev[i]
        pred = professional_predict(train, miss_streak, dual=dual)
        if dual:
            ok = actual in pred
        else:
            ok = actual == pred

        if ok:
            hits += 1
            miss_streak = 0
        else:
            miss_streak += 1
            max_miss = max(max_miss, miss_streak)

        # 只保留最近10期详情（按 i 越大越新）
        if i >= 80+total-10:
            details.append({
                "offset": total - (i-80) - 1,   # 0 = 最新
                "pred": pred,
                "actual": actual,
                "hit": ok
            })

    return hits / total, max_miss, details[::-1]   # 反转成从旧到新

# =========================
# 主程序
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", action="store_true", help="单色模式")
    parser.add_argument("--show", action="store_true", help="简洁模式，只显示预测结果")
    args = parser.parse_args()

    print("正在获取最新数据...")
    rows = fetch_hk(300)
    if not rows:
        print("数据获取失败")
        return

    colors = get_color_history(rows)
    latest_issue = rows[0]["issue_no"]

    # 进行预测
    if args.single:
        pred = professional_predict(colors, dual=False)
        mode_name = "职业版特单色"
    else:
        pred = professional_predict(colors, dual=True)
        mode_name = "职业版特二色"

    # 输出预测结果（无论 --show 与否都显示）
    print(f"\n预测期号: {next_issue(latest_issue)}")
    print(f"\n【{mode_name}】")
    if args.single:
        print(pred)
    else:
        print("、".join(pred))

    # 如果只是 --show，到此结束
    if args.show:
        return

    # 否则输出回测统计 + 最近10期详情
    lookback10 = 10
    lookback100 = 100

    hr10, miss10, details10 = backtest(colors, dual=not args.single, lookback=lookback10)
    hr100, miss100, _ = backtest(colors, dual=not args.single, lookback=lookback100)

    print(f"\n===== 回测统计 =====")
    print(f"近10期命中率: {hr10:.1%} | 最大连空: {miss10}")
    print(f"近100期命中率: {hr100:.1%} | 最大连空: {miss100}")

    # 输出最近10期详情表格
    if details10:
        print(f"\n===== 最近10期回测详情（0=最新）=====")
        if args.single:
            print(f"{'距今天数':<8} {'预测单色':<6} {'实际颜色':<6} {'结果':<5}")
        else:
            print(f"{'距今天数':<8} {'预测二色':<12} {'实际颜色':<6} {'结果':<5}")
        print("-" * 50)
        for d in details10:
            offset_day = d["offset"] + 1   # 1 = 上一期
            if args.single:
                pred_str = d["pred"]
            else:
                pred_str = ",".join(d["pred"])
            hit_mark = "✓" if d["hit"] else "✗"
            print(f"{offset_day:<8} {pred_str:<12} {d['actual']:<6} {hit_mark:<5}")
        print()

if __name__ == "__main__":
    main()