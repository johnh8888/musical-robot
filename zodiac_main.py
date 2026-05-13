#!/usr/bin/env python3
# zodiac_main.py - 香港一二三生肖预测（纯在线120期版）
import argparse, gzip, json, re, time, urllib.request
from collections import Counter
from itertools import combinations
from typing import List, Dict

# ===================== 生肖映射 =====================
ZODIAC_MAP = {
    "马": [1,13,25,37,49], "蛇": [2,14,26,38], "龙": [3,15,27,39],
    "兔": [4,16,28,40], "虎": [5,17,29,41], "牛": [6,18,30,42],
    "鼠": [7,19,31,43], "猪": [8,20,32,44], "狗": [9,21,33,45],
    "鸡": [10,22,34,46], "猴": [11,23,35,47], "羊": [12,24,36,48],
}
ZODIAC_LIST = list(ZODIAC_MAP.keys())
API_URL = "https://marksix6.net/index.php?api=1"

# ===================== 工具函数 =====================
def get_zodiac(n):
    for z, ns in ZODIAC_MAP.items():
        if n in ns: return z
    return "马"

def next_issue(issue_no: str) -> str:
    try:
        if '/' in issue_no:
            y, s = issue_no.split('/')
        else:
            y = issue_no[:4]
            s = issue_no[4:].lstrip('0') or '0'
        return f"{y}/{str(int(s)+1).zfill(3)}"
    except:
        return issue_no

# ===================== 在线数据获取（只取香港六合彩） =====================
def fetch_hk_online(limit=120):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://marksix6.net/",
    }
    req = urllib.request.Request(API_URL, headers=headers)
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read()
                if "gzip" in resp.headers.get("Content-Encoding", "").lower():
                    raw = gzip.decompress(raw)
                data = json.loads(raw.decode("utf-8"))
                # 遍历 lottery_data 找香港六合彩
                records = []
                for item in data.get("lottery_data", []):
                    name = item.get("name", "")
                    if "香港" in name or "六合彩" in name or "Mark Six" in name:
                        for line in item.get("history", []):
                            m = re.match(r"(\d{7})\s*期[：:]\s*([\d,]+)", line)
                            if not m: continue
                            nums = _parse_nums(m.group(2))
                            if len(nums) < 7: continue
                            raw_issue = m.group(1)
                            if len(raw_issue) >= 7:
                                y = raw_issue[2:4]
                                s = str(int(raw_issue[4:]))
                                issue_no = f"{y}/{s.zfill(3)}"
                            else:
                                issue_no = raw_issue
                            records.append({
                                "issue_no": issue_no,
                                "numbers": nums[:6],
                                "special_number": nums[6],
                            })
                        break
                # 去重排序
                dedup = {r["issue_no"]: r for r in records}
                sorted_rec = sorted(dedup.values(), key=lambda x: x["issue_no"], reverse=True)
                print(f"📡 在线获取到 {len(sorted_rec)} 期香港六合彩数据")
                return sorted_rec[:limit]
        except Exception as e:
            if attempt == 3:
                print(f"❌ 在线获取失败: {e}")
            time.sleep(2 ** attempt)
    return []

def _parse_nums(value):
    out = []
    for t in value.replace("，", ",").split(","):
        t = t.strip()
        if not t: continue
        try:
            n = int(t)
            if 1 <= n <= 49: out.append(n)
        except: pass
    return out

# ===================== 策略参数 =====================
CANDIDATE_SINGLE = [4,6,8,10,12,15,18,20,24,30]  # 去掉极短窗口
CANDIDATE_TWO    = [4,6,8,10,12,15,18,20,24,30,36,42]
CANDIDATE_THREE  = [4,6,8,10,12,15,18,20,24,30,36,42,48]
OPTIMAL_SINGLE = [6,10,12,18]
OPTIMAL_TWO    = [4,6,8,10,12]
OPTIMAL_THREE  = [8,10,12,15,30]
SINGLE_BOOST = 3.2
TWO_BOOST    = 3.0
MISS_PROTECT = 1

def w_weight(w, base=42):
    return round(base/w, 2)

# ===================== 策略核心 =====================
def omission_map(rows):
    if not rows: return {z:0 for z in ZODIAC_LIST}
    om = {z:0 for z in ZODIAC_LIST}
    for r in reversed(rows):
        appeared = set(get_zodiac(n) for n in r["numbers"])
        appeared.add(get_zodiac(r["special_number"]))
        for z in ZODIAC_LIST:
            om[z] = 0 if z in appeared else om[z]+1
    return om

def pred_single(train, w, boost=SINGLE_BOOST):
    recent = train[-w:] if len(train)>=w else train
    cnt = Counter()
    for r in recent:
        for n in r["numbers"]: cnt[get_zodiac(n)] += 1
        cnt[get_zodiac(r["special_number"])] += boost
    return cnt.most_common(1)[0][0] if cnt else "马"

def pred_two(train, w, boost=TWO_BOOST):
    recent = train[-w:] if len(train)>=w else train
    cnt = Counter()
    for r in recent:
        for n in r["numbers"]: cnt[get_zodiac(n)] += 1
        cnt[get_zodiac(r["special_number"])] += boost
    return [z for z,_ in cnt.most_common(2)]

def pred_three(train, w):
    recent = train[-w:] if len(train)>=w else train
    cnt = Counter()
    for r in recent:
        for n in r["numbers"]: cnt[get_zodiac(n)] += 1
        cnt[get_zodiac(r["special_number"])] += 1
    return [z for z,_ in cnt.most_common(3)]

def predict_all(history, w_s, w_t, w_th):
    vs = Counter()
    for w in w_s: vs[pred_single(history, w)] += w_weight(w)
    single = vs.most_common(1)[0][0]
    vt = Counter()
    for w in w_t:
        for z in pred_two(history, w): vt[z] += w_weight(w)
    two = [z for z,_ in vt.most_common(2)]
    vth = Counter()
    for w in w_th:
        for z in pred_three(history, w): vth[z] += w_weight(w, 48)
    three = [z for z,_ in vth.most_common(3)]
    return single, two, three

# ===================== 回测 =====================
def backtest(rows, lookback, w_s, w_t, w_th):
    rev = list(reversed(rows))
    total = min(lookback, len(rev)-20)
    if total <=0: return None
    hit_s=hit_t=hit_th=0
    miss_s=miss_t=miss_th=max_s=max_t=max_th=0
    for i in range(total):
        train = rev[i+20:]
        if len(train)<20: continue
        actual = rev[i]
        win_z = set(get_zodiac(n) for n in actual["numbers"])
        win_z.add(get_zodiac(actual["special_number"]))
        om = omission_map(train)
        # 一肖
        vs = Counter()
        for w in w_s: vs[pred_single(train,w)] += w_weight(w)
        ps = vs.most_common(1)[0][0]
        if miss_s >= MISS_PROTECT and om:
            ps = max(om, key=om.get)
        # 二肖
        vt = Counter()
        for w in w_t:
            for z in pred_two(train,w): vt[z] += w_weight(w)
        pt = [z for z,_ in vt.most_common(2)]
        if miss_t >= MISS_PROTECT and om:
            cold = max(om, key=om.get)
            if cold not in pt: pt[-1] = cold
        # 三肖
        vth = Counter()
        for w in w_th:
            for z in pred_three(train,w): vth[z] += w_weight(w, 48)
        pth = [z for z,_ in vth.most_common(3)]
        if miss_th >= MISS_PROTECT and om:
            cold2 = sorted(om, key=om.get, reverse=True)[:2]
            pth = [pth[0]] + [c for c in cold2 if c != pth[0]]
            while len(pth) < 3:
                for z,_ in vth.most_common():
                    if z not in pth: pth.append(z); break
                else: pth.append("马")
        # 统计
        if ps in win_z: hit_s+=1; miss_s=0
        else: miss_s+=1; max_s=max(max_s,miss_s)
        if any(z in win_z for z in pt): hit_t+=1; miss_t=0
        else: miss_t+=1; max_t=max(max_t,miss_t)
        if sum(1 for z in pth if z in win_z)>=2: hit_th+=1; miss_th=0
        else: miss_th+=1; max_th=max(max_th,miss_th)
    return {
        "s_hit": hit_s/total, "s_miss": max_s,
        "t_hit": hit_t/total, "t_miss": max_t,
        "th_hit": hit_th/total, "th_miss": max_th,
    }

def opt_single(rows, look=100):
    best, score = [6,10,12,18], -1
    for combo in combinations(CANDIDATE_SINGLE, 4):
        st = backtest(rows, look, list(combo), OPTIMAL_TWO, OPTIMAL_THREE)
        if st is None: continue
        s = st["s_hit"] - st["s_miss"]*0.1
        if s>score: score=s; best=list(combo)
    return best

def opt_two(rows, look=100):
    best, score = [4,6,8,10,12], -1
    for combo in combinations(CANDIDATE_TWO, 5):
        st = backtest(rows, look, OPTIMAL_SINGLE, list(combo), OPTIMAL_THREE)
        if st is None: continue
        s = st["t_hit"] - st["t_miss"]*0.1
        if s>score: score=s; best=list(combo)
    return best

def opt_three(rows, look=100):
    best, score = [8,10,12,15,30], -1
    for combo in combinations(CANDIDATE_THREE, 5):
        st = backtest(rows, look, OPTIMAL_SINGLE, OPTIMAL_TWO, list(combo))
        if st is None: continue
        s = st["th_hit"] - st["th_miss"]*0.1
        if s>score: score=s; best=list(combo)
    return best

# ===================== 主程序 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    rows = fetch_hk_online(120)
    if not rows:
        print("没有获取到在线数据，退出。")
        return

    global OPTIMAL_SINGLE, OPTIMAL_TWO, OPTIMAL_THREE
    OPTIMAL_SINGLE = opt_single(rows)
    OPTIMAL_TWO = opt_two(rows)
    OPTIMAL_THREE = opt_three(rows)
    print(f"窗口: 一肖{OPTIMAL_SINGLE} 二肖{OPTIMAL_TWO} 三肖{OPTIMAL_THREE}")

    if args.show:
        latest = rows[0]["issue_no"]
        pred = next_issue(latest)
        print(f"\n预测期号: {pred}")
        s, t, th = predict_all(rows, OPTIMAL_SINGLE, OPTIMAL_TWO, OPTIMAL_THREE)
        print(f"一生肖: {s}")
        print(f"二生肖: {'、'.join(t)}")
        print(f"三生肖: {'、'.join(th)}")
        st = backtest(rows, 10, OPTIMAL_SINGLE, OPTIMAL_TWO, OPTIMAL_THREE)
        if st:
            print(f"\n近10期回测：一生肖 {st['s_hit']:.1%} 连空{st['s_miss']}")
            print(f"二生肖 {st['t_hit']:.1%} 连空{st['t_miss']}")
            print(f"三生肖 {st['th_hit']:.1%} 连空{st['th_miss']}")

if __name__ == "__main__":
    main()