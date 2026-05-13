#!/usr/bin/env python3
# zodiac_main.py - 香港一二三生肖预测（一三肖连空压缩强化版）
import argparse, gzip, json, re, time, urllib.request
from collections import Counter
from itertools import combinations
from typing import List, Dict

ZODIAC_MAP = {
    "马": [1,13,25,37,49], "蛇": [2,14,26,38], "龙": [3,15,27,39],
    "兔": [4,16,28,40], "虎": [5,17,29,41], "牛": [6,18,30,42],
    "鼠": [7,19,31,43], "猪": [8,20,32,44], "狗": [9,21,33,45],
    "鸡": [10,22,34,46], "猴": [11,23,35,47], "羊": [12,24,36,48],
}
ZODIAC_LIST = list(ZODIAC_MAP.keys())
API_URL = "https://marksix6.net/index.php?api=1"

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

# ===================== 在线获取香港六合彩（120期） =====================
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
                if "gzip" in resp.headers.get("Content-Encoding","").lower():
                    raw = gzip.decompress(raw)
                data = json.loads(raw.decode("utf-8"))
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
                dedup = {r["issue_no"]: r for r in records}
                sorted_rec = sorted(dedup.values(), key=lambda x: x["issue_no"], reverse=True)
                print(f"📡 在线获取到 {len(sorted_rec)} 期香港六合彩数据")
                return sorted_rec[:limit]
        except Exception as e:
            if attempt == 3: print(f"❌ 在线获取失败: {e}")
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

# ===================== 参数 =====================
CANDIDATE_SINGLE = [4,6,8,10,12,15,18,20,24,30]
CANDIDATE_TWO    = [4,6,8,10,12,15,18,20,24,30,36,42]
CANDIDATE_THREE  = [4,6,8,10,12,15,18,20,24,30]

OPTIMAL_SINGLE = [6,10,12,18]
OPTIMAL_TWO    = [4,6,8,10,12]
OPTIMAL_THREE  = [8,10,12,15]

# 一肖强化参数
SINGLE_BOOST = 4.0                 # 继续提高特码权重
SINGLE_MISS_PROTECT = 1
SINGLE_PENALTY = 0.2               # 提高惩罚

# 二肖不变
TWO_BOOST = 3.0
TWO_MISS_PROTECT = 1

# 三肖强化参数
THREE_NORMAL_WEIGHT = 0.3
THREE_SIGNAL_THRESHOLD = 0.5
THREE_COLD_BASE = 1.2              # 提高冷号基础
THREE_COLD_STEP = 0.3
THREE_COLD_MAX = 1.8               # 提高上限
THREE_MISS_PROTECT = 1
THREE_PENALTY = 0.3                # 惩罚加大

# 三肖趋势感知
TREND_LOOKBACK = 12
HOT_OVERHEAT_THRESHOLD = 0.7
HOT_COLD_MULT = 2.2                # 过热时冷号乘数提高
COLD_HOT_BOOST = 1.1

def w_weight(w, base=42):
    return round(base/w, 2)

# ===================== 通用工具 =====================
def omission_map(rows):
    if not rows: return {z:0 for z in ZODIAC_LIST}
    om = {z:0 for z in ZODIAC_LIST}
    for r in reversed(rows):
        appeared = set(get_zodiac(n) for n in r["numbers"])
        appeared.add(get_zodiac(r["special_number"]))
        for z in ZODIAC_LIST:
            om[z] = 0 if z in appeared else om[z]+1
    return om

# ===================== 三肖辅助函数 =====================
def three_normal_signal(history):
    if len(history) < 10+1:
        return False, 0.0
    recent = history[-(10+1):]
    hits = 0
    for i in range(len(recent)-1):
        cur_set = set(get_zodiac(n) for n in recent[i]["numbers"])
        if get_zodiac(recent[i+1]["special_number"]) in cur_set:
            hits += 1
    rate = hits/(len(recent)-1)
    if rate >= THREE_SIGNAL_THRESHOLD:
        return True, THREE_NORMAL_WEIGHT
    return False, 0.0

def three_trend_factor(history):
    if len(history) < TREND_LOOKBACK+1:
        return 1.0, 1.0
    recent = history[-TREND_LOOKBACK:]
    hot_hit = 0
    total = len(recent)-1
    if total <= 0: return 1.0, 1.0
    for i in range(total):
        prev = recent[i]
        cur_sp = get_zodiac(recent[i+1]["special_number"])
        sub = history[:history.index(prev)+1][-30:]
        cnt = Counter()
        for r in sub:
            cnt[get_zodiac(r["special_number"])] += 1
        top5 = [z for z,_ in cnt.most_common(5)]
        if cur_sp in top5:
            hot_hit += 1
    ratio = hot_hit/total
    if ratio > HOT_OVERHEAT_THRESHOLD:
        return 0.9, HOT_COLD_MULT
    elif ratio < 0.3:
        return COLD_HOT_BOOST, 0.8
    return 1.0, 1.0

def three_cold_bonus(om_val):
    if om_val <= 0: return 0.0
    return min(THREE_COLD_BASE + (om_val//10)*THREE_COLD_STEP, THREE_COLD_MAX)

# ===================== 预测函数 =====================
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

def pred_three(train, w, use_normal=False, normal_w=0.0):
    recent = train[-w:] if len(train)>=w else train
    cnt = Counter()
    for r in recent:
        cnt[get_zodiac(r["special_number"])] += 1
        if use_normal:
            for n in r["numbers"]:
                cnt[get_zodiac(n)] += normal_w
    return [z for z,_ in cnt.most_common(3)]

def predict_all(history, w_s, w_t, w_th):
    vs = Counter()
    for w in w_s: vs[pred_single(history, w)] += w_weight(w)
    single = vs.most_common(1)[0][0]

    vt = Counter()
    for w in w_t:
        for z in pred_two(history, w): vt[z] += w_weight(w)
    two = [z for z,_ in vt.most_common(2)]

    use_n, nw = three_normal_signal(history)
    vth = Counter()
    for w in w_th:
        for z in pred_three(history, w, use_n, nw):
            vth[z] += w_weight(w, 30)
    om = omission_map(history)
    hot_b, cold_m = three_trend_factor(history)
    for z in om:
        bonus = three_cold_bonus(om[z])
        if bonus > 0:
            vth[z] += bonus * cold_m
    three = [z for z,_ in vth.most_common(3)]
    return single, two, three

# ===================== 回测（包含增强保护） =====================
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
        sorted_cold = sorted(om, key=om.get, reverse=True)

        # === 一肖保护：连空1期时，从遗漏前3中选一个不在最近3期出现的冷号 ===
        vs = Counter()
        for w in w_s: vs[pred_single(train,w)] += w_weight(w)
        ps = vs.most_common(1)[0][0]
        if miss_s >= SINGLE_MISS_PROTECT:
            # 获取最近3期出现过的生肖（避免选刚出过的）
            recent3 = train[-3:] if len(train)>=3 else train
            appeared_recent = set()
            for r in recent3:
                appeared_recent.update(get_zodiac(n) for n in r["numbers"])
                appeared_recent.add(get_zodiac(r["special_number"]))
            # 从遗漏前3中选一个未在最近3期出现的
            for z in sorted_cold[:3]:
                if z not in appeared_recent:
                    ps = z
                    break
            else:
                ps = sorted_cold[0]  # 兜底用最冷
        if ps in win_z: hit_s+=1; miss_s=0
        else: miss_s+=1; max_s=max(max_s,miss_s)

        # === 二肖不变 ===
        vt = Counter()
        for w in w_t:
            for z in pred_two(train,w): vt[z] += w_weight(w)
        pt = [z for z,_ in vt.most_common(2)]
        if miss_t >= TWO_MISS_PROTECT and sorted_cold:
            cold = sorted_cold[0]
            if cold not in pt: pt[-1] = cold
        if any(z in win_z for z in pt): hit_t+=1; miss_t=0
        else: miss_t+=1; max_t=max(max_t,miss_t)

        # === 三肖保护：保留第1名，后2名换最冷2个 ===
        use_n, nw = three_normal_signal(train)
        vth = Counter()
        for w in w_th:
            for z in pred_three(train, w, use_n, nw):
                vth[z] += w_weight(w, 30)
        hot_b, cold_m = three_trend_factor(train)
        for z in sorted_cold:
            bonus = three_cold_bonus(om[z])
            if bonus > 0:
                vth[z] += bonus * cold_m
        pth = [z for z,_ in vth.most_common(3)]
        if miss_th >= THREE_MISS_PROTECT and len(sorted_cold)>=2:
            keep = pth[:1]                       # 只保留第一名
            new_cold = [z for z in sorted_cold[:2] if z != keep[0]]
            pth = keep + new_cold
            while len(pth) < 3:
                for z,_ in vth.most_common():
                    if z not in pth:
                        pth.append(z)
                        break
                else: pth.append("马")
        if sum(1 for z in pth if z in win_z) >= 2: hit_th+=1; miss_th=0
        else: miss_th+=1; max_th=max(max_th,miss_th)

    return {
        "s_hit": hit_s/total, "s_miss": max_s,
        "t_hit": hit_t/total, "t_miss": max_t,
        "th_hit": hit_th/total, "th_miss": max_th,
    }

# ===================== 窗口优化 =====================
def opt_single(rows, look=100):
    best, score = [6,10,12,18], -1
    for combo in combinations(CANDIDATE_SINGLE, 4):
        st = backtest(rows, look, list(combo), OPTIMAL_TWO, OPTIMAL_THREE)
        if st is None: continue
        s = st["s_hit"] - st["s_miss"]*SINGLE_PENALTY
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
    best, score = [8,10,12,15], -1
    for combo in combinations(CANDIDATE_THREE, 4):
        st = backtest(rows, look, OPTIMAL_SINGLE, OPTIMAL_TWO, list(combo))
        if st is None: continue
        s = st["th_hit"] - st["th_miss"]*THREE_PENALTY
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