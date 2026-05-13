#!/usr/bin/env python3
# special_only.py - 香港特五肖预测（在线120期 + 增强策略）
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

# ===================== 在线获取香港数据 =====================
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
                records = []
                for item in data.get("lottery_data", []):
                    if "香港" in item.get("name","") or "六合彩" in item.get("name",""):
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

# ===================== 增强参数 =====================
CANDIDATE_WINDOWS = [4,6,8,10,12,15,18,20,24,30,36,42,48,54,60]
OPTIMAL_WINDOWS = [4,6,8,10,12]
SPECIAL_WEIGHT = 1.0
COLD_BASE = 0.8
COLD_STEP = 0.3
COLD_MAX = 1.5
MISS_PENALTY = 0.1
ADAPTIVE_LOOKBACK = 10
BASE_NORMAL_WEIGHT = 0.5
SIGNAL_THRESHOLD = 0.6
RECOMMEND_COUNT = 5

def w_weight(w, base=84):
    return round(base/w, 2)

def omission_map(rows):
    if not rows: return {z:0 for z in ZODIAC_LIST}
    om = {z:0 for z in ZODIAC_LIST}
    for r in reversed(rows):
        spz = get_zodiac(r["special_number"])
        for z in ZODIAC_LIST:
            om[z] = 0 if z==spz else om[z]+1
    return om

def normal_signal(history):
    if len(history) < ADAPTIVE_LOOKBACK+1:
        return False, 0.0
    recent = history[-(ADAPTIVE_LOOKBACK+1):]
    hits = 0
    for i in range(len(recent)-1):
        cur_set = set(get_zodiac(n) for n in recent[i]["numbers"])
        if get_zodiac(recent[i+1]["special_number"]) in cur_set:
            hits += 1
    rate = hits/(len(recent)-1)
    if rate >= SIGNAL_THRESHOLD:
        factor = 0.5 + rate
        w = BASE_NORMAL_WEIGHT * factor
        return True, max(0.1, min(0.6, w))
    return False, 0.0

def dynamic_cold_bonus(om_val):
    if om_val <= 0: return 0.0
    return min(COLD_BASE + (om_val//10)*COLD_STEP, COLD_MAX)

def recommend(history, windows, force_cold=False):
    if not history: return ZODIAC_LIST[:RECOMMEND_COUNT]
    use_normal, normal_w = normal_signal(history)
    om = omission_map(history)
    sorted_cold = sorted(om, key=om.get, reverse=True)
    votes = Counter()
    for w in windows:
        recent = history[-w:] if len(history)>=w else history
        cnt = Counter()
        for r in recent:
            spz = get_zodiac(r["special_number"])
            cnt[spz] += SPECIAL_WEIGHT
            if use_normal:
                for n in r["numbers"]:
                    cnt[get_zodiac(n)] += normal_w
        for z,_ in cnt.most_common(RECOMMEND_COUNT):
            votes[z] += w_weight(w)
    # 动态冷号加票
    for z in sorted_cold:
        bonus = dynamic_cold_bonus(om[z])
        if bonus > 0: votes[z] += bonus
    preds = [z for z,_ in votes.most_common(RECOMMEND_COUNT)]
    if force_cold and len(preds)>=RECOMMEND_COUNT:
        keep = preds[:RECOMMEND_COUNT-2]
        new_cold = [z for z in sorted_cold[:2] if z not in keep]
        preds = keep + new_cold
        while len(preds) < RECOMMEND_COUNT:
            for z,_ in votes.most_common():
                if z not in preds: preds.append(z); break
            else: preds.append("马")
    return preds[:RECOMMEND_COUNT]

def backtest(rows, lookback, windows):
    rev = list(reversed(rows))
    total = min(lookback, len(rev)-20)
    if total <=0: return None, None
    hits, cur_miss, max_miss = 0,0,0
    for i in range(total):
        train = rev[i+20:]
        if len(train)<20: continue
        actual_z = get_zodiac(rev[i]["special_number"])
        preds = recommend(train, windows, force_cold=(cur_miss>=1))
        if actual_z in preds:
            hits += 1; cur_miss = 0
        else:
            cur_miss += 1; max_miss = max(max_miss, cur_miss)
    return hits/total, max_miss

def optimize_windows(rows, look=100):
    best_combo, best_score = None, -float("inf")
    for combo in combinations(CANDIDATE_WINDOWS, 5):
        hr, miss = backtest(rows, look, list(combo))
        if hr is None: continue
        score = hr - miss*MISS_PENALTY
        if score > best_score:
            best_score = score
            best_combo = list(combo)
    return best_combo if best_combo else [4,6,8,10,12]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    rows = fetch_hk_online(120)
    if not rows:
        print("在线数据获取失败，退出。")
        return

    global OPTIMAL_WINDOWS
    OPTIMAL_WINDOWS = optimize_windows(rows)
    print(f"自动选择最优窗口: {OPTIMAL_WINDOWS}")

    if args.show:
        latest = rows[0]["issue_no"]
        pred = next_issue(latest)
        print(f"预测期号: {pred}")
        use_n, nw = normal_signal(rows)
        print(f"正码增强: {'开启' if use_n else '关闭'} (权重 {nw:.2f})")
        preds = recommend(rows, OPTIMAL_WINDOWS, force_cold=False)
        print(f"\n【特五肖推荐】: {'、'.join(preds)}")
        hr10, miss10 = backtest(rows, 10, OPTIMAL_WINDOWS)
        hr100, miss100 = backtest(rows, min(100, len(rows)), OPTIMAL_WINDOWS)
        if hr10 is not None:
            print(f"\n近10期回测：命中率 {hr10:.1%}，最大连空 {miss10}")
            print(f"近100期回测：命中率 {hr100:.1%}，最大连空 {miss100}")

if __name__ == "__main__":
    main()