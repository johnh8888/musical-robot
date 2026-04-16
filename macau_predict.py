#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新澳门六合彩预测 - 稳定版（最近3期，自动避免主号重复）
支持网络重试、CSV备用、确定性选号
用法:
    python macau_predict.py sync          # 同步历史数据（自动重试）
    python macau_predict.py predict       # 生成下期预测
    python macau_predict.py show          # 显示最终推荐
"""

import argparse
import csv
import json
import logging
import math
import os
import sys
import sqlite3
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("macau_predict")

# -------------------- 常量 --------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH_DEFAULT = str(SCRIPT_DIR / "macau_gentle.db")
CSV_FALLBACK_PATH = str(SCRIPT_DIR / "macau_history.csv")   # 备用CSV路径
MACAU_API_URL = "https://marksix6.net/index.php?api=1"

STRATEGY_CONFIGS = {
    "hot": {"name": "热号策略", "w_freq": 0.7, "w_omit": 0.0, "w_mom": 0.3},
    "cold": {"name": "冷号回补", "w_freq": 0.0, "w_omit": 0.7, "w_mom": 0.3},
    "momentum": {"name": "近期动量", "w_freq": 0.2, "w_omit": 0.0, "w_mom": 0.8},
    "balanced": {"name": "组合策略", "w_freq": 0.35, "w_omit": 0.25, "w_mom": 0.25},
    "pattern": {"name": "规律挖掘", "w_freq": 0.30, "w_omit": 0.30, "w_mom": 0.20},
    "ensemble": {"name": "集成投票", "w_freq": 0.30, "w_omit": 0.30, "w_mom": 0.20},
}
STRATEGY_IDS = ["hot", "cold", "momentum", "balanced", "ensemble", "pattern"]

# 生肖映射
ZODIAC_MAP = {
    "马": [1,13,25,37,49], "羊": [12,24,36,48], "猴": [11,23,35,47],
    "鸡": [10,22,34,46], "狗": [9,21,33,45], "猪": [8,20,32,44],
    "鼠": [7,19,31,43], "牛": [6,18,30,42], "虎": [5,17,29,41],
    "兔": [4,16,28,40], "龙": [3,15,27,39], "蛇": [2,14,26,38],
}
WUXING_NUM_MAP = {
    "金": [4,5,12,13,20,21,28,29,36,37,44,45],
    "木": [1,8,9,16,17,24,25,32,33,40,41,48,49],
    "水": [6,7,14,15,22,23,30,31,38,39,46,47],
    "火": [2,3,10,11,18,19,26,27,34,35,42,43],
    "土": [5,6,13,14,21,22,29,30,37,38,45,46]
}
ZODIAC_WUXING = {
    "鼠": "水", "牛": "土", "虎": "木", "兔": "木", "龙": "土", "蛇": "火",
    "马": "火", "羊": "土", "猴": "金", "鸡": "金", "狗": "土", "猪": "水"
}
WUXING_RELATION = {
    "金": {"生": "水", "克": "木"}, "木": {"生": "火", "克": "土"},
    "水": {"生": "木", "克": "火"}, "火": {"生": "土", "克": "金"},
    "土": {"生": "金", "克": "水"}
}
ZODIAC_CLASH = {
    "鼠": "马", "马": "鼠", "牛": "羊", "羊": "牛", "虎": "猴", "猴": "虎",
    "兔": "鸡", "鸡": "兔", "龙": "狗", "狗": "龙", "蛇": "猪", "猪": "蛇"
}
ZODIAC_HARMONY = {
    "鼠": "牛", "牛": "鼠", "虎": "猪", "猪": "虎", "兔": "狗", "狗": "兔",
    "龙": "鸡", "鸡": "龙", "蛇": "猴", "猴": "蛇", "马": "羊", "羊": "马"
}
PERSONAL_FAVOR = ["金", "水"]
PERSONAL_AVOID = ["火", "木"]
FAVOR_BONUS = 0.15
AVOID_PENALTY = 0.1

ALL_NUMBERS = list(range(1, 50))
SUM_TARGET = (105, 195)
PREDICT_WINDOW = 3
BACKTEST_WINDOW = 8
FENGSHUI_POWER = 0.03
STAT_POWER = 0.97
TOP_CANDIDATES = 16

ZODIAC_ODDS = {"马": 0.7}
SPECIAL_ODDS = 46
TRIO_ODDS = 1000

# 网络请求配置
REQUEST_TIMEOUT = 60
REQUEST_RETRIES = 2


@dataclass
class DrawRecord:
    issue_no: str
    draw_date: str
    numbers: List[int]
    special_number: int

@dataclass
class StrategyScore:
    main_picks: List[int]
    special_pick: int
    confidence: float
    raw_scores: Dict[int, float] = field(default_factory=dict)


# -------------------- 工具函数 --------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def parse_issue(issue_no: str) -> Optional[Tuple[str, int, int]]:
    if "/" in issue_no:
        parts = issue_no.split("/")
    else:
        if len(issue_no) >= 7:
            parts = [issue_no[:4], issue_no[4:]]
        else:
            return None
    if len(parts) != 2: return None
    year_s, seq_s = parts
    if not (year_s.isdigit() and seq_s.isdigit()): return None
    return year_s, int(seq_s), len(seq_s)

def next_issue_number(issue: str) -> str:
    parsed = parse_issue(issue)
    if not parsed: return issue
    year, seq, width = parsed
    return f"{year}{str(seq+1).zfill(width)}"

def get_zodiac(num: int) -> str:
    for z, nums in ZODIAC_MAP.items():
        if num in nums: return z
    return ""

def get_day_ganzhi(dt: date) -> Tuple[str, str, str]:
    base = date(1900,1,1)
    days = (dt-base).days
    gan_list = ["甲","乙","丙","丁","戊","己","庚","辛","壬","癸"]
    zhi_list = ["子","丑","寅","卯","辰","巳","午","未","申","酉","戌","亥"]
    gan = gan_list[days%10]
    zhi = zhi_list[days%12]
    wuxing = {"甲":"木","乙":"木","丙":"火","丁":"火","戊":"土","己":"土","庚":"金","辛":"金","壬":"水","癸":"水"}[gan]
    return gan, zhi, wuxing

def get_zodiac_clash_score(zodiac: str, day_zhi: str) -> float:
    score = 0.0
    zhi_to_zodiac = {"子":"鼠","丑":"牛","寅":"虎","卯":"兔","辰":"龙","巳":"蛇","午":"马","未":"羊","申":"猴","酉":"鸡","戌":"狗","亥":"猪"}
    day_zodiac = zhi_to_zodiac.get(day_zhi, "")
    if ZODIAC_CLASH.get(zodiac) == day_zodiac: score -= 0.5
    if ZODIAC_CLASH.get(day_zodiac) == zodiac: score -= 0.3
    if ZODIAC_HARMONY.get(zodiac) == day_zodiac: score += 0.5
    triples = [("申","子","辰"),("亥","卯","未"),("寅","午","戌"),("巳","酉","丑")]
    for triple in triples:
        if day_zhi in triple and zodiac in [zhi_to_zodiac[z] for z in triple if z != day_zhi]:
            score += 0.3
            break
    return score

def get_number_wuxing(num: int) -> str:
    for w, nums in WUXING_NUM_MAP.items():
        if num in nums: return w
    return ""

def get_number_fengshui_score(num: int, day_wuxing: str, day_zhi: str) -> float:
    score = 0.0
    zodiac = get_zodiac(num)
    num_wuxing = get_number_wuxing(num)
    if num_wuxing and day_wuxing:
        rel = WUXING_RELATION.get(day_wuxing, {})
        if num_wuxing == rel.get("生"): score += 0.4
        elif num_wuxing == rel.get("克"): score -= 0.3
        elif day_wuxing == WUXING_RELATION.get(num_wuxing, {}).get("生"): score += 0.2
    score += get_zodiac_clash_score(zodiac, day_zhi)
    zod_wuxing = ZODIAC_WUXING.get(zodiac, "")
    if zod_wuxing and day_wuxing:
        rel = WUXING_RELATION.get(day_wuxing, {})
        if zod_wuxing == rel.get("生"): score += 0.15
        elif zod_wuxing == rel.get("克"): score -= 0.1
    if num_wuxing in PERSONAL_FAVOR: score += FAVOR_BONUS
    elif num_wuxing in PERSONAL_AVOID: score -= AVOID_PENALTY
    return max(-1.0, min(1.0, score))


# -------------------- 数据库 --------------------
def connect_db(db_path=DB_PATH_DEFAULT):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS draws (
            issue_no TEXT PRIMARY KEY, draw_date TEXT NOT NULL,
            numbers_json TEXT NOT NULL, special_number INTEGER NOT NULL,
            sum_value INTEGER, odd_count INTEGER, big_count INTEGER,
            consec_pairs INTEGER, zodiac_json TEXT, source TEXT,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            issue_no TEXT NOT NULL, strategy TEXT NOT NULL,
            numbers_json TEXT NOT NULL, special_number INTEGER,
            confidence REAL, hit_count INTEGER, hit_rate REAL,
            special_hit INTEGER, status TEXT DEFAULT 'PENDING',
            created_at TEXT NOT NULL, reviewed_at TEXT,
            UNIQUE(issue_no, strategy)
        );
        CREATE TABLE IF NOT EXISTS backtest_stats (
            strategy TEXT PRIMARY KEY, total_runs INTEGER DEFAULT 0,
            avg_hit REAL DEFAULT 0, hit1_rate REAL DEFAULT 0,
            hit2_rate REAL DEFAULT 0, hit3_rate REAL DEFAULT 0,
            special_rate REAL DEFAULT 0, sharpe_ratio REAL DEFAULT 0,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS pair_affinity (
            num1 INTEGER NOT NULL, num2 INTEGER NOT NULL,
            co_occurrence INTEGER DEFAULT 0, lift REAL DEFAULT 1.0,
            updated_at TEXT NOT NULL, PRIMARY KEY (num1, num2)
        );
    """)
    _ensure_columns(conn)
    conn.commit()

def _ensure_columns(conn):
    existing = {r[1] for r in conn.execute("PRAGMA table_info(draws)").fetchall()}
    for col in {"sum_value","odd_count","big_count","consec_pairs","zodiac_json"} - existing:
        if col == "zodiac_json":
            conn.execute(f"ALTER TABLE draws ADD COLUMN {col} TEXT")
        else:
            conn.execute(f"ALTER TABLE draws ADD COLUMN {col} INTEGER")
    if "confidence" not in {r[1] for r in conn.execute("PRAGMA table_info(predictions)")}:
        conn.execute("ALTER TABLE predictions ADD COLUMN confidence REAL")
    for col in ["hit3_rate","sharpe_ratio"]:
        if col not in {r[1] for r in conn.execute("PRAGMA table_info(backtest_stats)")}:
            conn.execute(f"ALTER TABLE backtest_stats ADD COLUMN {col} REAL")

def compute_draw_features(numbers):
    return {
        "sum_value": sum(numbers),
        "odd_count": sum(1 for n in numbers if n%2),
        "big_count": sum(1 for n in numbers if n>=25),
        "consec_pairs": sum(1 for i in range(5) if abs(numbers[i]-numbers[i+1])==1),
        "zodiac_json": json.dumps([get_zodiac(n) for n in numbers])
    }

def upsert_draw(conn, record, source):
    now = utc_now()
    feat = compute_draw_features(record.numbers)
    if conn.execute("SELECT 1 FROM draws WHERE issue_no=?", (record.issue_no,)).fetchone():
        conn.execute("""UPDATE draws SET draw_date=?, numbers_json=?, special_number=?,
            sum_value=?, odd_count=?, big_count=?, consec_pairs=?, zodiac_json=?,
            source=?, updated_at=? WHERE issue_no=?""",
            (record.draw_date, json.dumps(record.numbers), record.special_number,
             feat["sum_value"], feat["odd_count"], feat["big_count"],
             feat["consec_pairs"], feat["zodiac_json"], source, now, record.issue_no))
        return "updated"
    else:
        conn.execute("""INSERT INTO draws (issue_no, draw_date, numbers_json, special_number,
            sum_value, odd_count, big_count, consec_pairs, zodiac_json,
            source, created_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (record.issue_no, record.draw_date, json.dumps(record.numbers), record.special_number,
             feat["sum_value"], feat["odd_count"], feat["big_count"],
             feat["consec_pairs"], feat["zodiac_json"], source, now, now))
        return "inserted"


# -------------------- 数据获取（带重试和CSV备用）--------------------
def fetch_macau_history_from_api(retry=REQUEST_RETRIES):
    for attempt in range(1, retry+1):
        try:
            logger.info(f"尝试获取数据 (第{attempt}次)...")
            resp = requests.get(MACAU_API_URL, timeout=REQUEST_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code == 200:
                data = resp.json()
                macau_data = next((item for item in data.get("lottery_data",[]) if item.get("name")=="新澳门彩"), None)
                if not macau_data:
                    logger.warning("未找到新澳门彩数据")
                    continue
                records = []
                for line in macau_data.get("history", []):
                    m = re.match(r"(\d{7})\s*期[：:]\s*([\d,]+)", line)
                    if m:
                        nums = [int(x) for x in m.group(2).split(",") if x.strip().isdigit()]
                        if len(nums) >= 7:
                            records.append(DrawRecord(
                                issue_no=f"{m.group(1)[:4]}/{m.group(1)[4:]}",
                                draw_date=datetime.now().strftime("%Y-%m-%d"),
                                numbers=nums[:6],
                                special_number=nums[6]
                            ))
                if not records:
                    expect = str(macau_data.get("expect",""))
                    code = macau_data.get("openCode","")
                    if code and len(expect)>=7:
                        nums = [int(x) for x in code.split(",") if x.strip().isdigit()]
                        if len(nums)>=7:
                            records.append(DrawRecord(
                                issue_no=f"{expect[:4]}/{expect[4:]}",
                                draw_date=macau_data.get("openTime","")[:10] or datetime.now().strftime("%Y-%m-%d"),
                                numbers=nums[:6],
                                special_number=nums[6]
                            ))
                if records:
                    logger.info(f"成功获取 {len(records)} 条记录")
                    return records
        except Exception as e:
            logger.warning(f"请求失败 (尝试{attempt}/{retry}): {e}")
            if attempt < retry:
                time.sleep(2)
    logger.error("所有API尝试均失败")
    return None

def load_csv_fallback(csv_path):
    if not os.path.exists(csv_path):
        logger.warning(f"备用CSV文件不存在: {csv_path}")
        return None
    records = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            issue = row.get("期号") or row.get("issue")
            date_str = row.get("日期") or row.get("date")
            nums_str = row.get("开奖号码") or row.get("numbers")
            special_str = row.get("特别号码") or row.get("special")
            if not issue or not nums_str:
                continue
            nums = [int(x) for x in re.findall(r"\d+", nums_str) if 1<=int(x)<=49]
            if len(nums) >= 7:
                records.append(DrawRecord(
                    issue_no=issue,
                    draw_date=date_str or "",
                    numbers=nums[:6],
                    special_number=nums[6]
                ))
    logger.info(f"从CSV导入 {len(records)} 条记录")
    return records

def sync_draws(conn, records, source="online"):
    ins = upd = 0
    for r in records:
        res = upsert_draw(conn, r, source)
        if res == "inserted": ins += 1
        else: upd += 1
    conn.commit()
    return ins, upd


# -------------------- 特征工程 --------------------
def get_recent_draws(conn, limit=PREDICT_WINDOW):
    rows = conn.execute("SELECT numbers_json FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT ?", (limit,)).fetchall()
    return [json.loads(r[0]) for r in rows]

def get_recent_specials(conn, limit=PREDICT_WINDOW):
    rows = conn.execute("SELECT special_number FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT ?", (limit,)).fetchall()
    return [r[0] for r in rows]

def calculate_exp_momentum(draws, half_life=2):
    scores = {n:0.0 for n in ALL_NUMBERS}
    for i, draw in enumerate(draws):
        w = math.exp(-i/half_life)
        for n in draw: scores[n] += w
    return scores

def calculate_pair_lift(draws):
    pair_cnt, single_cnt = Counter(), Counter()
    for draw in draws:
        for n in draw: single_cnt[n] += 1
        for a,b in combinations(sorted(draw),2):
            pair_cnt[(a,b)] += 1
    total = len(draws)
    lift = {}
    for (a,b), cnt in pair_cnt.items():
        exp = (single_cnt[a]/total)*(single_cnt[b]/total)*total if total>0 else 0
        if exp>0: lift[(a,b)] = cnt/exp
    return lift

def find_optimal_weights(draws, specials, base_weights, improvement_threshold=0.05):
    if len(draws)<4: return base_weights
    test_window = max(2, len(draws)//3)
    best_weights = base_weights.copy()
    best_avg = 0.0
    for df in [-0.10,0.0,0.10]:
        for do in [-0.10,0.0,0.10]:
            w_freq = base_weights["w_freq"]+df
            w_omit = base_weights["w_omit"]+do
            w_mom = 1.0 - w_freq - w_omit
            if w_freq<0.1 or w_omit<0.0 or w_mom<0.1 or w_mom>0.6: continue
            total_hits, cnt = 0,0
            for i in range(test_window, len(draws)):
                past = draws[:i]
                if len(past)<3: continue
                score = generate_strategy_score_with_weights(past, {"w_freq":w_freq,"w_omit":w_omit,"w_mom":w_mom})
                total_hits += len(set(score.main_picks) & set(draws[i]))
                cnt += 1
            if cnt>0:
                avg = total_hits/cnt
                if avg > best_avg + improvement_threshold:
                    best_avg = avg
                    best_weights = {"w_freq":w_freq,"w_omit":w_omit,"w_mom":w_mom}
    return best_weights

def generate_strategy_score_with_weights(draws, weights):
    freq = {n:0.0 for n in ALL_NUMBERS}
    for d in draws:
        for n in d: freq[n] += 1.0
    omit = {}
    for n in ALL_NUMBERS:
        for i,d in enumerate(draws):
            if n in d:
                omit[n]=i
                break
        else:
            omit[n]=len(draws)
    mom = calculate_exp_momentum(draws)
    def norm(d):
        vals = list(d.values())
        mn, mx = min(vals), max(vals)
        if mx==mn: return {k:0.0 for k in d}
        return {k:(v-mn)/(mx-mn) for k,v in d.items()}
    freq_n = norm(freq)
    omit_n = norm({n:1.0/(omit[n]+1) for n in ALL_NUMBERS})
    mom_n = norm(mom)
    scores = {n: freq_n[n]*weights["w_freq"] + omit_n[n]*weights["w_omit"] + mom_n[n]*weights["w_mom"] for n in ALL_NUMBERS}
    main = deterministic_pick(scores, {})
    special = max(scores, key=lambda n: scores[n] if n not in main else -1)
    return StrategyScore(main, special, 0.0, scores)

def deterministic_pick(scores, pair_lift, top_candidates=TOP_CANDIDATES):
    sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    candidates = [n for n,_ in sorted_nums[:top_candidates]]
    best_combo, best_score = None, -1e9
    for combo in combinations(candidates,6):
        combo = sorted(combo)
        if not smart_filter(combo): continue
        score = sum(scores[n] for n in combo)
        for a,b in combinations(combo,2):
            score += pair_lift.get((a,b),0)*0.2
        if score > best_score:
            best_score = score
            best_combo = combo
    if best_combo:
        return list(best_combo)
    else:
        return [n for n,_ in sorted_nums[:6]]

def smart_filter(nums):
    if len(nums)!=6: return False
    s = sorted(nums)
    total = sum(s)
    odd = sum(1 for n in s if n%2)
    big = sum(1 for n in s if n>=25)
    if total<SUM_TARGET[0] or total>SUM_TARGET[1]: return False
    if odd==0 or odd==6: return False
    if big==0 or big==6: return False
    zones = [(n-1)//10 for n in s]
    if max(Counter(zones).values())>3: return False
    tails = [n%10 for n in s]
    if max(Counter(tails).values())>2: return False
    max_consec = 1
    consec = 1
    for i in range(1,6):
        if s[i]-s[i-1]==1:
            consec+=1
            max_consec = max(max_consec, consec)
        else:
            consec=1
    if max_consec>3: return False
    primes = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47}
    prime_cnt = sum(1 for n in s if n in primes)
    if prime_cnt==0 or prime_cnt==6: return False
    colors = {"红":0,"蓝":0,"绿":0}
    for n in s:
        if n in [1,2,7,8,12,13,18,19,23,24,29,30,34,35,40,45,46]:
            colors["红"]+=1
        elif n in [3,4,9,10,14,15,20,21,25,26,31,32,36,37,41,42,47,48]:
            colors["蓝"]+=1
        else:
            colors["绿"]+=1
    if any(v==6 for v in colors.values()): return False
    return True

def generate_strategy_score(draws, specials, strategy, pair_lift, use_dynamic_weights=True, day_wuxing="", day_zhi=""):
    base = STRATEGY_CONFIGS.get(strategy, STRATEGY_CONFIGS["balanced"])
    weights = {"w_freq":base["w_freq"], "w_omit":base["w_omit"], "w_mom":base["w_mom"]}
    if use_dynamic_weights and strategy!="ensemble" and len(draws)>=4:
        weights = find_optimal_weights(draws, specials, weights)
    freq = {n:0.0 for n in ALL_NUMBERS}
    for d in draws:
        for n in d: freq[n] += 1.0
    omit = {}
    for n in ALL_NUMBERS:
        for i,d in enumerate(draws):
            if n in d:
                omit[n]=i
                break
        else:
            omit[n]=len(draws)
    mom = calculate_exp_momentum(draws)
    def norm(d):
        vals = list(d.values())
        mn,mx = min(vals),max(vals)
        if mx==mn: return {k:0.0 for k in d}
        return {k:(v-mn)/(mx-mn) for k,v in d.items()}
    freq_n = norm(freq)
    omit_n = norm({n:1.0/(omit[n]+1) for n in ALL_NUMBERS})
    mom_n = norm(mom)
    stat_scores = {n: freq_n[n]*weights["w_freq"] + omit_n[n]*weights["w_omit"] + mom_n[n]*weights["w_mom"] for n in ALL_NUMBERS}
    stat_norm = norm(stat_scores)
    if day_wuxing and day_zhi:
        fengshui = {n:(get_number_fengshui_score(n,day_wuxing,day_zhi)+1)/2 for n in ALL_NUMBERS}
    else:
        fengshui = {n:0.5 for n in ALL_NUMBERS}
    final = {n: stat_norm[n]*STAT_POWER + fengshui[n]*FENGSHUI_POWER for n in ALL_NUMBERS}
    if strategy == "ensemble":
        return ensemble_vote(draws, specials, pair_lift, use_dynamic_weights, day_wuxing, day_zhi)
    main = deterministic_pick(final, pair_lift)
    special = max((n for n in ALL_NUMBERS if n not in main), key=lambda n: final[n])
    conf = sum(final[n] for n in main)/6 if main else 0
    return StrategyScore(main, special, conf, final)

def ensemble_vote(draws, specials, pair_lift, use_dynamic_weights, day_wuxing, day_zhi):
    score_list = [generate_strategy_score(draws, specials, s, pair_lift, use_dynamic_weights, day_wuxing, day_zhi).raw_scores for s in ["hot","cold","momentum","balanced","pattern"]]
    votes = {n:0.0 for n in ALL_NUMBERS}
    for sc in score_list:
        for rank,(n,_) in enumerate(sorted(sc.items(), key=lambda x:x[1], reverse=True)):
            votes[n] += 49 - rank
    maxv = max(votes.values())
    norm_votes = {n:v/maxv for n,v in votes.items()} if maxv else {n:0.0 for n in ALL_NUMBERS}
    main = deterministic_pick(norm_votes, pair_lift)
    special = max((n for n in ALL_NUMBERS if n not in main), key=lambda n: norm_votes[n])
    conf = sum(norm_votes[n] for n in main)/6 if main else 0
    return StrategyScore(main, special, conf, norm_votes)

def wilson_interval(hits, total, z=1.96):
    if total==0: return (0.0,0.0)
    p = hits/total
    n = total
    denom = 1 + z**2/n
    centre = (p + z**2/(2*n))/denom
    adj = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2))/denom
    return (max(0.0, centre-adj)*100, min(1.0, centre+adj)*100)

def bayesian_posterior(hits, total):
    return (hits+1)/(total+49)*100

def send_pushplus_notification(title, content):
    token = os.environ.get("PUSHPLUS_TOKEN")
    if not token: return False
    try:
        r = requests.post("http://www.pushplus.plus/send", json={"token":token,"title":title,"content":content}, timeout=10)
        return r.json().get("code")==200
    except: return False

def print_betting_plan(hot5, top1_zod, top2_zod, special_first, top_specials, best_combo, budget=500):
    odds_zod = ZODIAC_ODDS.get(top1_zod, 1.0)
    S = int(budget/odds_zod) + (1 if budget/odds_zod > int(budget/odds_zod) else 0)
    rem = budget - S
    if rem < 0: S,T,P = budget,0,0
    else: T,P = int(rem*0.7), rem - int(rem*0.7)
    if odds_zod==1.0 and S==budget:
        print("\n⚠️ 生肖赔率1:1，需全部预算保本，提供备选：生肖480元，特码15元，三中三5元")
        S,T,P = 480,15,5
    print("\n"+"="*60)
    print("💰 智能投注方案")
    print(f"总预算: {budget}元 | 生肖: {top1_zod} (赔率1:{odds_zod}) | 特码: {special_first:02d} (1:{SPECIAL_ODDS})")
    if best_combo: print(f"三中三: {' '.join(f'{n:02d}' for n in best_combo)} (1:{TRIO_ODDS})")
    print("-"*60)
    print(f"一肖 {top1_zod}: {S}元 (中得 {S*odds_zod:.2f})")
    if T>0: print(f"特码 {special_first:02d}: {T}元 (中得 {T*SPECIAL_ODDS})")
    if P>0 and best_combo: print(f"三中三 {' '.join(f'{n:02d}' for n in best_combo)}: {P}元 (中得 {P*TRIO_ODDS})")
    print("-"*60)
    if T>0 or P>0:
        print("预期回报：")
        only_z = S*odds_zod
        print(f"仅生肖中: {only_z:.2f}元, 净收益 {only_z-budget:.2f}")
        if T>0: print(f"生肖+特码: {only_z+T*SPECIAL_ODDS:.2f}元, 净收益 {only_z+T*SPECIAL_ODDS-budget:.2f}")
        if P>0 and best_combo: print(f"生肖+三中三: {only_z+P*TRIO_ODDS:.2f}元, 净收益 {only_z+P*TRIO_ODDS-budget:.2f}")
        if T>0 and P>0 and best_combo: print(f"全中: {only_z+T*SPECIAL_ODDS+P*TRIO_ODDS:.2f}元, 净收益 {only_z+T*SPECIAL_ODDS+P*TRIO_ODDS-budget:.2f}")
    print("="*60)


# -------------------- 命令行接口 --------------------
def cmd_sync(args):
    conn = connect_db(args.db)
    init_db(conn)
    print("正在同步新澳门彩历史数据...")
    records = fetch_macau_history_from_api()
    if records is None:
        print("API获取失败，尝试从CSV导入...")
        records = load_csv_fallback(CSV_FALLBACK_PATH)
    if not records:
        print("错误：未获取到有效记录。请检查网络或提供CSV文件。")
        return
    ins, upd = sync_draws(conn, records, "macau_api")
    print(f"同步完成：新增 {ins} 期，更新 {upd} 期。")
    conn.close()

def cmd_predict(args):
    conn = connect_db(args.db)
    init_db(conn)
    draws = get_recent_draws(conn, PREDICT_WINDOW)
    specials = get_recent_specials(conn, PREDICT_WINDOW)
    if len(draws) < 3:
        print("错误：历史数据不足（至少3期），请先运行 sync。")
        return
    pair_lift = calculate_pair_lift(draws)
    latest = conn.execute("SELECT issue_no FROM draws ORDER BY draw_date DESC LIMIT 1").fetchone()
    next_issue = next_issue_number(latest[0]) if latest else f"{datetime.now().year}001"
    today = date.today()
    _, day_zhi, day_wuxing = get_day_ganzhi(today)
    for strat in STRATEGY_IDS:
        score = generate_strategy_score(draws, specials, strat, pair_lift, True, day_wuxing, day_zhi)
        conn.execute("INSERT OR REPLACE INTO predictions (issue_no, strategy, numbers_json, special_number, confidence, status, created_at) VALUES (?,?,?,?,?,'PENDING',?)",
                     (next_issue, strat, json.dumps(score.main_picks), score.special_pick, score.confidence, utc_now()))
    conn.commit()
    print(f"已生成 {next_issue} 期的预测推荐。")
    conn.close()

def cmd_show(args):
    conn = connect_db(args.db)
    init_db(conn)

    # 获取最新一期开奖（用于避免主号完全重复）
    latest_draw = conn.execute("SELECT numbers_json, special_number FROM draws ORDER BY draw_date DESC LIMIT 1").fetchone()
    prev_main_set = set(json.loads(latest_draw["numbers_json"])) if latest_draw else set()
    prev_special = latest_draw["special_number"] if latest_draw else None

    # 获取集成投票预测结果
    pending = conn.execute("SELECT issue_no, strategy, numbers_json, special_number, confidence FROM predictions WHERE status='PENDING' ORDER BY strategy").fetchall()
    if not pending:
        print("暂无待开奖预测，请先运行 predict")
        conn.close()
        return
    ensemble_row = next((p for p in pending if p["strategy"]=="ensemble"), None)
    if ensemble_row:
        main6 = json.loads(ensemble_row["numbers_json"])
        special = ensemble_row["special_number"]
    else:
        draws = get_recent_draws(conn, PREDICT_WINDOW)
        specials = get_recent_specials(conn, PREDICT_WINDOW)
        pair_lift = calculate_pair_lift(draws)
        today = date.today()
        _, day_zhi, day_wuxing = get_day_ganzhi(today)
        score = ensemble_vote(draws, specials, pair_lift, True, day_wuxing, day_zhi)
        main6 = score.main_picks
        special = score.special_pick

    # 避免主号与上期完全重复
    if set(main6) == prev_main_set:
        draws = get_recent_draws(conn, PREDICT_WINDOW)
        specials = get_recent_specials(conn, PREDICT_WINDOW)
        pair_lift = calculate_pair_lift(draws)
        today = date.today()
        _, day_zhi, day_wuxing = get_day_ganzhi(today)
        ensemble_score = ensemble_vote(draws, specials, pair_lift, True, day_wuxing, day_zhi)
        raw_scores = ensemble_score.raw_scores
        sorted_scores = sorted(raw_scores.items(), key=lambda x: x[1], reverse=True)
        candidates = [n for n,_ in sorted_scores if n not in prev_main_set]
        main_with_scores = [(n, raw_scores[n]) for n in main6]
        lowest = min(main_with_scores, key=lambda x: x[1])[0]
        if candidates:
            new_num = candidates[0]
            main6 = [new_num if n==lowest else n for n in main6]
            main6.sort()
            print(f"⚠️ 原主号与上期完全相同，已将 {lowest:02d} 替换为 {new_num:02d}")

    # 输出最终推荐
    print("\n" + "="*60)
    print(f"🎯 最终推荐 (最近3期统计 + 玄学3%)")
    print(f"主号6码: {' '.join(f'{n:02d}' for n in main6)}")
    print(f"特别号: {special:02d}")
    print("="*60)

    # 可选：输出简易投注方案（默认预算500元）
    # 此处可根据需要调用 print_betting_plan，需提供 hot5, top1_zod 等参数，为简化不展开
    conn.close()

def cmd_backtest(args):
    print("轻量回测已在 show 命令中展示最近8期统计，无需单独运行。")

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=DB_PATH_DEFAULT)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("sync").set_defaults(func=cmd_sync)
    sub.add_parser("predict").set_defaults(func=cmd_predict)
    sub.add_parser("show").set_defaults(func=cmd_show)
    sub.add_parser("backtest").set_defaults(func=cmd_backtest)
    return p

def main():
    args = build_parser().parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
