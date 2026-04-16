#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新澳门六合彩 - 温和玄学版（统计主导 + 金水微调 + 动态权重）
玄学影响力仅 3%，统计占 97%
用法:
    python macau_predict.py sync --year 2026
    python macau_predict.py predict
    python macau_predict.py show
"""

import argparse
import hashlib
import json
import logging
import math
import os
import random
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

# -------------------- 日志系统 --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("macau_gentle")

# -------------------- 常量与配置 --------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH_DEFAULT = str(SCRIPT_DIR / "macau_gentle.db")

MACAU_HISTORY_URL = "https://history.macaumarksix.com/history/macaujc2/y/{}"

# 策略配置
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
    "马": [1, 13, 25, 37, 49],
    "羊": [12, 24, 36, 48],
    "猴": [11, 23, 35, 47],
    "鸡": [10, 22, 34, 46],
    "狗": [9, 21, 33, 45],
    "猪": [8, 20, 32, 44],
    "鼠": [7, 19, 31, 43],
    "牛": [6, 18, 30, 42],
    "虎": [5, 17, 29, 41],
    "兔": [4, 16, 28, 40],
    "龙": [3, 15, 27, 39],
    "蛇": [2, 14, 26, 38],
}

# 号码五行映射
WUXING_NUM_MAP = {
    "金": [4,5,12,13,20,21,28,29,36,37,44,45],
    "木": [1,8,9,16,17,24,25,32,33,40,41,48,49],
    "水": [6,7,14,15,22,23,30,31,38,39,46,47],
    "火": [2,3,10,11,18,19,26,27,34,35,42,43],
    "土": [5,6,13,14,21,22,29,30,37,38,45,46]
}

# 生肖五行
ZODIAC_WUXING = {
    "鼠": "水", "牛": "土", "虎": "木", "兔": "木",
    "龙": "土", "蛇": "火", "马": "火", "羊": "土",
    "猴": "金", "鸡": "金", "狗": "土", "猪": "水"
}

# 五行生克关系
WUXING_RELATION = {
    "金": {"生": "水", "克": "木", "被生": "土", "被克": "火"},
    "木": {"生": "火", "克": "土", "被生": "水", "被克": "金"},
    "水": {"生": "木", "克": "火", "被生": "金", "被克": "土"},
    "火": {"生": "土", "克": "金", "被生": "木", "被克": "水"},
    "土": {"生": "金", "克": "水", "被生": "火", "被克": "木"}
}

# 生肖冲煞（六冲）
ZODIAC_CLASH = {
    "鼠": "马", "马": "鼠",
    "牛": "羊", "羊": "牛",
    "虎": "猴", "猴": "虎",
    "兔": "鸡", "鸡": "兔",
    "龙": "狗", "狗": "龙",
    "蛇": "猪", "猪": "蛇"
}

# 生肖六合
ZODIAC_HARMONY = {
    "鼠": "牛", "牛": "鼠",
    "虎": "猪", "猪": "虎",
    "兔": "狗", "狗": "兔",
    "龙": "鸡", "鸡": "龙",
    "蛇": "猴", "猴": "蛇",
    "马": "羊", "羊": "马"
}

# 个人八字喜忌（温和版：金水微加，火木微扣）
PERSONAL_FAVOR = ["金", "水"]
PERSONAL_AVOID = ["火", "木"]
FAVOR_BONUS = 0.15      # 温和加分
AVOID_PENALTY = 0.1     # 温和扣分

ALL_NUMBERS = list(range(1, 50))
MONTE_CARLO_TRIALS = 5000
SUM_TARGET = (115, 185)
PREDICT_WINDOW = 7

# 玄学影响力（固定为 3%，可通过环境变量覆盖）
FENGSHUI_POWER = float(os.environ.get("FENGSHUI_POWER", "0.03"))   # 修改：默认 3%
STAT_POWER = 1.0 - FENGSHUI_POWER


# -------------------- 数据结构 --------------------
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


# -------------------- 日干支与玄学推算 --------------------
def get_day_ganzhi(dt: date) -> Tuple[str, str, str]:
    """返回 (天干, 地支, 日柱五行)"""
    base = date(1900, 1, 1)
    days = (dt - base).days
    gan_list = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"]
    zhi_list = ["子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥"]
    gan_idx = days % 10
    zhi_idx = days % 12
    gan = gan_list[gan_idx]
    zhi = zhi_list[zhi_idx]
    gan_wuxing = {
        "甲": "木", "乙": "木", "丙": "火", "丁": "火", "戊": "土",
        "己": "土", "庚": "金", "辛": "金", "壬": "水", "癸": "水"
    }
    wuxing = gan_wuxing[gan]
    return gan, zhi, wuxing


def get_zodiac_clash_score(zodiac: str, day_zhi: str) -> float:
    """根据日支与生肖的关系计算冲合得分"""
    score = 0.0
    zhi_to_zodiac = {
        "子": "鼠", "丑": "牛", "寅": "虎", "卯": "兔",
        "辰": "龙", "巳": "蛇", "午": "马", "未": "羊",
        "申": "猴", "酉": "鸡", "戌": "狗", "亥": "猪"
    }
    day_zodiac = zhi_to_zodiac.get(day_zhi, "")
    
    if ZODIAC_CLASH.get(zodiac) == day_zodiac:
        score -= 0.5
    if ZODIAC_CLASH.get(day_zodiac) == zodiac:
        score -= 0.3
    if ZODIAC_HARMONY.get(zodiac) == day_zodiac:
        score += 0.5
    
    triples = [("申", "子", "辰"), ("亥", "卯", "未"), ("寅", "午", "戌"), ("巳", "酉", "丑")]
    for triple in triples:
        if day_zhi in triple and zodiac in [zhi_to_zodiac[z] for z in triple if z != day_zhi]:
            score += 0.3
            break
    return score


def get_number_wuxing(num: int) -> str:
    for w, nums in WUXING_NUM_MAP.items():
        if num in nums:
            return w
    return ""


def get_number_fengshui_score(num: int, day_wuxing: str, day_zhi: str) -> float:
    """计算单个号码的玄学综合得分（含个人八字喜忌，温和版）"""
    score = 0.0
    zodiac = get_zodiac(num)
    num_wuxing = get_number_wuxing(num)
    
    # 1. 号码五行与日五行生克
    if num_wuxing and day_wuxing:
        relation = WUXING_RELATION.get(day_wuxing, {})
        if num_wuxing == relation.get("生"):
            score += 0.4
        elif num_wuxing == relation.get("克"):
            score -= 0.3
        elif day_wuxing == WUXING_RELATION.get(num_wuxing, {}).get("生"):
            score += 0.2
    
    # 2. 冲煞与六合
    score += get_zodiac_clash_score(zodiac, day_zhi)
    
    # 3. 生肖五行与日五行生克
    zod_wuxing = ZODIAC_WUXING.get(zodiac, "")
    if zod_wuxing and day_wuxing:
        relation = WUXING_RELATION.get(day_wuxing, {})
        if zod_wuxing == relation.get("生"):
            score += 0.15
        elif zod_wuxing == relation.get("克"):
            score -= 0.1

    # 4. 个人八字喜忌（温和加成）
    if num_wuxing in PERSONAL_FAVOR:
        score += FAVOR_BONUS
    elif num_wuxing in PERSONAL_AVOID:
        score -= AVOID_PENALTY

    return max(-1.0, min(1.0, score))


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
    if len(parts) != 2:
        return None
    year_s, seq_s = parts
    if not (year_s.isdigit() and seq_s.isdigit()):
        return None
    return year_s, int(seq_s), len(seq_s)


def next_issue_number(issue: str) -> str:
    parsed = parse_issue(issue)
    if not parsed:
        return issue
    year, seq, width = parsed
    return f"{year}{str(seq + 1).zfill(width)}"


def get_zodiac(num: int) -> str:
    for z, nums in ZODIAC_MAP.items():
        if num in nums:
            return z
    return ""


# -------------------- 数据库操作 --------------------
def connect_db(db_path: str = DB_PATH_DEFAULT) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS draws (
            issue_no TEXT PRIMARY KEY,
            draw_date TEXT NOT NULL,
            numbers_json TEXT NOT NULL,
            special_number INTEGER NOT NULL,
            sum_value INTEGER,
            odd_count INTEGER,
            big_count INTEGER,
            consec_pairs INTEGER,
            zodiac_json TEXT,
            source TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            issue_no TEXT NOT NULL,
            strategy TEXT NOT NULL,
            numbers_json TEXT NOT NULL,
            special_number INTEGER,
            confidence REAL,
            hit_count INTEGER,
            hit_rate REAL,
            special_hit INTEGER,
            status TEXT DEFAULT 'PENDING',
            created_at TEXT NOT NULL,
            reviewed_at TEXT,
            UNIQUE(issue_no, strategy)
        );

        CREATE TABLE IF NOT EXISTS backtest_stats (
            strategy TEXT PRIMARY KEY,
            total_runs INTEGER DEFAULT 0,
            avg_hit REAL DEFAULT 0,
            hit1_rate REAL DEFAULT 0,
            hit2_rate REAL DEFAULT 0,
            hit3_rate REAL DEFAULT 0,
            special_rate REAL DEFAULT 0,
            sharpe_ratio REAL DEFAULT 0,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS pair_affinity (
            num1 INTEGER NOT NULL,
            num2 INTEGER NOT NULL,
            co_occurrence INTEGER DEFAULT 0,
            lift REAL DEFAULT 1.0,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (num1, num2)
        );
    """)
    _ensure_columns(conn)
    conn.commit()


def _ensure_columns(conn: sqlite3.Connection) -> None:
    existing = {r[1] for r in conn.execute("PRAGMA table_info(draws)").fetchall()}
    desired = {"sum_value", "odd_count", "big_count", "consec_pairs", "zodiac_json"}
    for col in desired - existing:
        if col == "zodiac_json":
            conn.execute(f"ALTER TABLE draws ADD COLUMN {col} TEXT")
        else:
            conn.execute(f"ALTER TABLE draws ADD COLUMN {col} INTEGER")
    existing = {r[1] for r in conn.execute("PRAGMA table_info(predictions)").fetchall()}
    if "confidence" not in existing:
        conn.execute("ALTER TABLE predictions ADD COLUMN confidence REAL")
    existing = {r[1] for r in conn.execute("PRAGMA table_info(backtest_stats)").fetchall()}
    for col in ["hit3_rate", "sharpe_ratio"]:
        if col not in existing:
            conn.execute(f"ALTER TABLE backtest_stats ADD COLUMN {col} REAL")


def compute_draw_features(numbers: List[int]) -> Dict:
    zodiacs = [get_zodiac(n) for n in numbers]
    return {
        "sum_value": sum(numbers),
        "odd_count": sum(1 for n in numbers if n % 2 == 1),
        "big_count": sum(1 for n in numbers if n >= 25),
        "consec_pairs": sum(1 for i in range(5) if abs(numbers[i]-numbers[i+1]) == 1),
        "zodiac_json": json.dumps(zodiacs, ensure_ascii=False),
    }


def upsert_draw(conn: sqlite3.Connection, record: DrawRecord, source: str) -> str:
    now = utc_now()
    features = compute_draw_features(record.numbers)
    existing = conn.execute("SELECT issue_no FROM draws WHERE issue_no = ?", (record.issue_no,)).fetchone()
    if existing:
        conn.execute("""
            UPDATE draws SET draw_date=?, numbers_json=?, special_number=?,
                sum_value=?, odd_count=?, big_count=?, consec_pairs=?, zodiac_json=?,
                source=?, updated_at=?
            WHERE issue_no=?
        """, (record.draw_date, json.dumps(record.numbers), record.special_number,
              features["sum_value"], features["odd_count"], features["big_count"],
              features["consec_pairs"], features["zodiac_json"], source, now, record.issue_no))
        return "updated"
    else:
        conn.execute("""
            INSERT INTO draws (issue_no, draw_date, numbers_json, special_number,
                sum_value, odd_count, big_count, consec_pairs, zodiac_json,
                source, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (record.issue_no, record.draw_date, json.dumps(record.numbers), record.special_number,
              features["sum_value"], features["odd_count"], features["big_count"],
              features["consec_pairs"], features["zodiac_json"], source, now, now))
        return "inserted"


# -------------------- 数据获取 --------------------
def fetch_macau_history(year: int) -> List[DrawRecord]:
    url = MACAU_HISTORY_URL.format(year)
    logger.info(f"正在获取 {year} 年新澳门数据...")
    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            logger.error(f"请求失败: HTTP {resp.status_code}")
            return []
        data = resp.json()
        if data.get("code") != 200 or not data.get("data"):
            logger.error("接口返回错误或数据为空")
            return []
        records = []
        for item in data["data"]:
            issue = str(item.get("expect", "")).strip()
            open_code = item.get("openCode", "")
            open_time = item.get("openTime", "")
            if not issue or not open_code:
                continue
            nums = [int(x) for x in open_code.split(",") if x.strip().isdigit()]
            if len(nums) >= 7:
                records.append(DrawRecord(
                    issue_no=issue,
                    draw_date=open_time[:10] if open_time else "",
                    numbers=nums[:6],
                    special_number=nums[6]
                ))
        logger.info(f"获取到 {len(records)} 条记录")
        return records
    except Exception as e:
        logger.error(f"获取数据异常: {e}")
        return []


def sync_draws(conn: sqlite3.Connection, records: List[DrawRecord], source: str = "online") -> Tuple[int, int]:
    inserted = updated = 0
    for r in records:
        res = upsert_draw(conn, r, source)
        if res == "inserted":
            inserted += 1
        else:
            updated += 1
    conn.commit()
    return inserted, updated


# -------------------- 特征工程 --------------------
def get_recent_draws(conn: sqlite3.Connection, limit: int = PREDICT_WINDOW) -> List[List[int]]:
    rows = conn.execute(
        "SELECT numbers_json FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT ?",
        (limit,)
    ).fetchall()
    return [json.loads(r[0]) for r in rows]


def get_recent_specials(conn: sqlite3.Connection, limit: int = PREDICT_WINDOW) -> List[int]:
    rows = conn.execute(
        "SELECT special_number FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT ?",
        (limit,)
    ).fetchall()
    return [r[0] for r in rows]


def calculate_exp_momentum(draws: List[List[int]], half_life: int = 2) -> Dict[int, float]:
    scores = {n: 0.0 for n in ALL_NUMBERS}
    for i, draw in enumerate(draws):
        weight = math.exp(-i / half_life)
        for n in draw:
            scores[n] += weight
    return scores


def calculate_pair_lift(draws: List[List[int]]) -> Dict[Tuple[int, int], float]:
    pair_count = Counter()
    single_count = Counter()
    for draw in draws:
        for n in draw:
            single_count[n] += 1
        for a, b in combinations(sorted(draw), 2):
            pair_count[(a, b)] += 1
    total = len(draws)
    lift_map = {}
    for (a, b), cnt in pair_count.items():
        expected = (single_count[a] / total) * (single_count[b] / total) * total if total > 0 else 0
        if expected > 0:
            lift_map[(a, b)] = cnt / expected
    return lift_map


# -------------------- 动态权重优化 --------------------
def find_optimal_weights(
    draws: List[List[int]],
    specials: List[int],
    base_weights: Dict[str, float],
) -> Dict[str, float]:
    if len(draws) < 4:
        return base_weights

    test_window = max(2, len(draws) // 3)
    best_weights = base_weights.copy()
    best_hits = 0.0

    for dw_freq in [-0.10, 0.0, 0.10]:
        for dw_omit in [-0.10, 0.0, 0.10]:
            w_freq = base_weights["w_freq"] + dw_freq
            w_omit = base_weights["w_omit"] + dw_omit
            w_mom = 1.0 - w_freq - w_omit
            if w_freq < 0.1 or w_omit < 0.0 or w_mom < 0.1 or w_mom > 0.6:
                continue

            total_hits = 0
            count = 0
            for i in range(test_window, len(draws)):
                past_draws = draws[:i]
                past_specials = specials[:i]
                if len(past_draws) < 3:
                    continue
                score_obj = generate_strategy_score_with_weights(
                    past_draws, past_specials, {"w_freq": w_freq, "w_omit": w_omit, "w_mom": w_mom}
                )
                actual = set(draws[i])
                total_hits += len(set(score_obj.main_picks) & actual)
                count += 1

            if count > 0:
                avg_hits = total_hits / count
                if avg_hits > best_hits:
                    best_hits = avg_hits
                    best_weights = {"w_freq": w_freq, "w_omit": w_omit, "w_mom": w_mom}

    return best_weights


def generate_strategy_score_with_weights(
    draws: List[List[int]],
    specials: List[int],
    weights: Dict[str, float]
) -> StrategyScore:
    freq = {n: 0.0 for n in ALL_NUMBERS}
    for d in draws:
        for n in d:
            freq[n] += 1.0

    omit = {}
    for n in ALL_NUMBERS:
        for i, d in enumerate(draws):
            if n in d:
                omit[n] = i
                break
        else:
            omit[n] = len(draws)

    mom = calculate_exp_momentum(draws, half_life=2)

    def norm(d):
        vals = list(d.values())
        mn, mx = min(vals), max(vals)
        if mx == mn:
            return {k: 0.0 for k in d}
        return {k: (v - mn) / (mx - mn) for k, v in d.items()}

    freq_n = norm(freq)
    omit_n = norm({n: 1.0 / (omit[n] + 1) for n in ALL_NUMBERS})
    mom_n = norm(mom)

    scores = {}
    for n in ALL_NUMBERS:
        scores[n] = (
            freq_n[n] * weights["w_freq"]
            + omit_n[n] * weights["w_omit"]
            + mom_n[n] * weights["w_mom"]
        )

    main_picks = [n for n, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:6]]
    special = max(scores, key=lambda n: scores[n])
    return StrategyScore(main_picks, special, 0.0, scores)


# -------------------- 智能过滤 --------------------
def smart_filter(nums: List[int]) -> bool:
    if len(nums) != 6:
        return False
    s = sorted(nums)
    total = sum(s)
    odd = sum(1 for n in s if n % 2 == 1)
    big = sum(1 for n in s if n >= 25)
    if total < SUM_TARGET[0] or total > SUM_TARGET[1]:
        return False
    if odd == 0 or odd == 6:
        return False
    if big == 0 or big == 6:
        return False
    zones = [(n - 1) // 10 for n in s]
    if max(Counter(zones).values()) > 3:
        return False
    tails = [n % 10 for n in s]
    if max(Counter(tails).values()) > 2:
        return False
    consec = max_consec = 1
    for i in range(1, 6):
        if s[i] - s[i-1] == 1:
            consec += 1
            max_consec = max(max_consec, consec)
        else:
            consec = 1
    if max_consec > 3:
        return False
    primes = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47}
    prime_count = sum(1 for n in s if n in primes)
    if prime_count == 0 or prime_count == 6:
        return False
    color_counts = {"红": 0, "蓝": 0, "绿": 0}
    for n in s:
        if n in [1,2,7,8,12,13,18,19,23,24,29,30,34,35,40,45,46]:
            color_counts["红"] += 1
        elif n in [3,4,9,10,14,15,20,21,25,26,31,32,36,37,41,42,47,48]:
            color_counts["蓝"] += 1
        else:
            color_counts["绿"] += 1
    if any(v == 6 for v in color_counts.values()):
        return False
    return True


# -------------------- 蒙特卡洛选号 --------------------
def monte_carlo_pick(
    scores: Dict[int, float],
    pair_lift: Dict[Tuple[int, int], float],
    trials: int = MONTE_CARLO_TRIALS,
    fixed_seed: bool = True,
) -> List[int]:
    if fixed_seed:
        random.seed(42)

    candidates = [n for n, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:35]]
    combo_counter = Counter()
    best_combo = []
    best_score = -1e9

    for _ in range(trials):
        combo = sorted(random.sample(candidates, 6))
        if not smart_filter(combo):
            continue
        score = sum(scores[n] for n in combo)
        for a, b in combinations(combo, 2):
            score += pair_lift.get((a, b), 0) * 0.2
        combo_counter[tuple(combo)] += 1
        if score > best_score:
            best_score = score
            best_combo = combo

    if combo_counter:
        return list(combo_counter.most_common(1)[0][0])
    elif best_combo:
        return best_combo
    else:
        return [n for n, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:6]]


# -------------------- 特别号马尔可夫模型 --------------------
class SpecialMarkovModel:
    def __init__(self, order: int = 2):
        self.order = order
        self.transitions = defaultdict(Counter)

    def train(self, specials: List[int]):
        for i in range(len(specials) - self.order):
            state = tuple(specials[i:i+self.order])
            self.transitions[state][specials[i+self.order]] += 1

    def predict(self, recent: List[int]) -> int:
        if len(recent) < self.order:
            return max(set(recent), key=recent.count) if recent else random.randint(1, 49)
        state = tuple(recent[-self.order:])
        if state in self.transitions and self.transitions[state]:
            return self.transitions[state].most_common(1)[0][0]
        return max(set(recent), key=recent.count)


# -------------------- 策略核心（温和玄学融合）--------------------
def generate_strategy_score(
    draws: List[List[int]],
    specials: List[int],
    strategy: str,
    pair_lift: Dict[Tuple[int, int], float],
    use_dynamic_weights: bool = True,
    day_wuxing: str = "",
    day_zhi: str = ""
) -> StrategyScore:
    base_cfg = STRATEGY_CONFIGS.get(strategy, STRATEGY_CONFIGS["balanced"])
    weights = {"w_freq": base_cfg["w_freq"], "w_omit": base_cfg["w_omit"], "w_mom": base_cfg["w_mom"]}

    if use_dynamic_weights and strategy != "ensemble" and len(draws) >= 4:
        weights = find_optimal_weights(draws, specials, weights)

    freq = {n: 0.0 for n in ALL_NUMBERS}
    for d in draws:
        for n in d:
            freq[n] += 1.0

    omit = {}
    for n in ALL_NUMBERS:
        for i, d in enumerate(draws):
            if n in d:
                omit[n] = i
                break
        else:
            omit[n] = len(draws)

    mom = calculate_exp_momentum(draws, half_life=2)

    def norm(d):
        vals = list(d.values())
        mn, mx = min(vals), max(vals)
        if mx == mn:
            return {k: 0.0 for k in d}
        return {k: (v - mn) / (mx - mn) for k, v in d.items()}

    freq_n = norm(freq)
    omit_n = norm({n: 1.0 / (omit[n] + 1) for n in ALL_NUMBERS})
    mom_n = norm(mom)

    # 统计得分（归一化到 0~1）
    stat_scores = {}
    for n in ALL_NUMBERS:
        stat_scores[n] = (
            freq_n[n] * weights["w_freq"]
            + omit_n[n] * weights["w_omit"]
            + mom_n[n] * weights["w_mom"]
        )
    stat_scores_norm = norm(stat_scores)

    # 玄学得分（映射到 0~1）
    fengshui_scores = {}
    if day_wuxing and day_zhi:
        for n in ALL_NUMBERS:
            raw_fs = get_number_fengshui_score(n, day_wuxing, day_zhi)
            fengshui_scores[n] = (raw_fs + 1) / 2  # -1~1 -> 0~1
    else:
        fengshui_scores = {n: 0.5 for n in ALL_NUMBERS}

    # 融合得分：统计主导，玄学微调（玄学权重 3%）
    final_scores = {}
    for n in ALL_NUMBERS:
        final_scores[n] = stat_scores_norm[n] * STAT_POWER + fengshui_scores[n] * FENGSHUI_POWER

    if strategy == "ensemble":
        return ensemble_vote(draws, specials, pair_lift, use_dynamic_weights, day_wuxing, day_zhi)

    main_picks = monte_carlo_pick(final_scores, pair_lift)

    markov = SpecialMarkovModel(2)
    markov.train(specials)
    special_pick = markov.predict(specials[-5:] if len(specials) >= 5 else specials)

    # 多样化特别号
    seed = int(hashlib.md5(strategy.encode()).hexdigest()[:8], 16) % 10000
    random.seed(seed)
    sorted_scores = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    candidates = [n for n, _ in sorted_scores if n not in main_picks][:10]
    if candidates:
        if random.random() < 0.8 or len(candidates) == 1:
            special_pick = candidates[0]
        else:
            special_pick = random.choice(candidates[1:min(4, len(candidates))])
    else:
        special_pick = max(final_scores, key=lambda n: final_scores[n] if n not in main_picks else -1)
    while special_pick in main_picks:
        special_pick = (special_pick % 49) + 1
    random.seed(42)

    confidence = sum(final_scores[n] for n in main_picks) / 6 if main_picks else 0
    return StrategyScore(main_picks, special_pick, confidence, final_scores)


def ensemble_vote(
    draws: List[List[int]], specials: List[int], pair_lift: Dict, use_dynamic_weights: bool = True,
    day_wuxing: str = "", day_zhi: str = ""
) -> StrategyScore:
    scores_list = []
    for s in ["hot", "cold", "momentum", "balanced", "pattern"]:
        score_obj = generate_strategy_score(draws, specials, s, pair_lift, use_dynamic_weights, day_wuxing, day_zhi)
        scores_list.append(score_obj.raw_scores)
    votes = {n: 0.0 for n in ALL_NUMBERS}
    for sc in scores_list:
        ranked = sorted(sc.items(), key=lambda x: x[1], reverse=True)
        for rank, (n, _) in enumerate(ranked):
            votes[n] += 49 - rank
    max_vote = max(votes.values()) if votes else 1
    norm_votes = {n: v / max_vote for n, v in votes.items()}
    main_picks = monte_carlo_pick(norm_votes, pair_lift)
    special = SpecialMarkovModel(2)
    special.train(specials)
    sp = special.predict(specials[-5:] if len(specials) >= 5 else specials)
    while sp in main_picks:
        sp = (sp % 49) + 1
    confidence = sum(norm_votes[n] for n in main_picks) / 6 if main_picks else 0
    return StrategyScore(main_picks, sp, confidence, norm_votes)


# -------------------- 概率计算工具 --------------------
def wilson_interval(hits: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    p = hits / total
    n = total
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    adjustment = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    low = max(0.0, centre - adjustment) * 100
    high = min(1.0, centre + adjustment) * 100
    return (low, high)


def bayesian_posterior(hits: int, total: int) -> float:
    return (hits + 1) / (total + 49) * 100


# -------------------- 微信推送 --------------------
def send_pushplus_notification(title: str, content: str) -> bool:
    token = os.environ.get("PUSHPLUS_TOKEN", "")
    if not token:
        print("ℹ️ 未配置 PUSHPLUS_TOKEN，跳过微信推送。")
        return False

    url = "http://www.pushplus.plus/send"
    data = {"token": token, "title": title, "content": content}
    try:
        resp = requests.post(url, json=data, timeout=10)
        if resp.status_code == 200:
            result = resp.json()
            if result.get("code") == 200:
                print("✅ 已通过 PushPlus 推送到微信")
                return True
            else:
                print(f"❌ PushPlus 推送失败: {result.get('msg')}")
                return False
        else:
            print(f"❌ PushPlus 请求失败: HTTP {resp.status_code}")
            return False
    except Exception as e:
        print(f"❌ PushPlus 推送异常: {e}")
        return False


# -------------------- 智能投注方案 --------------------
def print_betting_plan(hot5, top1_zod, special_first, top_specials, best_combo, budget=200):
    print("\n" + "=" * 60)
    print("💰 智能投注方案 (特码/正码/一肖/三中三)")
    print("=" * 60)
    print(f"📊 推荐预算: {budget}元\n")
    special_high = top_specials[0][0] if top_specials else special_first
    main_focus = best_combo if best_combo and len(best_combo) == 3 else hot5[:3]

    print("【稳健型方案】")
    print(f"  一肖 {top1_zod}: {int(budget * 0.3)}元")
    print(f"  特码 {special_first:02d}({int(budget * 0.15)}元) + {special_high:02d}({int(budget * 0.1)}元)")
    print(f"  正码 {' '.join(f'{n:02d}' for n in main_focus)} 各{int(budget * 0.1)}元")
    if best_combo:
        print(f"  三全中 {' '.join(f'{n:02d}' for n in best_combo)}: {int(budget * 0.15)}元")

    print("\n【进取型方案】")
    print(f"  一肖 {top1_zod}: {int(budget * 0.3)}元")
    print(f"  特码 {special_high:02d}({int(budget * 0.2)}元) + {special_first:02d}({int(budget * 0.1)}元)")
    print(f"  正码5个均注: {int(budget * 0.25)}元")
    if best_combo:
        print(f"  三全中 {' '.join(f'{n:02d}' for n in best_combo)}: {int(budget * 0.15)}元")

    print("\n【极限精简型】")
    print(f"  一肖 {top1_zod}: {int(budget * 0.4)}元")
    print(f"  特码 {special_high:02d}: {int(budget * 0.4)}元")
    print(f"  正码 {main_focus[0]:02d},{main_focus[1]:02d}: 各{int(budget * 0.1)}元")
    print("=" * 60)


# -------------------- 命令行接口 --------------------
def cmd_sync(args: argparse.Namespace) -> None:
    conn = connect_db(args.db)
    init_db(conn)
    print(f"正在从新澳门数据源同步 {args.year} 年历史开奖数据...")
    records = fetch_macau_history(args.year)
    if not records:
        print("错误：未获取到有效记录。")
        return
    ins, upd = sync_draws(conn, records, "macau_online")
    print(f"同步完成：新增 {ins} 期，更新 {upd} 期。")
    conn.close()


def cmd_predict(args: argparse.Namespace) -> None:
    conn = connect_db(args.db)
    init_db(conn)
    draws = get_recent_draws(conn, PREDICT_WINDOW)
    specials = get_recent_specials(conn, PREDICT_WINDOW)
    if len(draws) < 4:
        print("错误：历史数据不足（至少需要4期），请先运行 sync。")
        return
    pair_lift = calculate_pair_lift(draws)
    latest = conn.execute("SELECT issue_no FROM draws ORDER BY draw_date DESC LIMIT 1").fetchone()
    next_issue = next_issue_number(latest[0]) if latest else f"{datetime.now().year}001"

    today = date.today()
    day_gan, day_zhi, day_wuxing = get_day_ganzhi(today)
    print(f"今日玄学: {today} {day_gan}{day_zhi}日 五行{day_wuxing} (玄学权重 {FENGSHUI_POWER*100:.0f}%)")

    for strat in STRATEGY_IDS:
        score = generate_strategy_score(draws, specials, strat, pair_lift, use_dynamic_weights=True,
                                        day_wuxing=day_wuxing, day_zhi=day_zhi)
        conn.execute("""
            INSERT OR REPLACE INTO predictions (issue_no, strategy, numbers_json, special_number, confidence, status, created_at)
            VALUES (?, ?, ?, ?, ?, 'PENDING', ?)
        """, (next_issue, strat, json.dumps(score.main_picks), score.special_pick, score.confidence, utc_now()))
    conn.commit()
    print(f"已生成 {next_issue} 期的预测推荐。")
    conn.close()


def cmd_show(args: argparse.Namespace) -> None:
    conn = connect_db(args.db)
    init_db(conn)

    today = date.today()
    day_gan, day_zhi, day_wuxing = get_day_ganzhi(today)

    # ---------- 最新开奖 ----------
    latest = conn.execute(
        "SELECT issue_no, draw_date, numbers_json, special_number FROM draws ORDER BY draw_date DESC LIMIT 1"
    ).fetchone()
    if latest:
        nums = json.loads(latest["numbers_json"])
        print(f"最新开奖: {latest['issue_no']} {latest['draw_date']} | 主号: {' '.join(f'{n:02d}' for n in nums)} | 特别号: {latest['special_number']:02d}")
    else:
        print("暂无开奖数据。")

    # ---------- 上期预测复盘 ----------
    prev_draw = conn.execute("""
        SELECT issue_no, numbers_json, special_number 
        FROM draws 
        ORDER BY draw_date DESC, issue_no DESC 
        LIMIT 1 OFFSET 1
    """).fetchone()
    prev_issue = prev_top1 = prev_top2 = prev_picked_special = None
    zodiac_hit1 = zodiac_hit2 = special_hit = main_hits = "—"

    if prev_draw:
        prev_issue = prev_draw["issue_no"]
        prev_nums = json.loads(prev_draw["numbers_json"])
        prev_special = prev_draw["special_number"]
        prev_pred = conn.execute("""
            SELECT numbers_json, special_number 
            FROM predictions 
            WHERE issue_no = ? AND strategy = 'ensemble'
        """, (prev_issue,)).fetchone()

        if prev_pred:
            prev_picked = json.loads(prev_pred["numbers_json"])
            prev_picked_special = prev_pred["special_number"]
            prev_zodiac_score = Counter()
            for n in prev_picked:
                for z, nums in ZODIAC_MAP.items():
                    if n in nums:
                        prev_zodiac_score[z] += 1
                        break
            top_prev_zod = prev_zodiac_score.most_common(2)
            prev_top1 = top_prev_zod[0][0] if top_prev_zod else "—"
            prev_top2 = top_prev_zod[1][0] if len(top_prev_zod) > 1 else "—"
            zodiac_hit1 = "✅" if any(n in ZODIAC_MAP[prev_top1] for n in prev_nums) else "❌"
            zodiac_hit2 = "✅" if prev_top2 != "—" and any(n in ZODIAC_MAP[prev_top2] for n in prev_nums) else "❌"
            special_hit = "✅" if prev_picked_special == prev_special else "❌"
            main_hits = len(set(prev_picked) & set(prev_nums))
            print(f"📋 上期预测复盘 ({prev_issue})")
            print(f"   最强生肖: {prev_top1} {zodiac_hit1}  次强: {prev_top2} {zodiac_hit2} | 特别号: {prev_picked_special:02d} {special_hit} | 正码中{main_hits}")
        else:
            print(f"📋 上期预测复盘 ({prev_issue}): 无预测记录")
    else:
        print("📋 上期预测复盘: 历史数据不足")

    # ---------- 多策略推荐 ----------
    pending = conn.execute(
        "SELECT issue_no, strategy, numbers_json, special_number, confidence FROM predictions WHERE status='PENDING' ORDER BY strategy"
    ).fetchall()
    if pending:
        print("\n本期多策略推荐 (6码池，统计+玄学融合):")
        for p in pending:
            nums = json.loads(p["numbers_json"])
            conf_str = f" (置信度: {p['confidence']*100:.1f}%)" if p["confidence"] else ""
            strategy_name = STRATEGY_CONFIGS.get(p['strategy'], {}).get('name', p['strategy'])

            strat_zodiac = Counter()
            for n in nums:
                for z, z_nums in ZODIAC_MAP.items():
                    if n in z_nums:
                        strat_zodiac[z] += 1
                        break
            top_zod = strat_zodiac.most_common(2)
            z1 = top_zod[0][0] if top_zod else "—"
            z2 = top_zod[1][0] if len(top_zod) > 1 else "—"

            print(f"  [{p['issue_no']}] {strategy_name}{conf_str}: {' '.join(f'{n:02d}' for n in nums)} | 特别号: {p['special_number']:02d} | 极强:{z1} 次强:{z2}")
    else:
        print("\n暂无待开奖预测，请先运行 predict")

    # ---------- 简洁投注推荐 ----------
    print("\n" + "=" * 60)
    print(f"🎯 本期投注推荐单 (统计 {STAT_POWER*100:.0f}% + 玄学 {FENGSHUI_POWER*100:.0f}% · {day_gan}{day_zhi}日 五行{day_wuxing})")
    print("=" * 60)

    draws = get_recent_draws(conn, PREDICT_WINDOW)
    specials = get_recent_specials(conn, PREDICT_WINDOW)
    if len(draws) < 4:
        print("历史数据不足，无法生成投注推荐。")
        conn.close()
        return

    ensemble_pred = conn.execute(
        "SELECT numbers_json, special_number FROM predictions WHERE status='PENDING' AND strategy='ensemble'"
    ).fetchone()
    if ensemble_pred:
        picked_6 = json.loads(ensemble_pred["numbers_json"])
        picked_special = ensemble_pred["special_number"]
    else:
        pair_lift = calculate_pair_lift(draws)
        score = ensemble_vote(draws, specials, pair_lift, use_dynamic_weights=True, day_wuxing=day_wuxing, day_zhi=day_zhi)
        picked_6 = score.main_picks
        picked_special = score.special_pick

    hot5 = picked_6[:5] if len(picked_6) >= 5 else picked_6

    # 综合投票选出最终生肖
    vote_zodiac = Counter()
    for p in pending:
        nums = json.loads(p["numbers_json"])
        strat_zodiac = Counter()
        for n in nums:
            for z, z_nums in ZODIAC_MAP.items():
                if n in z_nums:
                    strat_zodiac[z] += 1
                    break
        top = strat_zodiac.most_common(2)
        if top:
            vote_zodiac[top[0][0]] += 3
        if len(top) > 1:
            vote_zodiac[top[1][0]] += 1

    recent_zodiac = Counter()
    for draw in draws[-5:]:
        for n in draw:
            for z, nums in ZODIAC_MAP.items():
                if n in nums:
                    recent_zodiac[z] += 1
    for z, cnt in recent_zodiac.items():
        vote_zodiac[z] += cnt * 2

    top_zod = vote_zodiac.most_common(2)
    top1 = top_zod[0][0] if top_zod else "龙"
    top2 = top_zod[1][0] if len(top_zod) > 1 else "马"

    special_zod = get_zodiac(picked_special)

    def zodiac_hit_rate(zod, limit=5):
        if len(draws) < limit:
            limit = len(draws)
        if limit == 0:
            return 0
        hits = 0
        for draw in draws[-limit:]:
            if any(n in ZODIAC_MAP[zod] for n in draw):
                hits += 1
        return hits / limit * 100

    rate1 = zodiac_hit_rate(top1)
    rate2 = zodiac_hit_rate(top2)

    next_issue_str = pending[0]['issue_no'] if pending else (next_issue_number(latest['issue_no']) if latest else "未知")
    print(f"📅 参考期号: {next_issue_str}")
    print("-" * 60)
    print(f"🐉 最强生肖: {top1}  (近{min(5, len(draws))}期命中率 {rate1:.0f}%)")
    print(f"🐉 次强生肖: {top2}  (近{min(5, len(draws))}期命中率 {rate2:.0f}%)")
    print("🎲 正码5个 (科学概率评估，基于最近7期):")

    for n in hot5:
        hits_7 = sum(1 for draw in draws if n in draw)
        low, high = wilson_interval(hits_7, len(draws))
        posterior = bayesian_posterior(hits_7, len(draws))
        print(f"      {n:02d} ({get_zodiac(n)})  ─ 威尔逊区间 [{low:.0f}%-{high:.0f}%]  后验概率 {posterior:.1f}%")

    print(f"🔮 特别号 (首选): {picked_special:02d} ({special_zod})")

    # ---------- 特别号6码推荐 ----------
    pair_lift = calculate_pair_lift(draws)
    scores_list = []
    for s in ["hot", "cold", "momentum", "balanced", "pattern"]:
        score_obj = generate_strategy_score(draws, specials, s, pair_lift, use_dynamic_weights=True,
                                            day_wuxing=day_wuxing, day_zhi=day_zhi)
        scores_list.append(score_obj.raw_scores)

    special_scores = {n: 0.0 for n in ALL_NUMBERS}
    votes = {n: 0.0 for n in ALL_NUMBERS}
    for sc in scores_list:
        ranked = sorted(sc.items(), key=lambda x: x[1], reverse=True)
        for rank, (n, _) in enumerate(ranked):
            votes[n] += 49 - rank
    max_vote = max(votes.values()) if votes else 1
    norm_votes = {n: v / max_vote for n, v in votes.items()}

    markov = SpecialMarkovModel(2)
    markov.train(specials)
    special_exp = {n: 0.0 for n in ALL_NUMBERS}
    for i, sp in enumerate(specials):
        weight = math.exp(-i / 2)
        special_exp[sp] += weight * 2.0

    recent_specials = specials
    state = tuple(recent_specials[-2:]) if len(recent_specials) >= 2 else None
    markov_probs = {n: 0.0 for n in ALL_NUMBERS}
    if state and state in markov.transitions:
        total = sum(markov.transitions[state].values())
        for n, cnt in markov.transitions[state].items():
            markov_probs[n] = cnt / total if total > 0 else 0

    max_exp = max(special_exp.values()) if special_exp.values() else 1
    for n in ALL_NUMBERS:
        special_scores[n] = (norm_votes[n] * 0.3 +
                             (special_exp[n] / max_exp) * 0.4 +
                             markov_probs[n] * 0.3)

    for n in hot5:
        special_scores[n] = -1.0

    top6_specials = sorted(special_scores.items(), key=lambda x: x[1], reverse=True)[:6]

    print("\n🔯 特别号6码推荐 (按综合评分排序):")
    special_lines = []
    for i, (num, score) in enumerate(top6_specials, 1):
        line = f"   {i}. {num:02d} ({get_zodiac(num)})  ─ 综合评分: {score*100:.1f}%"
        print(line)
        special_lines.append(line.strip())

    # ---------- 三中三推荐 ----------
    best_combo = None
    combo_hits_dict = {}
    if len(hot5) >= 3:
        print("\n🎰 三中三推荐 (从正码5个组合生成，共10组):")
        three_combos = list(combinations(sorted(hot5), 3))
        for combo in three_combos:
            hits = 0
            for draw in draws:
                if all(n in draw for n in combo):
                    hits += 1
            combo_hits_dict[combo] = hits
        for i, combo in enumerate(three_combos, 1):
            hits = combo_hits_dict[combo]
            prob = hits / len(draws) * 100 if draws else 0
            combo_str = " ".join(f"{n:02d}({get_zodiac(n)})" for n in combo)
            print(f"   {i:2d}. {combo_str}  ─ 近{len(draws)}期同时出现 {hits} 次 ({prob:.1f}%)")

        if combo_hits_dict:
            best_combo = max(three_combos, key=lambda c: (combo_hits_dict[c], -three_combos.index(c)))
            best_hits = combo_hits_dict[best_combo]
            best_prob = best_hits / len(draws) * 100 if draws else 0
            best_combo_str = " ".join(f"{n:02d}({get_zodiac(n)})" for n in best_combo)
            print(f"\n🏆 极强推荐组合: {best_combo_str}")
            print(f"   近{len(draws)}期同时出现 {best_hits} 次 ({best_prob:.1f}%)，是近期最稳定的三码组合。")

    # ---------- 最近7期特别号策略命中统计 ----------
    strat_stats_summary = []
    recent_7_rows = conn.execute("""
        SELECT issue_no, special_number FROM draws 
        ORDER BY draw_date DESC, issue_no DESC LIMIT 7
    """).fetchall()
    if len(recent_7_rows) >= 3:
        print("\n📊 最近7期特别号策略命中统计:")
        recent_7 = list(reversed(recent_7_rows))
        strat_special_stats = {s: {"hits": 0, "total": 0} for s in STRATEGY_IDS}
        for row in recent_7:
            issue = row["issue_no"]
            winning_special = row["special_number"]
            for s in STRATEGY_IDS:
                pred = conn.execute("""
                    SELECT special_number FROM predictions 
                    WHERE issue_no = ? AND strategy = ? AND status = 'REVIEWED'
                """, (issue, s)).fetchone()
                if pred:
                    strat_special_stats[s]["total"] += 1
                    if pred["special_number"] == winning_special:
                        strat_special_stats[s]["hits"] += 1
        print(f"  {'策略':<12} {'命中/总数':<8} {'命中率':<8}")
        print("  " + "-" * 30)
        for s in STRATEGY_IDS:
            stats = strat_special_stats[s]
            total = stats["total"]
            if total == 0:
                continue
            hits = stats["hits"]
            rate = hits / total * 100
            strategy_name = STRATEGY_CONFIGS.get(s, {}).get('name', s)
            line = f"  {strategy_name:<12} {hits}/{total:<6} {rate:.0f}%"
            print(line)
            strat_stats_summary.append(f"{strategy_name}:{hits}/{total}({rate:.0f}%)")
    else:
        print("\n最近7期数据不足，无法显示特别号命中统计。")

    print("=" * 60)
    print("⚠️ 数据仅供参考，理性投注。")

    # ---------- 智能投注方案 ----------
    print_betting_plan(hot5, top1, picked_special, top6_specials, best_combo, budget=200)

    # ---------- 微信推送（温和玄学版）----------
    push_lines = []
    push_lines.append(f"【新澳门·{next_issue_str}期推荐】")
    push_lines.append(f"今日{day_gan}{day_zhi}日 五行{day_wuxing} · 玄学权重{FENGSHUI_POWER*100:.0f}%")
    push_lines.append("💰 喜忌微调：金水略优，火木稍避")
    push_lines.append("⏰ 最佳投注：申时(15-17点) 酉时(17-19点)")
    push_lines.append("")

    if prev_draw and prev_pred:
        push_lines.append(f"📋 上期{prev_issue}复盘：")
        push_lines.append(f"   生肖 {prev_top1}{zodiac_hit1} {prev_top2}{zodiac_hit2} ｜ 特号 {prev_picked_special:02d}{special_hit} ｜ 正码中{main_hits}")
        push_lines.append("")

    push_lines.append(f"🐉 本期主攻生肖：{top1}")
    push_lines.append(f"🎲 正码5个：{' '.join(f'{n:02d}' for n in hot5)}")
    push_lines.append(f"🔮 特别号首选：{picked_special:02d}")
    push_lines.append("")

    if top6_specials:
        push_lines.append("💡 特号备选：")
        for i, (num, _) in enumerate(top6_specials[:3], 1):
            push_lines.append(f"   {i}. {num:02d}")
    push_lines.append("")

    if best_combo:
        push_lines.append(f"🏆 三中三极强组合：{' '.join(f'{n:02d}' for n in best_combo)}")
        push_lines.append("")

    if strat_stats_summary:
        push_lines.append("📊 近期特号命中率：")
        for summary in strat_stats_summary[:2]:
            push_lines.append(f"   {summary}")

    push_content = "\n".join(push_lines)
    send_pushplus_notification("新澳门预测", push_content)

    conn.close()


def cmd_backtest(args: argparse.Namespace) -> None:
    print("回测功能需较长历史数据，当前7期模式不建议运行完整回测。")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="新澳门六合彩 - 温和玄学版")
    parser.add_argument("--db", default=DB_PATH_DEFAULT, help="数据库路径")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sync = sub.add_parser("sync", help="同步历史数据")
    p_sync.add_argument("--year", type=int, default=datetime.now().year, help="指定年份")
    p_sync.set_defaults(func=cmd_sync)

    p_predict = sub.add_parser("predict", help="生成下期预测")
    p_predict.set_defaults(func=cmd_predict)

    p_show = sub.add_parser("show", help="显示推荐和统计")
    p_show.set_defaults(func=cmd_show)

    p_backtest = sub.add_parser("backtest", help="回测（不推荐）")
    p_backtest.set_defaults(func=cmd_backtest)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
