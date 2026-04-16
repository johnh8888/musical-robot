#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新澳门六合彩 - 统计主导版（确定性选号 + 轻玄学微调 + 最近6期预测）
玄学影响力固定 3%，组合枚举候选数优化为 16，基于最近6期数据预测
根据生肖赔率自动计算投注分配（马1:0.7，其他1:1；特码46倍；三中三1000倍）
用法:
    python macau_predict.py sync          # 同步历史数据
    python macau_predict.py predict       # 生成下期预测
    python macau_predict.py show          # 显示推荐和回测统计，并输出智能投注方案
"""

import argparse
import json
import logging
import math
import os
import sqlite3
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
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
logger = logging.getLogger("macau_predict")

# -------------------- 常量与配置 --------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH_DEFAULT = str(SCRIPT_DIR / "macau_gentle.db")

# 可靠数据源（使用 marksix6.net 中的“新澳门彩”）
MACAU_API_URL = "https://marksix6.net/index.php?api=1"

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

# 号码五行映射（仅用于玄学微调，权重3%）
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
FAVOR_BONUS = 0.15
AVOID_PENALTY = 0.1

ALL_NUMBERS = list(range(1, 50))
SUM_TARGET = (105, 195)          # 放宽和值范围
PREDICT_WINDOW = 6               # 预测使用最近6期（原7期改为6期）
BACKTEST_WINDOW = 8              # 回测最近8期

# 玄学影响力固定 3%
FENGSHUI_POWER = 0.03
STAT_POWER = 0.97

# 优化点：组合枚举候选数从 30 降至 16（原15改为16）
TOP_CANDIDATES = 16

# 赔率配置
ZODIAC_ODDS = {
    "马": 0.7,      # 马赔率 1:0.7
    # 其他生肖默认为 1.0
}
SPECIAL_ODDS = 46
TRIO_ODDS = 1000   # 三中三赔率（假设）


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


# -------------------- 日干支与玄学推算（保留，仅3%影响）--------------------
def get_day_ganzhi(dt: date) -> Tuple[str, str, str]:
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
    score = 0.0
    zodiac = get_zodiac(num)
    num_wuxing = get_number_wuxing(num)
    if num_wuxing and day_wuxing:
        relation = WUXING_RELATION.get(day_wuxing, {})
        if num_wuxing == relation.get("生"):
            score += 0.4
        elif num_wuxing == relation.get("克"):
            score -= 0.3
        elif day_wuxing == WUXING_RELATION.get(num_wuxing, {}).get("生"):
            score += 0.2
    score += get_zodiac_clash_score(zodiac, day_zhi)
    zod_wuxing = ZODIAC_WUXING.get(zodiac, "")
    if zod_wuxing and day_wuxing:
        relation = WUXING_RELATION.get(day_wuxing, {})
        if zod_wuxing == relation.get("生"):
            score += 0.15
        elif zod_wuxing == relation.get("克"):
            score -= 0.1
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


# -------------------- 数据获取（可靠源）--------------------
def fetch_macau_history_from_api() -> List[DrawRecord]:
    """从 marksix6.net API 获取新澳门彩历史数据"""
    logger.info("正在从 marksix6.net 获取新澳门彩历史数据...")
    try:
        resp = requests.get(MACAU_API_URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            logger.error(f"请求失败: HTTP {resp.status_code}")
            return []
        data = resp.json()
        lottery_list = data.get("lottery_data", [])
        macau_data = None
        for item in lottery_list:
            if item.get("name") == "新澳门彩":
                macau_data = item
                break
        if not macau_data:
            logger.error("未找到新澳门彩数据")
            return []
        records = []
        history_list = macau_data.get("history", [])
        if history_list:
            for line in history_list:
                match = re.match(r"(\d{7})\s*期[：:]\s*([\d,]+)", line)
                if not match:
                    continue
                expect_raw = match.group(1)
                numbers_str = match.group(2)
                nums = [int(x) for x in numbers_str.split(",") if x.strip().isdigit()]
                if len(nums) >= 7:
                    issue_no = f"{expect_raw[:4]}/{expect_raw[4:]}"
                    draw_date = datetime.now().strftime("%Y-%m-%d")  # 无历史日期，暂用当日
                    records.append(DrawRecord(
                        issue_no=issue_no,
                        draw_date=draw_date,
                        numbers=nums[:6],
                        special_number=nums[6]
                    ))
        if not records:
            expect_raw = str(macau_data.get("expect", ""))
            numbers_raw = macau_data.get("openCode") or macau_data.get("numbers")
            if numbers_raw and isinstance(numbers_raw, str):
                nums = [int(x) for x in numbers_raw.split(",") if x.strip().isdigit()]
                if len(nums) >= 7:
                    issue_no = f"{expect_raw[:4]}/{expect_raw[4:]}" if len(expect_raw) >= 7 else expect_raw
                    draw_date = macau_data.get("openTime", "").split()[0] if macau_data.get("openTime") else datetime.now().strftime("%Y-%m-%d")
                    records.append(DrawRecord(
                        issue_no=issue_no,
                        draw_date=draw_date,
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


# -------------------- 动态权重优化（带阈值）--------------------
def find_optimal_weights(
    draws: List[List[int]],
    specials: List[int],
    base_weights: Dict[str, float],
    improvement_threshold: float = 0.05,
) -> Dict[str, float]:
    if len(draws) < 4:
        return base_weights

    test_window = max(2, len(draws) // 3)
    best_weights = base_weights.copy()
    best_avg_hits = 0.0

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
                if len(past_draws) < 3:
                    continue
                score_obj = generate_strategy_score_with_weights(
                    past_draws, {"w_freq": w_freq, "w_omit": w_omit, "w_mom": w_mom}
                )
                actual = set(draws[i])
                total_hits += len(set(score_obj.main_picks) & actual)
                count += 1

            if count > 0:
                avg_hits = total_hits / count
                if avg_hits > best_avg_hits + improvement_threshold:
                    best_avg_hits = avg_hits
                    best_weights = {"w_freq": w_freq, "w_omit": w_omit, "w_mom": w_mom}

    return best_weights


def generate_strategy_score_with_weights(
    draws: List[List[int]],
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

    main_picks = deterministic_pick(scores, {})  # 确定性选号
    special = max(scores, key=lambda n: scores[n] if n not in main_picks else -1)
    return StrategyScore(main_picks, special, 0.0, scores)


# -------------------- 确定性选号（候选数优化为 TOP_CANDIDATES=16）--------------------
def deterministic_pick(
    scores: Dict[int, float],
    pair_lift: Dict[Tuple[int, int], float],
    top_candidates: int = TOP_CANDIDATES,
) -> List[int]:
    """
    确定性选择：从得分最高的 top_candidates 个号码中，枚举所有6码组合，
    通过 smart_filter 过滤后，返回总得分（加上pair_lift）最高的组合。
    若没有组合通过过滤，则返回得分最高的6个号码（不进行过滤）。
    """
    sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    candidates = [n for n, _ in sorted_nums[:top_candidates]]
    best_combo = None
    best_score = -1e9

    for combo in combinations(candidates, 6):
        combo = sorted(combo)
        if not smart_filter(combo):
            continue
        combo_score = sum(scores[n] for n in combo)
        for a, b in combinations(combo, 2):
            combo_score += pair_lift.get((a, b), 0) * 0.2
        if combo_score > best_score:
            best_score = combo_score
            best_combo = combo

    if best_combo:
        return list(best_combo)
    else:
        return [n for n, _ in sorted_nums[:6]]


# -------------------- 智能过滤（放宽版）--------------------
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
    max_consec = 1
    consec = 1
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


# -------------------- 策略核心（融合玄学3%）--------------------
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

    stat_scores = {}
    for n in ALL_NUMBERS:
        stat_scores[n] = (
            freq_n[n] * weights["w_freq"]
            + omit_n[n] * weights["w_omit"]
            + mom_n[n] * weights["w_mom"]
        )
    stat_scores_norm = norm(stat_scores)

    # 玄学得分（仅3%权重）
    fengshui_scores = {}
    if day_wuxing and day_zhi:
        for n in ALL_NUMBERS:
            raw_fs = get_number_fengshui_score(n, day_wuxing, day_zhi)
            fengshui_scores[n] = (raw_fs + 1) / 2
    else:
        fengshui_scores = {n: 0.5 for n in ALL_NUMBERS}

    final_scores = {}
    for n in ALL_NUMBERS:
        final_scores[n] = stat_scores_norm[n] * STAT_POWER + fengshui_scores[n] * FENGSHUI_POWER

    if strategy == "ensemble":
        return ensemble_vote(draws, specials, pair_lift, use_dynamic_weights, day_wuxing, day_zhi)

    main_picks = deterministic_pick(final_scores, pair_lift)
    special_candidates = [(n, final_scores[n]) for n in ALL_NUMBERS if n not in main_picks]
    special_pick = max(special_candidates, key=lambda x: x[1])[0] if special_candidates else 1

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
    main_picks = deterministic_pick(norm_votes, pair_lift)
    special_candidates = [(n, norm_votes[n]) for n in ALL_NUMBERS if n not in main_picks]
    special_pick = max(special_candidates, key=lambda x: x[1])[0] if special_candidates else 1
    confidence = sum(norm_votes[n] for n in main_picks) / 6 if main_picks else 0
    return StrategyScore(main_picks, special_pick, confidence, norm_votes)


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


# -------------------- 智能投注方案（根据赔率优化）--------------------
def print_betting_plan(hot5, top1_zod, top2_zod, special_first, top_specials, best_combo, budget=500):
    """
    根据生肖赔率动态调整投注方案
    - 马赔率 1:0.7（即投1元中得0.7元，净亏0.3元）
    - 其他生肖赔率 1:1（投1元中得1元，保本）
    - 特码赔率 1:46
    - 三中三赔率 1:1000（假设）
    """
    odds_zodiac = ZODIAC_ODDS.get(top1_zod, 1.0)
    odds_special = SPECIAL_ODDS
    odds_trio = TRIO_ODDS

    # 计算实现“保本”所需的生肖投注额（即生肖中奖后至少收回总本金）
    min_S = budget / odds_zodiac if odds_zodiac > 0 else budget
    S = int(min_S) + (1 if min_S > int(min_S) else 0)
    remaining = budget - S
    if remaining < 0:
        S = budget
        T = 0
        P = 0
    else:
        # 剩余资金分配：特码占70%，三中三占30%（可调）
        T = int(remaining * 0.7)
        P = remaining - T

    # 若生肖赔率为1:1且 S == budget，则无法购买其他项，提供备选方案
    if odds_zodiac == 1.0 and S == budget:
        print("\n⚠️ 当前生肖赔率为1:1，实现严格保本需将全部预算投入生肖，无法购买特码和三中三。")
        print("   建议降低总预算或接受生肖中时微亏的方案。")
        # 备选方案：生肖投 budget-20，特码15，三中三5
        S = budget - 20
        T = 15
        P = 5
        print(f"   备选方案：生肖 {S} 元（中奖得 {S}，亏20），特码 {T} 元，三中三 {P} 元。")

    print("\n" + "=" * 60)
    print("💰 智能投注方案 (根据实际赔率优化)")
    print("=" * 60)
    print(f"📊 总预算: {budget}元")
    print(f"🐉 生肖: {top1_zod} (赔率 1:{odds_zodiac})")
    print(f"🎲 特码: {special_first:02d} (赔率 1:{odds_special})")
    if best_combo:
        print(f"🏆 三中三: {' '.join(f'{n:02d}' for n in best_combo)} (赔率 1:{odds_trio})")
    print("-" * 60)
    print(f"【推荐投注】")
    print(f"  一肖 {top1_zod}: {S} 元  (中奖得 {S * odds_zodiac:.2f} 元)")
    if T > 0:
        print(f"  特码 {special_first:02d}: {T} 元  (中奖得 {T * odds_special} 元)")
    if P > 0:
        trio_str = ' '.join(f'{n:02d}' for n in best_combo) if best_combo else "无"
        print(f"  三中三 {trio_str}: {P} 元  (中奖得 {P * odds_trio} 元)")
    print("-" * 60)
    # 计算各种中奖情况下的总回报
    if T > 0 or P > 0:
        print("【预期回报】")
        only_zodiac = S * odds_zodiac
        print(f"  仅生肖中: 总回报 {only_zodiac:.2f} 元, 净收益 {only_zodiac - budget:.2f} 元")
        if T > 0:
            zodiac_special = S * odds_zodiac + T * odds_special
            print(f"  生肖+特码: {zodiac_special:.2f} 元, 净收益 {zodiac_special - budget:.2f} 元")
        if P > 0 and best_combo:
            zodiac_trio = S * odds_zodiac + P * odds_trio
            print(f"  生肖+三中三: {zodiac_trio:.2f} 元, 净收益 {zodiac_trio - budget:.2f} 元")
        if T > 0 and P > 0 and best_combo:
            all_win = S * odds_zodiac + T * odds_special + P * odds_trio
            print(f"  全部中奖: {all_win:.2f} 元, 净收益 {all_win - budget:.2f} 元")
    print("=" * 60)
    print("⚠️ 数据仅供参考，理性投注。")


# -------------------- 命令行接口 --------------------
def cmd_sync(args: argparse.Namespace) -> None:
    conn = connect_db(args.db)
    init_db(conn)
    print("正在从可靠数据源同步新澳门彩历史数据...")
    records = fetch_macau_history_from_api()
    if not records:
        print("错误：未获取到有效记录。")
        return
    ins, upd = sync_draws(conn, records, "macau_api")
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

    # 最新开奖
    latest = conn.execute(
        "SELECT issue_no, draw_date, numbers_json, special_number FROM draws ORDER BY draw_date DESC LIMIT 1"
    ).fetchone()
    if latest:
        nums = json.loads(latest["numbers_json"])
        print(f"最新开奖: {latest['issue_no']} {latest['draw_date']} | 主号: {' '.join(f'{n:02d}' for n in nums)} | 特别号: {latest['special_number']:02d}")
    else:
        print("暂无开奖数据。")

    # 最近8期回测统计
    print(f"\n📊 最近 {BACKTEST_WINDOW} 期策略回测统计:")
    backtest_draws = conn.execute(
        f"SELECT issue_no, numbers_json, special_number FROM draws ORDER BY draw_date ASC, issue_no ASC LIMIT {BACKTEST_WINDOW}"
    ).fetchall()
    if len(backtest_draws) >= 3:
        strat_stats = {s: {"total": 0, "hits": 0, "special_hits": 0} for s in STRATEGY_IDS}
        for draw in backtest_draws:
            issue = draw["issue_no"]
            actual_main = set(json.loads(draw["numbers_json"]))
            actual_special = draw["special_number"]
            for strat in STRATEGY_IDS:
                pred = conn.execute(
                    "SELECT numbers_json, special_number FROM predictions WHERE issue_no = ? AND strategy = ? AND status = 'REVIEWED'",
                    (issue, strat)
                ).fetchone()
                if pred:
                    strat_stats[strat]["total"] += 1
                    pred_main = set(json.loads(pred["numbers_json"]))
                    hit = len(pred_main & actual_main)
                    strat_stats[strat]["hits"] += hit
                    if pred["special_number"] == actual_special:
                        strat_stats[strat]["special_hits"] += 1
        print(f"  {'策略':<12} {'期数':<6} {'平均命中':<10} {'特别号命中率':<12}")
        for strat in STRATEGY_IDS:
            stats = strat_stats[strat]
            if stats["total"] == 0:
                continue
            avg_hit = stats["hits"] / stats["total"]
            special_rate = stats["special_hits"] / stats["total"] * 100
            name = STRATEGY_CONFIGS[strat]["name"]
            print(f"  {name:<12} {stats['total']:<6} {avg_hit:.2f}         {special_rate:.1f}%")
    else:
        print("  历史数据不足，无法回测。")

    # 多策略推荐
    pending = conn.execute(
        "SELECT issue_no, strategy, numbers_json, special_number, confidence FROM predictions WHERE status='PENDING' ORDER BY strategy"
    ).fetchall()
    if pending:
        print("\n本期多策略推荐 (6码池，统计+玄学融合):")
        for p in pending:
            nums = json.loads(p["numbers_json"])
            conf_str = f" (置信度: {p['confidence']*100:.1f}%)" if p["confidence"] else ""
            strategy_name = STRATEGY_CONFIGS.get(p['strategy'], {}).get('name', p['strategy'])
            print(f"  [{p['issue_no']}] {strategy_name}{conf_str}: {' '.join(f'{n:02d}' for n in nums)} | 特别号: {p['special_number']:02d}")
    else:
        print("\n暂无待开奖预测，请先运行 predict")

    # 最终推荐（集成投票）
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

    # ---------- 修正生肖选择：直接基于最近6期实际命中率 ----------
    # 统计最近6期开奖中每个生肖出现的次数（主号）
    zodiac_hit_count = {z: 0 for z in ZODIAC_MAP.keys()}
    recent_draws_for_zodiac = draws[-6:]  # 最近6期（与预测窗口一致）
    for draw in recent_draws_for_zodiac:
        for n in draw:
            z = get_zodiac(n)
            if z:
                zodiac_hit_count[z] += 1
    total_nums = len(recent_draws_for_zodiac) * 6
    zodiac_rate = {z: cnt / total_nums * 100 for z, cnt in zodiac_hit_count.items()}
    sorted_zodiac = sorted(zodiac_rate.items(), key=lambda x: x[1], reverse=True)
    top1 = sorted_zodiac[0][0] if sorted_zodiac else "龙"
    top2 = sorted_zodiac[1][0] if len(sorted_zodiac) > 1 else "马"
    rate1 = zodiac_rate[top1]
    rate2 = zodiac_rate[top2]

    special_zod = get_zodiac(picked_special)

    next_issue_str = pending[0]['issue_no'] if pending else (next_issue_number(latest['issue_no']) if latest else "未知")
    print(f"📅 参考期号: {next_issue_str}")
    print("-" * 60)
    print(f"🐉 最强生肖: {top1}  (近{len(recent_draws_for_zodiac)}期命中率 {rate1:.0f}%)")
    print(f"🐉 次强生肖: {top2}  (近{len(recent_draws_for_zodiac)}期命中率 {rate2:.0f}%)")
    print("🎲 正码5个 (科学概率评估，基于最近6期):")
    for n in hot5:
        hits_6 = sum(1 for draw in draws if n in draw)
        low, high = wilson_interval(hits_6, len(draws))
        posterior = bayesian_posterior(hits_6, len(draws))
        print(f"      {n:02d} ({get_zodiac(n)})  ─ 威尔逊区间 [{low:.0f}%-{high:.0f}%]  后验概率 {posterior:.1f}%")
    print(f"🔮 特别号 (首选): {picked_special:02d} ({special_zod})")

    # 特别号6码推荐
    pair_lift = calculate_pair_lift(draws)
    scores_list = []
    for s in ["hot", "cold", "momentum", "balanced", "pattern"]:
        score_obj = generate_strategy_score(draws, specials, s, pair_lift, use_dynamic_weights=True,
                                            day_wuxing=day_wuxing, day_zhi=day_zhi)
        scores_list.append(score_obj.raw_scores)
    votes = {n: 0.0 for n in ALL_NUMBERS}
    for sc in scores_list:
        ranked = sorted(sc.items(), key=lambda x: x[1], reverse=True)
        for rank, (n, _) in enumerate(ranked):
            votes[n] += 49 - rank
    max_vote = max(votes.values()) if votes else 1
    norm_votes = {n: v / max_vote for n, v in votes.items()}
    special_scores = {n: norm_votes[n] for n in ALL_NUMBERS}
    for n in hot5:
        special_scores[n] = -1.0
    top6_specials = sorted(special_scores.items(), key=lambda x: x[1], reverse=True)[:6]
    print("\n🔯 特别号6码推荐 (按综合评分排序):")
    for i, (num, score) in enumerate(top6_specials, 1):
        print(f"   {i}. {num:02d} ({get_zodiac(num)})  ─ 综合评分: {score*100:.1f}%")

    # 三中三推荐
    best_combo = None
    if len(hot5) >= 3:
        print("\n🎰 三中三推荐 (从正码5个组合生成，共10组):")
        three_combos = list(combinations(sorted(hot5), 3))
        combo_hits = {}
        for combo in three_combos:
            hits = sum(1 for draw in draws if all(n in draw for n in combo))
            combo_hits[combo] = hits
        for i, combo in enumerate(three_combos, 1):
            hits = combo_hits[combo]
            prob = hits / len(draws) * 100 if draws else 0
            combo_str = " ".join(f"{n:02d}({get_zodiac(n)})" for n in combo)
            print(f"   {i:2d}. {combo_str}  ─ 近{len(draws)}期同时出现 {hits} 次 ({prob:.1f}%)")
        if combo_hits:
            best_combo = max(three_combos, key=lambda c: combo_hits[c])
            best_hits = combo_hits[best_combo]
            best_prob = best_hits / len(draws) * 100 if draws else 0
            best_combo_str = " ".join(f"{n:02d}({get_zodiac(n)})" for n in best_combo)
            print(f"\n🏆 极强推荐组合: {best_combo_str} (近{len(draws)}期出现 {best_hits} 次, {best_prob:.1f}%)")

    print("=" * 60)
    print("⚠️ 数据仅供参考，理性投注。")

    # 智能投注方案（使用预算500元，可修改）
    print_betting_plan(hot5, top1, top2, picked_special, top6_specials, best_combo, budget=500)

    # 微信推送（可选）
    push_lines = []
    push_lines.append(f"【新澳门·{next_issue_str}期推荐】")
    push_lines.append(f"今日{day_gan}{day_zhi}日 五行{day_wuxing} · 玄学权重{FENGSHUI_POWER*100:.0f}%")
    push_lines.append(f"🐉 主攻生肖：{top1}")
    push_lines.append(f"🎲 正码5个：{' '.join(f'{n:02d}' for n in hot5)}")
    push_lines.append(f"🔮 特别号：{picked_special:02d}")
    push_lines.append(f"🏆 三中三组合：{' '.join(f'{n:02d}' for n in best_combo) if best_combo else '无'}")
    send_pushplus_notification("新澳门预测", "\n".join(push_lines))

    conn.close()


def cmd_backtest(args: argparse.Namespace) -> None:
    print("轻量回测已在 show 命令中展示最近8期统计，无需单独运行。")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="新澳门六合彩 - 确定性选号版")
    parser.add_argument("--db", default=DB_PATH_DEFAULT, help="数据库路径")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sync = sub.add_parser("sync", help="同步历史数据")
    p_sync.set_defaults(func=cmd_sync)

    p_predict = sub.add_parser("predict", help="生成下期预测")
    p_predict.set_defaults(func=cmd_predict)

    p_show = sub.add_parser("show", help="显示推荐和最近8期回测统计")
    p_show.set_defaults(func=cmd_show)

    p_backtest = sub.add_parser("backtest", help="轻量回测（已集成在show中）")
    p_backtest.set_defaults(func=cmd_backtest)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
