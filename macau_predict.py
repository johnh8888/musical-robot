#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新澳门六合彩 - 终极科学版（仅使用最近7期数据 + 动态权重自适应）
用法:
    python macau_predict.py sync --year 2026
    python macau_predict.py predict
    python macau_predict.py show
    python macau_predict.py backtest
"""

import argparse
import json
import logging
import math
import random
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
logger = logging.getLogger("macau_7")

# -------------------- 常量与配置 --------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH_DEFAULT = str(SCRIPT_DIR / "macau_7.db")

# 新澳门历史数据接口
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

# 波色映射（用于过滤）
COLOR_MAP = {
    "红": [1,2,7,8,12,13,18,19,23,24,29,30,34,35,40,45,46],
    "蓝": [3,4,9,10,14,15,20,21,25,26,31,32,36,37,41,42,47,48],
    "绿": [5,6,11,16,17,22,27,28,33,38,39,43,44,49]
}

ALL_NUMBERS = list(range(1, 50))
MONTE_CARLO_TRIALS = 5000
SUM_TARGET = (115, 185)

# 预测时使用的历史期数（只取最近7期）
PREDICT_WINDOW = 7


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


# -------------------- 数据获取（新澳门专用） --------------------
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


# -------------------- 高级特征工程（适配短窗口）--------------------
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
    """指数衰减动量，半衰期调短以适配7期数据"""
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


# -------------------- 动态权重优化（适配7期短窗口）--------------------
def find_optimal_weights(
    draws: List[List[int]],
    specials: List[int],
    base_weights: Dict[str, float],
) -> Dict[str, float]:
    """在7期数据下，用最近2期作为验证集进行快速网格搜索"""
    if len(draws) < 4:  # 至少需要4期才能做简单回测
        return base_weights

    test_window = max(2, len(draws) // 3)  # 7期时约2-3期
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

    mom = calculate_exp_momentum(draws, half_life=2)  # 短半衰期

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
    color_counts = {c: 0 for c in COLOR_MAP}
    for n in s:
        for c, nums in COLOR_MAP.items():
            if n in nums:
                color_counts[c] += 1
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


# -------------------- 策略核心 --------------------
def generate_strategy_score(
    draws: List[List[int]],
    specials: List[int],
    strategy: str,
    pair_lift: Dict[Tuple[int, int], float],
    use_dynamic_weights: bool = True
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

    scores = {}
    for n in ALL_NUMBERS:
        scores[n] = (
            freq_n[n] * weights["w_freq"]
            + omit_n[n] * weights["w_omit"]
            + mom_n[n] * weights["w_mom"]
        )

    if strategy == "ensemble":
        return ensemble_vote(draws, specials, pair_lift, use_dynamic_weights)

    main_picks = monte_carlo_pick(scores, pair_lift)

    markov = SpecialMarkovModel(2)
    markov.train(specials)
    special_pick = markov.predict(specials[-5:] if len(specials) >= 5 else specials)
    while special_pick in main_picks:
        special_pick = (special_pick % 49) + 1

    confidence = sum(scores[n] for n in main_picks) / 6 if main_picks else 0
    return StrategyScore(main_picks, special_pick, confidence, scores)


def ensemble_vote(
    draws: List[List[int]], specials: List[int], pair_lift: Dict, use_dynamic_weights: bool = True
) -> StrategyScore:
    scores_list = []
    for s in ["hot", "cold", "momentum", "balanced", "pattern"]:
        score_obj = generate_strategy_score(draws, specials, s, pair_lift, use_dynamic_weights)
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


# -------------------- 概率计算工具（基于7期）--------------------
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
    for strat in STRATEGY_IDS:
        score = generate_strategy_score(draws, specials, strat, pair_lift, use_dynamic_weights=True)
        conn.execute("""
            INSERT OR REPLACE INTO predictions (issue_no, strategy, numbers_json, special_number, confidence, status, created_at)
            VALUES (?, ?, ?, ?, ?, 'PENDING', ?)
        """, (next_issue, strat, json.dumps(score.main_picks), score.special_pick, score.confidence, utc_now()))
    conn.commit()
    print(f"已生成 {next_issue} 期的预测推荐（基于最近{len(draws)}期数据）。")
    conn.close()


def cmd_show(args: argparse.Namespace) -> None:
    conn = connect_db(args.db)
    init_db(conn)

    # ---------- 最新开奖 ----------
    latest = conn.execute(
        "SELECT issue_no, draw_date, numbers_json, special_number FROM draws ORDER BY draw_date DESC LIMIT 1"
    ).fetchone()
    if latest:
        nums = json.loads(latest["numbers_json"])
        print(f"最新开奖: {latest['issue_no']} {latest['draw_date']} | 主号: {' '.join(f'{n:02d}' for n in nums)} | 特别号: {latest['special_number']:02d}")
    else:
        print("暂无开奖数据。")

    # ---------- 多策略推荐 ----------
    pending = conn.execute(
        "SELECT issue_no, strategy, numbers_json, special_number, confidence FROM predictions WHERE status='PENDING' ORDER BY strategy"
    ).fetchall()
    if pending:
        print("\n本期多策略推荐 (6码池，基于最近7期):")
        for p in pending:
            nums = json.loads(p["numbers_json"])
            conf_str = f" (置信度: {p['confidence']*100:.1f}%)" if p["confidence"] else ""
            strategy_name = STRATEGY_CONFIGS.get(p['strategy'], {}).get('name', p['strategy'])
            print(f"  [{p['issue_no']}] {strategy_name}{conf_str}: {' '.join(f'{n:02d}' for n in nums)} | 特别号: {p['special_number']:02d}")
    else:
        print("\n暂无待开奖预测，请先运行 predict")

    # ---------- 简洁投注推荐 ----------
    print("\n" + "=" * 60)
    print("🎯 本期投注推荐单 (基于最近7期 + 动态权重)")
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
        score = ensemble_vote(draws, specials, pair_lift, use_dynamic_weights=True)
        picked_6 = score.main_picks
        picked_special = score.special_pick

    hot5 = picked_6[:5] if len(picked_6) >= 5 else picked_6

    zodiac_score = Counter()
    for draw in draws[-5:]:
        for n in draw:
            for z, nums in ZODIAC_MAP.items():
                if n in nums:
                    zodiac_score[z] += 1
    top_zod = zodiac_score.most_common(2)
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

    print(f"📅 参考期号: {pending[0]['issue_no'] if pending else next_issue_number(latest['issue_no'])}")
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
        score_obj = generate_strategy_score(draws, specials, s, pair_lift, use_dynamic_weights=True)
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
    for i, (num, score) in enumerate(top6_specials, 1):
        print(f"   {i}. {num:02d} ({get_zodiac(num)})  ─ 综合评分: {score*100:.1f}%")

    # ---------- 极强推荐（三中三中最佳组合） ----------
    if len(hot5) >= 3 and combo_hits:
        best_combo = max(three_combos, key=lambda c: (combo_hits[c], -three_combos.index(c)))
        best_hits = combo_hits[best_combo]
        best_prob = best_hits / len(draws) * 100 if draws else 0
        best_combo_str = " ".join(f"{n:02d}({get_zodiac(n)})" for n in best_combo)
        print(f"\n🏆 极强推荐组合: {best_combo_str}")
        print(f"   近{len(draws)}期同时出现 {best_hits} 次 ({best_prob:.1f}%)，是近期最稳定的三码组合。")

    # ---------- 最近7期特别号策略命中统计 ----------
    recent_7_rows = conn.execute("""
        SELECT issue_no, special_number FROM draws 
        ORDER BY draw_date DESC, issue_no DESC LIMIT 7
    """).fetchall()
    if len(recent_7_rows) >= 3:
        print("\n📊 最近7期特别号策略命中统计:")
        recent_7 = list(reversed(recent_7_rows))  # 按时间正序
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
            print(f"  {strategy_name:<12} {hits}/{total:<6} {rate:.0f}%")
    else:
        print("\n最近7期数据不足，无法显示特别号命中统计。")

    print("=" * 60)
    print("⚠️ 数据仅供参考，理性投注。")

    conn.close()


def cmd_backtest(args: argparse.Namespace) -> None:
    print("回测功能需较长历史数据，当前7期模式不建议运行完整回测。")
    print("如需回测，请使用完整版脚本。")
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="新澳门六合彩 - 7期短窗口预测")
    parser.add_argument("--db", default=DB_PATH_DEFAULT, help="数据库路径")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sync = sub.add_parser("sync", help="同步历史数据")
    p_sync.add_argument("--year", type=int, default=datetime.now().year, help="指定年份")
    p_sync.set_defaults(func=cmd_sync)

    p_predict = sub.add_parser("predict", help="生成下期预测（基于最近7期）")
    p_predict.set_defaults(func=cmd_predict)

    p_show = sub.add_parser("show", help="显示推荐和统计")
    p_show.set_defaults(func=cmd_show)

    p_backtest = sub.add_parser("backtest", help="回测（7期模式下不推荐）")
    p_backtest.set_defaults(func=cmd_backtest)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
