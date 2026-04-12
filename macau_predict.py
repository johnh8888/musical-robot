#!/usr/bin/env python3
# ==================== 澳门六合彩终极智能预测系统 ====================
import argparse
import json
import sqlite3
import requests
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any
from itertools import combinations

# ==================== 配置 ====================
SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH = str(SCRIPT_DIR / "macau_lottery.db")

API_HISTORY = "https://history.macaumarksix.com/history/macaujc2/y/{}"
API_LATEST = "https://macaumarksix.com/api/macaujc2.com"

ZODIAC_MAP = {
    "马": [1, 13, 25, 37, 49], "羊": [12, 24, 36, 48], "猴": [11, 23, 35, 47],
    "鸡": [10, 22, 34, 46], "狗": [9, 21, 33, 45], "猪": [8, 20, 32, 44],
    "鼠": [7, 19, 31, 43], "牛": [6, 18, 30, 42], "虎": [5, 17, 29, 41],
    "兔": [4, 16, 28, 40], "龙": [3, 15, 27, 39], "蛇": [2, 14, 26, 38]
}

COLOR_MAP = {
    "红": [1,2,7,8,12,13,18,19,23,24,29,30,34,35,40,45,46],
    "蓝": [3,4,9,10,14,15,20,21,25,26,31,32,36,37,41,42,47,48],
    "绿": [5,6,11,16,17,22,27,28,33,38,39,43,44,49]
}

WUXING_MAP = {
    "金": [4,5,12,13,20,21,28,29,36,37,44,45],
    "木": [1,8,9,16,17,24,25,32,33,40,41,48,49],
    "水": [6,7,14,15,22,23,30,31,38,39,46,47],
    "火": [2,3,10,11,18,19,26,27,34,35,42,43],
    "土": [5,6,13,14,21,22,29,30,37,38,45,46]
}

ALL_NUMBERS = list(range(1, 50))

STRATEGIES = {
    "hot": {"name": "热号追踪", "w_freq": 0.7, "w_omit": 0.0, "w_mom": 0.3, "w_conn": 0.0, "w_zod": 0.0},
    "cold": {"name": "冷号回补", "w_freq": 0.0, "w_omit": 0.7, "w_mom": 0.3, "w_conn": 0.0, "w_zod": 0.0},
    "momentum": {"name": "近期动量", "w_freq": 0.2, "w_omit": 0.0, "w_mom": 0.8, "w_conn": 0.0, "w_zod": 0.0},
    "balanced": {"name": "均衡策略", "w_freq": 0.35, "w_omit": 0.25, "w_mom": 0.25, "w_conn": 0.10, "w_zod": 0.05},
    "zodiac": {"name": "生肖强化", "w_freq": 0.40, "w_omit": 0.20, "w_mom": 0.20, "w_conn": 0.05, "w_zod": 0.15},
    "pattern": {"name": "形态追踪", "w_freq": 0.20, "w_omit": 0.20, "w_mom": 0.30, "w_conn": 0.20, "w_zod": 0.10},
    "ml_optimized": {"name": "机器学习", "w_freq": 0.30, "w_omit": 0.30, "w_mom": 0.20, "w_conn": 0.10, "w_zod": 0.10},
}


@dataclass
class DrawRecord:
    issue: str
    date: str
    numbers: List[int]
    special: int


@dataclass
class PredictionResult:
    strategy: str
    numbers: List[int]
    special: int
    confidence: float
    expected_hits: float
    risk_level: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ==================== 数据库操作 ====================
def connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS draws (
            issue TEXT PRIMARY KEY,
            draw_date TEXT NOT NULL,
            numbers_json TEXT NOT NULL,
            special_number INTEGER NOT NULL,
            sum_value INTEGER,
            odd_count INTEGER,
            big_count INTEGER,
            consecutive_count INTEGER,
            created_at TEXT NOT NULL
        );
        
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            issue TEXT NOT NULL,
            strategy TEXT NOT NULL,
            numbers_json TEXT NOT NULL,
            special_number INTEGER,
            confidence REAL,
            expected_hits REAL,
            hit_count INTEGER,
            hit_rate REAL,
            special_hit INTEGER,
            status TEXT DEFAULT 'PENDING',
            created_at TEXT NOT NULL,
            reviewed_at TEXT,
            UNIQUE(issue, strategy)
        );
        
        CREATE TABLE IF NOT EXISTS backtest_stats (
            strategy TEXT PRIMARY KEY,
            total_runs INTEGER DEFAULT 0,
            avg_hit REAL DEFAULT 0,
            hit1_rate REAL DEFAULT 0,
            hit2_rate REAL DEFAULT 0,
            hit3_rate REAL DEFAULT 0,
            special_rate REAL DEFAULT 0,
            avg_confidence REAL DEFAULT 0,
            sharpe_ratio REAL DEFAULT 0,
            max_drawdown REAL DEFAULT 0,
            win_streak INTEGER DEFAULT 0,
            updated_at TEXT NOT NULL
        );
        
        CREATE TABLE IF NOT EXISTS ml_weights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            w_freq REAL,
            w_omit REAL,
            w_mom REAL,
            w_conn REAL,
            w_zod REAL,
            score REAL,
            created_at TEXT NOT NULL
        );
        
        CREATE TABLE IF NOT EXISTS number_features (
            number INTEGER PRIMARY KEY,
            freq_10 REAL,
            freq_30 REAL,
            freq_50 REAL,
            avg_gap REAL,
            max_gap INTEGER,
            hot_score REAL,
            updated_at TEXT NOT NULL
        );
    """)
    
    try:
        conn.execute("ALTER TABLE draws ADD COLUMN sum_value INTEGER")
        conn.execute("ALTER TABLE draws ADD COLUMN odd_count INTEGER")
        conn.execute("ALTER TABLE draws ADD COLUMN big_count INTEGER")
        conn.execute("ALTER TABLE draws ADD COLUMN consecutive_count INTEGER")
    except:
        pass
    
    try:
        conn.execute("ALTER TABLE predictions ADD COLUMN confidence REAL")
        conn.execute("ALTER TABLE predictions ADD COLUMN expected_hits REAL")
    except:
        pass
    
    try:
        conn.execute("ALTER TABLE backtest_stats ADD COLUMN hit3_rate REAL")
        conn.execute("ALTER TABLE backtest_stats ADD COLUMN avg_confidence REAL")
        conn.execute("ALTER TABLE backtest_stats ADD COLUMN sharpe_ratio REAL")
        conn.execute("ALTER TABLE backtest_stats ADD COLUMN max_drawdown REAL")
        conn.execute("ALTER TABLE backtest_stats ADD COLUMN win_streak INTEGER")
    except:
        pass
    
    conn.commit()


# ==================== 数据获取 ====================
def fetch_online_data(year: int = None) -> List[DrawRecord]:
    records = []
    if year is None:
        year = datetime.now().year
    
    url = API_HISTORY.format(year)
    print(f"正在获取 {year} 年数据...")
    
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, timeout=15, headers=headers)
        
        if r.status_code != 200:
            print(f"请求失败: {r.status_code}")
            return []
        
        data = r.json()
        
        if data.get("code") != 200 or not data.get("data"):
            print("接口返回数据为空")
            return []
        
        for item in data["data"]:
            issue = str(item.get("expect", "")).strip()
            if not issue.startswith(str(year)):
                continue
                
            open_code = item.get("openCode", "")
            open_time = item.get("openTime", "")
            
            if not open_code:
                continue
            
            try:
                nums = [int(x.strip()) for x in open_code.split(",") if x.strip().isdigit()]
            except:
                continue
            
            if len(nums) >= 7:
                records.append(DrawRecord(
                    issue=issue,
                    date=open_time[:10] if open_time else "",
                    numbers=nums[:6],
                    special=nums[6]
                ))
        
        records.sort(key=lambda x: x.issue)
        print(f"获取到 {len(records)} 条记录")
        
    except Exception as e:
        print(f"获取失败: {e}")
    
    return records


def calculate_draw_features(numbers: List[int]) -> Dict:
    return {
        "sum_value": sum(numbers),
        "odd_count": sum(1 for n in numbers if n % 2 == 1),
        "big_count": sum(1 for n in numbers if n >= 25),
        "consecutive_count": sum(1 for i in range(len(numbers)-1) if abs(sorted(numbers)[i] - sorted(numbers)[i+1]) == 1)
    }


def sync_draws(conn: sqlite3.Connection, records: List[DrawRecord]) -> Tuple[int, int]:
    inserted, updated = 0, 0
    now = utc_now()
    
    for r in records:
        features = calculate_draw_features(r.numbers)
        existing = conn.execute(
            "SELECT issue FROM draws WHERE issue = ?", (r.issue,)
        ).fetchone()
        
        if existing:
            conn.execute("""
                UPDATE draws 
                SET draw_date = ?, numbers_json = ?, special_number = ?,
                    sum_value = ?, odd_count = ?, big_count = ?, consecutive_count = ?
                WHERE issue = ?
            """, (r.date, json.dumps(r.numbers), r.special,
                  features["sum_value"], features["odd_count"], 
                  features["big_count"], features["consecutive_count"], r.issue))
            updated += 1
        else:
            conn.execute("""
                INSERT INTO draws (issue, draw_date, numbers_json, special_number,
                    sum_value, odd_count, big_count, consecutive_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (r.issue, r.date, json.dumps(r.numbers), r.special,
                  features["sum_value"], features["odd_count"], 
                  features["big_count"], features["consecutive_count"], now))
            inserted += 1
    
    conn.commit()
    return inserted, updated


# ==================== 高级特征计算 ====================
def get_recent_draws(conn: sqlite3.Connection, limit: int = 100) -> List[List[int]]:
    rows = conn.execute("""
        SELECT numbers_json FROM draws 
        ORDER BY issue DESC LIMIT ?
    """, (limit,)).fetchall()
    return [json.loads(r["numbers_json"]) for r in rows]


def calculate_frequency(draws: List[List[int]]) -> Dict[int, float]:
    freq = {n: 0.0 for n in ALL_NUMBERS}
    for draw in draws:
        for n in draw:
            freq[n] += 1.0
    return freq


def calculate_omission(draws: List[List[int]]) -> Dict[int, float]:
    omission = {n: float(len(draws)) for n in ALL_NUMBERS}
    for i, draw in enumerate(draws):
        for n in draw:
            omission[n] = min(omission[n], float(i))
    return omission


def calculate_momentum(draws: List[List[int]]) -> Dict[int, float]:
    momentum = {n: 0.0 for n in ALL_NUMBERS}
    for i, draw in enumerate(draws):
        weight = math.exp(-i / 10)
        for n in draw:
            momentum[n] += weight
    return momentum


def calculate_connection_score(draws: List[List[int]]) -> Dict[int, float]:
    """计算连号关联分数"""
    conn_score = {n: 0.0 for n in ALL_NUMBERS}
    
    for draw in draws[:30]:
        sorted_nums = sorted(draw)
        for i in range(len(sorted_nums)):
            n = sorted_nums[i]
            if i > 0 and sorted_nums[i] - sorted_nums[i-1] == 1:
                conn_score[n] += 2.0
                conn_score[sorted_nums[i-1]] += 1.5
            if i < len(sorted_nums)-1 and sorted_nums[i+1] - sorted_nums[i] == 1:
                conn_score[n] += 2.0
    
    return conn_score


def calculate_zodiac_heat(draws: List[List[int]]) -> Dict[int, float]:
    zodiac_freq = {z: 0.0 for z in ZODIAC_MAP}
    for draw in draws:
        for n in draw:
            for z, nums in ZODIAC_MAP.items():
                if n in nums:
                    zodiac_freq[z] += 1.0
    
    scores = {n: 0.0 for n in ALL_NUMBERS}
    for n in ALL_NUMBERS:
        for z, nums in ZODIAC_MAP.items():
            if n in nums:
                scores[n] = zodiac_freq[z]
    return scores


def calculate_wuxing_balance(draws: List[List[int]]) -> Dict[int, float]:
    """五行平衡分析"""
    wuxing_count = {w: 0 for w in WUXING_MAP}
    for draw in draws[:20]:
        for n in draw:
            for w, nums in WUXING_MAP.items():
                if n in nums:
                    wuxing_count[w] += 1
    
    avg_count = sum(wuxing_count.values()) / 5
    wuxing_score = {w: avg_count - count for w, count in wuxing_count.items()}
    
    scores = {n: 0.0 for n in ALL_NUMBERS}
    for n in ALL_NUMBERS:
        for w, nums in WUXING_MAP.items():
            if n in nums:
                scores[n] = wuxing_score[w]
    
    return scores


def normalize_scores(scores: Dict[int, float]) -> Dict[int, float]:
    values = list(scores.values())
    mn, mx = min(values), max(values)
    if mx == mn:
        return {k: 0.0 for k in scores}
    return {k: (v - mn) / (mx - mn) for k, v in scores.items()}


# ==================== 智能选号系统 ====================
def calculate_sum_probability(draws: List[List[int]]) -> Dict[str, float]:
    """计算和值概率分布"""
    sums = [sum(d) for d in draws[:100]]
    avg_sum = sum(sums) / len(sums)
    std_sum = math.sqrt(sum((s - avg_sum) ** 2 for s in sums) / len(sums))
    
    return {
        "avg": avg_sum,
        "std": std_sum,
        "low": avg_sum - std_sum,
        "high": avg_sum + std_sum,
        "min_30": min(sums[-30:]) if len(sums) >= 30 else min(sums),
        "max_30": max(sums[-30:]) if len(sums) >= 30 else max(sums)
    }


def smart_filter(numbers: List[int], draw_features: Dict) -> bool:
    """智能过滤 - 排除不合理组合"""
    if len(numbers) != 6:
        return False
    
    sorted_nums = sorted(numbers)
    total = sum(sorted_nums)
    odd = sum(1 for n in numbers if n % 2 == 1)
    big = sum(1 for n in numbers if n >= 25)
    
    # 和值过滤
    if total < 80 or total > 220:
        return False
    
    # 单双过滤
    if odd == 0 or odd == 6:
        return False
    
    # 大小过滤
    if big == 0 or big == 6:
        return False
    
    # 连号过滤（最多3连）
    consec = 1
    max_consec = 1
    for i in range(1, 6):
        if sorted_nums[i] - sorted_nums[i-1] == 1:
            consec += 1
            max_consec = max(max_consec, consec)
        else:
            consec = 1
    if max_consec > 3:
        return False
    
    # 区间分布（每个区间最多3个）
    zones = [(n - 1) // 10 for n in numbers]
    if max(Counter(zones).values()) > 3:
        return False
    
    # 尾数分布（同尾数最多2个）
    tails = [n % 10 for n in numbers]
    if max(Counter(tails).values()) > 2:
        return False
    
    return True


def pick_top_numbers_with_filter(
    scores: Dict[int, float], 
    count: int = 6,
    sum_range: Tuple[float, float] = None,
    max_attempts: int = 1000
) -> List[int]:
    """带智能过滤的选号"""
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # 尝试多次生成，找到最佳组合
    best_pick = None
    best_score = -1
    
    for _ in range(max_attempts):
        picked = []
        available = ranked.copy()
        
        while len(picked) < count and available:
            # 带概率的选择
            total_score = sum(s for _, s in available[:20])
            if total_score > 0:
                r = random.random() * total_score
                cumsum = 0
                for n, s in available[:20]:
                    cumsum += s
                    if r <= cumsum:
                        picked.append(n)
                        available = [(x, sc) for x, sc in available if x != n]
                        break
            else:
                picked.append(available[0][0])
                available = available[1:]
        
        if len(picked) == count and smart_filter(picked, {}):
            pick_score = sum(scores[n] for n in picked)
            if sum_range:
                total = sum(picked)
                if sum_range[0] <= total <= sum_range[1]:
                    if pick_score > best_score:
                        best_score = pick_score
                        best_pick = picked
            else:
                if pick_score > best_score:
                    best_score = pick_score
                    best_pick = picked
    
    if best_pick:
        return best_pick
    
    # 降级：使用基础选号
    picked = []
    for n, _ in ranked:
        if len(picked) == count:
            break
        if n not in picked:
            picked.append(n)
    
    while len(picked) < count:
        for n, _ in ranked:
            if n not in picked:
                picked.append(n)
                break
    
    return picked


def generate_advanced_picks(draws: List[List[int]], strategy: str) -> Tuple[List[int], int, float]:
    """高级策略选号"""
    window = min(80, len(draws))
    recent = draws[:window]
    
    # 计算所有特征
    freq = normalize_scores(calculate_frequency(recent))
    omission = normalize_scores(calculate_omission(recent))
    momentum = normalize_scores(calculate_momentum(recent))
    connection = normalize_scores(calculate_connection_score(recent))
    zodiac = normalize_scores(calculate_zodiac_heat(recent))
    wuxing = normalize_scores(calculate_wuxing_balance(recent))
    
    cfg = STRATEGIES.get(strategy, STRATEGIES["balanced"])
    
    # 综合评分
    scores = {}
    for n in ALL_NUMBERS:
        scores[n] = (
            freq[n] * cfg["w_freq"] +
            omission[n] * cfg["w_omit"] +
            momentum[n] * cfg["w_mom"] +
            connection[n] * cfg.get("w_conn", 0.1) +
            zodiac[n] * cfg.get("w_zod", 0.05) +
            wuxing[n] * 0.05
        )
    
    # 计算和值范围
    sum_prob = calculate_sum_probability(draws)
    sum_range = (sum_prob["low"], sum_prob["high"])
    
    # 智能选号
    main_picks = pick_top_numbers_with_filter(scores, 6, sum_range)
    
    # 计算置信度
    confidence = sum(scores[n] for n in main_picks) / 6
    
    # 选特别号
    special = None
    for n, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        if n not in main_picks:
            special = n
            break
    
    return main_picks, special or main_picks[0], confidence


def ensemble_vote_advanced(draws: List[List[int]]) -> Tuple[List[int], int, float, Dict[str, List[int]]]:
    """高级集成投票"""
    all_picks = {}
    all_scores = []
    confidences = []
    
    for strategy in STRATEGIES.keys():
        picks, special, conf = generate_advanced_picks(draws, strategy)
        all_picks[strategy] = picks
        confidences.append(conf)
        
        # 构建分数用于投票
        scores = {n: 0.0 for n in ALL_NUMBERS}
        for i, n in enumerate(picks):
            scores[n] = 6 - i
        all_scores.append(scores)
    
    # 加权投票
    votes = {n: 0.0 for n in ALL_NUMBERS}
    for scores, conf in zip(all_scores, confidences):
        weight = conf
        for n, s in scores.items():
            votes[n] += s * weight
    
    # 选号
    sum_prob = calculate_sum_probability(draws)
    sum_range = (sum_prob["low"], sum_prob["high"])
    final_scores = normalize_scores(votes)
    main_picks = pick_top_numbers_with_filter(final_scores, 6, sum_range)
    
    special = None
    for n, _ in sorted(final_scores.items(), key=lambda x: x[1], reverse=True):
        if n not in main_picks:
            special = n
            break
    
    avg_confidence = sum(confidences) / len(confidences)
    
    return main_picks, special or main_picks[0], avg_confidence, all_picks


# ==================== 机器学习优化 ====================
def optimize_ml_weights(conn: sqlite3.Connection) -> Dict[str, float]:
    """使用网格搜索优化权重"""
    draws = get_recent_draws(conn, limit=200)
    if len(draws) < 50:
        return STRATEGIES["ml_optimized"]
    
    best_weights = STRATEGIES["ml_optimized"].copy()
    best_score = -1
    
    # 权重搜索空间
    weight_options = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    test_draws = draws[20:50]
    history = draws[50:150]
    
    for w_freq in weight_options:
        for w_omit in weight_options:
            for w_mom in weight_options:
                if w_freq + w_omit + w_mom > 0.9:
                    continue
                
                w_conn = 0.1
                w_zod = 0.1
                
                total_score = 0
                valid_tests = 0
                
                for i, target in enumerate(test_draws):
                    hist = history[i:i+30]
                    if len(hist) < 20:
                        continue
                    
                    test_weights = {
                        "w_freq": w_freq, "w_omit": w_omit, "w_mom": w_mom,
                        "w_conn": w_conn, "w_zod": w_zod
                    }
                    
                    # 临时使用这些权重
                    freq = normalize_scores(calculate_frequency(hist))
                    omission = normalize_scores(calculate_omission(hist))
                    momentum = normalize_scores(calculate_momentum(hist))
                    connection = normalize_scores(calculate_connection_score(hist))
                    zodiac = normalize_scores(calculate_zodiac_heat(hist))
                    
                    scores = {}
                    for n in ALL_NUMBERS:
                        scores[n] = (
                            freq[n] * w_freq +
                            omission[n] * w_omit +
                            momentum[n] * w_mom +
                            connection[n] * w_conn +
                            zodiac[n] * w_zod
                        )
                    
                    picked = pick_top_numbers_with_filter(scores, 6)
                    hits = len(set(picked) & set(target))
                    total_score += hits
                    valid_tests += 1
                
                if valid_tests > 0:
                    avg_score = total_score / valid_tests
                    if avg_score > best_score:
                        best_score = avg_score
                        best_weights = {
                            "name": "机器学习",
                            "w_freq": w_freq, "w_omit": w_omit, "w_mom": w_mom,
                            "w_conn": w_conn, "w_zod": w_zod
                        }
    
    # 保存到数据库
    now = utc_now()
    conn.execute("""
        INSERT INTO ml_weights (w_freq, w_omit, w_mom, w_conn, w_zod, score, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (best_weights["w_freq"], best_weights["w_omit"], best_weights["w_mom"],
          best_weights["w_conn"], best_weights["w_zod"], best_score, now))
    conn.commit()
    
    STRATEGIES["ml_optimized"] = best_weights
    return best_weights


# ==================== 预测和复盘 ====================
def generate_predictions(conn: sqlite3.Connection, issue: str = None) -> str:
    row = conn.execute("SELECT issue FROM draws ORDER BY issue DESC LIMIT 1").fetchone()
    if not row:
        raise RuntimeError("没有开奖数据，请先同步")
    
    if issue is None:
        last_issue = row["issue"]
        parts = last_issue.split("/") if "/" in last_issue else (last_issue[:4], last_issue[4:])
        year = parts[0]
        seq = int(parts[1]) if len(parts) > 1 else int(last_issue[4:])
        issue = f"{year}/{seq + 1:03d}"
    
    draws = get_recent_draws(conn, limit=150)
    if len(draws) < 30:
        raise RuntimeError("历史数据不足，至少需要30期")
    
    # 优化ML权重
    optimize_ml_weights(conn)
    
    now = utc_now()
    
    for strategy in STRATEGIES.keys():
        conn.execute(
            "DELETE FROM predictions WHERE issue = ? AND strategy = ?",
            (issue, strategy)
        )
        
        if strategy == "ensemble":
            main_picks, special, confidence, _ = ensemble_vote_advanced(draws)
            expected_hits = confidence * 6 / 49
        else:
            main_picks, special, confidence = generate_advanced_picks(draws, strategy)
            expected_hits = confidence * 6 / 49
        
        conn.execute("""
            INSERT INTO predictions 
            (issue, strategy, numbers_json, special_number, confidence, expected_hits, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, 'PENDING', ?)
        """, (issue, strategy, json.dumps(main_picks), special, confidence, expected_hits, now))
    
    conn.commit()
    return issue


def review_prediction(conn: sqlite3.Connection, issue: str) -> int:
    draw = conn.execute(
        "SELECT numbers_json, special_number FROM draws WHERE issue = ?",
        (issue,)
    ).fetchone()
    
    if not draw:
        return 0
    
    winning = set(json.loads(draw["numbers_json"]))
    winning_special = draw["special_number"]
    
    predictions = conn.execute(
        "SELECT id, numbers_json, special_number FROM predictions WHERE issue = ? AND status = 'PENDING'",
        (issue,)
    ).fetchall()
    
    reviewed = 0
    now = utc_now()
    
    for p in predictions:
        picked = json.loads(p["numbers_json"])
        hit_count = len([n for n in picked if n in winning])
        hit_rate = hit_count / 6.0
        special_hit = 1 if p["special_number"] == winning_special else 0
        
        conn.execute("""
            UPDATE predictions 
            SET status = 'REVIEWED', hit_count = ?, hit_rate = ?, special_hit = ?, reviewed_at = ?
            WHERE id = ?
        """, (hit_count, hit_rate, special_hit, now, p["id"]))
        
        reviewed += 1
    
    update_backtest_stats(conn)
    conn.commit()
    
    return reviewed


def update_backtest_stats(conn: sqlite3.Connection) -> None:
    now = utc_now()
    
    for strategy in STRATEGIES.keys():
        stats = conn.execute("""
            SELECT 
                COUNT(*) as total,
                AVG(hit_count) as avg_hit,
                AVG(CASE WHEN hit_count >= 1 THEN 1.0 ELSE 0.0 END) as hit1_rate,
                AVG(CASE WHEN hit_count >= 2 THEN 1.0 ELSE 0.0 END) as hit2_rate,
                AVG(CASE WHEN hit_count >= 3 THEN 1.0 ELSE 0.0 END) as hit3_rate,
                AVG(special_hit) as special_rate,
                AVG(confidence) as avg_confidence
            FROM predictions 
            WHERE strategy = ? AND status = 'REVIEWED'
        """, (strategy,)).fetchone()
        
        if stats and stats["total"] > 0:
            conn.execute("""
                INSERT INTO backtest_stats 
                (strategy, total_runs, avg_hit, hit1_rate, hit2_rate, hit3_rate, 
                 special_rate, avg_confidence, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(strategy) DO UPDATE SET
                    total_runs = excluded.total_runs,
                    avg_hit = excluded.avg_hit,
                    hit1_rate = excluded.hit1_rate,
                    hit2_rate = excluded.hit2_rate,
                    hit3_rate = excluded.hit3_rate,
                    special_rate = excluded.special_rate,
                    avg_confidence = excluded.avg_confidence,
                    updated_at = excluded.updated_at
            """, (
                strategy,
                stats["total"],
                stats["avg_hit"] or 0,
                stats["hit1_rate"] or 0,
                stats["hit2_rate"] or 0,
                stats["hit3_rate"] or 0,
                stats["special_rate"] or 0,
                stats["avg_confidence"] or 0,
                now
            ))


# ==================== 高级分析显示 ====================
def print_advanced_recommendations(conn: sqlite3.Connection) -> None:
    print("\n" + "=" * 90)
    print("🎯 澳门六合彩 - 终极智能预测系统")
    print("=" * 90)
    
    # 最新开奖
    latest = conn.execute("""
        SELECT issue, draw_date, numbers_json, special_number, sum_value, odd_count, big_count
        FROM draws ORDER BY issue DESC LIMIT 1
    """).fetchone()
    
    if latest:
        nums = json.loads(latest["numbers_json"])
        print(f"\n📅 最新开奖: {latest['issue']} ({latest['draw_date']})")
        print(f"   号码: {' '.join(f'{n:02d}' for n in nums)} | 特别号: {latest['special_number']:02d}")
        print(f"   和值: {latest['sum_value']} | 单:{latest['odd_count']} 双:{6-latest['odd_count']} | 大:{latest['big_count']} 小:{6-latest['big_count']}")
    
    # 趋势分析
    draws = get_recent_draws(conn, limit=100)
    
    print("\n" + "-" * 90)
    print("📊 一、多维度趋势分析")
    print("-" * 90)
    
    if draws:
        sum_prob = calculate_sum_probability(draws)
        print(f"   📈 和值预测区间: [{sum_prob['low']:.0f} - {sum_prob['high']:.0f}] (历史平均: {sum_prob['avg']:.0f})")
        
        # 号码热度
        all_nums = [n for d in draws[:30] for n in d]
        freq = Counter(all_nums)
        hot = [n for n, _ in freq.most_common(8)]
        cold = [n for n in ALL_NUMBERS if n not in all_nums][:8]
        
        print(f"   🔥 热号(近30期): {' '.join(f'{n:02d}' for n in hot)}")
        print(f"   ❄️ 冷号(近30期未出): {' '.join(f'{n:02d}' for n in cold)}")
        
        # 生肖分析
        zodiac_hot = Counter()
        for d in draws[:20]:
            for n in d:
                for z, nums in ZODIAC_MAP.items():
                    if n in nums:
                        zodiac_hot[z] += 1
        top_zodiacs = [z for z, _ in zodiac_hot.most_common(4)]
        print(f"   🐉 热门生肖: {' > '.join(top_zodiacs)}")
    
    # 各策略推荐
    print("\n" + "-" * 90)
    print("🎲 二、多策略智能推荐")
    print("-" * 90)
    
    if len(draws) >= 30:
        # 集成投票
        ensemble_nums, ensemble_special, confidence, all_picks = ensemble_vote_advanced(draws)
        
        print(f"\n   🌟 集成投票推荐 (置信度: {confidence*100:.1f}%)")
        print(f"      6码: {' '.join(f'{n:02d}' for n in ensemble_nums)}")
        print(f"      特别号: {ensemble_special:02d}")
        
        print("\n   各策略详细推荐:")
        for strategy, cfg in STRATEGIES.items():
            if strategy in all_picks:
                nums = all_picks[strategy]
                _, special, conf = generate_advanced_picks(draws, strategy)
                expected = conf * 6 / 49
                stars = "⭐" * min(5, int(conf * 5) + 1)
                print(f"      {stars} {cfg['name']}: {' '.join(f'{n:02d}' for n in nums)} | 特{special:02d} | 预期命中:{expected:.2f}")
    
    # 历史表现
    print("\n" + "-" * 90)
    print("📈 三、策略历史表现评估")
    print("-" * 90)
    
    stats = conn.execute("""
        SELECT strategy, total_runs, avg_hit, hit1_rate, hit2_rate, hit3_rate, 
               special_rate, avg_confidence
        FROM backtest_stats ORDER BY avg_hit DESC
    """).fetchall()
    
    if stats:
        print(f"\n   {'策略':<12} {'测试':<6} {'平均命中':<8} {'≥1码':<8} {'≥2码':<8} {'≥3码':<8} {'特号率':<8} {'置信度':<8}")
        print("   " + "-" * 80)
        for s in stats:
            name = STRATEGIES.get(s["strategy"], {}).get("name", s["strategy"])[:10]
            conf = s["avg_confidence"] * 100 if s["avg_confidence"] else 0
            print(f"   {name:<12} {s['total_runs']:<6} {s['avg_hit']:.2f}     {s['hit1_rate']*100:.1f}%    {s['hit2_rate']*100:.1f}%    {s['hit3_rate']*100:.1f}%    {s['special_rate']*100:.1f}%   {conf:.1f}%")
    
    # ML权重
    ml_weights = conn.execute("""
        SELECT w_freq, w_omit, w_mom, w_conn, w_zod, score, created_at
        FROM ml_weights ORDER BY created_at DESC LIMIT 1
    """).fetchone()
    
    if ml_weights:
        print("\n" + "-" * 90)
        print("🤖 四、机器学习优化权重")
        print("-" * 90)
        print(f"   频率权重: {ml_weights['w_freq']:.2f} | 遗漏权重: {ml_weights['w_omit']:.2f} | 动量权重: {ml_weights['w_mom']:.2f}")
        print(f"   连号权重: {ml_weights['w_conn']:.2f} | 生肖权重: {ml_weights['w_zod']:.2f}")
        print(f"   验证得分: {ml_weights['score']:.3f} | 更新时间: {ml_weights['created_at'][:10]}")
    
    # 下期推荐汇总
    print("\n" + "-" * 90)
    print("🎯 五、下期综合投注建议")
    print("-" * 90)
    
    pending = conn.execute("""
        SELECT strategy, numbers_json, special_number, confidence, expected_hits
        FROM predictions WHERE status = 'PENDING'
        ORDER BY confidence DESC
    """).fetchall()
    
    if pending:
        # 统计投票
        vote_count = Counter()
        special_votes = Counter()
        
        for p in pending:
            nums = json.loads(p["numbers_json"])
            weight = p["confidence"] if p["confidence"] else 1.0
            for n in nums:
                vote_count[n] += weight
            special_votes[p["special_number"]] += weight
        
        top6 = [n for n, _ in vote_count.most_common(6)]
        top_special = special_votes.most_common(1)[0][0] if special_votes else None
        
        print(f"\n   🏆 加权投票推荐6码: {' '.join(f'{n:02d}' for n in top6)}")
        if top_special:
            print(f"   🏆 加权投票特别号: {top_special:02d}")
        
        # 最佳策略
        best = pending[0]
        best_nums = json.loads(best["numbers_json"])
        best_name = STRATEGIES.get(best["strategy"], {}).get("name", best["strategy"])
        print(f"\n   💡 最佳单策略({best_name}): {' '.join(f'{n:02d}' for n in best_nums)}")
        print(f"      置信度: {best['confidence']*100:.1f}% | 预期命中: {best['expected_hits']:.2f}码")
    
    print("\n" + "=" * 90)
    print("⚠️ 重要提醒:")
    print("   1. 以上推荐基于机器学习+多策略集成，仅供参考")
    print("   2. 历史表现不代表未来结果，请理性投注")
    print("   3. 建议采用多策略分散，控制单期投入")
    print("   4. 系统持续学习优化，长期使用效果更佳")
    print("=" * 90)


# ==================== 命令行接口 ====================
def cmd_sync(args: argparse.Namespace) -> None:
    conn = connect_db()
    try:
        init_db(conn)
        year = args.year or datetime.now().year
        records = fetch_online_data(year)
        
        if records:
            inserted, updated = sync_draws(conn, records)
            print(f"同步完成: 新增 {inserted} 期, 更新 {updated} 期")
            
            latest = conn.execute("SELECT issue FROM draws ORDER BY issue DESC LIMIT 1").fetchone()
            if latest:
                reviewed = review_prediction(conn, latest["issue"])
                print(f"自动复盘: {reviewed} 个策略")
        else:
            print("未获取到数据")
    finally:
        conn.close()


def cmd_predict(args: argparse.Namespace) -> None:
    conn = connect_db()
    try:
        init_db(conn)
        issue = generate_predictions(conn, args.issue)
        print(f"已生成预测: {issue}")
    finally:
        conn.close()


def cmd_review(args: argparse.Namespace) -> None:
    conn = connect_db()
    try:
        init_db(conn)
        if args.issue:
            reviewed = review_prediction(conn, args.issue)
        else:
            latest = conn.execute("SELECT issue FROM draws ORDER BY issue DESC LIMIT 1").fetchone()
            if latest:
                reviewed = review_prediction(conn, latest["issue"])
            else:
                reviewed = 0
        print(f"已复盘: {reviewed} 个预测")
    finally:
        conn.close()


def cmd_show(args: argparse.Namespace) -> None:
    conn = connect_db()
    try:
        init_db(conn)
        print_advanced_recommendations(conn)
    finally:
        conn.close()


def cmd_auto(args: argparse.Namespace) -> None:
    conn = connect_db()
    try:
        init_db(conn)
        
        print("=" * 50)
        print("1. 同步最新数据...")
        year = datetime.now().year
        records = fetch_online_data(year)
        if records:
            inserted, updated = sync_draws(conn, records)
            print(f"   新增 {inserted} 期, 更新 {updated} 期")
        
        print("\n2. 复盘最新预测...")
        latest = conn.execute("SELECT issue FROM draws ORDER BY issue DESC LIMIT 1").fetchone()
        if latest:
            reviewed = review_prediction(conn, latest["issue"])
            print(f"   已复盘 {reviewed} 个策略")
        
        print("\n3. 优化ML权重...")
        ml_weights = optimize_ml_weights(conn)
        print(f"   ML权重已优化")
        
        print("\n4. 生成下期预测...")
        issue = generate_predictions(conn)
        print(f"   目标期号: {issue}")
        
        print("\n5. 终极智能推荐结果:")
        print_advanced_recommendations(conn)
        
    finally:
        conn.close()


def cmd_export(args: argparse.Namespace) -> None:
    """导出推荐单"""
    conn = connect_db()
    try:
        init_db(conn)
        
        pending = conn.execute("""
            SELECT strategy, numbers_json, special_number, confidence
            FROM predictions WHERE status = 'PENDING'
            ORDER BY confidence DESC
        """).fetchall()
        
        if not pending:
            print("暂无待开奖预测")
            return
        
        filename = args.output or f"recommend_{datetime.now().strftime('%Y%m%d')}.txt"
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write(f"澳门六合彩智能推荐单 - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write("=" * 60 + "\n\n")
            
            for p in pending[:5]:
                name = STRATEGIES.get(p["strategy"], {}).get("name", p["strategy"])
                nums = json.loads(p["numbers_json"])
                f.write(f"[{name}]\n")
                f.write(f"  6码: {' '.join(f'{n:02d}' for n in nums)}\n")
                f.write(f"  特别号: {p['special_number']:02d}\n")
                f.write(f"  置信度: {p['confidence']*100:.1f}%\n\n")
            
            f.write("=" * 60 + "\n")
            f.write("⚠️ 仅供参考，理性投注\n")
        
        print(f"已导出到: {filename}")
        
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="澳门六合彩终极智能预测系统")
    sub = parser.add_subparsers(dest="command")
    
    p_sync = sub.add_parser("sync", help="同步开奖数据")
    p_sync.add_argument("--year", type=int, help="指定年份")
    p_sync.set_defaults(func=cmd_sync)
    
    p_predict = sub.add_parser("predict", help="生成预测")
    p_predict.add_argument("--issue", help="目标期号")
    p_predict.set_defaults(func=cmd_predict)
    
    p_review = sub.add_parser("review", help="复盘预测")
    p_review.add_argument("--issue", help="指定期号")
    p_review.set_defaults(func=cmd_review)
    
    p_show = sub.add_parser("show", help="显示智能推荐")
    p_show.set_defaults(func=cmd_show)
    
    p_auto = sub.add_parser("auto", help="一键自动运行")
    p_auto.set_defaults(func=cmd_auto)
    
    p_export = sub.add_parser("export", help="导出推荐单")
    p_export.add_argument("--output", help="输出文件名")
    p_export.set_defaults(func=cmd_export)
    
    args = parser.parse_args()
    
    if args.command is None:
        cmd_show(args)
    else:
        args.func(args)


if __name__ == "__main__":
    print("⚠️ 理性提醒：数据仅供娱乐参考，请严格控制投注金额！")
    main()
