#!/usr/bin/env python3
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
warnings.filterwarnings("ignore", category=UserWarning, module='lightgbm')
import sys
from pathlib import Path
import traceback
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import argparse, json, math, os, re, socket, sqlite3, time, pickle, subprocess
import pandas as pd
from urllib.error import URLError
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple, Union
from urllib.request import Request, urlopen
import random

# ---------- 外部模块 (可选) ----------
try:
    from history_utils import HistoryProvider, DrawRecord
except Exception:
    class HistoryProvider:
        def __init__(self, conn: sqlite3.Connection):
            self.conn = conn

        def get_draws_up_to(self, issue_no: str, limit: int = 50) -> List[DrawRecord]:
            target = self.conn.execute(
                "SELECT draw_date FROM draws WHERE issue_no = ?", (issue_no,)
            ).fetchone()
            if not target:
                return []
            draw_date = target["draw_date"]
            rows = self.conn.execute(
                """
                SELECT issue_no, draw_date, numbers_json, special_number
                FROM draws
                WHERE draw_date < ? OR (draw_date = ? AND issue_no < ?)
                ORDER BY draw_date DESC, issue_no DESC
                LIMIT ?
                """,
                (draw_date, draw_date, issue_no, limit),
            ).fetchall()
            return [
                DrawRecord(
                    issue_no=r["issue_no"],
                    draw_date=r["draw_date"],
                    numbers=json.loads(r["numbers_json"]),
                    special_number=int(r["special_number"]),
                )
                for r in rows
            ]

        def get_recent_draws_for_prediction(self, limit: int = 30) -> List[DrawRecord]:
            last_issue_row = self.conn.execute(
                "SELECT issue_no, draw_date FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT 1"
            ).fetchone()
            if not last_issue_row:
                return []
            return self.get_draws_up_to(last_issue_row["issue_no"], limit=limit)

try:
    from transformer_sp import train_transformer, predict_transformer
except ImportError:
    def train_transformer(*args, **kwargs): pass
    def predict_transformer(*args, **kwargs): return None

try:
    import numpy as np
except ImportError:
    np = None

_BEST_PARAMS_PATH = Path(__file__).resolve().parent / "best_params_zodiac.json"
_BEST_PARAMS_OTHER = Path(__file__).resolve().parent / "best_params.json"

def load_best_zodiac_params():
    if _BEST_PARAMS_PATH.exists():
        with open(_BEST_PARAMS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def load_best_params():
    for path in (_BEST_PARAMS_PATH, _BEST_PARAMS_OTHER):
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return None

for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

try:
    from tail_predictor import get_best_tail, backtest_tail
except Exception:
    def get_best_tail(*args, **kwargs): return []
    def backtest_tail(*args, **kwargs): return 0.0, 0, 0

try:
    from zodiac_strict import get_three_zodiac_picks
except Exception:
    def get_three_zodiac_picks(*args, **kwargs): return ["马", "蛇", "龙"]

try:
    from lstm_predictor import predict_lstm_proba
except ImportError:
    predict_lstm_proba = None

def safe_get_hmm_state_proba(conn):
    try:
        from hmm_features import get_hmm_state_proba
        return get_hmm_state_proba(conn)
    except Exception:
        return None

try:
    from risk_manager import RiskManager
except Exception:
    class RiskManager:
        def __init__(self, bankroll: float = 1000.0): self.bankroll = bankroll
        def get_bet_recommendation(self, *_args, **_kwargs): return {"suspended": False, "recommended_stake": 0.0}

try:
    from xgboost_predictor import XGBoostPredictor
except Exception:
    class XGBoostPredictor:
        def train(self, conn): return None
        def predict_pool(self, conn, top_k: int = 20): return []

try:
    from lightgbm_predictor import LightGBMPredictor
except Exception:
    class LightGBMPredictor:
        def train(self, conn): return None
        def predict_pool(self, conn, top_k: int = 20): return []

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH_DEFAULT = str(SCRIPT_DIR / "hongkong_lottery.db")
HK_MARK_SIX_API_URL = "https://marksix6.net/index.php?api=1"
API_TIMEOUT_DEFAULT = 20
API_RETRIES_DEFAULT = 4
API_RETRY_BACKOFF_SECONDS = 2.0
MINED_CONFIG_KEY = "mined_strategy_config_v1"
ALL_NUMBERS = list(range(1, 50))
FEATURE_WINDOW_DEFAULT = 10

STRATEGY_BASE_WINDOWS = {
    "hot_v1": 6, "momentum_v1": 7, "cold_rebound_v1": 13,
    "balanced_v1": 10, "pattern_mined_v1": 6, "ensemble_v2": 10,
}
WEIGHT_WINDOW_DEFAULT = 30
HEALTH_WINDOW_DEFAULT = 18
BACKTEST_ISSUES_DEFAULT = 20
ZERO_HIT_TRIGGER_THRESHOLD = float(os.environ.get("ZERO_HIT_TRIGGER_THRESHOLD", "0.5"))

ENSEMBLE_DIVERSITY_BONUS = 0.18
BIAS_THRESHOLD = 0.65
BIAS_ADJUSTMENT = 0.40
FORCED_BIAS_COEFFICIENT = 0.75

STRATEGY_LABELS = {
    "balanced_v1": "组合策略", "hot_v1": "热号策略", "cold_rebound_v1": "冷号回补",
    "momentum_v1": "近期动量", "ensemble_v2": "集成投票", "pattern_mined_v1": "规律挖掘",
}
STRATEGY_IDS = ["balanced_v1", "hot_v1", "cold_rebound_v1", "momentum_v1", "ensemble_v2", "pattern_mined_v1"]
SPECIAL_ANALYSIS_ORDER = ["pattern_mined_v1", "ensemble_v2", "momentum_v1", "cold_rebound_v1", "hot_v1", "balanced_v1"]
TEXIAO5_SIZE_DEFAULT = 5

ZODIAC_MAP = {
    "马": [1, 13, 25, 37, 49], "蛇": [2, 14, 26, 38], "龙": [3, 15, 27, 39],
    "兔": [4, 16, 28, 40], "虎": [5, 17, 29, 41], "牛": [6, 18, 30, 42],
    "鼠": [7, 19, 31, 43], "猪": [8, 20, 32, 44], "狗": [9, 21, 33, 45],
    "鸡": [10, 22, 34, 46], "猴": [11, 23, 35, 47], "羊": [12, 24, 36, 48],
}
ZODIAC_PAIR = {
    "鼠": "牛", "牛": "鼠", "虎": "猪", "猪": "虎", "兔": "狗", "狗": "兔",
    "龙": "鸡", "鸡": "龙", "蛇": "猴", "猴": "蛇", "马": "羊", "羊": "马"
}

PUSHPLUS_TOKEN = os.environ.get("PUSHPLUS_TOKEN", "")

# ---------- LightGBM 生肖预测器 ----------
_MODEL_PATH = SCRIPT_DIR / "lgb_zodiac_model.pkl"

class ZodiacLightGBM:
    """使用 LightGBM 预测每个生肖的出现概率"""
    def __init__(self):
        self.model = None
        self.feature_names = None

    def train(self, conn):
        try:
            import lightgbm as lgb
            import numpy as np
        except ImportError:
            print("[LightGBM] 未安装 lightgbm，跳过训练")
            return

        draws = conn.execute(
            "SELECT issue_no, draw_date, numbers_json, special_number "
            "FROM draws ORDER BY draw_date, issue_no"
        ).fetchall()
        if len(draws) < 50:
            print("[LightGBM] 数据不足，跳过训练")
            return

        X, y = [], []
        for i in range(30, len(draws)):
            target_date = draws[i]["draw_date"]
            target_issue = draws[i]["issue_no"]
            hist = conn.execute(
                """SELECT numbers_json, special_number FROM draws
                   WHERE draw_date < ? OR (draw_date = ? AND issue_no < ?)
                   ORDER BY draw_date DESC, issue_no DESC LIMIT 50""",
                (target_date, target_date, target_issue)
            ).fetchall()
            if len(hist) < 10:
                continue

            actual_main = json.loads(draws[i]["numbers_json"])
            actual_sp = draws[i]["special_number"]
            actual_zodiacs = set(get_zodiac_by_number(n) for n in actual_main)
            actual_zodiacs.add(get_zodiac_by_number(actual_sp))

            base_feats = self._build_features(hist)
            for z in ZODIAC_MAP:
                label = 1 if z in actual_zodiacs else 0
                row = base_feats.copy()
                row.extend(self._zodiac_specific_features(hist, z))
                X.append(row)
                y.append(label)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        self.feature_names = [f"f_{i}" for i in range(X.shape[1])]

        self.model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=10,
            num_leaves=63,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        self.model.fit(X, y, eval_set=[(X, y)], eval_metric='logloss')
        print("[LightGBM] 生肖模型训练完成")

        with open(_MODEL_PATH, "wb") as f:
            pickle.dump(self, f)

    def _build_features(self, hist_rows):
        nums_all = []
        sp_all = []
        for r in hist_rows:
            nums_all.extend(json.loads(r["numbers_json"]))
            sp_all.append(r["special_number"])
        feats = [
            len(hist_rows),
            sum(1 for n in nums_all if n % 2 == 1) / max(1, len(nums_all)),
            sum(1 for n in nums_all if n > 24) / max(1, len(nums_all)),
            sum(1 for t in sp_all[-5:] if t % 2 == 1) / max(1, len(sp_all[-5:])),
            sum(1 for t in sp_all[-5:] if t % 10 == sp_all[-1] % 10) / 5.0,
        ]
        return feats

    def _zodiac_specific_features(self, hist_rows, zodiac):
        omission = _zodiac_omission_map(hist_rows)
        recent_zodiacs = []
        for r in hist_rows:
            nums = json.loads(r["numbers_json"])
            sp = r["special_number"]
            z_set = set(get_zodiac_by_number(n) for n in nums)
            z_set.add(get_zodiac_by_number(sp))
            recent_zodiacs.append(z_set)
        cnt_5 = sum(1 for z_set in recent_zodiacs[:5] if zodiac in z_set)
        cnt_10 = sum(1 for z_set in recent_zodiacs[:10] if zodiac in z_set)
        cnt_20 = sum(1 for z_set in recent_zodiacs[:20] if zodiac in z_set)
        miss_val = omission.get(zodiac, len(hist_rows))
        last_seen = 0
        for i, z_set in enumerate(recent_zodiacs):
            if zodiac in z_set:
                last_seen = i
                break
        else:
            last_seen = len(recent_zodiacs)

        pair_z = ZODIAC_PAIR.get(zodiac)
        pair_cnt = 0
        if pair_z:
            pair_cnt = sum(1 for z_set in recent_zodiacs[:3] if pair_z in z_set)
        last5_sp_tails = [int(r["special_number"]) % 10 for r in hist_rows[:5]]
        nums_in_zodiac = ZODIAC_MAP.get(zodiac, [])
        tail_match_ratio = sum(1 for n in nums_in_zodiac if n % 10 in last5_sp_tails) / max(1, len(nums_in_zodiac))

        specials = [int(r["special_number"]) for r in hist_rows]
        special_zodiacs = [get_zodiac_by_number(sp) for sp in specials]
        last_special_seen = 0
        for i, z in enumerate(special_zodiacs):
            if z == zodiac:
                last_special_seen = i
                break
        else:
            last_special_seen = len(specials)
        cnt_special_10 = sum(1 for z in special_zodiacs[:10] if z == zodiac)

        omit_ratio = miss_val / max(1, len(hist_rows))
        recent_sp_z = [get_zodiac_by_number(int(r["special_number"])) for r in hist_rows[:7]]
        follow_cnt = 0
        for i in range(len(recent_sp_z)-1):
            if recent_sp_z[i] == zodiac and recent_sp_z[i+1] == zodiac:
                follow_cnt += 1
        follow_freq = follow_cnt / max(1, len(recent_sp_z)-1)

        zodiac_tails = {n % 10 for n in nums_in_zodiac}
        recent_sp_tails = [sp % 10 for sp in specials[:12]]
        tail_overlap = sum(1 for t in recent_sp_tails if t in zodiac_tails) / max(1, len(recent_sp_tails))

        recent_omits = []
        temp_omit = 0
        for r in hist_rows[:20]:
            temp_omit += 1
            nums = json.loads(r["numbers_json"])
            sp = r["special_number"]
            z_set = set(get_zodiac_by_number(n) for n in nums)
            z_set.add(get_zodiac_by_number(sp))
            if zodiac in z_set:
                recent_omits.append(temp_omit)
                temp_omit = 0
        avg_recent_omit = sum(recent_omits) / max(1, len(recent_omits)) if recent_omits else miss_val
        omit_trend = avg_recent_omit / max(1, miss_val)

        if len(recent_omits) >= 2:
            mean_omit = sum(recent_omits) / len(recent_omits)
            var_omit = sum((x - mean_omit)**2 for x in recent_omits) / len(recent_omits)
            std_omit = var_omit ** 0.5
        else:
            std_omit = miss_val

        opposite_z = ZODIAC_PAIR.get(zodiac)
        opposite_follow_cnt = 0
        for i in range(len(special_zodiacs)-1):
            if special_zodiacs[i] == opposite_z and special_zodiacs[i+1] == zodiac:
                opposite_follow_cnt += 1
        opposite_follow_rate = opposite_follow_cnt / max(1, len(special_zodiacs)-1)

        zodiac_nums = ZODIAC_MAP.get(zodiac, [])
        recent_sp_nums = specials[:10]
        hot_num_cnt = sum(1 for n in recent_sp_nums if n in zodiac_nums)

        odd_even_ratio = sum(1 for n in nums_in_zodiac if n % 2 == 1) / max(1, len(nums_in_zodiac))
        zone_idx = (nums_in_zodiac[0] - 1) // 10 if nums_in_zodiac else 0
        zone_hot = sum(1 for sp in specials[:12] if (sp - 1) // 10 == zone_idx) / max(1, len(specials[:12]))
        latest_tail = specials[0] % 10 if specials else 0
        latest_tail_match = 1 if latest_tail in zodiac_tails else 0
        mean_interval = sum(recent_omits) / len(recent_omits) if recent_omits else miss_val

        if len(recent_omits) >= 3:
            x_arr = np.arange(len(recent_omits), dtype=float)
            y_arr = np.array(recent_omits, dtype=float)
            n = len(x_arr)
            denom = n * np.sum(x_arr * x_arr) - np.sum(x_arr) ** 2
            if abs(denom) > 1e-10:
                trend_strength = (n * np.sum(x_arr * y_arr) - np.sum(x_arr) * np.sum(y_arr)) / denom
            else:
                trend_strength = 0.0
        else:
            trend_strength = 0.0

        zod_cnt_20 = {z: sum(1 for z_set in recent_zodiacs[:20] if z in z_set) for z in ZODIAC_MAP}
        sorted_zod = sorted(zod_cnt_20.items(), key=lambda x: x[1], reverse=True)
        rank = [z for z, _ in sorted_zod].index(zodiac) / 11.0

        latest_main_zodiacs = [get_zodiac_by_number(int(n)) for n in json.loads(hist_rows[0]["numbers_json"])]
        main_overlap = sum(1 for z in latest_main_zodiacs if z == zodiac) / max(1, len(latest_main_zodiacs))

        if len(specials) >= 2:
            diff = abs(specials[0] - specials[1])
            diff_zodiac = get_zodiac_by_number(diff)
            diff_match = 1 if diff_zodiac == zodiac else 0
        else:
            diff_match = 0

        if len(recent_omits) >= 3:
            omit_acceleration = recent_omits[-1] - 2*recent_omits[-2] + recent_omits[-3]
        else:
            omit_acceleration = 0.0

        main_special_match = 1 if zodiac in latest_main_zodiacs else 0

        main_num_count_5 = sum(1 for r in hist_rows[:5] for n in json.loads(r["numbers_json"]) if n in zodiac_nums)

        zod_cnt_10 = {z: sum(1 for z_set in recent_zodiacs[:10] if z in z_set) for z in ZODIAC_MAP}
        sorted_zod_10 = sorted(zod_cnt_10.items(), key=lambda x: x[1], reverse=True)
        rank_10 = [z for z, _ in sorted_zod_10].index(zodiac) / 11.0 if zodiac in zod_cnt_10 else 1.0
        rank_change_10 = rank_10 - rank

        if specials:
            latest_sp_odd = specials[0] % 2
            main_odd = 1 if odd_even_ratio > 0.5 else 0
            odd_even_match = 1 if latest_sp_odd == main_odd else 0
        else:
            odd_even_match = 0

        big_small_match = 0
        if specials and nums_in_zodiac:
            latest_sp_big = 1 if specials[0] >= 25 else 0
            zodiac_big_ratio = sum(1 for n in nums_in_zodiac if n >= 25) / len(nums_in_zodiac)
            zodiac_big = 1 if zodiac_big_ratio > 0.5 else 0
            big_small_match = 1 if latest_sp_big == zodiac_big else 0

        zodiac_avg_num = sum(nums_in_zodiac) / len(nums_in_zodiac) if nums_in_zodiac else 25.0

        consecutive_special = 0
        for z in special_zodiacs:
            if z == zodiac:
                consecutive_special += 1
            else:
                break

        latest_sp_zodiac = 1 if (specials and get_zodiac_by_number(specials[0]) == zodiac) else 0

        if len(recent_sp_nums) > 0 and nums_in_zodiac:
            zodiac_span = max(nums_in_zodiac) - min(nums_in_zodiac) + 1
            recent_sp_span = max(recent_sp_nums) - min(recent_sp_nums) + 1
            spread_score = zodiac_span / recent_sp_span if recent_sp_span > 0 else 1.0
        else:
            spread_score = 1.0

        repeat_tendency = 0
        consecutive_count = 0
        for z_set in recent_zodiacs[:20]:
            if zodiac in z_set:
                consecutive_count += 1
            else:
                if consecutive_count >= 2:
                    repeat_tendency += 1
                consecutive_count = 0
        if consecutive_count >= 2:
            repeat_tendency += 1

        cold_to_hot = 1 if (miss_val > 10 and last_seen <= 3) else 0

        latest_main_sum = sum(json.loads(hist_rows[0]["numbers_json"])) if hist_rows else 0
        zodiac_sum = sum(nums_in_zodiac) if nums_in_zodiac else 0
        zodiac_sum_trend = 1.0 / (abs(latest_main_sum - zodiac_sum) + 1) if latest_main_sum > 0 else 0.0

        same_last_digit_small = 0
        if specials and nums_in_zodiac:
            last_dgt = specials[0] % 10
            for n in nums_in_zodiac:
                if n % 10 == last_dgt and n <= 12:
                    same_last_digit_small = 1
                    break

        return [cnt_5, cnt_10, cnt_20, miss_val, last_seen, pair_cnt, tail_match_ratio,
                last_special_seen, cnt_special_10,
                omit_ratio, follow_freq, tail_overlap, omit_trend, std_omit,
                opposite_follow_rate, hot_num_cnt,
                odd_even_ratio, zone_hot, latest_tail_match, mean_interval,
                trend_strength, rank, main_overlap, diff_match,
                omit_acceleration, main_special_match, main_num_count_5,
                rank_change_10, odd_even_match, big_small_match,
                zodiac_avg_num, consecutive_special,
                latest_sp_zodiac, spread_score, repeat_tendency,
                cold_to_hot, zodiac_sum_trend, same_last_digit_small]

    def predict_proba(self, conn, issue_no):
        if self.model is None:
            if _MODEL_PATH.exists():
                with open(_MODEL_PATH, "rb") as f:
                    saved = pickle.load(f)
                    self.model = saved.model
                    self.feature_names = saved.feature_names
            else:
                return None
        target = conn.execute("SELECT draw_date FROM draws WHERE issue_no = ?", (issue_no,)).fetchone()
        if not target:
            return None
        draw_date = target["draw_date"]
        hist = conn.execute(
            """SELECT numbers_json, special_number FROM draws
               WHERE draw_date < ? OR (draw_date = ? AND issue_no < ?)
               ORDER BY draw_date DESC, issue_no DESC LIMIT 50""",
            (draw_date, draw_date, issue_no)
        ).fetchall()
        if len(hist) < 10:
            return None
        base_feats = self._build_features(hist)
        probs = {}
        for z in ZODIAC_MAP:
            row = base_feats + self._zodiac_specific_features(hist, z)
            prob = self.model.predict_proba([row])[0][1]
            probs[z] = prob
        return probs

lgb_zodiac_model = ZodiacLightGBM()

# ---------- 全自动在线调整器 ----------
class OnlineAdjuster:
    """全自动在线调整：根据近期表现自动修正关键策略参数及权重"""
    def __init__(self, conn=None):
        self.conn = conn
        self.single_temperature = 0.5
        self.four_boost_strength = 0.4
        self.w_single = 0.15
        self.w_two = 0.20
        self.w_three = 0.325
        self.w_four = 0.325

    def load_state(self):
        if self.conn is None:
            return
        state = get_model_state(self.conn, "online_adjust_params")
        if state:
            try:
                params = json.loads(state)
                self.single_temperature = float(params.get("single_temperature", 0.5))
                self.four_boost_strength = float(params.get("four_boost_strength", 0.4))
                self.w_single = float(params.get("w_single", 0.15))
                self.w_two = float(params.get("w_two", 0.20))
                self.w_three = float(params.get("w_three", 0.325))
                self.w_four = float(params.get("w_four", 0.325))
            except:
                pass

    def save_state(self):
        if self.conn is None:
            return
        params = {
            "single_temperature": self.single_temperature,
            "four_boost_strength": self.four_boost_strength,
            "w_single": self.w_single,
            "w_two": self.w_two,
            "w_three": self.w_three,
            "w_four": self.w_four
        }
        set_model_state(self.conn, "online_adjust_params", json.dumps(params))

    def adjust(self):
        if self.conn is None:
            return

        one_rep = get_recent_single_zodiac_report(self.conn, lookback=10)
        two_rep = get_recent_two_zodiac_report(self.conn, lookback=10)
        three_rep = get_recent_three_zodiac_report(self.conn, lookback=10)
        four_rep = get_recent_four_zodiac_report(self.conn, lookback=10)

        rates = {
            'single': one_rep.get('hit_rate', 0.0),
            'two': two_rep.get('hit_rate', 0.0),
            'three': three_rep.get('hit_rate', 0.0),
            'four': four_rep.get('hit_rate', 0.0)
        }
        max_miss = {
            'single': one_rep.get('max_miss_streak', 0),
            'two': two_rep.get('max_miss_streak', 0),
            'three': three_rep.get('max_miss_streak', 0),
            'four': four_rep.get('max_miss_streak', 0)
        }

        if rates['single'] < 0.50 or max_miss['single'] > 2:
            self.single_temperature = min(1.5, self.single_temperature + 0.1)
        else:
            self.single_temperature = max(0.3, self.single_temperature - 0.05)

        if max_miss['four'] > 4 or rates['four'] < 0.30:
            self.four_boost_strength = min(1.0, self.four_boost_strength + 0.08)
        else:
            self.four_boost_strength = max(0.2, self.four_boost_strength - 0.03)

        weights = {
            'single': self.w_single,
            'two': self.w_two,
            'three': self.w_three,
            'four': self.w_four
        }

        targets = {
            'single': (0.60, 2),
            'two': (0.80, 2),
            'three': (0.45, 4),
            'four': (0.40, 4)
        }

        for key in weights:
            target_rate, target_miss = targets[key]
            rate_diff = rates[key] - target_rate
            miss_diff = target_miss - max_miss[key]
            score = rate_diff * 0.6 + miss_diff * 0.1
            if score > 0:
                weights[key] *= 0.97
            else:
                weights[key] *= 1.05

        total = sum(weights.values())
        for key in weights:
            weights[key] = weights[key] / total

        for key in weights:
            if weights[key] < 0.10:
                weights[key] = 0.10
            elif weights[key] > 0.45:
                weights[key] = 0.45

        total = sum(weights.values())
        for key in weights:
            weights[key] = weights[key] / total

        self.w_single = weights['single']
        self.w_two = weights['two']
        self.w_three = weights['three']
        self.w_four = weights['four']

        print(f"[在线调整] 温度={self.single_temperature:.2f}, 增强={self.four_boost_strength:.2f}, "
              f"权重: 一肖={self.w_single:.3f}, 二肖={self.w_two:.3f}, 三肖={self.w_three:.3f}, 特五生肖={self.w_four:.3f}")

        if max_miss['four'] >= 1 and rates['four'] < 0.35:
            self.w_four = min(0.45, self.w_four * 1.08)
            self.four_boost_strength = min(1.5, self.four_boost_strength + 0.1)

        if max_miss['four'] >= 1:
            self.w_four = min(0.45, self.w_four * 1.08)
            self.four_boost_strength = min(1.5, self.four_boost_strength + 0.10)

        self.save_state()

online_adjuster = OnlineAdjuster()
_WEIGHT_PROTECTION_PRINTED: set[str] = set()
_PROTECTION_PRINT_COUNTER = 0

# =================== 数据库操作 ===================
@dataclass
class DrawRecord:
    issue_no: str
    draw_date: str
    numbers: List[int]
    special_number: int

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def connect_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS draws (
            issue_no TEXT PRIMARY KEY,
            draw_date TEXT NOT NULL,
            numbers_json TEXT NOT NULL,
            special_number INTEGER NOT NULL,
            source TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS prediction_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            issue_no TEXT NOT NULL,
            strategy TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'PENDING',
            hit_count INTEGER,
            hit_rate REAL,
            hit_count_10 INTEGER,
            hit_rate_10 REAL,
            hit_count_14 INTEGER,
            hit_rate_14 REAL,
            hit_count_20 INTEGER,
            hit_rate_20 REAL,
            special_hit INTEGER,
            created_at TEXT NOT NULL,
            reviewed_at TEXT,
            UNIQUE(issue_no, strategy)
        );

        CREATE TABLE IF NOT EXISTS prediction_picks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            pick_type TEXT NOT NULL DEFAULT 'MAIN',
            number INTEGER NOT NULL,
            rank INTEGER NOT NULL,
            score REAL NOT NULL,
            reason TEXT NOT NULL,
            UNIQUE(run_id, number),
            FOREIGN KEY(run_id) REFERENCES prediction_runs(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS prediction_pools (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            pool_size INTEGER NOT NULL,
            numbers_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(run_id, pool_size),
            FOREIGN KEY(run_id) REFERENCES prediction_runs(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS model_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS special_picks_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            issue_no TEXT NOT NULL,
            picks_json TEXT NOT NULL,
            hit_count INTEGER,
            special_hit INTEGER,
            created_at TEXT NOT NULL,
            UNIQUE(issue_no)
        );

        CREATE TABLE IF NOT EXISTS strategy_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            issue_no TEXT NOT NULL,
            strategy TEXT NOT NULL,
            main_hit_count INTEGER NOT NULL,
            special_hit INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(issue_no, strategy)
        );
    """)
    _ensure_migrations(conn)
    conn.commit()

def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r["name"] == column for r in rows)

def _ensure_migrations(conn: sqlite3.Connection) -> None:
    if not _column_exists(conn, "prediction_picks", "pick_type"):
        conn.execute("ALTER TABLE prediction_picks ADD COLUMN pick_type TEXT NOT NULL DEFAULT 'MAIN'")
    if not _column_exists(conn, "prediction_runs", "special_hit"):
        conn.execute("ALTER TABLE prediction_runs ADD COLUMN special_hit INTEGER")
    if not _column_exists(conn, "prediction_runs", "hit_count_10"):
        conn.execute("ALTER TABLE prediction_runs ADD COLUMN hit_count_10 INTEGER")
    if not _column_exists(conn, "prediction_runs", "hit_rate_10"):
        conn.execute("ALTER TABLE prediction_runs ADD COLUMN hit_rate_10 REAL")
    if not _column_exists(conn, "prediction_runs", "hit_count_14"):
        conn.execute("ALTER TABLE prediction_runs ADD COLUMN hit_count_14 INTEGER")
    if not _column_exists(conn, "prediction_runs", "hit_rate_14"):
        conn.execute("ALTER TABLE prediction_runs ADD COLUMN hit_rate_14 REAL")
    if not _column_exists(conn, "prediction_runs", "hit_count_20"):
        conn.execute("ALTER TABLE prediction_runs ADD COLUMN hit_count_20 INTEGER")
    if not _column_exists(conn, "prediction_runs", "hit_rate_20"):
        conn.execute("ALTER TABLE prediction_runs ADD COLUMN hit_rate_20 REAL")

def get_model_state(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute("SELECT value FROM model_state WHERE key = ?", (key,)).fetchone()
    return str(row["value"]) if row else None

def set_model_state(conn: sqlite3.Connection, key: str, value: str) -> None:
    now = utc_now()
    conn.execute(
        """INSERT INTO model_state(key, value, updated_at) VALUES (?, ?, ?)
           ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at""",
        (key, value, now),
    )

def _pick(row: Dict[str, str], keys: Sequence[str]) -> str:
    for k in keys:
        if k in row and str(row[k]).strip():
            return str(row[k]).strip()
    return ""

def _parse_date(date_text: str) -> Optional[str]:
    text = date_text.strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(text).strftime("%Y-%m-%d")
    except ValueError:
        return None

def _parse_numbers(value: str) -> List[int]:
    out: List[int] = []
    for token in value.replace("，", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            n = int(token)
        except ValueError:
            continue
        if 1 <= n <= 49:
            out.append(n)
    return out

# =================== API 解析（香港六合彩） ===================
def parse_hk_mark_six_from_api(payload: dict) -> List[DrawRecord]:
    records: List[DrawRecord] = []
    lottery_list = payload.get("lottery_data", [])
    if not isinstance(lottery_list, list):
        return records

    hk_data = None
    hk_names = {"香港六合彩", "香港彩", "HK六合彩", "HK Mark Six"}
    for item in lottery_list:
        if isinstance(item, dict) and str(item.get("name", "")).strip() in hk_names:
            hk_data = item
            break
    if not hk_data:
        return records

    history_list = hk_data.get("history", [])
    if history_list and isinstance(history_list, list):
        for line in history_list:
            match = re.match(r"(\d{7})\s*期[：:]\s*([\d,]+)", line)
            if not match:
                continue
            expect_raw = match.group(1)
            numbers_str = match.group(2)
            num_list = _parse_numbers(numbers_str)
            if len(num_list) < 7:
                continue
            main_numbers = num_list[:6]
            special = num_list[6]

            if len(expect_raw) >= 7:
                year = expect_raw[2:4]
                seq = str(int(expect_raw[4:]))
                issue_no = f"{year}/{seq.zfill(3)}"
            else:
                issue_no = expect_raw

            draw_date = _parse_date(hk_data.get("openTime", "").split()[0]) if hk_data.get("openTime") else None
            if not draw_date:
                draw_date = "2026-01-01"
            records.append(DrawRecord(
                issue_no=issue_no,
                draw_date=draw_date,
                numbers=main_numbers,
                special_number=special,
            ))
    else:
        expect_raw = str(hk_data.get("expect", ""))
        numbers_raw = hk_data.get("openCode") or hk_data.get("numbers")
        if numbers_raw:
            if isinstance(numbers_raw, str):
                num_list = _parse_numbers(numbers_raw)
            elif isinstance(numbers_raw, list):
                num_list = [int(x) for x in numbers_raw if str(x).isdigit()]
            else:
                num_list = []
            if len(num_list) >= 7:
                main_numbers = num_list[:6]
                special = num_list[6]
                if len(expect_raw) >= 7:
                    year = expect_raw[2:4]
                    seq = str(int(expect_raw[4:]))
                    issue_no = f"{year}/{seq.zfill(3)}"
                else:
                    issue_no = expect_raw
                draw_date = _parse_date(hk_data.get("openTime", "").split()[0]) if hk_data.get("openTime") else None
                if draw_date:
                    records.append(DrawRecord(
                        issue_no=issue_no,
                        draw_date=draw_date,
                        numbers=main_numbers,
                        special_number=special,
                    ))

    dedup: Dict[str, DrawRecord] = {}
    for r in records:
        dedup[r.issue_no] = r
    return sorted(dedup.values(), key=lambda r: (r.draw_date, r.issue_no))

def fetch_hk_mark_six_records(
    timeout: int = API_TIMEOUT_DEFAULT,
    retries: int = API_RETRIES_DEFAULT,
    backoff_seconds: float = API_RETRY_BACKOFF_SECONDS,
) -> List[DrawRecord]:
    req = Request(
        HK_MARK_SIX_API_URL,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; hk-local/1.0)",
            "Accept": "application/json",
        },
    )
    attempts = max(1, int(retries))
    last_error: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            with urlopen(req, timeout=int(timeout)) as resp:
                raw = resp.read().decode("utf-8-sig")
                payload = json.loads(raw)
                records = parse_hk_mark_six_from_api(payload)
                if not records:
                    raise RuntimeError("香港六合彩数据解析失败，请检查API返回格式")
                return records
        except (TimeoutError, socket.timeout, URLError, json.JSONDecodeError, RuntimeError) as exc:
            last_error = exc
            if attempt >= attempts:
                break
            delay = backoff_seconds * (2 ** (attempt - 1))
            print(f"[sync] API attempt {attempt}/{attempts} failed: {exc}. retry in {delay:.1f}s", flush=True)
            time.sleep(delay)
    raise RuntimeError(f"香港API请求失败，已重试 {attempts} 次。last_error={last_error}")

def fetch_hk_mark_six_recent_records(
    limit: int = 120,
    timeout: int = API_TIMEOUT_DEFAULT,
    retries: int = API_RETRIES_DEFAULT,
    backoff_seconds: float = API_RETRY_BACKOFF_SECONDS,
) -> List[DrawRecord]:
    records = fetch_hk_mark_six_records(timeout=timeout, retries=retries, backoff_seconds=backoff_seconds)
    if limit > 0:
        records = records[-int(limit):]
    return records

def upsert_draw(conn: sqlite3.Connection, record: DrawRecord, source: str) -> str:
    now = utc_now()
    existing = conn.execute("SELECT issue_no FROM draws WHERE issue_no = ?", (record.issue_no,)).fetchone()
    if existing:
        conn.execute(
            """UPDATE draws SET draw_date = ?, numbers_json = ?, special_number = ?, source = ?, updated_at = ? WHERE issue_no = ?""",
            (record.draw_date, json.dumps(record.numbers), record.special_number, source, now, record.issue_no),
        )
        return "updated"
    conn.execute(
        """INSERT INTO draws(issue_no, draw_date, numbers_json, special_number, source, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (record.issue_no, record.draw_date, json.dumps(record.numbers), record.special_number, source, now, now),
    )
    return "inserted"

def sync_from_records(conn: sqlite3.Connection, records: List[DrawRecord], source: str) -> Tuple[int, int, int]:
    inserted, updated = 0, 0
    for r in records:
        result = upsert_draw(conn, r, source)
        if result == "inserted":
            inserted += 1
        else:
            updated += 1
    conn.commit()
    return len(records), inserted, updated

def has_any_draw(conn: sqlite3.Connection) -> bool:
    row = conn.execute("SELECT 1 FROM draws LIMIT 1").fetchone()
    return row is not None

def parse_issue(issue_no: str) -> Optional[Tuple[str, int, int]]:
    parts = issue_no.split("/")
    if len(parts) != 2:
        return None
    year_s, seq_s = parts
    if not (year_s.isdigit() and seq_s.isdigit()):
        return None
    return year_s, int(seq_s), len(seq_s)

def issue_sort_key(issue_no: str) -> Optional[int]:
    parsed = parse_issue(issue_no)
    if not parsed:
        return None
    year_s, seq, _ = parsed
    return int(year_s) * 1000 + seq

def build_issue(year_s: str, seq: int, width: int) -> str:
    return f"{year_s}/{str(seq).zfill(width)}"

def next_issue(issue_no: str) -> str:
    parsed = parse_issue(issue_no)
    if not parsed:
        return issue_no
    year, seq, width = parsed
    return f"{year}/{str(seq + 1).zfill(width)}"

def missing_issues_since_latest(conn: sqlite3.Connection, incoming: List[DrawRecord]) -> List[str]:
    latest_row = conn.execute("SELECT issue_no FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT 1").fetchone()
    if not latest_row:
        return []
    latest_issue = str(latest_row["issue_no"])
    latest_parsed = parse_issue(latest_issue)
    latest_key = issue_sort_key(latest_issue)
    if not latest_parsed or latest_key is None:
        return []
    incoming_set = {r.issue_no for r in incoming}
    incoming_keys = [issue_sort_key(r.issue_no) for r in incoming if issue_sort_key(r.issue_no) is not None]
    if not incoming_keys:
        return []
    max_key = max(incoming_keys)
    if max_key <= latest_key:
        return []
    year_s, seq, width = latest_parsed
    missing: List[str] = []
    probe_key = latest_key
    probe_year = int(year_s)
    probe_seq = seq
    while probe_key < max_key:
        probe_seq += 1
        if probe_seq > 366:
            probe_year += 1
            probe_seq = 1
            width = 3
        issue = build_issue(str(probe_year).zfill(len(year_s)), probe_seq, width)
        probe_key = probe_year * 1000 + probe_seq
        if issue not in incoming_set:
            exists = conn.execute("SELECT 1 FROM draws WHERE issue_no = ? LIMIT 1", (issue,)).fetchone()
            if not exists:
                missing.append(issue)
    return missing

def load_recent_draws(conn: sqlite3.Connection, limit: int = 3) -> List[List[int]]:
    rows = conn.execute(
        "SELECT numbers_json FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT ?", (limit,)
    ).fetchall()
    return [json.loads(r["numbers_json"]) for r in rows]

# =================== 特征工程函数 ===================
def _normalize(score_map: Dict[int, float]) -> Dict[int, float]:
    values = list(score_map.values())
    mn, mx = min(values), max(values)
    if mx == mn:
        return {k: 0.0 for k in score_map}
    return {k: (v - mn) / (mx - mn) for k, v in score_map.items()}

def _freq_map(draws: List[List[int]]) -> Dict[int, float]:
    freq = {n: 0.0 for n in ALL_NUMBERS}
    for draw in draws:
        for n in draw:
            freq[n] += 1.0
    return freq

def _omission_map(draws: List[List[int]]) -> Dict[int, float]:
    omission = {n: float(len(draws) + 1) for n in ALL_NUMBERS}
    for i, draw in enumerate(draws):
        for n in draw:
            omission[n] = min(omission[n], float(i + 1))
    return omission

def _momentum_map(draws: List[List[int]]) -> Dict[int, float]:
    m = {n: 0.0 for n in ALL_NUMBERS}
    for i, draw in enumerate(draws):
        w = 1.0 / (1.0 + i)
        for n in draw:
            m[n] += w
    return m

def _pair_affinity_map(draws: List[List[int]], window: int = 3) -> Dict[int, float]:
    pair_count: Dict[Tuple[int, int], int] = {}
    for draw in draws[:window]:
        s = sorted(draw)
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                key = (s[i], s[j])
                pair_count[key] = pair_count.get(key, 0) + 1
    social = {n: 0.0 for n in ALL_NUMBERS}
    for (a, b), c in pair_count.items():
        social[a] += float(c)
        social[b] += float(c)
    return social

def _zone_heat_map(draws: List[List[int]], window: int = 3) -> Dict[int, float]:
    zone_counts = [0.0] * 5
    w = draws[:window]
    if not w:
        return {n: 0.0 for n in ALL_NUMBERS}
    for draw in w:
        for n in draw:
            zone = min(4, (n - 1) // 10)
            zone_counts[zone] += 1.0
    expected = 6.0 * len(w) / 5.0
    zone_score = [expected - c for c in zone_counts]
    return {n: zone_score[min(4, (n - 1) // 10)] for n in ALL_NUMBERS}

def _adjacency_compensation_map(draws: List[List[int]], window: int = 5) -> Dict[int, float]:
    adjacency = {n: 0.0 for n in ALL_NUMBERS}
    w = draws[:window]
    if not w:
        return adjacency
    for idx, draw in enumerate(w):
        recency_w = 1.0 / (1.0 + idx * 0.35)
        for base in draw:
            for delta, bonus in ((1, 1.6), (2, 1.0), (3, 0.5)):
                for candidate in (base - delta, base + delta):
                    if 1 <= candidate <= 49:
                        adjacency[candidate] += bonus * recency_w
    return adjacency

def _pick_top_six(scores: Dict[int, float], reason: str) -> List[Tuple[int, int, float, str]]:
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    picked: List[Tuple[int, float]] = []
    for n, s in ranked:
        if len(picked) == 6:
            break
        proposal = [pn for pn, _ in picked] + [n]
        odd_count = sum(1 for x in proposal if x % 2 == 1)
        if len(proposal) >= 4 and (odd_count == 0 or odd_count == len(proposal)):
            continue
        zone_counts: Dict[int, int] = {}
        for x in proposal:
            z = min(4, (x - 1) // 10)
            zone_counts[z] = zone_counts.get(z, 0) + 1
        if any(c >= 4 for c in zone_counts.values()):
            continue
        picked.append((n, s))
    while len(picked) < 6:
        for n, s in ranked:
            if n not in [pn for pn, _ in picked]:
                picked.append((n, s))
                break

    target_low, target_high = 95, 205
    top6 = [n for n, _ in picked[:6]]
    total = sum(top6)
    if not (target_low <= total <= target_high):
        for i in range(5, -1, -1):
            replaced = False
            for alt_n, alt_s in ranked:
                if alt_n in top6:
                    continue
                candidate = list(top6)
                candidate[i] = alt_n
                csum = sum(candidate)
                if target_low <= csum <= target_high:
                    picked[i] = (alt_n, alt_s)
                    top6 = candidate
                    replaced = True
                    break
            if replaced:
                break

    return [(n, idx + 1, s, f"{reason} score={s:.4f}") for idx, (n, s) in enumerate(picked)]

def _default_mined_config() -> Dict[str, float]:
    return {
        "window": 6.0,
        "w_freq": 0.30,
        "w_omit": 0.45,
        "w_mom": 0.15,
        "w_pair": 0.00,
        "w_zone": 0.10,
        "w_adj": 0.10,
        "special_bonus": 0.10,
    }

def _candidate_mined_configs() -> List[Dict[str, float]]:
    windows = [6, 9, 12, 18]
    weight_triplets = [
        (0.50, 0.30, 0.20),
        (0.45, 0.35, 0.20),
        (0.40, 0.40, 0.20),
        (0.35, 0.45, 0.20),
        (0.30, 0.50, 0.20),
        (0.60, 0.20, 0.20),
        (0.20, 0.60, 0.20),
        (0.40, 0.30, 0.30),
        (0.30, 0.40, 0.30),
    ]
    pair_zone_sets = [
        (0.00, 0.00),
        (0.05, 0.05),
        (0.10, 0.00),
        (0.00, 0.10),
    ]
    out: List[Dict[str, float]] = []
    for w in windows:
        for wf, wo, wm in weight_triplets:
            for wp, wz in pair_zone_sets:
                out.append({
                    "window": float(w),
                    "w_freq": wf,
                    "w_omit": wo,
                    "w_mom": wm,
                    "w_pair": wp,
                    "w_zone": wz,
                    "w_adj": 0.10,
                    "special_bonus": 0.10,
                })
    return out

def _apply_weight_config(
    draws: List[List[int]],
    config: Dict[str, float],
    reason: str,
) -> Tuple[List[Tuple[int, int, float, str]], int, float, Dict[int, float]]:
    window_size = int(config.get("window", FEATURE_WINDOW_DEFAULT))
    window = draws[: max(3, window_size)]
    freq = _normalize(_freq_map(window))
    omission = _normalize(_omission_map(window))
    momentum = _normalize(_momentum_map(window))
    pair = _normalize(_pair_affinity_map(window, window=min(3, len(window))))
    zone = _normalize(_zone_heat_map(window, window=min(3, len(window))))
    adjacency = _normalize(_adjacency_compensation_map(window, window=min(5, len(window))))

    w_freq = float(config.get("w_freq", 0.40))
    w_omit = float(config.get("w_omit", 0.28))
    w_mom = float(config.get("w_mom", 0.16))
    w_pair = float(config.get("w_pair", 0.00))
    w_zone = float(config.get("w_zone", 0.06))
    w_adj = float(config.get("w_adj", 0.10))

    scores: Dict[int, float] = {}
    for n in ALL_NUMBERS:
        scores[n] = (
            freq[n] * w_freq
            + omission[n] * w_omit
            + momentum[n] * w_mom
            + pair[n] * w_pair
            + zone[n] * w_zone
            + adjacency[n] * w_adj
        )

    main_picks = _pick_top_six(scores, reason)
    main_set = {n for n, _, _, _ in main_picks}
    special_candidates = [(n, s) for n, s in sorted(scores.items(), key=lambda x: x[1], reverse=True) if n not in main_set]
    if not special_candidates:
        special_candidates = [(n, s) for n, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
    special_number, special_score = special_candidates[0]
    return main_picks, special_number, special_score, scores

def mine_pattern_config_from_rows(rows: Sequence[sqlite3.Row]) -> Dict[str, float]:
    if len(rows) < 3:
        return _default_mined_config()

    candidates = _candidate_mined_configs()
    best_cfg = _default_mined_config()
    best_score = -1.0

    min_history = 3
    eval_span = min(500, len(rows) - min_history)
    start = max(min_history, len(rows) - eval_span)

    parsed_main = [json.loads(r["numbers_json"]) for r in rows]
    parsed_special = [int(r["special_number"]) for r in rows]

    for cfg in candidates:
        score_sum = 0.0
        count = 0
        for i in range(start, len(rows)):
            hist_start = max(0, i - int(cfg["window"]))
            history_desc = [parsed_main[j] for j in range(i - 1, hist_start - 1, -1)]
            if len(history_desc) < min_history:
                continue
            picks, special, _, _ = _apply_weight_config(history_desc, cfg, "规律挖掘")
            picked_main = [n for n, _, _, _ in picks]
            win_main = set(parsed_main[i])
            hit_count = len([n for n in picked_main if n in win_main])
            special_hit = 1 if int(special) == parsed_special[i] else 0
            score_sum += hit_count / 6.0 + float(cfg.get("special_bonus", 0.10)) * special_hit
            count += 1

        if count == 0:
            continue
        score = score_sum / count
        if score > best_score:
            best_score = score
            best_cfg = cfg

    return best_cfg

def ensure_mined_pattern_config(conn: sqlite3.Connection, force: bool = False) -> Dict[str, float]:
    if not force:
        cached = get_model_state(conn, MINED_CONFIG_KEY)
        if cached:
            try:
                obj = json.loads(cached)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

    rows = _draws_ordered_asc(conn)
    cfg = mine_pattern_config_from_rows(rows)
    set_model_state(conn, MINED_CONFIG_KEY, json.dumps(cfg, ensure_ascii=False))
    conn.commit()
    return cfg

def _rank_vote_score(score_maps: Sequence[Dict[int, float]]) -> Dict[int, float]:
    votes = {n: 0.0 for n in ALL_NUMBERS}
    for m in score_maps:
        ranked = sorted(m.items(), key=lambda x: x[1], reverse=True)
        for rank, (n, _) in enumerate(ranked):
            votes[n] += float(49 - rank)
    return _normalize(votes)

def _build_candidate_pools(scores: Dict[int, float], main6: List[int]) -> Dict[int, List[int]]:
    ranked = [n for n, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
    main_unique = []
    for n in main6:
        if n not in main_unique:
            main_unique.append(n)

    rest = [n for n in ranked if n not in main_unique]
    pool10 = main_unique + rest[: max(0, 10 - len(main_unique))]
    pool14 = main_unique + rest[: max(0, 14 - len(main_unique))]
    pool20 = main_unique + rest[: max(0, 20 - len(main_unique))]
    return {6: main_unique[:6], 10: pool10[:10], 14: pool14[:14], 20: pool20[:20]}

def _pool_hit_count(pool_numbers: Sequence[int], winning: set[int]) -> int:
    return len([n for n in pool_numbers if n in winning])

def _save_prediction_pools(conn: sqlite3.Connection, run_id: int, pools: Dict[int, List[int]]) -> None:
    conn.execute("DELETE FROM prediction_pools WHERE run_id = ?", (run_id,))
    now = utc_now()
    for pool_size, numbers in pools.items():
        conn.execute(
            """INSERT INTO prediction_pools(run_id, pool_size, numbers_json, created_at) VALUES (?, ?, ?, ?)""",
            (run_id, int(pool_size), json.dumps(numbers), now),
        )

def get_pool_numbers_for_run(conn: sqlite3.Connection, run_id: int, pool_size: int = 6) -> List[int]:
    row = conn.execute(
        "SELECT numbers_json FROM prediction_pools WHERE run_id = ? AND pool_size = ?",
        (run_id, int(pool_size)),
    ).fetchone()
    if not row:
        return []
    try:
        nums = json.loads(row["numbers_json"])
    except Exception:
        return []
    valid_numbers: List[int] = []
    for n in nums:
        if isinstance(n, int) and 1 <= n <= 49:
            valid_numbers.append(n)
            continue
        if isinstance(n, str) and n.isdigit():
            parsed = int(n)
            if 1 <= parsed <= 49:
                valid_numbers.append(parsed)
    return valid_numbers

def get_adaptive_strategy_window(strategy: str, conn: sqlite3.Connection) -> int:
    base = STRATEGY_BASE_WINDOWS.get(strategy, FEATURE_WINDOW_DEFAULT)
    health = get_strategy_health(conn, window=20)
    h = health.get(strategy, {})
    recent_avg = float(h.get("recent_avg_hit", 0.65))
    cold_streak = int(h.get("cold_streak", 0))

    if strategy == "cold_rebound_v1":
        rows = conn.execute("SELECT numbers_json FROM draws ORDER BY draw_date DESC LIMIT 60").fetchall()
        all_nums = []
        for r in rows:
            all_nums.extend(json.loads(r["numbers_json"]))
        freq = Counter(all_nums)
        cold_count = sum(1 for n in ALL_NUMBERS if freq.get(n, 0) == 0)
        if cold_count >= 5:
            return min(20, base + 8)

    if recent_avg >= 0.95:
        return max(5, base - 2)
    elif recent_avg >= 0.80:
        return max(6, base - 1)
    elif recent_avg <= 0.55 or cold_streak >= 4:
        return min(15, base + 3)
    elif recent_avg <= 0.65:
        return min(13, base + 2)
    return base    
# ========== 偏态检测函数（强制偏态模式） ==========
def detect_bias(conn: sqlite3.Connection, window: int = 10) -> Tuple[float, Dict[str, float]]:
    """强制偏态模式：固定偏态系数 0.75"""
    return 0.75, {
        "forced": True,
        "zone_bias": 0.75,
        "parity_bias": 0.70,
        "hot_cold_bias": 0.70,
        "zone_dist": [0]*5,
        "odd_ratio": 0.5
    }

def adjust_weights_for_bias(weights: Dict[str, float], bias_score: float) -> Dict[str, float]:
    if bias_score < BIAS_THRESHOLD:
        return weights
    adjusted = weights.copy()
    cold_boost = 1 + BIAS_ADJUSTMENT * bias_score
    adjusted["cold_rebound_v1"] = weights.get("cold_rebound_v1", 0.15) * cold_boost
    adjusted["hot_v1"] = weights.get("hot_v1", 0.15) * (1 - BIAS_ADJUSTMENT * bias_score * 0.7)
    adjusted["momentum_v1"] = weights.get("momentum_v1", 0.15) * (1 - BIAS_ADJUSTMENT * bias_score * 0.5)
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: v / total for k, v in adjusted.items()}
    return adjusted

# ========== 特别号 v4 增强版 ==========
def _generate_special_number_v4(
    conn: sqlite3.Connection,
    main_pool: List[int],
    issue_no: str
) -> Tuple[int, float, List[int]]:
    latest_row = conn.execute(
        "SELECT issue_no, draw_date FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT 1"
    ).fetchone()
    if not latest_row:
        return 1, 0.0, [2, 3, 4]

    latest_issue = str(latest_row["issue_no"])
    latest_date = str(latest_row["draw_date"])
    recent_specials = [int(r["special_number"]) for r in conn.execute(
        "SELECT special_number FROM draws WHERE draw_date < ? OR (draw_date = ? AND issue_no < ?) ORDER BY draw_date DESC, issue_no DESC LIMIT 80",
        (latest_date, latest_date, latest_issue),
    ).fetchall()]

    latest_sp_row = conn.execute(
        "SELECT special_number FROM draws ORDER BY draw_date DESC LIMIT 1"
    ).fetchone()
    latest_sp = int(latest_sp_row["special_number"]) if latest_sp_row else None

    prev_special = recent_specials[0] if recent_specials else None
    main_set = set(main_pool)

    omission = {n: 80 for n in ALL_NUMBERS}
    for i, num in enumerate(recent_specials):
        omission[num] = min(omission.get(num, 80), i + 1)

    tail_counter = Counter([n % 10 for n in recent_specials[:40]])
    coldest_tail = min(tail_counter.keys(), key=lambda t: tail_counter[t]) if tail_counter else 0

    scores = {}
    for n in ALL_NUMBERS:
        if n == latest_sp or n in main_set:
            continue
        score = 0.0
        if prev_special is not None:
            diff = abs(n - prev_special)
            if diff == 1:
                score += 8.8
            elif diff == 2:
                score += 6.6
            elif diff == 3:
                score += 3.8
        if recent_specials and n == recent_specials[0]:
            score *= 0.75
        if recent_specials and n in recent_specials[:3]:
            score *= 0.80
        omit = omission.get(n, 80)
        if omit >= 10:
            score += 8.0 * (omit / 15.0)
        elif omit >= 6:
            score += 4.5
        if n % 10 == coldest_tail:
            score += 5.0
        scores[n] = max(0.0, score)

    if not scores:
        return 1, 0.0, [2, 3, 4]

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best = ranked[0][0]
    confidence = min(1.0, ranked[0][1] / 29.0)
    defenses = [n for n, _ in ranked[1:] if n not in main_set][:3]
    return best, round(confidence, 3), defenses

# ========== 三中三相关逻辑（集成策略v3.1） ==========
def _ensemble_strategy_v3_1(draws, mined_config, strategy_weights, conn, issue_no, history_records=None):
    sub_scores = {}
    for sub in ["hot_v1", "cold_rebound_v1", "momentum_v1", "balanced_v1", "pattern_mined_v1"]:
        _, _, _, score_map = generate_strategy(draws, sub, conn=conn, issue_no=issue_no, history_records=history_records)
        sub_scores[sub] = score_map
    voted = {n: 0.0 for n in ALL_NUMBERS}
    for score_map in sub_scores.values():
        for n, v in score_map.items():
            voted[n] += float(v)
    voted = _normalize(voted)
    main_picked = _pick_top_six(voted, "集成投票v3.1")
    main_set = {n for n, _, _, _ in main_picked}
    special_number, confidence, _ = _generate_special_number_v4(conn, main_set, issue_no)
    return main_picked, special_number, confidence, voted

def generate_strategy(
    draws: List[List[int]],
    strategy: str,
    mined_config: Optional[Dict[str, float]] = None,
    strategy_weights: Optional[Dict[str, float]] = None,
    conn: Optional[sqlite3.Connection] = None,
    issue_no: Optional[str] = None,
    history_records: Optional[List[DrawRecord]] = None,
) -> Tuple[List[Tuple[int, int, float, str]], int, float, Dict[int, float]]:

    window_size = STRATEGY_BASE_WINDOWS.get(strategy, FEATURE_WINDOW_DEFAULT)
    strategy_draws = draws[:window_size] if len(draws) > window_size else draws

    if strategy == "hot_v1":
        return _apply_weight_config(
            strategy_draws,
            {"window": float(window_size), "w_freq": 0.74, "w_omit": 0.06, "w_mom": 0.14, "w_zone": 0.06, "w_adj": 0.10},
            "热号策略"
        )
    elif strategy == "cold_rebound_v1":
        return _apply_weight_config(
            strategy_draws,
            {"window": float(window_size), "w_freq": 0.06, "w_omit": 0.62, "w_mom": 0.22, "w_zone": 0.05, "w_adj": 0.12},
            "冷号回补"
        )
    elif strategy == "momentum_v1":
        return _apply_weight_config(
            strategy_draws,
            {"window": float(window_size), "w_freq": 0.10, "w_omit": 0.05, "w_mom": 0.75, "w_zone": 0.05, "w_adj": 0.05},
            "近期动量"
        )
    elif strategy == "balanced_v1":
        return _apply_weight_config(
            strategy_draws,
            {
                "window": float(window_size),
                "w_freq": 0.36,
                "w_omit": 0.26,
                "w_mom": 0.18,
                "w_pair": 0.05,
                "w_zone": 0.06,
                "w_adj": 0.14,
            },
            "组合策略",
        )
    elif strategy == "pattern_mined_v1":
        cfg = mined_config or _default_mined_config()
        cfg["window"] = float(window_size)
        return _apply_weight_config(strategy_draws, cfg, "规律挖掘")
    elif strategy in ("ensemble_v2", "ensemble_v3"):
        if strategy_weights is None:
            strategy_weights = get_strategy_weights(conn, window=WEIGHT_WINDOW_DEFAULT, for_issue=issue_no) if conn else {s: 1.0/len(STRATEGY_IDS) for s in STRATEGY_IDS}
        if conn is None:
            raise ValueError("ensemble_v2/v3 requires database connection")
        if issue_no is None:
            raise ValueError("ensemble_v2/v3 requires issue_no parameter")
        return _ensemble_strategy_v3_1(strategy_draws, mined_config, strategy_weights, conn, issue_no)

    return _apply_weight_config(
        strategy_draws,
        {
            "window": float(window_size),
            "w_freq": 0.40,
            "w_omit": 0.30,
            "w_mom": 0.20,
            "w_pair": 0.05,
            "w_zone": 0.05,
        },
        "组合策略",
    )

def generate_predictions(conn: sqlite3.Connection, issue_no: Optional[str] = None) -> str:
    row = conn.execute("SELECT issue_no FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT 1").fetchone()
    if not row:
        raise RuntimeError("No draws found. Run sync/bootstrap first.")
    target_issue = issue_no or next_issue(row["issue_no"])
    hp = HistoryProvider(conn)
    draws_records = hp.get_recent_draws_for_prediction(limit=FEATURE_WINDOW_DEFAULT)
    draws = [rec.numbers for rec in draws_records]
    if len(draws) < 3:
        raise RuntimeError("Need at least 3 draws to generate predictions.")
    mined_cfg = ensure_mined_pattern_config(conn, force=False)

    strategy_weights = get_strategy_weights(conn, window=WEIGHT_WINDOW_DEFAULT, for_issue=issue_no)

    for strategy in STRATEGY_IDS:
        now = utc_now()
        existing = conn.execute(
            "SELECT id FROM prediction_runs WHERE issue_no = ? AND strategy = ?",
            (target_issue, strategy),
        ).fetchone()
        if existing:
            run_id = existing["id"]
            conn.execute(
                """
                   UPDATE prediction_runs
                   SET status='PENDING', hit_count=NULL, hit_rate=NULL,
                       hit_count_10=NULL, hit_rate_10=NULL,
                       hit_count_14=NULL, hit_rate_14=NULL,
                       hit_count_20=NULL, hit_rate_20=NULL,
                       special_hit=NULL, reviewed_at=NULL, created_at=?
                   WHERE id=?
                   """,
                (now, run_id),
            )
            conn.execute("DELETE FROM prediction_picks WHERE run_id = ?", (run_id,))
        else:
            cur = conn.execute(
                """
                   INSERT INTO prediction_runs(issue_no, strategy, status, created_at)
                   VALUES (?, ?, 'PENDING', ?)
                   """,
                (target_issue, strategy, now),
            )
            run_id = cur.lastrowid

        picks, special_number, special_score, score_map = generate_strategy(
            draws, strategy, mined_config=mined_cfg, strategy_weights=strategy_weights, conn=conn, issue_no=target_issue, history_records=draws_records
        )
        main_numbers = [n for n, _, _, _ in picks]
        conn.executemany(
            """
               INSERT INTO prediction_picks(run_id, pick_type, number, rank, score, reason)
               VALUES (?, ?, ?, ?, ?, ?)
               """,
            [(run_id, "MAIN", n, rank, score, reason) for n, rank, score, reason in picks]
            + [(run_id, "SPECIAL", special_number, 1, special_score, "特别号候选")],
        )
        pools = _build_candidate_pools(score_map, main_numbers)
        _save_prediction_pools(conn, int(run_id), pools)
        conn.commit()
    return target_issue

def _draws_ordered_asc(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    return conn.execute(
        "SELECT issue_no, draw_date, numbers_json, special_number FROM draws ORDER BY draw_date ASC, issue_no ASC"
    ).fetchall()

def run_historical_backtest(
    conn: sqlite3.Connection,
    min_history: int = 3,
    rebuild: bool = False,
    progress_every: int = 20,
    max_issues: int = BACKTEST_ISSUES_DEFAULT,
) -> Tuple[int, int]:
    draws = _draws_ordered_asc(conn)
    if len(draws) <= min_history:
        return 0, 0

    if max_issues > 0 and len(draws) > max_issues + min_history:
        draws = draws[-(max_issues + min_history):]
        print(f"[backtest] 限制回测范围为最近 {max_issues} 期（实际处理 {len(draws) - min_history} 期）", flush=True)

    if rebuild:
        conn.execute("""
               DELETE FROM prediction_pools
               WHERE run_id IN (SELECT id FROM prediction_runs WHERE issue_no IN (SELECT issue_no FROM draws))
               """)
        conn.execute("""
               DELETE FROM prediction_runs
               WHERE issue_no IN (SELECT issue_no FROM draws)
               """)
        conn.execute("DELETE FROM strategy_performance WHERE issue_no IN (SELECT issue_no FROM draws)")
        conn.commit()

    issues_processed = 0
    runs_processed = 0
    total_targets = len(draws) - min_history
    started_at = time.time()

    mined_cfg_cache: Dict[int, Dict[str, float]] = {}
    print(
        f"[backtest] start: total_issues={total_targets}, strategies_per_issue={len(STRATEGY_IDS)}, rebuild={rebuild}",
        flush=True,
    )

    for i in range(min_history, len(draws)):
        target = draws[i]
        issue_no = str(target["issue_no"])
        existing = conn.execute(
            """
               SELECT COUNT(*) AS c
               FROM prediction_runs
               WHERE issue_no = ? AND status = 'REVIEWED'
               """,
            (issue_no,),
        ).fetchone()
        if existing and int(existing["c"]) >= len(STRATEGY_IDS):
            continue

        history_desc = [
            json.loads(draws[j]["numbers_json"])
            for j in range(i - 1, max(-1, i - FEATURE_WINDOW_DEFAULT - 1), -1)
        ]
        if len(history_desc) < min_history:
            continue
        winning_main = set(json.loads(target["numbers_json"]))
        winning_special = int(target["special_number"])

        for strategy in STRATEGY_IDS:
            mined_cfg = None
            if strategy == "pattern_mined_v1":
                bucket = i // 3
                if bucket not in mined_cfg_cache:
                    mined_cfg_cache[bucket] = mine_pattern_config_from_rows(draws[:i])
                mined_cfg = mined_cfg_cache[bucket]
            main_picks, special_number, special_score, score_map = generate_strategy(
                history_desc,
                strategy,
                mined_config=mined_cfg,
                conn=conn,
                issue_no=issue_no,
            )
            picked_main = [n for n, _, _, _ in main_picks]
            pools = _build_candidate_pools(score_map, picked_main)
            hit_count = len([n for n in picked_main if n in winning_main])
            hit_rate = round(hit_count / 6.0, 4)
            hit_count_10 = _pool_hit_count(pools[10], winning_main)
            hit_count_14 = _pool_hit_count(pools[14], winning_main)
            hit_count_20 = _pool_hit_count(pools[20], winning_main)
            hit_rate_10 = round(hit_count_10 / 6.0, 4)
            hit_rate_14 = round(hit_count_14 / 6.0, 4)
            hit_rate_20 = round(hit_count_20 / 6.0, 4)
            special_hit = 1 if special_number == winning_special else 0

            now = utc_now()
            row = conn.execute(
                "SELECT id FROM prediction_runs WHERE issue_no = ? AND strategy = ?",
                (issue_no, strategy),
            ).fetchone()
            if row:
                run_id = int(row["id"])
                conn.execute(
                    """
                       UPDATE prediction_runs
                       SET status='REVIEWED', hit_count=?, hit_rate=?,
                           hit_count_10=?, hit_rate_10=?,
                           hit_count_14=?, hit_rate_14=?,
                           hit_count_20=?, hit_rate_20=?,
                           special_hit=?, created_at=?, reviewed_at=?
                       WHERE id=?
                       """,
                    (
                        hit_count,
                        hit_rate,
                        hit_count_10,
                        hit_rate_10,
                        hit_count_14,
                        hit_rate_14,
                        hit_count_20,
                        hit_rate_20,
                        special_hit,
                        now,
                        now,
                        run_id,
                    ),
                )
                conn.execute("DELETE FROM prediction_picks WHERE run_id = ?", (run_id,))
            else:
                cur = conn.execute(
                    """
                       INSERT INTO prediction_runs(
                         issue_no, strategy, status, hit_count, hit_rate,
                         hit_count_10, hit_rate_10, hit_count_14, hit_rate_14, hit_count_20, hit_rate_20,
                         special_hit, created_at, reviewed_at
                       )
                       VALUES (?, ?, 'REVIEWED', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       """,
                    (
                        issue_no,
                        strategy,
                        hit_count,
                        hit_rate,
                        hit_count_10,
                        hit_rate_10,
                        hit_count_14,
                        hit_rate_14,
                        hit_count_20,
                        hit_rate_20,
                        special_hit,
                        now,
                        now,
                    ),
                )
                run_id = int(cur.lastrowid)

            conn.executemany(
                """
                   INSERT INTO prediction_picks(run_id, pick_type, number, rank, score, reason)
                   VALUES (?, ?, ?, ?, ?, ?)
                   """,
                [(run_id, "MAIN", n, rank, score, reason) for n, rank, score, reason in main_picks]
                + [(run_id, "SPECIAL", special_number, 1, special_score, "特别号候选")],
            )
            _save_prediction_pools(conn, run_id, pools)

            conn.execute(
                """
                   INSERT OR REPLACE INTO strategy_performance(issue_no, strategy, main_hit_count, special_hit, created_at)
                   VALUES (?, ?, ?, ?, ?)
                   """,
                (issue_no, strategy, hit_count, special_hit, now),
            )
            runs_processed += 1

        issues_processed += 1
        if (
            issues_processed == 1
            or issues_processed == total_targets
            or (progress_every > 0 and issues_processed % progress_every == 0)
        ):
            elapsed = max(time.time() - started_at, 1e-9)
            pct = (issues_processed / total_targets) * 100.0 if total_targets > 0 else 100.0
            speed = issues_processed / elapsed
            eta = ((total_targets - issues_processed) / speed) if speed > 0 else 0.0
            print(
                f"[backtest] progress: {issues_processed}/{total_targets} ({pct:.1f}%), "
                f"runs={runs_processed}, elapsed={elapsed:.0f}s, eta={eta:.0f}s",
                flush=True,
            )

    conn.commit()
    return issues_processed, runs_processed

def review_issue(conn: sqlite3.Connection, issue_no: str) -> int:
    draw = conn.execute("SELECT numbers_json, special_number FROM draws WHERE issue_no = ?", (issue_no,)).fetchone()
    if not draw:
        return 0
    winning = set(json.loads(draw["numbers_json"]))
    winning_special = int(draw["special_number"])
    runs = conn.execute(
        "SELECT id, strategy FROM prediction_runs WHERE issue_no = ? AND status = 'PENDING'",
        (issue_no,),
    ).fetchall()
    count = 0
    for run in runs:
        run_id = run["id"]
        picks = conn.execute(
            "SELECT pick_type, number FROM prediction_picks WHERE run_id = ?",
            (run_id,),
        ).fetchall()
        main_picked = [p["number"] for p in picks if p["pick_type"] in (None, "MAIN")]
        special_picked = [p["number"] for p in picks if p["pick_type"] == "SPECIAL"]
        pool10 = get_pool_numbers_for_run(conn, int(run_id), 10) or main_picked
        pool14 = get_pool_numbers_for_run(conn, int(run_id), 14) or main_picked
        pool20 = get_pool_numbers_for_run(conn, int(run_id), 20) or main_picked
        hit_count = len([n for n in main_picked if n in winning])
        hit_rate = round(hit_count / 6.0, 4)
        hit_count_10 = _pool_hit_count(pool10, winning)
        hit_count_14 = _pool_hit_count(pool14, winning)
        hit_count_20 = _pool_hit_count(pool20, winning)
        hit_rate_10 = round(hit_count_10 / 6.0, 4)
        hit_rate_14 = round(hit_count_14 / 6.0, 4)
        hit_rate_20 = round(hit_count_20 / 6.0, 4)
        special_hit = 1 if (special_picked and special_picked[0] == winning_special) else 0
        conn.execute(
            """
               UPDATE prediction_runs
               SET status='REVIEWED', hit_count=?, hit_rate=?,
                   hit_count_10=?, hit_rate_10=?,
                   hit_count_14=?, hit_rate_14=?,
                   hit_count_20=?, hit_rate_20=?,
                   special_hit=?, reviewed_at=?
               WHERE id=?
               """,
            (
                hit_count,
                hit_rate,
                hit_count_10,
                hit_rate_10,
                hit_count_14,
                hit_rate_14,
                hit_count_20,
                hit_rate_20,
                special_hit,
                utc_now(),
                run_id,
            ),
        )
        conn.execute(
            """
               INSERT OR REPLACE INTO strategy_performance(issue_no, strategy, main_hit_count, special_hit, created_at)
               VALUES (?, ?, ?, ?, ?)
               """,
            (issue_no, run["strategy"], hit_count, special_hit, utc_now()),
        )
        count += 1
    conn.commit()
    return count

def review_latest(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT issue_no FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT 1").fetchone()
    if not row:
        return 0
    return review_issue(conn, row["issue_no"])

def _fmt_num(n: int) -> str:
    return str(n).zfill(2)

def get_latest_draw(conn: sqlite3.Connection) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT issue_no, draw_date, numbers_json, special_number FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT 1"
    ).fetchone()

def get_pending_runs(conn: sqlite3.Connection, limit: int = 12) -> List[sqlite3.Row]:
    return conn.execute(
        "SELECT id, issue_no, strategy, created_at FROM prediction_runs WHERE status='PENDING' ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()

def get_review_stats(conn: sqlite3.Connection, window: Optional[int] = None) -> List[sqlite3.Row]:
    if window:
        recent_issues = conn.execute(
            "SELECT issue_no FROM draws ORDER BY draw_date DESC LIMIT ?", (window,)
        ).fetchall()
        issue_list = [r['issue_no'] for r in recent_issues]
        if not issue_list:
            return []
        placeholders = ','.join('?' for _ in issue_list)
        query = f"""
            SELECT
              strategy,
              COUNT(*) AS c,
              AVG(hit_count) AS avg_hit,
              AVG(hit_rate) AS avg_rate,
              AVG(hit_count_10) AS avg_hit_10,
              AVG(hit_rate_10) AS avg_rate_10,
              AVG(hit_count_14) AS avg_hit_14,
              AVG(hit_rate_14) AS avg_rate_14,
              AVG(hit_count_20) AS avg_hit_20,
              AVG(hit_rate_20) AS avg_rate_20,
              AVG(COALESCE(special_hit, 0)) AS special_rate,
              AVG(CASE WHEN hit_count >= 1 THEN 1.0 ELSE 0.0 END) AS hit1_rate,
              AVG(CASE WHEN hit_count >= 2 THEN 1.0 ELSE 0.0 END) AS hit2_rate
            FROM prediction_runs
            WHERE status='REVIEWED' AND issue_no IN ({placeholders})
            GROUP BY strategy
            ORDER BY avg_rate DESC
        """
        rows = conn.execute(query, issue_list).fetchall()
    else:
        rows = conn.execute("""
            SELECT
              strategy,
              COUNT(*) AS c,
              AVG(hit_count) AS avg_hit,
              AVG(hit_rate) AS avg_rate,
              AVG(hit_count_10) AS avg_hit_10,
              AVG(hit_rate_10) AS avg_rate_10,
              AVG(hit_count_14) AS avg_hit_14,
              AVG(hit_rate_14) AS avg_rate_14,
              AVG(hit_count_20) AS avg_hit_20,
              AVG(hit_rate_20) AS avg_rate_20,
              AVG(COALESCE(special_hit, 0)) AS special_rate,
              AVG(CASE WHEN hit_count >= 1 THEN 1.0 ELSE 0.0 END) AS hit1_rate,
              AVG(CASE WHEN hit_count >= 2 THEN 1.0 ELSE 0.0 END) AS hit2_rate
            FROM prediction_runs
            WHERE status='REVIEWED'
            GROUP BY strategy
            ORDER BY avg_rate DESC
        """).fetchall()
    out = []
    for r in rows:
        strat = str(r["strategy"])
        ordered = conn.execute(
            """
            SELECT hit_count
            FROM prediction_runs
            WHERE status='REVIEWED' AND strategy = ?
            ORDER BY reviewed_at ASC, created_at ASC, id ASC
            """,
            (strat,),
        ).fetchall()
        miss_streak = 0
        max_miss_streak = 0
        for x in ordered:
            if int(x["hit_count"] or 0) == 0:
                miss_streak += 1
                max_miss_streak = max(max_miss_streak, miss_streak)
            else:
                miss_streak = 0
        row_dict = dict(r)
        row_dict["max_miss_streak"] = max_miss_streak
        out.append(row_dict)
    return out

def get_recent_reviews(conn: sqlite3.Connection, limit: int = 20) -> List[sqlite3.Row]:
    return conn.execute(
        """
           SELECT issue_no, strategy, hit_count, hit_rate, COALESCE(special_hit, 0) AS special_hit, reviewed_at
           FROM prediction_runs
           WHERE status='REVIEWED'
           ORDER BY reviewed_at DESC
           LIMIT ?
           """,
        (limit,),
    ).fetchall()

def get_draw_issues_desc(conn: sqlite3.Connection, limit: int = 300) -> List[str]:
    rows = conn.execute(
        "SELECT issue_no FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [str(r["issue_no"]) for r in rows]

def get_reviewed_runs_for_issue(conn: sqlite3.Connection, issue_no: str) -> List[sqlite3.Row]:
    return conn.execute(
        """
           SELECT
             id, issue_no, strategy,
             hit_count, hit_rate,
             hit_count_10, hit_rate_10,
             hit_count_14, hit_rate_14,
             hit_count_20, hit_rate_20,
             COALESCE(special_hit, 0) AS special_hit
           FROM prediction_runs
           WHERE issue_no = ? AND status = 'REVIEWED'
           ORDER BY strategy ASC
           """,
        (issue_no,),
    ).fetchall()

def get_picks_for_run(conn: sqlite3.Connection, run_id: int) -> Tuple[List[int], Optional[int]]:
    picks = conn.execute(
        "SELECT pick_type, number FROM prediction_picks WHERE run_id = ? ORDER BY rank ASC",
        (run_id,),
    ).fetchall()
    mains = [p["number"] for p in picks if p["pick_type"] in (None, "MAIN")]
    specials = [p["number"] for p in picks if p["pick_type"] == "SPECIAL"]
    return mains, (specials[0] if specials else None)

def backfill_missing_special_picks(conn: sqlite3.Connection) -> int:
    draws = load_recent_draws(conn, FEATURE_WINDOW_DEFAULT)
    if len(draws) < 3:
        return 0
    mined_cfg = ensure_mined_pattern_config(conn, force=False)

    runs = conn.execute(
        """
           SELECT id, strategy, issue_no
           FROM prediction_runs
           WHERE status='PENDING'
           """
    ).fetchall()
    patched = 0
    for run in runs:
        run_id = int(run["id"])
        existing_special = conn.execute(
            "SELECT 1 FROM prediction_picks WHERE run_id = ? AND pick_type = 'SPECIAL' LIMIT 1",
            (run_id,),
        ).fetchone()
        if existing_special:
            continue

        mains = conn.execute(
            "SELECT number FROM prediction_picks WHERE run_id = ? AND (pick_type = 'MAIN' OR pick_type IS NULL)",
            (run_id,),
        ).fetchall()
        main_set = {int(r["number"]) for r in mains}
        strategy_name = str(run["strategy"])
        run_issue = str(run["issue_no"])
        cfg = mined_cfg if strategy_name == "pattern_mined_v1" else None
        _, special_number, special_score, _ = generate_strategy(
            draws,
            strategy_name,
            mined_config=cfg,
            conn=conn,
            issue_no=run_issue,
        )

        if special_number in main_set:
            for n in ALL_NUMBERS:
                if n not in main_set:
                    special_number = n
                    break

        conn.execute(
            """
               INSERT OR IGNORE INTO prediction_picks(run_id, pick_type, number, rank, score, reason)
               VALUES (?, 'SPECIAL', ?, 1, ?, '特别号补齐')
               """,
            (run_id, special_number, float(special_score)),
        )
        patched += 1

    if patched > 0:
        conn.commit()
    return patched

def print_recommendation_sheet(conn: sqlite3.Connection, limit: int = 8) -> None:
    backfill_missing_special_picks(conn)
    rows = get_pending_runs(conn, limit=limit)
    print("\n6/10/14/20 推荐单:")
    if not rows:
        print("  (空)")
        return

    for r in rows:
        mains, special = get_picks_for_run(conn, int(r["id"]))
        pool6 = [int(n) for n in mains]
        pool10 = [int(n) for n in (get_pool_numbers_for_run(conn, int(r["id"]), 10) or pool6)]
        pool14 = [int(n) for n in (get_pool_numbers_for_run(conn, int(r["id"]), 14) or pool6)]
        pool20 = [int(n) for n in (get_pool_numbers_for_run(conn, int(r["id"]), 20) or pool6)]
        strategy_name = STRATEGY_LABELS.get(r["strategy"], r["strategy"])
        special_text = _fmt_num(special) if special is not None else "--"
        p6 = " ".join(_fmt_num(n) for n in pool6)
        p10 = " ".join(_fmt_num(n) for n in pool10)
        p14 = " ".join(_fmt_num(n) for n in pool14)
        p20 = " ".join(_fmt_num(n) for n in pool20)
        print(f"  [{r['issue_no']}] {strategy_name}")
        print(f"    6号池 : {p6} | 特别号: {special_text}")
        print(f"    10号池: {p10} | 特别号: {special_text}")
        print(f"    14号池: {p14} | 特别号: {special_text}")
        print(f"    20号池: {p20} | 特别号: {special_text}")

# ========== 动态权重相关函数 ==========
def get_strategy_weights(conn, window=WEIGHT_WINDOW_DEFAULT, for_issue: Optional[str] = None):
    if for_issue:
        target = conn.execute("SELECT draw_date FROM draws WHERE issue_no = ?", (for_issue,)).fetchone()
        if target:
            rows = conn.execute("""
                SELECT strategy,
                       AVG(main_hit_count) as avg_hit,
                       AVG(COALESCE(main_hit_count, 0) / 6.0) as avg_rate,
                       AVG(CASE WHEN main_hit_count >= 1 THEN 1.0 ELSE 0.0 END) AS hit1_rate,
                       AVG(CASE WHEN main_hit_count >= 2 THEN 1.0 ELSE 0.0 END) AS hit2_rate
                FROM strategy_performance
                WHERE issue_no IN (
                    SELECT issue_no FROM draws
                    WHERE draw_date < ? OR (draw_date = ? AND issue_no < ?)
                    ORDER BY draw_date DESC, issue_no DESC LIMIT ?
                )
                GROUP BY strategy
            """, (target["draw_date"], target["draw_date"], for_issue, window)).fetchall()
        else:
            rows = []
    else:
        rows = conn.execute("""
            SELECT strategy,
                   AVG(main_hit_count) as avg_hit,
                   AVG(COALESCE(main_hit_count, 0) / 6.0) as avg_rate,
                   AVG(CASE WHEN main_hit_count >= 1 THEN 1.0 ELSE 0.0 END) AS hit1_rate,
                   AVG(CASE WHEN main_hit_count >= 2 THEN 1.0 ELSE 0.0 END) AS hit2_rate
            FROM strategy_performance
            WHERE issue_no IN (
                SELECT issue_no FROM draws ORDER BY draw_date DESC LIMIT ?
            )
            GROUP BY strategy
        """, (window,)).fetchall()

    baseline = 0.6
    weights = {s: baseline for s in STRATEGY_IDS}
    protection_msgs = []

    for r in rows:
        strategy = str(r["strategy"])
        avg_hit = float(r["avg_hit"] or 0.0)
        if strategy in weights:
            weights[strategy] = max(avg_hit, baseline)

    health = get_strategy_health(conn, window=HEALTH_WINDOW_DEFAULT)
    for strategy, h in health.items():
        if strategy not in weights:
            continue
        hit1_rate = float(h.get("hit1_rate", 0.0))
        cold_streak = int(h.get("cold_streak", 0))
        shrink = 1.0
        if strategy == "cold_rebound_v1":
            if cold_streak >= 2:
                shrink *= 0.85
        if strategy == "pattern_mined_v1":
            if cold_streak >= 5:
                shrink *= 0.65
            elif cold_streak >= 1:
                shrink *= 0.82
            weights[strategy] = max(0.12, weights[strategy] * shrink)
            if cold_streak >= 1:
                protection_msgs.append(f"[保护] 规律挖掘连挂 {cold_streak}，权重已平滑下调")
        else:
            if hit1_rate < 0.52:
                shrink *= 0.90
            if cold_streak >= 2:
                shrink *= 0.78
            if strategy == "momentum_v1":
                avg_rate_6 = float(h.get("recent_avg_hit", 0.0))
                if avg_rate_6 < 0.15:
                    shrink *= 0.80
                    protection_msgs.append(f"[保护] 动量策略6码命中率过低({avg_rate_6*100:.1f}%)，下调权重")
            weights[strategy] = max(0.10, weights[strategy] * shrink)

    long_rows = conn.execute("""
        SELECT strategy, AVG(main_hit_count) as avg_hit_long
        FROM strategy_performance
        WHERE issue_no IN (
            SELECT issue_no FROM draws ORDER BY draw_date DESC LIMIT 50
        )
        GROUP BY strategy
    """).fetchall()
    long_dict = {r["strategy"]: r["avg_hit_long"] for r in long_rows}

    for strategy in STRATEGY_IDS:
        short_avg = weights[strategy]
        long_avg = float(long_dict.get(strategy, short_avg) or short_avg)
        combined = 0.6 * short_avg + 0.4 * max(long_avg, baseline)
        weights[strategy] = combined

    total = sum(weights.values())
    for msg in protection_msgs:
        print(msg, flush=True)
    return {k: round(v / total, 4) for k, v in weights.items()}

def get_trio_weights(conn: sqlite3.Connection, window: int = WEIGHT_WINDOW_DEFAULT) -> Tuple[float, float, float]:
    rows = conn.execute("""
           SELECT strategy, AVG(main_hit_count) as avg_hit
           FROM strategy_performance
           WHERE strategy IN ('momentum_v1', 'hot_v1', 'cold_rebound_v1')
           AND issue_no IN (SELECT issue_no FROM draws ORDER BY draw_date DESC LIMIT ?)
           GROUP BY strategy
       """, (window,)).fetchall()
    stats = {r["strategy"]: r["avg_hit"] for r in rows}
    w_mom = max(float(stats.get('momentum_v1', 0.0) or 0.0), 0.6)
    w_hot = max(float(stats.get('hot_v1', 0.0) or 0.0), 0.6)
    w_cold = max(float(stats.get('cold_rebound_v1', 0.0) or 0.0), 0.6)
    total = w_mom + w_hot + w_cold
    return w_mom/total, w_hot/total, w_cold/total

def get_strategy_health(conn: sqlite3.Connection, window: int = HEALTH_WINDOW_DEFAULT) -> Dict[str, Dict[str, float]]:
    health: Dict[str, Dict[str, float]] = {}
    for strategy in STRATEGY_IDS:
        rows = conn.execute(
            """
               SELECT hit_count
               FROM prediction_runs
               WHERE strategy = ? AND status = 'REVIEWED'
               ORDER BY reviewed_at DESC
               LIMIT ?
               """,
            (strategy, window),
        ).fetchall()
        if not rows:
            health[strategy] = {
                "samples": 0.0,
                "recent_avg_hit": 0.0,
                "hit1_rate": 0.0,
                "hit2_rate": 0.0,
                "cold_streak": 0.0,
            }
            continue

        hit_counts = [int(r["hit_count"] or 0) for r in rows]
        samples = len(hit_counts)
        hit1_rate = sum(1 for x in hit_counts if x >= 1) / samples
        hit2_rate = sum(1 for x in hit_counts if x >= 2) / samples
        recent_avg_hit = sum(hit_counts) / samples

        cold_streak = 0
        for x in hit_counts:
            if x == 0:
                cold_streak += 1
            else:
                break

        health[strategy] = {
            "samples": float(samples),
            "recent_avg_hit": float(recent_avg_hit),
            "hit1_rate": float(hit1_rate),
            "hit2_rate": float(hit2_rate),
            "cold_streak": float(cold_streak),
        }
    return health    
# ========== 生肖相关函数（优化版） ==========
def get_zodiac_by_number(number: int) -> str:
    for zodiac, nums in ZODIAC_MAP.items():
        if number in nums:
            return zodiac
    return "马"

def _get_previous_issue(conn: sqlite3.Connection, current_issue: str) -> Optional[str]:
    """获取当前期号的上一期"""
    row = conn.execute(
        """
           SELECT issue_no FROM draws 
           WHERE draw_date < (SELECT draw_date FROM draws WHERE issue_no = ?)
              OR (draw_date = (SELECT draw_date FROM draws WHERE issue_no = ?) AND issue_no < ?)
           ORDER BY draw_date DESC, issue_no DESC 
           LIMIT 1
           """,
        (current_issue, current_issue, current_issue)
    ).fetchone()
    return row["issue_no"] if row else None

def _check_two_zodiac_hit(conn: sqlite3.Connection, issue_no: str) -> bool:
    """检查指定期号的双生肖推荐是否命中"""
    draw = conn.execute(
        "SELECT numbers_json, special_number FROM draws WHERE issue_no = ?",
        (issue_no,)
    ).fetchone()
    if not draw:
        return False

    winning_main = json.loads(draw["numbers_json"])
    winning_special = int(draw["special_number"])
    winning_zodiacs = {get_zodiac_by_number(n) for n in winning_main}
    winning_zodiacs.add(get_zodiac_by_number(winning_special))

    rows = conn.execute(
        """
           SELECT numbers_json, special_number FROM draws 
           WHERE draw_date < (SELECT draw_date FROM draws WHERE issue_no = ?)
              OR (draw_date = (SELECT draw_date FROM draws WHERE issue_no = ?) AND issue_no < ?)
           ORDER BY draw_date DESC, issue_no DESC 
           LIMIT ?
           """,
        (issue_no, issue_no, issue_no, 16)
    ).fetchall()
    if not rows:
        return False

    zodiac_scores = _build_zodiac_scores_from_rows(rows, decay=0.08)
    ranked = sorted(zodiac_scores.items(), key=lambda x: (-x[1], x[0]))
    picks = [ranked[0][0], ranked[1][0]] if len(ranked) >= 2 else ["马", "蛇"]

    return any(z in winning_zodiacs for z in picks)

def _zodiac_omission_map(rows: Sequence[sqlite3.Row]) -> Dict[str, int]:
    """计算每个生肖最近一次出现的期数距离（遗漏值）"""
    zodiac_omission = {z: len(rows) + 1 for z in ZODIAC_MAP.keys()}
    for i, row in enumerate(rows):
        numbers = json.loads(row["numbers_json"])
        special = int(row["special_number"])
        appeared_zodiacs = set()
        for n in numbers:
            appeared_zodiacs.add(get_zodiac_by_number(n))
        appeared_zodiacs.add(get_zodiac_by_number(special))
        for z in appeared_zodiacs:
            if zodiac_omission[z] > i + 1:
                zodiac_omission[z] = i + 1
    return zodiac_omission

def _zodiac_omission_map_from_rows(rows: Sequence[sqlite3.Row]) -> Dict[str, int]:
    if not rows:
        return {z: 0 for z in ZODIAC_MAP}
    return _zodiac_omission_map(rows)

def _build_zodiac_scores_from_rows(rows: Sequence[sqlite3.Row], decay: float = 0.08) -> Dict[str, float]:
    zodiac_scores: Dict[str, float] = {z: 0.0 for z in ZODIAC_MAP.keys()}
    omission_map = _zodiac_omission_map(rows)
    for idx, row in enumerate(rows):
        recency_w = 1.0 / (1.0 + idx * decay)
        numbers = json.loads(row["numbers_json"])
        for n in numbers:
            zodiac_scores[get_zodiac_by_number(int(n))] += 1.0 * recency_w
        zodiac_scores[get_zodiac_by_number(int(row["special_number"]))] += 1.8 * recency_w
    for z in zodiac_scores:
        omit = omission_map.get(z, len(rows))
        if omit >= 8:
            zodiac_scores[z] += 4.0
        elif omit >= 3:
            zodiac_scores[z] += omit / 6.0
    return zodiac_scores

def _zodiac_sequence_features(rows: List[sqlite3.Row], max_len=30) -> Dict[str, Dict]:
    """提取生肖序列特征：转移概率、间隔、最近5期频率"""
    zodiac_seq = []
    for row in rows[:max_len]:
        nums = json.loads(row["numbers_json"])
        sp = int(row["special_number"])
        z_set = set()
        for n in nums:
            z = get_zodiac_by_number(n)
            if z not in z_set:
                zodiac_seq.append(z)
                z_set.add(z)
        z_sp = get_zodiac_by_number(sp)
        if z_sp not in z_set:
            zodiac_seq.append(z_sp)
    trans = defaultdict(Counter)
    for i in range(len(zodiac_seq)-1):
        a, b = zodiac_seq[i], zodiac_seq[i+1]
        trans[a][b] += 1
    last_pos = {}
    intervals = defaultdict(list)
    for i, z in enumerate(zodiac_seq):
        if z in last_pos:
            intervals[z].append(i - last_pos[z])
        last_pos[z] = i
    avg_interval = {z: (sum(intervals[z])/len(intervals[z]) if intervals[z] else 99.0) for z in ZODIAC_MAP}
    last_seen = {z: len(zodiac_seq) - pos for z, pos in last_pos.items()} if last_pos else {z: 99 for z in ZODIAC_MAP}
    last5 = zodiac_seq[-5:] if len(zodiac_seq) >= 5 else zodiac_seq
    recent5_cnt = Counter(last5)
    result = {}
    for z in ZODIAC_MAP:
        incoming = sum(1 for a, cnt in trans.items() for b in cnt if b == z) / max(1, sum(len(c) for c in trans.values()))
        result[z] = {"avg_interval": avg_interval.get(z, 99.0), "last_seen": last_seen.get(z, 99), "incoming_prob": incoming, "recent5_cnt": recent5_cnt.get(z, 0)}
    return result

def smooth_penalty(miss: int, threshold: int, weight: float) -> float:
    """平滑连空惩罚，轻微指数增长避免爆炸"""
    if miss <= threshold:
        return 0.0
    excess = miss - threshold
    return weight * (excess ** 1.1)

def _markov_two_predictor(hist_rows, target_zodiac):
    """二阶马尔可夫链：基于前两期特别号生肖预测本期特别号生肖"""
    specials = [get_zodiac_by_number(int(r["special_number"])) for r in hist_rows]
    if len(specials) < 3:
        return 0.0
    
    trans = defaultdict(lambda: defaultdict(float))
    for i in range(len(specials) - 2):
        prev_pair = (specials[i], specials[i+1])
        next_z = specials[i+2]
        trans[prev_pair][next_z] += 1
    
    for pair in trans:
        total = sum(trans[pair].values())
        for z in trans[pair]:
            trans[pair][z] /= total
    
    last_pair = (specials[0], specials[1]) if len(specials) >= 2 else (None, None)
    
    if last_pair in trans and target_zodiac in trans[last_pair]:
        return trans[last_pair][target_zodiac]
    else:
        return 0.0

def get_hot_tails(rows, top_n=5):
    """获取最近 N 期最热的尾数"""
    tail_counter = Counter()
    for r in rows:
        nums = json.loads(r["numbers_json"])
        sp = int(r["special_number"])
        for n in nums:
            tail_counter[int(n) % 10] += 1
        tail_counter[sp % 10] += 2
    return {t for t, _ in tail_counter.most_common(top_n)}

# ========== 强力生肖预测函数 ==========
def predict_strong_single(hist_rows, params):
    """强力一肖：动态融合 + 锐化采样"""
    from math import exp
    import random

    scores_rule = _compute_twoinone_score(
        hist_rows,
        recent_window=params.get("single_recent_window", 12),
        special_boost=params.get("single_special_boost", 2.0),
        interval_penalty=params.get("single_interval_penalty", 0.5),
        recent5_boost=params.get("single_recent5_boost", 1.2),
        cold_window=25
    )

    markov_scores = {}
    for z in ZODIAC_MAP:
        markov_scores[z] = _markov_two_predictor(hist_rows, z)
    mx_m = max(markov_scores.values()) if markov_scores else 1
    markov_norm = {z: s / mx_m for z, s in markov_scores.items()} if mx_m > 0 else markov_scores

    mx_r = max(scores_rule.values()) if scores_rule else 1
    scores_rule_norm = {z: s / mx_r for z, s in scores_rule.items()}

    combined = {}
    for z in ZODIAC_MAP:
        combined[z] = scores_rule_norm[z] * 0.6 + markov_norm.get(z, 0) * 0.4

    sorted_items = sorted(combined.items(), key=lambda x: -x[1])
    top = sorted_items[:3]
    raw_scores = [s for _, s in top]
    if max(raw_scores) == min(raw_scores):
        return top[0][0]

    online_adjuster.load_state()
    temp = online_adjuster.single_temperature
    calibrated = [math.exp(s / temp) for s in raw_scores]
    total = sum(calibrated)
    probs = [c / total for c in calibrated]
    zodiacs = [z for z, _ in top]
    seed = hash(params.get("seed", 42)) % 10000
    random.seed(seed)
    return random.choices(zodiacs, weights=probs, k=1)[0]

def predict_strong_two(hist_rows, params):
    """强力二肖：动态融合 + 遗漏保护"""
    scores = _compute_twoinone_score(
        hist_rows,
        recent_window=params.get("two_recent_window", 12),
        special_boost=params.get("two_special_boost", 1.8),
        interval_penalty=params.get("two_interval_penalty", 0.6),
        recent5_boost=params.get("two_recent5_boost", 0.9),
        cold_window=params.get("two_cold_omit_threshold", 5) + 10
    )
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    picks = [ranked[0][0], ranked[1][0]] if len(ranked) >= 2 else ["马", "蛇"]

    if len(ranked) >= 2 and (ranked[0][1] - ranked[1][1]) < 0.1:
        omission = _zodiac_omission_map(hist_rows)
        coldest = max(omission, key=omission.get)
        if coldest not in picks:
            picks[1] = coldest

    return picks

def predict_strong_three(hist_rows, params):
    """最强三生肖（每期必给）"""
    scores = _build_zodiac_scores_from_rows(hist_rows, decay=0.028)
    
    recent_sp_z = [get_zodiac_by_number(int(r["special_number"])) for r in hist_rows[:18]]
    recent_main_z = []
    for r in hist_rows[:12]:
        recent_main_z.extend(get_zodiac_by_number(int(n)) for n in json.loads(r["numbers_json"]))

    counter_sp = Counter(recent_sp_z)
    counter_main = Counter(recent_main_z)
    
    for z, cnt in counter_sp.items():
        scores[z] += cnt * params.get("three_hot_weight", 3.2)
    for z, cnt in counter_main.items():
        scores[z] += cnt * 1.1

    latest = recent_sp_z[0] if recent_sp_z else "马"
    if latest in ZODIAC_PAIR:
        scores[ZODIAC_PAIR[latest]] += 2.4
    
    omission = _zodiac_omission_map(hist_rows)
    for z, o in omission.items():
        if o >= 5:
            scores[z] += o * 1.15 * params.get("three_omit_weight", 1.0)
    if recent_sp_z:
        latest_sp_z = recent_sp_z[0]
        scores[latest_sp_z] += 3.0
        pair = ZODIAC_PAIR.get(latest_sp_z)
        if pair:
            scores[pair] += 1.5
    hot_tails = get_hot_tails(hist_rows[:25])
    for z, nums in ZODIAC_MAP.items():
        if any(n % 10 in hot_tails for n in nums):
            scores[z] += params.get("three_tail_weight", 1.3)

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [ranked[0][0], ranked[1][0], ranked[2][0]]

def predict_strong_four(hist_rows, params):
    """最强四生肖（特别号专用）- 使用可调参数"""
    recent_special_window = int(params.get("four_recent_special_window", 12))
    cold_omit_threshold = int(params.get("four_cold_omit_threshold", 6))
    cold_boost = float(params.get("four_cold_boost", 2.0))
    transition_weight = float(params.get("four_transition_weight", 1.8))

    scores = {z: 0.0 for z in ZODIAC_MAP}
    specials = [int(r["special_number"]) for r in hist_rows]
    special_zodiacs = [get_zodiac_by_number(sp) for sp in specials]

    for i, z in enumerate(special_zodiacs[:recent_special_window]):
        w = 1.0 / (1.0 + i * 0.12)
        scores[z] += 3.0 * w

    omission = {z: 0 for z in ZODIAC_MAP}
    for i, z in enumerate(special_zodiacs):
        if omission[z] == 0:
            omission[z] = i + 1
    max_omit = max(omission.values()) if omission else 1
    for z, omit in omission.items():
        if omit >= cold_omit_threshold:
            online_adjuster.load_state()
            boost_factor = 1.0 + online_adjuster.four_boost_strength
            scores[z] += (omit / max_omit) ** 1.25 * cold_boost * boost_factor

    trans = defaultdict(Counter)
    for i in range(len(special_zodiacs) - 1):
        a, b = special_zodiacs[i], special_zodiacs[i+1]
        trans[a][b] += 1
    if special_zodiacs:
        last = special_zodiacs[0]
        for next_z, cnt in trans[last].most_common(3):
            scores[next_z] += transition_weight * cnt

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    picks = [ranked[i][0] for i in range(4)]

    if params.get('four_miss_streak', 0) >= 1 and hist_rows:
        latest_z = get_zodiac_by_number(int(hist_rows[0]["special_number"]))
        if latest_z not in picks:
            picks[-1] = latest_z

    return picks

def predict_strong_five(hist_rows, params):
    """特五肖预测 - 强化连空保护版"""
    miss_streak = params.get('four_miss_streak', 0)

    if miss_streak >= 2 and hist_rows:
        omission = _zodiac_omission_map(hist_rows)
        cold_sorted = sorted(omission.items(), key=lambda x: x[1], reverse=True)
        picks = [z for z, _ in cold_sorted[:5]]
        latest_z = get_zodiac_by_number(int(hist_rows[0]["special_number"]))
        if latest_z not in picks:
            picks[-1] = latest_z
        picks = list(dict.fromkeys(picks))
        while len(picks) < 5:
            for z in ZODIAC_MAP:
                if z not in picks:
                    picks.append(z)
                    break
        return picks[:5]

    recent_special_window = int(params.get("four_recent_special_window", 20))
    cold_omit_threshold = int(params.get("four_cold_omit_threshold", 5))
    scores = _compute_special_five_score(hist_rows, recent_special_window, cold_omit_threshold)
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    picks = [ranked[i][0] for i in range(5)]

    omission = _zodiac_omission_map(hist_rows)
    top2_cold = sorted(omission.items(), key=lambda x: x[1], reverse=True)[:2]
    if not any(z in picks for z in [top2_cold[0][0], top2_cold[1][0]]):
        picks[-1] = top2_cold[0][0]

    if miss_streak >= 1 and hist_rows:
        latest_z = get_zodiac_by_number(int(hist_rows[0]["special_number"]))
        if latest_z not in picks:
            picks[-1] = latest_z

    picks = list(dict.fromkeys(picks))
    while len(picks) < 5:
        for z in ZODIAC_MAP:
            if z not in picks:
                picks.append(z)
                break
    return picks[:5]

def _compute_twoinone_score(rows: Sequence[sqlite3.Row], recent_window=12, cold_window=25,
                            special_boost=2.0, interval_penalty=0.5, recent5_boost=1.2) -> Dict[str, float]:
    """增强版一生肖/二生肖评分"""
    scores = {z: 0.0 for z in ZODIAC_MAP}
    omission = _zodiac_omission_map(rows)
    max_omit = max(omission.values()) if omission else 1

    for idx, row in enumerate(rows[:recent_window]):
        w = 1.0 / (1.0 + idx * 0.08)
        nums = json.loads(row["numbers_json"])
        sp = int(row["special_number"])
        for n in nums:
            scores[get_zodiac_by_number(n)] += w * 0.6
        scores[get_zodiac_by_number(sp)] += special_boost * w

    for z, omit in omission.items():
        if omit >= 8:
            scores[z] += (omit / max_omit) ** 1.3 * 3.5
        elif omit >= 4:
            scores[z] += (omit / max_omit) ** 1.1 * 1.8

    seq_feat = _zodiac_sequence_features(list(rows))
    for z, feat in seq_feat.items():
        if feat["last_seen"] > 12:
            scores[z] += 2.2
        if feat["avg_interval"] > 18:
            scores[z] += 1.1
        scores[z] += recent5_boost * feat["recent5_cnt"] * 1.1

    return scores

def _compute_special_four_score(rows: Sequence[sqlite3.Row], recent_special_window=20, cold_omit_threshold=5) -> Dict[str, float]:
    """特别四生肖评分"""
    scores = {z: 0.0 for z in ZODIAC_MAP}
    specials = [int(row["special_number"]) for row in rows]
    seq = [get_zodiac_by_number(sp) for sp in specials]

    for i, z in enumerate(seq[:recent_special_window]):
        w_short = 1.0 / (1.0 + i * 0.07)
        w_mid = 0.68 / (1.0 + i * 0.14)
        scores[z] += 4.1 * w_short + 1.9 * w_mid

    omission = {z: 0 for z in ZODIAC_MAP}
    for i, z in enumerate(seq):
        if omission[z] == 0:
            omission[z] = i + 1
    max_omit = max(omission.values()) if omission else 1

    for z, omit in omission.items():
        if omit >= cold_omit_threshold:
            scores[z] += (omit / max_omit) ** 1.08 * 2.7

    trans1 = defaultdict(Counter)
    for i in range(1, len(seq)):
        p1 = seq[i-1]
        curr = seq[i]
        trans1[p1][curr] += 1
    if seq:
        last = seq[0]
        for nz, cnt in trans1[last].most_common(7):
            scores[nz] += 2.7 * cnt

    return scores

def _compute_special_five_score(rows: Sequence[sqlite3.Row], recent_special_window=20, cold_omit_threshold=5) -> Dict[str, float]:
    """特五肖评分"""
    scores = {z: 0.0 for z in ZODIAC_MAP}
    specials = [int(row["special_number"]) for row in rows]
    seq = [get_zodiac_by_number(sp) for sp in specials]

    for i, z in enumerate(seq[:recent_special_window]):
        w_short = 1.0 / (1.0 + i * 0.055)
        w_mid = 0.75 / (1.0 + i * 0.12)
        scores[z] += 4.6 * w_short + 2.2 * w_mid

    omission = {z: 0 for z in ZODIAC_MAP}
    for i, z in enumerate(seq):
        if omission[z] == 0:
            omission[z] = i + 1
    max_omit = max(omission.values()) if omission else 1

    for z, omit in omission.items():
        if omit >= cold_omit_threshold:
            scores[z] += (omit / max_omit) ** 1.10 * 3.1

    trans1 = defaultdict(Counter)
    for i in range(1, len(seq)):
        p1 = seq[i-1]
        curr = seq[i]
        trans1[p1][curr] += 1
    if seq:
        last = seq[0]
        for nz, cnt in trans1[last].most_common(7):
            scores[nz] += 3.0 * cnt

    return scores

# ========== 对外接口：生肖选择 ==========
def get_two_zodiac_picks(conn, issue_no=None, window: int = 16) -> List[str]:
    """融合规则评分和 LightGBM 概率，并加入连空保护"""
    fw = get_ultimate_two_in_one()
    rule_picks = fw.predict_two_zodiac(conn, issue_no)

    lgb_probs = lgb_zodiac_model.predict_proba(conn, issue_no)
    if lgb_probs is not None:
        combined = {}
        for z in ZODIAC_MAP:
            combined[z] = lgb_probs.get(z, 0.0)
        sorted_combined = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        lgb_picks = [sorted_combined[0][0], sorted_combined[1][0]]
        if rule_picks[0] in lgb_picks:
            picks = [rule_picks[0], lgb_picks[0] if lgb_picks[0] != rule_picks[0] else lgb_picks[1]]
        else:
            picks = [rule_picks[0], lgb_picks[0]]
    else:
        picks = rule_picks

    hist_rows = None
    if issue_no:
        target = conn.execute("SELECT draw_date FROM draws WHERE issue_no = ?", (issue_no,)).fetchone()
        if target:
            hist_rows = conn.execute(
                """SELECT numbers_json, special_number FROM draws
                   WHERE draw_date < ? OR (draw_date = ? AND issue_no < ?)
                   ORDER BY draw_date DESC, issue_no DESC LIMIT 50""",
                (target["draw_date"], target["draw_date"], issue_no)
            ).fetchall()
    if not hist_rows:
        hist_rows = conn.execute(
            "SELECT numbers_json, special_number FROM draws ORDER BY draw_date DESC LIMIT 50"
        ).fetchall()
    omission = _zodiac_omission_map(hist_rows) if hist_rows else {}
    cold_candidates = [z for z, omit in omission.items() if omit >= 5 and z not in picks]
    if cold_candidates:
        longest_cold = max(cold_candidates, key=lambda z: omission[z])
        picks[-1] = longest_cold
        if picks[0] == picks[1]:
            picks[1] = "蛇" if picks[0] != "蛇" else "马"
    return picks

def get_single_zodiac_pick(conn, issue_no=None, window=12):
    """动态权重建模融合一生肖"""
    params = load_best_zodiac_params()
    
    w_rule = float(params.get("single_w_rule", 0.5))
    w_lgb = float(params.get("single_w_lgb", 0.3))
    w_trans = float(params.get("single_w_transformer", 0.2))
    w_markov = float(params.get("single_w_markov", 0.0))
    total_w = w_rule + w_lgb + w_trans + w_markov
    if total_w == 0:
        total_w = 1.0

    if issue_no:
        target = conn.execute("SELECT draw_date FROM draws WHERE issue_no = ?", (issue_no,)).fetchone()
        if target:
            rows = conn.execute(
                """SELECT numbers_json, special_number FROM draws
                   WHERE draw_date < ? OR (draw_date = ? AND issue_no < ?)
                   ORDER BY draw_date DESC, issue_no DESC LIMIT 50""",
                (target["draw_date"], target["draw_date"], issue_no)
            ).fetchall()
        else:
            rows = conn.execute("SELECT numbers_json, special_number FROM draws ORDER BY draw_date DESC LIMIT 50").fetchall()
    else:
        rows = conn.execute("SELECT numbers_json, special_number FROM draws ORDER BY draw_date DESC LIMIT 50").fetchall()
    
    if not rows:
        return "马"

    recent_window = int(params.get("single_recent_window", 12))
    special_boost = float(params.get("single_special_boost", 2.0))
    interval_penalty = float(params.get("single_interval_penalty", 0.5))
    recent5_boost = float(params.get("single_recent5_boost", 1.2))
    scores_rule = _compute_twoinone_score(
        rows, recent_window=recent_window,
        special_boost=special_boost,
        interval_penalty=interval_penalty,
        recent5_boost=recent5_boost,
        cold_window=25
    )

    lgb_probs = lgb_zodiac_model.predict_proba(conn, issue_no)
    transformer_probs = predict_transformer(conn, SCRIPT_DIR / "transformer_sp.pth", issue_no)

    markov_scores = {}
    for z in ZODIAC_MAP:
        markov_scores[z] = _markov_two_predictor(rows, z)
    mx_markov = max(markov_scores.values()) if markov_scores else 1
    markov_norm = {z: s / mx_markov for z, s in markov_scores.items()} if mx_markov > 0 else markov_scores

    mx_rule = max(scores_rule.values()) if scores_rule else 1
    scores_rule_norm = {z: s / mx_rule for z, s in scores_rule.items()}

    combined = {}
    for z in ZODIAC_MAP:
        score = 0.0
        score += scores_rule_norm.get(z, 0) * (w_rule / total_w)
        if lgb_probs:
            score += lgb_probs.get(z, 0) * (w_lgb / total_w)
        if transformer_probs:
            score += transformer_probs.get(z, 0) * (w_trans / total_w)
        score += markov_norm.get(z, 0) * (w_markov / total_w)
        combined[z] = score

    seed = int(issue_no.split("/")[-1]) if issue_no and "/" in issue_no else 42
    best = sharpened_single_pick(combined, top_k=3, temperature=0.5, seed=seed)
    if best is None:
        best = "马"

    if max(combined.values()) < 0.3:
        omission = _zodiac_omission_map(rows)
        best = max(omission, key=omission.get)

    return best

def sharpened_single_pick(scores, top_k=3, temperature=0.5, seed=None):
    if not scores:
        return None
    if seed is not None:
        random.seed(seed)
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    zodiacs = [item[0] for item in sorted_items]
    raw_scores = [item[1] for item in sorted_items]
    if max(raw_scores) == min(raw_scores):
        return zodiacs[0]
    calibrated = [math.exp(s / temperature) for s in raw_scores]
    total = sum(calibrated)
    probs = [c / total for c in calibrated]
    return random.choices(zodiacs, weights=probs, k=1)[0]

def omission_boosted_four_pick(scores, omission_map, omit_threshold=6, boost_strength=0.4):
    boosted_scores = scores.copy()
    max_omit = max(omission_map.values()) if omission_map else 1
    for z in boosted_scores:
        omit = omission_map.get(z, 0)
        if omit >= omit_threshold:
            boost_factor = 1.0 + boost_strength * (omit / max_omit)
            boosted_scores[z] *= boost_factor
    sorted_items = sorted(boosted_scores.items(), key=lambda x: x[1], reverse=True)
    return [z for z, _ in sorted_items[:4]]

def get_hot_cold_zodiacs(conn: sqlite3.Connection, window: int = 12, top_n: int = 3) -> Tuple[List[str], List[str]]:
    rows = conn.execute(
        "SELECT numbers_json, special_number FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT ?",
        (window,)
    ).fetchall()
    if len(rows) < window:
        default = ["马", "蛇", "龙", "兔", "虎", "牛"]
        return default[:top_n], default[-top_n:]
    score_counter: Dict[str, float] = {z: 0.0 for z in ZODIAC_MAP.keys()}
    for idx, row in enumerate(rows):
        recency_w = 1.0 / (1.0 + idx * 0.35)
        numbers = json.loads(row["numbers_json"])
        for n in numbers:
            score_counter[get_zodiac_by_number(n)] += 1.0 * recency_w
        special = row["special_number"]
        score_counter[get_zodiac_by_number(special)] += 1.2 * recency_w
    sorted_by_freq = sorted(score_counter.items(), key=lambda x: x[1], reverse=True)
    hot = [z for z, _ in sorted_by_freq[:top_n]]
    all_zodiacs = list(ZODIAC_MAP.keys())
    cold_candidates = [(z, score_counter.get(z, 0.0)) for z in all_zodiacs]
    cold_candidates.sort(key=lambda x: x[1])
    cold = [z for z, _ in cold_candidates[:top_n]]
    return hot, cold

# 历史回溯辅助
def _get_rows_before_issue(conn, issue_no, limit=60):
    target = conn.execute("SELECT draw_date FROM draws WHERE issue_no = ?", (issue_no,)).fetchone()
    if not target:
        return []
    draw_date = target["draw_date"]
    rows = conn.execute(
        """SELECT numbers_json, special_number FROM draws
           WHERE draw_date < ? OR (draw_date = ? AND issue_no < ?)
           ORDER BY draw_date DESC, issue_no DESC LIMIT ?""",
        (draw_date, draw_date, issue_no, limit)
    ).fetchall()
    return rows

# 回测报告
def get_recent_single_zodiac_report(conn: sqlite3.Connection, lookback: int = 20) -> Dict[str, float]:
    rows = _draws_ordered_asc(conn)
    if len(rows) < 2:
        return {"samples": 0.0, "hit_rate": 0.0, "max_miss_streak": 0.0}
    start = max(1, len(rows) - lookback)
    hits = 0
    samples = 0
    miss_streak = 0
    max_streak = 0
    for i in range(start, len(rows)):
        pick = get_single_zodiac_pick(conn, rows[i]["issue_no"], window=16)
        win_main = json.loads(rows[i]["numbers_json"])
        win_sp = int(rows[i]["special_number"])
        win_zod = {get_zodiac_by_number(n) for n in win_main}
        win_zod.add(get_zodiac_by_number(win_sp))
        hit = 1 if pick in win_zod else 0
        hits += hit
        samples += 1
        if hit == 0:
            miss_streak += 1
            max_streak = max(max_streak, miss_streak)
        else:
            miss_streak = 0
    rate = hits / samples if samples else 0.0
    return {"samples": float(samples), "hit_rate": rate, "max_miss_streak": float(max_streak)}

def get_recent_two_zodiac_report(conn: sqlite3.Connection, lookback: int = 20, history_window: int = 16) -> Dict[str, float]:
    rows = _draws_ordered_asc(conn)
    if len(rows) < lookback + 5:
        return {"samples": 0.0, "hit_rate": 0.0, "max_miss_streak": 0.0}
    hits = 0
    miss_streak = 0
    max_streak = 0
    for i in range(len(rows) - lookback, len(rows)):
        issue = rows[i]["issue_no"]
        picks = get_two_zodiac_picks(conn, issue, window=history_window)
        win_main = json.loads(rows[i]["numbers_json"])
        win_sp = int(rows[i]["special_number"])
        win_zod = {get_zodiac_by_number(n) for n in win_main}
        win_zod.add(get_zodiac_by_number(win_sp))
        hit = any(z in win_zod for z in picks)
        hits += hit
        if hit:
            miss_streak = 0
        else:
            miss_streak += 1
            max_streak = max(max_streak, miss_streak)
    rate = hits / lookback if lookback else 0.0
    return {"samples": float(lookback), "hit_rate": rate, "max_miss_streak": float(max_streak)}

def get_recent_three_zodiac_report(conn: sqlite3.Connection, lookback: int = 20, history_window: int = 16) -> Dict[str, float]:
    rows = _draws_ordered_asc(conn)
    if len(rows) < history_window + 1:
        return {"samples": 0.0, "hit_rate": 0.0, "max_miss_streak": 0.0}
    start = max(history_window, len(rows) - lookback)
    hits = 0
    samples = 0
    miss_streak = 0
    max_miss_streak = 0
    for i in range(start, len(rows)):
        history_rows = rows[max(0, i - history_window):i]
        if len(history_rows) < history_window:
            continue
        picks = _get_three_zodiac_from_history_rows(history_rows, conn)
        win_main = json.loads(rows[i]["numbers_json"])
        win_special = int(rows[i]["special_number"])
        winning_zodiacs = {get_zodiac_by_number(int(n)) for n in win_main}
        winning_zodiacs.add(get_zodiac_by_number(win_special))
        hit_count = sum(1 for z in picks if z in winning_zodiacs)
        hit = 1 if hit_count >= 2 else 0
        hits += hit
        samples += 1
        if hit == 0:
            miss_streak += 1
            max_miss_streak = max(max_miss_streak, miss_streak)
        else:
            miss_streak = 0
    if samples == 0:
        return {"samples": 0.0, "hit_rate": 0.0, "max_miss_streak": 0.0}
    return {"samples": float(samples), "hit_rate": float(hits / samples), "max_miss_streak": float(max_miss_streak)}

def get_recent_four_zodiac_report(conn: sqlite3.Connection, lookback: int = 20, history_window: int = 16) -> Dict[str, float]:
    rows = _draws_ordered_asc(conn)
    if len(rows) < history_window + 1:
        return {"samples": 0.0, "hit_rate": 0.0, "max_miss_streak": 0.0}
    start = max(history_window, len(rows) - lookback)
    hits = 0
    samples = 0
    miss_streak = 0
    max_miss_streak = 0
    params = load_best_zodiac_params()
    for i in range(start, len(rows)):
        hist_rows = rows[max(0, i - history_window):i]
        if len(hist_rows) < history_window:
            continue
        picks = predict_strong_five(hist_rows, params)
        win_special = int(rows[i]["special_number"])
        win_zodiac = get_zodiac_by_number(win_special)
        if win_zodiac in picks:
            hits += 1
            miss_streak = 0
        else:
            miss_streak += 1
            max_miss_streak = max(max_miss_streak, miss_streak)
        samples += 1
    if samples == 0:
        return {"samples": 0.0, "hit_rate": 0.0, "max_miss_streak": 0.0}
    return {"samples": float(samples), "hit_rate": float(hits / samples), "max_miss_streak": float(max_miss_streak)}

def get_recent_texiao5_report(conn, lookback=10):
    rows = _draws_ordered_asc(conn)
    if len(rows) < 2:
        return {"samples":0, "hit_rate":0.0, "max_miss_streak":0}
    start = max(1, len(rows)-lookback)
    hits, samples, miss, max_miss = 0,0,0,0
    for i in range(start, len(rows)):
        issue_no = rows[i]["issue_no"]
        picks5 = get_texiao4_picks(conn, issue_no, status="REVIEWED", k=TEXIAO5_SIZE_DEFAULT)
        win_special = int(rows[i]["special_number"])
        win_zodiac = get_zodiac_by_number(win_special)
        if win_zodiac in set(picks5):
            hits += 1
            miss = 0
        else:
            miss += 1
            max_miss = max(max_miss, miss)
        samples += 1
    rate = hits/samples if samples>0 else 0.0
    return {"samples":samples, "hit_rate":rate, "max_miss_streak":max_miss}

def get_texiao4_picks(conn, issue_no, status="REVIEWED", k=TEXIAO5_SIZE_DEFAULT):
    params = load_best_zodiac_params()
    w_rule = float(params.get("four_w_rule", 0.5))
    w_lgb = float(params.get("four_w_lgb", 0.3))
    w_trans = float(params.get("four_w_transformer", 0.2))
    total_w = w_rule + w_lgb + w_trans
    if total_w == 0:
        total_w = 1.0

    if issue_no:
        target = conn.execute("SELECT draw_date FROM draws WHERE issue_no = ?", (issue_no,)).fetchone()
        if target:
            rows = conn.execute(
                """SELECT numbers_json, special_number FROM draws
                   WHERE draw_date < ? OR (draw_date = ? AND issue_no < ?)
                   ORDER BY draw_date DESC, issue_no DESC LIMIT 50""",
                (target["draw_date"], target["draw_date"], issue_no)
            ).fetchall()
        else:
            rows = conn.execute("SELECT numbers_json, special_number FROM draws ORDER BY draw_date DESC LIMIT 50").fetchall()
    else:
        rows = conn.execute("SELECT numbers_json, special_number FROM draws ORDER BY draw_date DESC LIMIT 50").fetchall()
    if not rows:
        return ["马", "蛇", "龙", "兔"][:k]

    recent_special_window = int(params.get("four_recent_special_window", 12))
    cold_omit_threshold = int(params.get("four_cold_omit_threshold", 6))
    scores_rule = _compute_special_four_score(rows, recent_special_window, cold_omit_threshold)

    lgb_probs = lgb_zodiac_model.predict_proba(conn, issue_no)
    transformer_probs = predict_transformer(conn, SCRIPT_DIR / "transformer_sp.pth", issue_no)

    mx_rule = max(scores_rule.values()) if scores_rule else 1
    scores_rule_norm = {z: s / mx_rule for z, s in scores_rule.items()}

    combined = {}
    for z in ZODIAC_MAP:
        score = 0.0
        score += scores_rule_norm.get(z, 0) * (w_rule / total_w)
        if lgb_probs:
            score += lgb_probs.get(z, 0) * (w_lgb / total_w)
        if transformer_probs:
            score += transformer_probs.get(z, 0) * (w_trans / total_w)
        combined[z] = score

    omission_map = _zodiac_omission_map(rows)
    picks = omission_boosted_four_pick(combined, omission_map, omit_threshold=6, boost_strength=0.45)

    cold_candidates = [z for z, omit in omission_map.items() if omit >= 6 and z not in picks]
    if cold_candidates:
        longest_cold = max(cold_candidates, key=lambda z: omission_map[z])
        picks[-1] = longest_cold
        picks = list(dict.fromkeys(picks))
        while len(picks) < 4:
            for z in ZODIAC_MAP:
                if z not in picks:
                    picks.append(z)
                    if len(picks) == 4: break
    return picks[:k]

# 内部历史数据生肖选择（用于回测防穿越）
def _get_three_zodiac_from_history_rows(rows: Sequence[sqlite3.Row], conn=None) -> List[str]:
    if not rows:
        return ["马", "蛇", "龙"]
    params = load_best_zodiac_params()
    scores = _build_zodiac_scores_from_rows(rows, decay=0.04)
    recent = rows[:14]
    recent_special_zodiacs = [get_zodiac_by_number(int(r["special_number"])) for r in recent]
    recent_main_zodiacs = [get_zodiac_by_number(int(n)) for r in recent for n in json.loads(r["numbers_json"])]
    special_counter = Counter(recent_special_zodiacs)
    main_counter = Counter(recent_main_zodiacs)
    for z, cnt in special_counter.items():
        scores[z] += cnt * 3.6 * params.get("three_hot_weight", 1.35)
    for z, cnt in main_counter.items():
        scores[z] += cnt * 1.00 * params.get("three_hot_weight", 1.35)
    latest_special_z = recent_special_zodiacs[0] if recent_special_zodiacs else None
    if latest_special_z:
        scores[latest_special_z] += 5.0
        pair_z = ZODIAC_PAIR.get(latest_special_z)
        if pair_z:
            scores[pair_z] += 2.5
    omission_zodiac = _zodiac_omission_map(rows)
    for z, omit in omission_zodiac.items():
        if omit >= 2:
            scores[z] += min(3.4, omit / 3.0) * params.get("three_omit_weight", 1.60)
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    picks = [ranked[0][0], ranked[1][0]] if len(ranked) >= 2 else ["马", "蛇"]
    for z, _ in ranked[2:]:
        if z not in picks:
            picks.append(z)
            break
    cold_candidates = [z for z, omit in omission_zodiac.items() if omit >= 5 and z not in picks]
    if cold_candidates:
        coldest = max(cold_candidates, key=lambda z: omission_zodiac[z])
        picks[-1] = coldest
    return picks[:3]

# 三生肖对外接口
def get_three_zodiac_picks(conn, issue_no=None):
    if issue_no:
        rows = _get_rows_before_issue(conn, issue_no, limit=16)
    else:
        rows = conn.execute("SELECT numbers_json, special_number FROM draws ORDER BY draw_date DESC LIMIT 16").fetchall()
    return _get_three_zodiac_from_history_rows(rows, conn) if rows else ["马", "蛇", "龙"]

# 特别号推荐
def get_top_special_votes(conn: sqlite3.Connection, issue_no: str, top_n: int = 3) -> List[int]:
    all_specials = []
    for strategy in STRATEGY_IDS:
        run = conn.execute(
            "SELECT id FROM prediction_runs WHERE issue_no = ? AND strategy = ? AND status='PENDING'",
            (issue_no, strategy)
        ).fetchone()
        if run:
            _, sp = get_picks_for_run(conn, run["id"])
            if sp is not None:
                all_specials.append(sp)
    if not all_specials:
        return []
    vote_counter = Counter(all_specials)
    sorted_items = sorted(vote_counter.items(), key=lambda x: (-x[1], x[0]))
    return [num for num, _ in sorted_items[:top_n]]

def get_special_recommendation(conn: sqlite3.Connection, issue_no: str, main6: Sequence[int], zodiac_two: Optional[Sequence[str]] = None) -> Tuple[Optional[int], List[int], bool]:
    top_votes = get_top_special_votes(conn, issue_no, top_n=8)
    if not top_votes:
        return None, [], False

    mains = {int(n) for n in main6}
    recent_3_specials = [int(r["special_number"]) for r in conn.execute(
        "SELECT special_number FROM draws ORDER BY draw_date DESC LIMIT 3"
    ).fetchall()]
    recent_12_specials = [int(r["special_number"]) for r in conn.execute(
        "SELECT special_number FROM draws ORDER BY draw_date DESC LIMIT 12"
    ).fetchall()]
    recent_8_specials = [int(r["special_number"]) for r in conn.execute(
        "SELECT special_number FROM draws ORDER BY draw_date DESC LIMIT 8"
    ).fetchall()]

    vote_scores = Counter(top_votes)
    candidates = sorted(set(top_votes) | set(recent_12_specials) | set(recent_8_specials))
    if zodiac_two:
        allowed = set()
        for z in zodiac_two:
            allowed.update(ZODIAC_MAP.get(z, []))
        filtered = [n for n in candidates if n in allowed]
        if filtered:
            candidates = filtered
    combined = []
    for n in candidates:
        if n in mains:
            continue
        score = vote_scores.get(n, 0) * 4.0
        if recent_12_specials:
            recent_special_tail = recent_12_specials[0] % 10
            recent_special_zone = (recent_12_specials[0] - 1) // 10
            if n % 10 == recent_special_tail: score += 0.45
            if (n - 1) // 10 == recent_special_zone: score += 0.25
        if n in recent_3_specials:
            score *= 0.8949
        combined.append((n, score))

    if not combined:
        return None, [], False
    combined.sort(key=lambda x: (-x[1], x[0]))
    primary = int(combined[0][0])
    conflict = primary in mains
    defenses = []
    for n, _ in combined[1:]:
        n_int = int(n)
        if n_int == primary or n_int in defenses or n_int in mains or n_int in recent_3_specials:
            continue
        defenses.append(n_int)
        if len(defenses) >= 3:
            break
    return primary, defenses, conflict

def _get_ml_special_votes(conn, issue_no, main_set, top_n=3):
    votes = Counter()
    for model_name in ("xgb", "lgb"):
        model_path = SCRIPT_DIR / f"{model_name}_model.pkl"
        if not model_path.exists():
            continue
        try:
            import pickle
            with open(model_path, "rb") as f:
                predictor = pickle.load(f)
            if hasattr(predictor, "predict_special_proba"):
                proba = predictor.predict_special_proba(conn, main_pool=list(main_set))
                if proba:
                    best_num = max(proba, key=proba.get)
                    votes[int(best_num)] += 1
            elif hasattr(predictor, "predict_special"):
                special = predictor.predict_special(conn)
                if special:
                    votes[int(special)] += 1
        except Exception:
            continue
    return [num for num, _ in votes.most_common(top_n)]

def get_strong_special_from_strategies(
    conn: sqlite3.Connection,
    issue_no: str,
    main6: Sequence[int],
) -> Tuple[List[int], List[str], Optional[int], Optional[str]]:
    strategy_weights = get_strategy_weights(conn, window=WEIGHT_WINDOW_DEFAULT, for_issue=issue_no)
    specials: List[int] = []
    weighted_items: List[Tuple[int, float]] = []
    for strategy in SPECIAL_ANALYSIS_ORDER:
        run = conn.execute(
            "SELECT id FROM prediction_runs WHERE issue_no = ? AND strategy = ? AND status='PENDING'",
            (issue_no, strategy),
        ).fetchone()
        if not run:
            continue
        _, sp = get_picks_for_run(conn, int(run["id"]))
        if sp is None:
            continue
        special_num = int(sp)
        specials.append(special_num)
        weighted_items.append((special_num, float(strategy_weights.get(strategy, 1.0 / max(len(STRATEGY_IDS), 1)))))
    if not specials:
        return [], [], None, None

    zodiac_list = [get_zodiac_by_number(n) for n in specials]
    zodiac_counter = Counter(zodiac_list)
    number_votes = Counter(specials)
    weighted_scores: Dict[int, float] = {}
    for n, w in weighted_items:
        weighted_scores[n] = weighted_scores.get(n, 0.0) + w

    recent_specials = [int(r["special_number"]) for r in conn.execute(
        "SELECT special_number FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT 30"
    ).fetchall()]
    omission = {n: 31 for n in ALL_NUMBERS}
    for idx, n in enumerate(recent_specials):
        omission[n] = min(omission.get(n, 31), idx + 1)

    recent_special_zodiacs = [get_zodiac_by_number(n) for n in recent_specials[:8]]
    recent_main_zodiacs: List[str] = []
    for row in conn.execute(
        "SELECT numbers_json FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT 8"
    ).fetchall():
        recent_main_zodiacs.extend(get_zodiac_by_number(int(n)) for n in json.loads(row["numbers_json"]))
    recent_zodiac_counter = Counter(recent_special_zodiacs + recent_main_zodiacs)

    model_score: Dict[str, float] = {z: 0.0 for z in ZODIAC_MAP.keys()}
    for z, cnt in zodiac_counter.items():
        model_score[z] += cnt * 2.8
    for z, cnt in recent_zodiac_counter.items():
        model_score[z] += cnt * 0.25
    hot_special = [z for z, _ in Counter(recent_special_zodiacs).most_common(1)]
    for z in hot_special:
        model_score[z] += 2.0
    omission_zodiac: Dict[str, int] = {z: 0 for z in ZODIAC_MAP.keys()}
    for idx, sp in enumerate(recent_specials):
        oz = get_zodiac_by_number(sp)
        omission_zodiac[oz] = max(omission_zodiac.get(oz, 0), 30 - idx)
    cold_zodiacs = [z for z, _ in sorted(omission_zodiac.items(), key=lambda x: (-x[1], x[0]))[:2]]
    for z in cold_zodiacs:
        model_score[z] += 4.2
    for z in ZODIAC_MAP.keys():
        if omission_zodiac.get(z, 0) >= 5:
            model_score[z] += 2.2
    ranked_zodiacs = sorted(model_score.items(), key=lambda x: (-x[1], x[0]))
    top_zodiacs = [z for z, _ in ranked_zodiacs[:4]]
    if len(top_zodiacs) < 4:
        for z, _ in ranked_zodiacs:
            if z not in top_zodiacs:
                top_zodiacs.append(z)
            if len(top_zodiacs) == 4:
                break

    mains = {int(x) for x in main6}
    candidate_scores: Dict[int, float] = {}
    for n in sorted(set(specials)):
        zodiac = get_zodiac_by_number(n)
        if zodiac not in top_zodiacs:
            continue
        score = 0.0
        score += number_votes.get(n, 0) * 2.4
        score += weighted_scores.get(n, 0.0) * 1.6
        score += zodiac_counter.get(zodiac, 0) * 1.0
        score += min(1.2, float(omission.get(n, 31)) / 24.0)
        if n in mains:
            score -= 0.8
        if zodiac in hot_special:
            score += 0.9
        if zodiac in cold_zodiacs:
            score += 0.6
        candidate_scores[n] = score

    ranked = sorted(candidate_scores.items(), key=lambda x: (-x[1], x[0]))
    best: Optional[int] = None
    for n, _ in ranked:
        if n not in mains:
            best = n
            break
    if best is None and ranked:
        best = ranked[0][0]
    ml_votes = _get_ml_special_votes(conn, issue_no, mains, top_n=3)
    for n in ml_votes:
        zodiac = get_zodiac_by_number(n)
        candidate_scores[n] = candidate_scores.get(n, 0.0) + 1.5
        if zodiac in top_zodiacs:
            candidate_scores[n] += 0.8
    ranked = sorted(candidate_scores.items(), key=lambda x: (-x[1], x[0]))
    best: Optional[int] = None
    for n, _ in ranked:
        if n not in mains:
            best = n
            break
    if best is None and ranked:
        best = ranked[0][0]
    if best is None:
        return specials, top_zodiacs, None, None
    return specials, top_zodiacs, best, get_zodiac_by_number(best)

def _weighted_consensus_pools(conn, issue_no):
    strategy_weights = get_strategy_weights(conn, window=WEIGHT_WINDOW_DEFAULT, for_issue=issue_no)
    number_scores = {}
    special_scores = {}
    for strategy in STRATEGY_IDS:
        run = conn.execute(
            "SELECT id FROM prediction_runs WHERE issue_no=? AND strategy=? AND status='PENDING'",
            (issue_no, strategy)
        ).fetchone()
        if not run: continue
        run_id = int(run["id"])
        w = float(strategy_weights.get(strategy, 1.0 / len(STRATEGY_IDS)))
        pool20 = get_pool_numbers_for_run(conn, run_id, 20)
        for idx, n in enumerate(pool20):
            if not (1 <= int(n) <= 49): continue
            rank_boost = (20 - idx) / 20.0
            number_scores[int(n)] = number_scores.get(int(n), 0.0) + w * rank_boost
        main6 = get_pool_numbers_for_run(conn, run_id, 6)
        for n in main6:
            if 1 <= int(n) <= 49:
                number_scores[int(n)] = number_scores.get(int(n), 0.0) + w * 0.35
        _, special = get_picks_for_run(conn, run_id)
        if special is not None and 1 <= int(special) <= 49:
            special_scores[int(special)] = special_scores.get(int(special), 0.0) + w

    if not number_scores: return [], [], [], [], None

    ranked_numbers = [n for n, _ in sorted(number_scores.items(), key=lambda x: (-x[1], x[0]))]
    pool20 = ranked_numbers[:20]
    pool14 = pool20[:14]
    pool10 = pool20[:10]
    main6 = pool20[:6]

    special = None
    if special_scores:
        special = sorted(special_scores.items(), key=lambda x: (-x[1], x[0]))[0][0]
    else:
        for n in pool20:
            if n not in main6:
                special = n
                break
    return main6, pool10, pool14, pool20, special

def get_trio_from_merged_pool20_v2(conn, issue_no):
    _, _, _, pool20, _ = _weighted_consensus_pools(conn, issue_no)
    if not pool20 or len(pool20) < 3: return [1, 2, 3]
    all_pools = []
    for strategy in STRATEGY_IDS:
        run = conn.execute(
            "SELECT id FROM prediction_runs WHERE issue_no=? AND strategy=? AND status='PENDING'",
            (issue_no, strategy)
        ).fetchone()
        if run:
            p20 = get_pool_numbers_for_run(conn, run["id"], 20)
            p20_filtered = [n for n in p20 if n in pool20]
            all_pools.extend(p20_filtered)
    if len(all_pools) < 3: return pool20[:3]
    app_count = Counter(all_pools)
    diff_numbers = [n for n, c in app_count.items() if 1 <= c <= 2 and n in pool20]
    if len(diff_numbers) < 6: diff_numbers = [n for n, c in app_count.items() if c <= 3 and n in pool20]
    if len(diff_numbers) < 3: diff_numbers = pool20[:15]
    draws = load_recent_draws(conn, FEATURE_WINDOW_DEFAULT)
    if len(draws) < 3: return diff_numbers[:3]
    momentum = _momentum_map(draws); freq = _freq_map(draws); omission = _omission_map(draws)
    momentum_norm = _normalize(momentum); freq_norm = _normalize(freq); omission_norm = _normalize(omission)
    w_mom, w_hot, w_cold = get_trio_weights(conn, window=WEIGHT_WINDOW_DEFAULT)
    scores = {}
    for n in diff_numbers[:15]:
        score = (w_mom * momentum_norm.get(n, 0) + w_hot * freq_norm.get(n, 0) + w_cold * omission_norm.get(n, 0))
        score += (6 - app_count.get(n, 3)) * 0.15
        scores[n] = score
    sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    candidates = [n for n, _ in sorted_nums[:10]]
    def is_valid(tri):
        odd_cnt = sum(1 for x in tri if x % 2 == 1)
        total = sum(tri)
        return 1 <= odd_cnt <= 2 and 80 <= total <= 130
    for i in range(len(candidates)):
        for j in range(i+1, len(candidates)):
            for k in range(j+1, len(candidates)):
                tri = (candidates[i], candidates[j], candidates[k])
                if is_valid(tri): return list(tri)
    for i in range(len(candidates)):
        for j in range(i+1, len(candidates)):
            for k in range(j+1, len(candidates)):
                tri = (candidates[i], candidates[j], candidates[k])
                odd_cnt = sum(1 for x in tri if x % 2 == 1)
                if 1 <= odd_cnt <= 2: return list(tri)
    return candidates[:3] if len(candidates) >= 3 else pool20[:3]

def get_final_recommendation(conn):
    row = conn.execute("SELECT issue_no FROM prediction_runs WHERE status='PENDING' ORDER BY created_at DESC LIMIT 1").fetchone()
    if not row: return None
    issue_no = row["issue_no"]
    main6, pool10, pool14, pool20, _ = _weighted_consensus_pools(conn, issue_no)
    if not main6: return None
    zodiac_single = get_single_zodiac_pick(conn, issue_no, window=16)
    zodiac_two = get_two_zodiac_picks(conn, issue_no, window=16)
    special, defs, conflict = get_special_recommendation(conn, issue_no, main6, zodiac_two)
    if special is None: return None
    strategy_specials, strategy_special_zodiacs, strategy_strong_special, strategy_strong_zodiac = get_strong_special_from_strategies(conn, issue_no, main6)
    predict_trio = get_trio_from_merged_pool20_v2(conn, issue_no)
  
    history_rows = conn.execute(
        "SELECT numbers_json, special_number FROM draws ORDER BY draw_date DESC LIMIT 16"
    ).fetchall()
    params = load_best_zodiac_params()
    special_zodiacs = predict_strong_five(history_rows, params)
    special_zodiacs = list(dict.fromkeys(special_zodiacs))[:5]
    return (issue_no, main6, special, pool10, pool14, pool20, predict_trio,
            defs, conflict, zodiac_single, zodiac_two, special_zodiacs,
            strategy_specials, strategy_special_zodiacs, strategy_strong_special, strategy_strong_zodiac)

class KellyManager:
    def __init__(self, bankroll: float = 1000.0):
        self.bankroll = bankroll
        self.loss_streak = 0

    def update_result(self, net_profit: float):
        if net_profit <= 0:
            self.loss_streak += 1
        else:
            self.loss_streak = 0
        self.bankroll += net_profit
        if self.bankroll < 0:
            self.bankroll = 0.0

    def kelly_stake(self, win_rate: float, odds: float, fraction: float = 0.5) -> float:
        b = odds - 1.0
        if win_rate <= 0 or b <= 0:
            return 0.0
        f = (win_rate * b - (1 - win_rate)) / b
        f = f * fraction
        if self.loss_streak >= 2:
            f *= 0.5
        f = min(f, 0.25)
        return max(0.0, f * self.bankroll)    
# ========== 最终推荐与仪表盘 ==========
def print_final_recommendation(conn, xgb_pool20=None):
    rec = get_final_recommendation(conn)
    if not rec:
        print("\n最终推荐: (暂无有效预测)")
        return
    (issue_no, main6, special, pool10, pool14, pool20, predict_trio,
     special_defenses, special_conflict, zodiac_single, zodiac_two,
     special_zodiacs, strategy_specials, strategy_special_zodiacs,
     strategy_strong_special, strategy_strong_zodiac) = rec
    if xgb_pool20 and len(xgb_pool20) >= 20:
        pool20 = xgb_pool20[:20]; pool14 = pool20[:14]; pool10 = pool20[:10]; main6 = pool20[:6]
        print("[XGB] 主号池已升级为 XGBoost 预测池")
    p6 = " ".join(f"{n:02d}" for n in main6)
    p10 = " ".join(f"{n:02d}" for n in pool10)
    p14 = " ".join(f"{n:02d}" for n in pool14)
    p20 = " ".join(f"{n:02d}" for n in pool20)
    trio_str = " ".join(f"{n:02d}" for n in predict_trio) if predict_trio else "无"
    special_text = f"{special:02d}"
    print()
    print(f"一生肖推荐: {zodiac_single}")
    print(f"二生肖推荐: {'、'.join(zodiac_two)}")
    print(f"三生肖推荐: {'、'.join(get_three_zodiac_picks(conn))}")
    print(f"特别生肖推荐: {'、'.join(special_zodiacs)}")
    latest = get_latest_draw(conn)
    if latest:
        print(f"推荐期数日期: {latest['issue_no']}（{latest['draw_date']}）")
    one_rep = get_recent_single_zodiac_report(conn, lookback=10)
    two_rep = get_recent_two_zodiac_report(conn, lookback=10)
    three_rep = get_recent_three_zodiac_report(conn, lookback=10)
    four_rep = get_recent_four_zodiac_report(conn, lookback=10)
    print(f"一生肖近10期命中率: {one_rep['hit_rate']*100:.1f}% 最大连空{int(one_rep['max_miss_streak'])}")
    print(f"二生肖近10期命中率: {two_rep['hit_rate']*100:.1f}% 最大连空{int(two_rep['max_miss_streak'])}")
    print(f"三生肖近10期命中率: {three_rep['hit_rate']*100:.1f}% 最大连空{int(three_rep['max_miss_streak'])}")
    print(f"特别生肖近10期命中率: {four_rep['hit_rate']*100:.1f}% 最大连空{int(four_rep['max_miss_streak'])}")
    sp_report = get_recent_special_picks_report(conn, lookback=10)
    print(f"特别号精选回测（最近10期）: 命中率={sp_report['hit_rate']*100:.1f}% 最大连空={int(sp_report['max_miss_streak'])}")
    enhanced_zodiacs = list(special_zodiacs) + [get_zodiac_by_number(int(r['special_number'])) for r in conn.execute("SELECT special_number FROM draws ORDER BY draw_date DESC LIMIT 3").fetchall()]
    enhanced_zodiacs = list(dict.fromkeys(enhanced_zodiacs))
    while len(enhanced_zodiacs) < 4:
        for z in ZODIAC_MAP:
            if z not in enhanced_zodiacs:
                enhanced_zodiacs.append(z)
                if len(enhanced_zodiacs) >= 4: break
    precise = get_precise_specials_for_issue(conn, issue_no, enhanced_zodiacs, top_n=3)
    if precise:
        ps_str = " ".join(f"{n:02d}" for n in precise)
        ps_detail = ", ".join(f"{n:02d}({get_zodiac_by_number(n)})" for n in precise)
        print(f"精选特别号 (3码): {ps_str}  ({ps_detail})")
        log_special_picks(conn, issue_no, precise, special)
    if four_rep['hit_rate'] < 0.65:
        latest_sp = conn.execute("SELECT special_number FROM draws ORDER BY draw_date DESC LIMIT 1").fetchone()["special_number"]
        trend_z = get_zodiac_by_number(int(latest_sp))
        if trend_z not in special_zodiacs:
            special_zodiacs[-1] = trend_z
            print(f"[修正] 特别生肖即时跟随: {trend_z}")
    km = KellyManager()
    km_stake = km.kelly_stake(four_rep['hit_rate'], 1.5)
    if km_stake > 0:
        print(f"特别生肖建议仓位: {km_stake:.2f} 元")
    else:
        print(f"特别生肖建议仓位: <未达正期望>, 试探仓位 {km.bankroll*0.02:.2f} 元")
    rm = RiskManager()
    z_rec = rm.get_bet_recommendation("zodiac_strict_two", 0.30, 5.0, rm.bankroll)
    s_rec = rm.get_bet_recommendation("special", 0.03, 45.0, rm.bankroll)
    print(f"风控: 生肖{'暂停' if z_rec['suspended'] else '继续'} | 特别号{'暂停' if s_rec['suspended'] else '继续'}")    
    latest_sp = conn.execute("SELECT special_number FROM draws ORDER BY draw_date DESC LIMIT 1").fetchone()
    if latest_sp:
        latest_sp_zodiac = get_zodiac_by_number(int(latest_sp["special_number"]))
        omission = _zodiac_omission_map(conn.execute("SELECT numbers_json, special_number FROM draws ORDER BY draw_date DESC LIMIT 50").fetchall())
        longest_cold = max(omission, key=omission.get)
        opposite = ZODIAC_PAIR.get(latest_sp_zodiac, "蛇")
        hedge_zodiacs = list(set([longest_cold, opposite]))
        print(f"🎯 对冲组合推荐: {'、'.join(hedge_zodiacs)} (遗漏最长: {longest_cold} | 相冲: {opposite})")
    print("=" * 50)

def send_pushplus_notification(title: str, content: str) -> bool:
    if not PUSHPLUS_TOKEN:
        print("[推送] 未配置 PUSHPLUS_TOKEN，跳过推送")
        return False
    import urllib.request
    import urllib.parse
    url = "https://www.pushplus.plus/send"
    data = {
        "token": PUSHPLUS_TOKEN,
        "title": title,
        "content": content,
        "template": "txt"
    }
    post_data = urllib.parse.urlencode(data).encode("utf-8")
    req = urllib.request.Request(url, data=post_data, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            if result.get("code") == 200:
                print("[推送] 成功")
                return True
            else:
                print(f"[推送] 失败: {result}")
                return False
    except Exception as e:
        print(f"[推送] 异常: {e}")
        return False

def review_latest_prediction(conn: sqlite3.Connection) -> str:
    latest_draw = get_latest_draw(conn)
    if not latest_draw:
        return "暂无开奖数据。"
    issue_no = latest_draw["issue_no"]
    draw_date = latest_draw["draw_date"]
    actual_numbers = set(json.loads(latest_draw["numbers_json"]))
    actual_special = int(latest_draw["special_number"])
    actual_main_str = " ".join(_fmt_num(n) for n in sorted(actual_numbers))
    actual_special_str = _fmt_num(actual_special)

    runs = conn.execute(
        "SELECT id, strategy FROM prediction_runs WHERE issue_no = ? AND status='REVIEWED'",
        (issue_no,)
    ).fetchall()
    if not runs:
        return f"最新一期 {issue_no} 无预测记录（可能未运行预测）。"

    lines = []
    lines.append(f"复盘最新一期 {issue_no}（{draw_date}）")
    lines.append(f"实际开奖: 主号 {actual_main_str}  特别号 {actual_special_str}")
    lines.append("")
    lines.append("各策略预测与命中情况：")
    for run in runs:
        strategy = run["strategy"]
        strategy_name = STRATEGY_LABELS.get(strategy, strategy)
        main6, special = get_picks_for_run(conn, run["id"])
        if not main6:
            continue
        hit_count = len([n for n in main6 if n in actual_numbers])
        special_hit = 1 if special == actual_special else 0
        main_str = " ".join(_fmt_num(n) for n in main6)
        special_str = _fmt_num(special) if special is not None else "--"
        lines.append(f"  {strategy_name}: 主号 {main_str} | 特别号 {special_str} | 中主号 {hit_count}/6 | 中特别号 {'Y' if special_hit else 'N'}")
    lines.append("")
    return "\n".join(lines)

def print_dashboard(conn: sqlite3.Connection, xgb_pool20: Optional[List[int]] = None) -> None:
    latest = get_latest_draw(conn)
    if latest:
        nums = " ".join(_fmt_num(n) for n in json.loads(latest["numbers_json"]))
        print(f"最新开奖: {latest['issue_no']} {latest['draw_date']} | 主号: {nums} | 特别号: {_fmt_num(int(latest['special_number']))}")
    else:
        print("暂无开奖数据。")

    print("\n=== 近期推荐 ===")
    print_recommendation_sheet(conn, limit=8)

    print("\n=== 近10期真实回测结果 ===")
    one_rep = get_recent_single_zodiac_report(conn, lookback=10)
    two_rep = get_recent_two_zodiac_report(conn, lookback=10)
    three_rep = get_recent_three_zodiac_report(conn, lookback=10)
    four_rep = get_recent_four_zodiac_report(conn, lookback=10, history_window=16)
    texiao5_rep = get_recent_texiao5_report(conn, lookback=10)
    sp_rep = get_recent_special_picks_report(conn, lookback=10)
    print(f"  一生肖: 命中率 {one_rep['hit_rate']*100:.1f}% | 样本 {int(one_rep['samples'])} | 最大连空 {int(one_rep['max_miss_streak'])}")
    print(f"  二生肖: 命中率 {two_rep['hit_rate']*100:.1f}% | 样本 {int(two_rep['samples'])} | 最大连空 {int(two_rep['max_miss_streak'])}")
    print(f"  三生肖(需中2): 命中率 {three_rep['hit_rate']*100:.1f}% | 样本 {int(three_rep['samples'])} | 最大连空 {int(three_rep['max_miss_streak'])}")
    print(f"  特五肖(中特别号生肖): 命中率 {four_rep['hit_rate']*100:.1f}% | 样本 {int(four_rep['samples'])} | 最大连空 {int(four_rep['max_miss_streak'])}")
    print(f"  特五肖(仅特别号): 命中率 {texiao5_rep['hit_rate']*100:.1f}% | 样本 {int(texiao5_rep['samples'])} | 最大连空 {int(texiao5_rep['max_miss_streak'])}")
    print(f"特别号精选回测（最近10期）: 命中率={sp_rep['hit_rate']*100:.1f}% 最大连空={int(sp_rep.get('max_miss_streak', 0))}")

    print("\n=== 策略健康度（最近10期复盘） ===")
    stats_10 = get_review_stats(conn, window=10)
    if not stats_10:
        print("  (近期暂无复盘数据，请先运行 sync)")
    for s in stats_10:
        strategy_name = STRATEGY_LABELS.get(s["strategy"], s["strategy"])
        print(
            f"  - {strategy_name}: 平均命中 {s['avg_hit']:.2f} | 6码 {s['avg_rate'] * 100:.1f}% | "
            f"10码 {float(s['avg_rate_10'] or 0) * 100:.1f}% | 14码 {float(s['avg_rate_14'] or 0) * 100:.1f}% | "
            f"20码 {float(s['avg_rate_20'] or 0) * 100:.1f}% | 特别号 {s['special_rate'] * 100:.1f}% | "
            f"至少中1个 {s['hit1_rate'] * 100:.1f}% | 至少中2个 {s['hit2_rate'] * 100:.1f}% | 最大连空 {int(s.get('max_miss_streak', 0))}"
        )

    confidence = 0.0
    max_miss = 0
    if stats_10:
        confidence = max(float(s.get('hit1_rate', 0.0)) for s in stats_10) * 100.0
        max_miss = max(int(s.get('max_miss_streak', 0)) for s in stats_10)
    if confidence >= 80 and max_miss < 3:
        advice = "🔥 高信心：可适当加大投入"
    elif confidence >= 60:
        advice = "👍 中等信心：正常投入"
    else:
        advice = "⚠️ 低信心：建议减少投入或观望"
    print(f"\n信心指数: {confidence:.1f}/100 | 建议投入: {advice}")

    print(f"\n=== 策略健康度（最近{HEALTH_WINDOW_DEFAULT}期） ===")
    weights = get_strategy_weights(conn, window=WEIGHT_WINDOW_DEFAULT)
    health = get_strategy_health(conn, window=HEALTH_WINDOW_DEFAULT)
    for strategy in STRATEGY_IDS:
        strategy_name = STRATEGY_LABELS.get(strategy, strategy)
        h = health.get(strategy, {})
        samples = int(h.get("samples", 0.0))
        avg_hit = float(h.get("recent_avg_hit", 0.0))
        hit1 = float(h.get("hit1_rate", 0.0)) * 100.0
        hit2 = float(h.get("hit2_rate", 0.0)) * 100.0
        cold = int(h.get("cold_streak", 0.0))
        weight = float(weights.get(strategy, 0.0)) * 100.0
        print(
            f"  - {strategy_name}: 样本={samples} 最近均中={avg_hit:.2f} "
            f"近1中率={hit1:.1f}% 近2中率={hit2:.1f}% 连挂={cold} 当前权重={weight:.1f}%"
        )

    one_rep = get_recent_single_zodiac_report(conn, lookback=10)
    two_rep = get_recent_two_zodiac_report(conn, lookback=10)
    four_rep = get_recent_four_zodiac_report(conn, lookback=10, history_window=16)
    texiao5_rep = get_recent_texiao5_report(conn, lookback=10)
    sp_rep = get_recent_special_picks_report(conn, lookback=10)

    print(f"近10期: 一生肖={one_rep['hit_rate']:.3f}(连空{int(one_rep['max_miss_streak'])}) "
          f"二肖={two_rep['hit_rate']:.3f}(连空{int(two_rep['max_miss_streak'])}) "
          f"三肖(需中2)={three_rep['hit_rate']:.3f}(连空{int(three_rep['max_miss_streak'])}) "
          f"五肖(中特别号生肖)={four_rep['hit_rate']:.3f}(连空{int(four_rep['max_miss_streak'])}) "
          f"特别号={sp_rep['hit_rate']:.3f}(连空{int(sp_rep.get('max_miss_streak',0))})")
    print(f"特五肖(仅特别号) 近10期命中率: {texiao5_rep['hit_rate']:.3f} 最大连空{int(texiao5_rep['max_miss_streak'])}")
    if one_rep['hit_rate'] >= 0.9 and two_rep['hit_rate'] >= 0.8 and four_rep['hit_rate'] >= 1.0:
        print("🎉 达标！")

    print("\n=== 当前推荐 ===")
    print_final_recommendation(conn, xgb_pool20=xgb_pool20)

    print("\n=== 最新一期复盘 ===")
    print(review_latest_prediction(conn))

    if PUSHPLUS_TOKEN:
        rec = get_final_recommendation(conn)
        if rec:
            (issue_no, main6, special, _, _, _, predict_trio,
             special_defenses, special_conflict, zodiac_single, zodiac_two,
             special_zodiacs, strategy_specials, strategy_special_zodiacs,
             strategy_strong_special, strategy_strong_zodiac) = rec
            special_text = _fmt_num(special)
            trio_str = " ".join(_fmt_num(n) for n in predict_trio) if predict_trio else "无"
            defense_text = " ".join(_fmt_num(n) for n in special_defenses) if special_defenses else "无"
            strong_special_text = _fmt_num(strategy_strong_special) if strategy_strong_special is not None else "无"
            strong_zodiac_text = strategy_strong_zodiac if strategy_strong_zodiac else "无"
            special_zodiacs_text = "、".join(special_zodiacs) if special_zodiacs else "无"
            strategy_special_text = " ".join(_fmt_num(n) for n in strategy_specials) if strategy_specials else "无"
            strategy_zodiac_text = "、".join(strategy_special_zodiacs) if strategy_special_zodiacs else "无"

            all_specials = []
            for strategy in STRATEGY_IDS:
                run = conn.execute(
                    "SELECT id FROM prediction_runs WHERE issue_no = ? AND strategy = ? AND status='PENDING'",
                    (issue_no, strategy)
                ).fetchone()
                if run:
                    _, sp = get_picks_for_run(conn, run["id"])
                    if sp is not None:
                        all_specials.append(sp)
            unique_specials = []
            for sp in all_specials:
                if sp not in unique_specials:
                    unique_specials.append(sp)
            all_specials_str = " ".join(_fmt_num(n) for n in unique_specials) if unique_specials else "无"

            top_special_votes = get_top_special_votes(conn, issue_no, top_n=3)
            top_special_str = " ".join(_fmt_num(n) for n in top_special_votes) if top_special_votes else "无"

            stats_10 = get_review_stats(conn, window=10)
            confidence = max((float(s.get("hit1_rate") or 0.0) for s in stats_10), default=0.0) * 100.0
            max_miss = max((int(s.get("max_miss_streak", 0)) for s in stats_10), default=0)
            if confidence >= 80 and max_miss < 3:
                advice = "🔥 高信心：可适当加大投入"
            elif confidence >= 60:
                advice = "👍 中等信心：正常投入"
            else:
                advice = "⚠️ 低信心：建议减少投入或观望"

            zodiac_single_text = zodiac_single if zodiac_single else "数据不足"
            zodiac_two_text = "、".join(zodiac_two) if zodiac_two else "数据不足"
            conflict_tip = "（已避开主号冲突）" if special_conflict else ""

            content = (
                f"【香港六合彩·{issue_no}期推荐】\n"
                f"2生肖推荐：{zodiac_two_text}\n"
                f"1生肖推荐：{zodiac_single_text}\n"
                f"特别生肖推荐：{special_zodiacs_text}\n"
                f"特别号主推：{special_text}{conflict_tip}\n"
                f"特别号防守：{defense_text}\n"
                f"信心指数：{confidence:.0f}/100\n"
                f"建议投入：{advice}\n"
                f"六策略极强号：{strong_special_text}（{strong_zodiac_text}）\n"
                f"六策略特别号组：{strategy_special_text}\n"
                f"六策略生肖组：{strategy_zodiac_text}\n"
                f"特别号综合汇总（各策略去重）：{all_specials_str}\n"
                f"最终投票特别号（前三热门）：{top_special_str}\n"
                f"三中三预测（综合20码池+动态权重）：{trio_str}\n"
                f"详情请运行 python hk_predict.py show"
            )
            send_pushplus_notification(f"香港六合彩预测 {issue_no}", content)

# ---------- 高级框架类 (UltimateTwoInOneFramework 等) ----------
class UltimateSingleZodiacFramework:
    """一生肖高级预测框架"""
    def __init__(self):
        self.recent_window = 10
        self.special_boost = 2.8
        self.omit_power = 1.0
        self.omit_max_boost = 2.1
        self.interval_penalty = 0.45
        self.recent5_boost = 1.15
        self.cold_fallback = True
        self._confidence_threshold = 0.72
        self._miss_streak = 0

    def predict_single_zodiac(self, conn, issue_no=None) -> str:
        if issue_no:
            rows = conn.execute(
                """SELECT numbers_json, special_number FROM draws 
                   WHERE draw_date < (SELECT draw_date FROM draws WHERE issue_no = ?)
                   OR (draw_date = (SELECT draw_date FROM draws WHERE issue_no = ?) AND issue_no < ?)
                   ORDER BY draw_date DESC, issue_no DESC LIMIT 50""",
                (issue_no, issue_no, issue_no)
            ).fetchall()
        else:
            rows = conn.execute("SELECT numbers_json, special_number FROM draws ORDER BY draw_date DESC LIMIT 50").fetchall()
        if not rows:
            return "马"
        scores = self._compute_single_score(rows)
        omission = _zodiac_omission_map(rows)
        latest_special_z = get_zodiac_by_number(int(rows[0]["special_number"]))
        scores[latest_special_z] += 1.2
        if self._miss_streak >= 1:
            cold_z = max(omission, key=omission.get)
            scores[cold_z] += self._miss_streak * 2.0
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        best = ranked[0][0]
        if self.cold_fallback and ranked[0][1] < self._confidence_threshold:
            recent_hot = Counter(get_zodiac_by_number(int(r["special_number"])) for r in rows[:6]).most_common(2)
            hot_z = recent_hot[0][0] if recent_hot else "马"
            cold_z = max(omission, key=omission.get)
            latest_z = get_zodiac_by_number(int(rows[0]["special_number"]))
            if omission.get(latest_z, 0) >= 3:
                best = latest_z
            elif omission.get(hot_z, 0) >= 4:
                best = hot_z
            else:
                best = cold_z
        return best

    def _compute_single_score(self, rows):
        scores = {z: 0.0 for z in ZODIAC_MAP}
        omission = _zodiac_omission_map(rows)
        max_omit = max(omission.values()) if omission else 1
        recent = rows[:self.recent_window]
        recent_specials = [get_zodiac_by_number(int(r["special_number"])) for r in recent]
        recent_main = [get_zodiac_by_number(int(n)) for r in recent for n in json.loads(r["numbers_json"])]
        recent_special_counter = Counter(recent_specials)
        recent_main_counter = Counter(recent_main)
        for idx, row in enumerate(recent):
            w = 1.0 / (1.0 + idx * 0.08)
            nums = json.loads(row["numbers_json"])
            sp = int(row["special_number"])
            for n in nums:
                scores[get_zodiac_by_number(n)] += w * 0.85
            scores[get_zodiac_by_number(sp)] += self.special_boost * w
        for z, cnt in recent_special_counter.items():
            scores[z] += cnt * 1.35
        for z, cnt in recent_main_counter.items():
            scores[z] += cnt * 0.25
        for z, omit in omission.items():
            boost = (omit / max_omit) ** self.omit_power * self.omit_max_boost
            if omit >= 4:
                boost += 1.1
            scores[z] += boost
        seq_feat = _zodiac_sequence_features(list(rows))
        for z, feat in seq_feat.items():
            if feat["last_seen"] > 8:
                scores[z] += (feat["last_seen"] / 28.0) * 1.7
            if feat["avg_interval"] > 18:
                scores[z] -= self.interval_penalty
            scores[z] += self.recent5_boost * feat["recent5_cnt"]
        if recent_specials:
            scores[recent_specials[0]] += 0.9
            if len(recent_specials) > 1:
                scores[recent_specials[1]] += 0.35    
        for z in ZODIAC_MAP:
            scores[z] += _markov_two_predictor(rows, z) * 0.5
        return scores

    def update_miss_streak(self, missed: bool):
        self._miss_streak = 0 if missed else self._miss_streak + 1

class UltimateTwoInOneFramework:
    """二中一高级预测框架"""
    def __init__(self):
        self.recent_window = 12
        self.special_boost = 1.8
        self.interval_penalty = 0.6
        self.recent5_boost = 0.9
        self.cold_omit_threshold = 5
        self._miss_streak = 0

    def predict_single_zodiac(self, conn, issue_no=None):
        picks = self.predict_two_zodiac(conn, issue_no)
        return picks[0] if picks else "马"

    def predict_two_zodiac(self, conn, issue_no=None) -> List[str]:
        if issue_no:
            rows = conn.execute(
                """SELECT numbers_json, special_number FROM draws 
                   WHERE draw_date < (SELECT draw_date FROM draws WHERE issue_no = ?)
                   OR (draw_date = (SELECT draw_date FROM draws WHERE issue_no = ?) AND issue_no < ?)
                   ORDER BY draw_date DESC, issue_no DESC LIMIT 50""",
                (issue_no, issue_no, issue_no)
            ).fetchall()
        else:
            rows = conn.execute("SELECT numbers_json, special_number FROM draws ORDER BY draw_date DESC LIMIT 50").fetchall()
        if not rows:
            return ["马", "蛇"]
        scores = _compute_twoinone_score(rows, self.recent_window, self.cold_omit_threshold + 10,
                                         self.special_boost, self.interval_penalty, self.recent5_boost)
        if self._miss_streak >= 2:
            omission = _zodiac_omission_map(rows)
            cold_z = max(omission, key=omission.get)
            scores[cold_z] += self._miss_streak * 2.5
            for z in list(scores.keys()):
                if z != cold_z:
                    scores[z] *= 0.85
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        picks = [ranked[0][0]]
        for z, _ in ranked[1:]:
            if z not in picks:
                picks.append(z)
                break
        if len(picks) < 2:
            picks.append("蛇" if picks[0] != "蛇" else "马")
        if picks[0] == picks[1]:
            picks[1] = "蛇" if picks[0] != "蛇" else "马"
        if max(scores.values()) < 0.65:
            omission = _zodiac_omission_map(rows)
            hot = ranked[0][0]
            cold = max(omission, key=omission.get)
            picks = [hot, cold] if hot != cold else ["马", "蛇"]
        return picks[:2]

    def update_miss_streak(self, missed: bool):
        self._miss_streak = 0 if missed else self._miss_streak + 1

class UltimateSpecialFourFramework:
    """特别号四生肖高级预测框架"""
    def __init__(self):
        self.recent_special_window = 10
        self.cold_omit_threshold = 5
        self._miss_streak = 0

    def predict_single_zodiac(self, conn, issue_no=None):
        picks = self.predict_four_zodiac(conn, issue_no)
        return picks[0] if picks else "马"

    def predict_four_zodiac(self, conn, issue_no=None) -> List[str]:
        if issue_no:
            rows = conn.execute(
                """SELECT numbers_json, special_number FROM draws 
                   WHERE draw_date < (SELECT draw_date FROM draws WHERE issue_no = ?)
                   OR (draw_date = (SELECT draw_date FROM draws WHERE issue_no = ?) AND issue_no < ?)
                   ORDER BY draw_date DESC, issue_no DESC LIMIT 50""",
                (issue_no, issue_no, issue_no)
            ).fetchall()
        else:
            rows = conn.execute("SELECT numbers_json, special_number FROM draws ORDER BY draw_date DESC LIMIT 50").fetchall()
        if not rows:
            return ["马", "蛇", "龙", "兔"]
        scores = _compute_special_four_score(rows, self.recent_special_window, self.cold_omit_threshold)
        if self._miss_streak >= 2:
            specials = [int(r["special_number"]) for r in rows]
            omission = {z: 0 for z in ZODIAC_MAP}
            for i, sp in enumerate(specials):
                z = get_zodiac_by_number(sp)
                if omission[z] == 0:
                    omission[z] = i + 1
            cold_z = max(omission, key=omission.get)
            scores[cold_z] += self._miss_streak * 2.5
            for z in list(scores.keys()):
                if z != cold_z:
                    scores[z] *= 0.85
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        picks = [z for z, _ in ranked[:4]]
        while len(picks) < 4:
            for z in ZODIAC_MAP:
                if z not in picks:
                    picks.append(z)
                    break
        return picks[:4]

    def update_miss_streak(self, missed: bool):
        self._miss_streak = 0 if missed else self._miss_streak + 1

def get_ultimate_single():
    return UltimateSingleZodiacFramework()

def get_ultimate_two_in_one():
    return UltimateTwoInOneFramework()

def get_ultimate_special_four():
    return UltimateSpecialFourFramework()

# ---------- 历史回溯专用 - 精选特别号 ----------
def get_precise_specials_from_history(history_rows, zodiac_pool, top_n=3):
    if not zodiac_pool: return []
    latest_row = history_rows[0]
    latest_special = int(latest_row['special_number'])
    recent_specials = [int(r['special_number']) for r in history_rows[:12]]
    omission = {}
    for i, sp in enumerate(recent_specials):
        if sp not in omission: omission[sp] = i + 1
    candidates = list(set(n for z in zodiac_pool for n in ZODIAC_MAP.get(z, [])))
    if not candidates: return []
    tail_counter = Counter()
    for row in history_rows[:8]:
        for n in json.loads(row['numbers_json']): tail_counter[n % 10] += 1
    for sp in recent_specials[:8]: tail_counter[sp % 10] += 3
    hot_tails = {t for t, _ in tail_counter.most_common(6)}
    last_tail = latest_special % 10
    neighbor_tails = {last_tail, (last_tail + 1) % 10, (last_tail - 1) % 10}
    selected = []
    penalty_nums = set(recent_specials[:2])
    neighbors = [n for n in candidates if abs(n - latest_special) == 1 and n not in penalty_nums]
    if not neighbors: neighbors = [n for n in candidates if abs(n - latest_special) == 1]
    if neighbors: selected.append(max(neighbors, key=lambda n: omission.get(n, 20)))
    if len(selected) < top_n:
        tail_candidates = [n for n in candidates if n not in selected and n % 10 == last_tail and n not in penalty_nums]
        if not tail_candidates: tail_candidates = [n for n in candidates if n not in selected and n % 10 in neighbor_tails and n not in penalty_nums]
        if not tail_candidates: tail_candidates = [n for n in candidates if n not in selected and n % 10 in hot_tails and n not in penalty_nums]
        if not tail_candidates: tail_candidates = [n for n in candidates if n not in selected and n % 10 == last_tail]
        if tail_candidates: selected.append(max(tail_candidates, key=lambda n: omission.get(n, 20)))
    if len(selected) < top_n:
        neighbors2 = [n for n in candidates if abs(n - latest_special) == 2 and n not in selected and n not in penalty_nums]
        if not neighbors2: neighbors2 = [n for n in candidates if abs(n - latest_special) == 2 and n not in selected]
        if neighbors2: selected.append(max(neighbors2, key=lambda n: omission.get(n, 20)))
    if len(selected) < top_n:
        cold_pool = [n for n in candidates if n not in selected and n != latest_special]
        if cold_pool: selected.append(max(cold_pool, key=lambda n: omission.get(n, 20)))
    if len(selected) < top_n:
        remaining = [n for n in candidates if n not in selected]
        remaining.sort(key=lambda n: omission.get(n, 20), reverse=True)
        for n in remaining:
            selected.append(n)
            if len(selected) >= top_n: break
    if not selected:
        omission_sp = {n: 30 for n in range(1, 50)}
        for idx, sp in enumerate(recent_specials):
            omission_sp[sp] = min(omission_sp[sp], idx + 1)
        selected = [n for n, _ in sorted(omission_sp.items(), key=lambda x: -x[1])[:top_n]]
    return selected[:top_n]

def get_precise_specials_for_issue(conn, issue_no, zodiac_pool, top_n=3):
    latest_row = conn.execute(
        "SELECT issue_no, draw_date FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT 1"
    ).fetchone()
    if not latest_row:
        return []
    latest_main = json.loads(conn.execute(
        "SELECT numbers_json FROM draws ORDER BY draw_date DESC LIMIT 1"
    ).fetchone()["numbers_json"])
    special, confidence, defenses = _generate_special_number_v4(conn, latest_main, issue_no)
    picks = [special] + defenses[:top_n-1]
    return picks[:top_n]

def log_special_picks(conn: sqlite3.Connection, issue_no: str, picks: Sequence[int], special_number: Optional[int] = None) -> None:
    try:
        special_hit = 0
        if special_number is not None:
            special_hit = 1 if int(special_number) in {int(n) for n in picks} else 0
        conn.execute(
            "INSERT OR REPLACE INTO special_picks_log(issue_no, picks_json, hit_count, special_hit, created_at) VALUES (?, ?, ?, ?, ?)",
            (issue_no, json.dumps([int(n) for n in picks], ensure_ascii=False), special_hit, special_hit, utc_now()),
        )
        conn.commit()
    except Exception:
        pass

def get_recent_special_picks_report(conn: sqlite3.Connection, lookback: int = 20) -> Dict[str, float]:
    rows = conn.execute(
        "SELECT issue_no, picks_json, special_hit FROM special_picks_log ORDER BY id DESC LIMIT ?",
        (lookback,),
    ).fetchall()
    if not rows: return {"samples": 0.0, "hit_rate": 0.0, "max_miss_streak": 0.0}
    hits = 0
    miss_streak = 0
    max_miss = 0
    for row in rows:
        hit = int(row["special_hit"] or 0)
        hits += hit
        if hit == 0:
            miss_streak += 1
            max_miss = max(max_miss, miss_streak)
        else:
            miss_streak = 0
    samples = len(rows)
    return {"samples": float(samples), "hit_rate": float(hits / samples), "max_miss_streak": float(max_miss)}

def backfill_special_picks_log(conn, max_issues=100):
    draws = conn.execute(
        "SELECT issue_no, draw_date, special_number FROM draws ORDER BY draw_date ASC"
    ).fetchall()
    if len(draws) < 16:
        print("数据不足，无法回溯（至少需要16期）。")
        return 0
    count = 0
    for i in range(12, len(draws)):
        target_issue = draws[i]['issue_no']
        target_date = draws[i]['draw_date']
        existing = conn.execute(
            "SELECT 1 FROM special_picks_log WHERE issue_no = ?", (target_issue,)
        ).fetchone()
        if existing:
            continue
        history = conn.execute(
            """SELECT numbers_json, special_number FROM draws 
               WHERE draw_date < ? OR (draw_date = ? AND issue_no < ?)
               ORDER BY draw_date DESC, issue_no DESC
               LIMIT 16""",
            (target_date, target_date, target_issue)
        ).fetchall()
        if len(history) < 12:
            continue
        base_four = _get_four_zodiac_from_history_rows(history, conn)
        recent_zodiacs = [get_zodiac_by_number(int(r['special_number'])) for r in history[:8]]
        zodiac_freq = Counter(recent_zodiacs)
        specials_hist = [int(r['special_number']) for r in history[:30]]
        omission_zodiac = {z: 0 for z in ZODIAC_MAP}
        for i_sp, sp in enumerate(specials_hist):
            z = get_zodiac_by_number(sp)
            if omission_zodiac[z] == 0:
                omission_zodiac[z] = i_sp + 1
        sorted_omit = sorted(omission_zodiac.items(), key=lambda x: -x[1])
        extra_freq = [z for z, _ in zodiac_freq.most_common(3) if z not in base_four][:2]
        extra_cold = [z for z, _ in sorted_omit if z not in base_four and z not in extra_freq][:2]
        last3_zodiacs = [get_zodiac_by_number(int(r['special_number'])) for r in history[:3]]
        latest_main = json.loads(history[0]['numbers_json'])
        main_counter = Counter(get_zodiac_by_number(n) for n in latest_main)
        top_main = main_counter.most_common(1)[0][0] if main_counter else None
        zodiac_pool = base_four + extra_freq + extra_cold + last3_zodiacs + ([top_main] if top_main else [])
        seen = set()
        final_pool = []
        for z in zodiac_pool:
            if z not in seen:
                seen.add(z)
                final_pool.append(z)
        while len(final_pool) < 8:
            for z in ZODIAC_MAP:
                if z not in final_pool:
                    final_pool.append(z)
                if len(final_pool) == 8: break
        zodiac_pool = final_pool[:8]
        picks = get_precise_specials_from_history(history, zodiac_pool, top_n=3)
        if picks:
            actual_special_row = conn.execute(
                "SELECT special_number FROM draws WHERE issue_no = ?", (target_issue,)
            ).fetchone()
            actual_special = actual_special_row['special_number'] if actual_special_row else None
            special_hit = 1 if actual_special is not None and actual_special in picks else 0
            conn.execute(
                "INSERT OR IGNORE INTO special_picks_log (issue_no, picks_json, special_hit, created_at) VALUES (?, ?, ?, ?)",
                (target_issue, json.dumps(picks), special_hit, utc_now())
            )
            count += 1
        if count >= max_issues:
            break
    conn.commit()
    return count

def _get_four_zodiac_from_history_rows(rows, conn=None):
    if len(rows) < 3:
        return ["马", "蛇", "龙", "兔"]
    params = load_best_zodiac_params()
    four_boost = float(params.get("four_boost", 1.4221))
    return _strategy_four_boosted(rows, four_boost)

def _strategy_four_boosted(rows, four_boost):
    omission = {z: 0 for z in ZODIAC_MAP}
    specials = [_row_special(r) for r in rows]
    for i, sp in enumerate(specials[::-1]):
        z = get_zodiac_by_number(sp)
        if omission[z] == 0: omission[z] = i + 1
    for z in omission: omission[z] *= four_boost
    sorted_cold = sorted(omission.items(), key=lambda x: -x[1])
    picks = [z for z, _ in sorted_cold[:3]]
    latest_z = get_zodiac_by_number(specials[-1]) if specials else None
    if latest_z and latest_z not in picks:
        picks.append(latest_z)
    else:
        for z, _ in sorted_cold[3:]:
            if z not in picks: picks.append(z); break
    return picks[:4]

def _row_numbers(r):
    if isinstance(r, dict):
        return json.loads(r["numbers_json"])
    if hasattr(r, "keys") and "numbers_json" in r.keys():
        return json.loads(r["numbers_json"])
    return json.loads(r[0]) if isinstance(r[0], str) else r[0]

def _row_special(r):
    if isinstance(r, dict):
        return int(r["special_number"])
    if hasattr(r, "keys") and "special_number" in r.keys():
        return int(r["special_number"])
    return int(r[1])

# ---------- 自动优化 ----------
def rolling_cv_score(conn, params, lookback=50, test_size=10):
    if np is None:
        return 0.0, 999, 0.0, 999
    draws = _draws_ordered_asc(conn)
    if len(draws) < lookback + test_size:
        return 0.0, 999, 0.0, 999

    single_fw = get_ultimate_single()
    two_fw = get_ultimate_two_in_one()
    four_fw = get_ultimate_special_four()
    for k, v in params.items():
        if k.startswith("single_"):
            setattr(single_fw, k[7:], v)
        elif k.startswith("two_"):
            setattr(two_fw, k[4:], v)
        elif k.startswith("four_"):
            setattr(four_fw, k[5:], v)

    hit_rates_two = []
    max_misses_two = []
    hit_rates_four = []
    max_misses_four = []
    start = lookback
    while start + test_size <= len(draws):
        test_issues = draws[start:start + test_size]
        hits_two = hits_four = 0
        max_miss_two = miss_two = 0
        max_miss_four = miss_four = 0
        for test_row in test_issues:
            issue = test_row["issue_no"]
            win_main = json.loads(test_row["numbers_json"])
            win_sp = int(test_row["special_number"])
            win_z = {get_zodiac_by_number(n) for n in win_main}
            win_z.add(get_zodiac_by_number(win_sp))
            picks_two = two_fw.predict_two_zodiac(conn, issue)
            if any(z in win_z for z in picks_two):
                hits_two += 1
                miss_two = 0
            else:
                miss_two += 1
                max_miss_two = max(max_miss_two, miss_two)
            picks_four = four_fw.predict_four_zodiac(conn, issue)
            if get_zodiac_by_number(win_sp) in picks_four:
                hits_four += 1
                miss_four = 0
            else:
                miss_four += 1
                max_miss_four = max(max_miss_four, miss_four)
        hit_rates_two.append(hits_two / test_size)
        max_misses_two.append(max_miss_two)
        hit_rates_four.append(hits_four / test_size)
        max_misses_four.append(max_miss_four)
        start += test_size

    avg_hit_two = float(np.mean(hit_rates_two)) if hit_rates_two else 0.0
    avg_max_miss_two = float(np.mean(max_misses_two)) if max_misses_two else 999.0
    avg_hit_four = float(np.mean(hit_rates_four)) if hit_rates_four else 0.0
    avg_max_miss_four = float(np.mean(max_misses_four)) if max_misses_four else 999.0
    return avg_hit_two, avg_max_miss_two, avg_hit_four, avg_max_miss_four

def evaluate_zodiac_performance(conn, params, lookback=20):
    import shutil
    backup_path = _BEST_PARAMS_PATH.with_suffix(".backup")
    if _BEST_PARAMS_PATH.exists():
        shutil.copy(_BEST_PARAMS_PATH, backup_path)
    try:
        with open(_BEST_PARAMS_PATH, "w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False)
        single_rep = get_recent_single_zodiac_report(conn, lookback=lookback)
        two_rep = get_recent_two_zodiac_report(conn, lookback=lookback)
        three_rep = get_recent_three_zodiac_report(conn, lookback=lookback)
        four_rep = get_recent_texiao5_report(conn, lookback=lookback)
        hit_rates = {
            'single': single_rep['hit_rate'],
            'two': two_rep['hit_rate'],
            'three': three_rep['hit_rate'],
            'special': four_rep['hit_rate']
        }
        max_misses = {
            'single': single_rep['max_miss_streak'],
            'two': two_rep['max_miss_streak'],
            'three': three_rep['max_miss_streak'],
            'special': four_rep['max_miss_streak']
        }
        return hit_rates, max_misses
    finally:
        if backup_path.exists():
            shutil.copy(backup_path, _BEST_PARAMS_PATH)
            backup_path.unlink()

def auto_optimize_loop(conn, target_hit_rate=0.90, target_max_miss=1,
                       timeout_hours=5, base_trials=500):
    try:
        import optuna
        from optuna.samplers import TPESampler
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("❌ 请安装 optuna: pip install optuna")
        return None
    online_adjuster.conn = conn
    online_adjuster.load_state()
    dyn_single_w = online_adjuster.w_single
    dyn_two_w = online_adjuster.w_two
    dyn_three_w = online_adjuster.w_three
    dyn_four_w = 0.55
    start_time = time.time()
    timeout_seconds = timeout_hours * 3600

    total_rows = conn.execute("SELECT COUNT(*) FROM draws").fetchone()[0]
    print(f"✅ 当前数据库共 {total_rows} 期开奖记录")

    elapsed = time.time() - start_time
    remaining_seconds = timeout_seconds - elapsed
    max_possible = max(1, int(remaining_seconds / 30))
    effective_trials = min(base_trials, max_possible)
    print(f"⏱️ 预估可运行 {effective_trials} 次试验（总限制 {timeout_hours}h，已用 {elapsed:.0f}s）")

    def get_params(trial):
        return {
            "single_recent_window": trial.suggest_int("single_recent_window", 4, 50),
            "single_special_boost": trial.suggest_float("single_special_boost", 1.0, 6.0),
            "single_omit_power": trial.suggest_float("single_omit_power", 0.5, 3.0),
            "single_omit_max_boost": trial.suggest_float("single_omit_max_boost", 1.0, 10.0),
            "single_interval_penalty": trial.suggest_float("single_interval_penalty", 0.1, 2.0),
            "single_recent5_boost": trial.suggest_float("single_recent5_boost", 0.3, 2.5),
            "two_recent_window": trial.suggest_int("two_recent_window", 5, 30),
            "two_special_boost": trial.suggest_float("two_special_boost", 1.0, 3.5),
            "two_interval_penalty": trial.suggest_float("two_interval_penalty", 0.1, 1.8),
            "two_recent5_boost": trial.suggest_float("two_recent5_boost", 0.3, 2.2),
            "two_cold_omit_threshold": trial.suggest_int("two_cold_omit_threshold", 2, 15),
            "three_hot_weight": trial.suggest_float("three_hot_weight", 1.5, 5.0),
            "three_omit_weight": trial.suggest_float("three_omit_weight", 0.8, 4.0),
            "three_tail_weight": trial.suggest_float("three_tail_weight", 0.5, 2.5),
            "four_recent_special_window": trial.suggest_int("four_recent_special_window", 8, 25),
            "four_cold_omit_threshold": trial.suggest_int("four_cold_omit_threshold", 3, 25),
            "single_w_rule": trial.suggest_float("single_w_rule", 0.2, 0.8),
            "single_w_lgb": trial.suggest_float("single_w_lgb", 0.2, 0.8),
            "single_w_transformer": trial.suggest_float("single_w_transformer", 0.0, 0.4),
            "single_w_markov": trial.suggest_float("single_w_markov", 0.0, 0.3),
            "four_w_rule": trial.suggest_float("four_w_rule", 0.2, 0.8),
            "four_w_lgb": trial.suggest_float("four_w_lgb", 0.2, 0.8),
            "four_w_transformer": trial.suggest_float("four_w_transformer", 0.0, 0.5),
            "four_w_markov": trial.suggest_float("four_w_markov", 0.0, 0.3),
        }

    def objective(trial):
        params = get_params(trial)
        draws = _draws_ordered_asc(conn)
        if len(draws) < 80:
            return 0.0
        test_start = max(30, len(draws) - 30)
        count = 0
        hits = {'single':0, 'two':0, 'three':0, 'four':0}
        max_miss = {'single':0, 'two':0, 'three':0, 'four':0}
        streak = {'single':0, 'two':0, 'three':0, 'four':0}
        for i in range(test_start, len(draws)):
            issue = draws[i]["issue_no"]
            hist_rows = _get_rows_before_issue(conn, issue, limit=80)
            if len(hist_rows) < 25:
                continue
            win_z = {get_zodiac_by_number(n) for n in json.loads(draws[i]["numbers_json"])}
            win_z.add(get_zodiac_by_number(int(draws[i]["special_number"])))
            params['four_miss_streak'] = streak['four']
            pred_single = get_single_zodiac_pick(conn, issue, window=16)
            pred_two = predict_strong_two(hist_rows, params)
            pred_three = predict_strong_three(hist_rows, params)
            pred_four = predict_strong_four(hist_rows, params)
            if pred_single in win_z:
                hits['single'] += 1
                streak['single'] = 0
            else:
                streak['single'] += 1
                max_miss['single'] = max(max_miss['single'], streak['single'])
            if any(z in win_z for z in pred_two):
                hits['two'] += 1
                streak['two'] = 0
            else:
                streak['two'] += 1
                max_miss['two'] = max(max_miss['two'], streak['two'])
            hit3 = sum(1 for z in pred_three if z in win_z)
            if hit3 >= 2:
                hits['three'] += 1
                streak['three'] = 0
            else:
                streak['three'] += 1
                max_miss['three'] = max(max_miss['three'], streak['three'])
            pred_five = predict_strong_five(hist_rows, params)
            actual_sp_z = get_zodiac_by_number(int(draws[i]["special_number"]))
            if actual_sp_z in pred_five:
                hits['four'] += 1
                streak['four'] = 0
            else:
                streak['four'] += 1
                max_miss['four'] = max(max_miss['four'], streak['four'])
            count += 1
        if count < 15:
            return 0.0
        rates = {k: hits[k] / count for k in hits}
        three_extra_boost = 1.3
        four_extra_boost = 3.5
        score = (rates['single'] * dyn_single_w +
                 rates['two']    * dyn_two_w +
                 rates['three']  * dyn_three_w * three_extra_boost +
                 rates['four']   * dyn_four_w * four_extra_boost)
        score = score * 10.0
        penalty_scale = max(0.05, rates['four'])
        score -= smooth_penalty(max_miss['single'], 1, 0.008)
        score -= smooth_penalty(max_miss['two'],    1, 0.008)
        score -= smooth_penalty(max_miss['three'], 3, 0.003)
        score -= smooth_penalty(max_miss['four'],  1, 0.015)
        for k in rates:
            trial.set_user_attr(f"rate_{k}", round(rates[k], 4))
            trial.set_user_attr(f"max_miss_{k}", int(max_miss[k]))
        return score

    def print_callback(study, trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            best = study.best_trial
            attrs = trial.user_attrs
            if "error" in attrs:
                print(f"[试验 {trial.number + 1}/{effective_trials}] 错误: {attrs['error']}")
                return
            print(
                f"[试验 {trial.number + 1}/{effective_trials}] "
                f"得分={trial.value:.4f} | "
                f"一肖={attrs.get('rate_single', 0.0):.1%}(连空{attrs.get('max_miss_single', 0)}) "
                f"二肖={attrs.get('rate_two', 0.0):.1%}(连空{attrs.get('max_miss_two', 0)}) "
                f"三肖={attrs.get('rate_three', 0.0):.1%}(连空{attrs.get('max_miss_three', 0)}) "
                f"四肖={attrs.get('rate_four', 0.0):.1%}(连空{attrs.get('max_miss_four', 0)}) | "
                f"当前最佳得分={best.value:.4f}",
                flush=True
            )

    storage_name = f"sqlite:///{SCRIPT_DIR / 'optuna_study.db'}"
    study = optuna.create_study(
        study_name="zodiac_strict",
        direction="maximize",
        sampler=TPESampler(seed=42, multivariate=True),
        storage=storage_name,
        load_if_exists=True
    )

    print(f"🚀 开始严格优化，试验次数={effective_trials}")
    study.optimize(objective, n_trials=effective_trials, timeout=timeout_seconds,
                   callbacks=[print_callback], show_progress_bar=False)

    best_params = study.best_params
    best_score = study.best_value

    final_trial = optuna.trial.FixedTrial(best_params)
    objective(final_trial)
    attrs = final_trial.user_attrs

    print(f"\n🏆 优化完成，最佳评分: {best_score:.4f}")
    print(f"   一生肖命中率: {attrs.get('rate_single', 0.0)*100:.1f}%  最大连空 {attrs.get('max_miss_single', 0)}")
    print(f"   二生肖命中率: {attrs.get('rate_two', 0.0)*100:.1f}%  最大连空 {attrs.get('max_miss_two', 0)}")
    print(f"   三生肖命中率: {attrs.get('rate_three', 0.0)*100:.1f}%  最大连空 {attrs.get('max_miss_three', 0)}")
    print(f"   四生肖命中率: {attrs.get('rate_four', 0.0)*100:.1f}%  最大连空 {attrs.get('max_miss_four', 0)}")

    with open(_BEST_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)
    print(f"✅ 最佳参数已保存至 {_BEST_PARAMS_PATH}")
    return best_params

def print_callback(study, trial):
    if trial.state == "COMPLETE":
        best = study.best_trial
        attrs = trial.user_attrs
        if "error" in attrs:
            print(f"[试验 {trial.number + 1}/{study.trials_count}] 错误: {attrs['error']}")
            return
        print(
            f"[试验 {trial.number + 1}/{study.trials_count}] "
            f"得分={trial.value:.4f} | "
            f"一肖={attrs.get('rate_single', 0.0):.1%}(连空{attrs.get('max_miss_single', 0)}) "
            f"二肖={attrs.get('rate_two', 0.0):.1%}(连空{attrs.get('max_miss_two', 0)}) "
            f"三肖={attrs.get('rate_three', 0.0):.1%}(连空{attrs.get('max_miss_three', 0)}) "
            f"四肖={attrs.get('rate_four', 0.0):.1%}(连空{attrs.get('max_miss_four', 0)}) | "
            f"当前最佳得分={best.value:.4f}",
            flush=True
        )

# ---------- 命令行处理函数 ----------
def cmd_bootstrap(args: argparse.Namespace) -> None:
    conn = connect_db(args.db)
    try:
        init_db(conn)
        records = fetch_hk_mark_six_records(timeout=args.api_timeout, retries=args.api_retries)
        total, inserted, updated = sync_from_records(conn, records, source="hk_mark_six_api")
        print(f"自动执行轻量回测（最近{BACKTEST_ISSUES_DEFAULT}期）...")
        run_historical_backtest(conn, rebuild=True, max_issues=BACKTEST_ISSUES_DEFAULT)
        issue = generate_predictions(conn)
        print(f"Bootstrap done. total={total}, inserted={inserted}, updated={updated}, next_prediction={issue}")
    finally:
        conn.close()

def cmd_train_xgb(args: argparse.Namespace) -> None:
    conn = connect_db(args.db)
    try:
        init_db(conn)
        predictor = XGBoostPredictor()
        predictor.train(conn)
        model_path = SCRIPT_DIR / 'xgb_model.pkl'
        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump(predictor, f)
        print(f"XGBoost 模型已训练并保存至 {model_path}")
    finally:
        conn.close()

def cmd_train_lgb(args: argparse.Namespace) -> None:
    conn = connect_db(args.db)
    try:
        init_db(conn)
        predictor = LightGBMPredictor()
        predictor.train(conn)
        model_path = SCRIPT_DIR / 'lgb_model.pkl'
        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump(predictor, f)
        print(f"LightGBM 模型已训练并保存至 {model_path}")
    finally:
        conn.close()

def cmd_force_retrain(args: argparse.Namespace) -> None:
    cmd_train_xgb(args)
    cmd_train_lgb(args)
    print("所有 ML 模型已重新训练。")

def cmd_sync(args: argparse.Namespace) -> None:
    conn = connect_db(args.db)
    try:
        init_db(conn)
        records = fetch_hk_mark_six_records(timeout=args.api_timeout, retries=args.api_retries)
        if args.require_continuity:
            missing = missing_issues_since_latest(conn, records)
            if missing:
                raise RuntimeError(
                    f"Continuity check failed. Missing {len(missing)} issues, sample={','.join(missing[:10])}"
                )
        total, inserted, updated = sync_from_records(conn, records, source="hk_mark_six_api")
        mined_cfg = ensure_mined_pattern_config(conn, force=args.remine)
        reviewed = review_latest(conn)
        bt_issues, bt_runs = 0, 0
        if args.with_backtest:
            bt_issues, bt_runs = run_historical_backtest(conn, rebuild=False, max_issues=BACKTEST_ISSUES_DEFAULT)
        issue = generate_predictions(conn)
        patched = backfill_missing_special_picks(conn)
        print(f"Sync done. total={total}, inserted={inserted}, updated={updated}, reviewed={reviewed}, next_prediction={issue}")
        print(f"Mined config: {json.dumps(mined_cfg, ensure_ascii=False)}")
        if bt_issues > 0:
            print(f"Backtest updated. issues={bt_issues}, strategy_runs={bt_runs}")
        if patched > 0:
            print(f"Patched missing special picks: {patched}")
        online_adjuster.conn = conn
        online_adjuster.load_state()
        online_adjuster.adjust()
    finally:
        conn.close()

def cmd_reset_and_auto(args: argparse.Namespace) -> None:
    conn = connect_db(args.db)
    try:
        init_db(conn)
        conn.execute("DELETE FROM prediction_picks")
        conn.execute("DELETE FROM prediction_pools")
        conn.execute("DELETE FROM prediction_runs")
        conn.execute("DELETE FROM strategy_performance")
        conn.execute("DELETE FROM special_picks_log")
        conn.execute("DELETE FROM model_state WHERE key = ?", (MINED_CONFIG_KEY,))
        conn.commit()
        optuna_path = SCRIPT_DIR / "optuna_macau_stable.db"
        if optuna_path.exists():
            os.remove(optuna_path)
            print(f"[重置] 已删除 {optuna_path}")
        records = fetch_hk_mark_six_recent_records(limit=120, timeout=args.api_timeout, retries=args.api_retries)
        total, inserted, updated = sync_from_records(conn, records, source="hk_mark_six_api_recent_120")
        mined_cfg = ensure_mined_pattern_config(conn, force=True)
        max_retries = 5
        trials = getattr(args, "trials", 1000)
        optimize_script = SCRIPT_DIR / "hyper_optimize_ultimate_target.py"
        debug_flag = ["--debug"] if getattr(args, "debug", False) else []
        if optimize_script.exists():
            for attempt in range(1, max_retries + 1):
                print(f"\n====== 第 {attempt} 次优化 (trials={trials}) ======")
                ret = subprocess.run(
                    [sys.executable, str(optimize_script),
                     "--db", args.db,
                     "--recent", "120",
                     "--trials", str(trials)] + debug_flag,
                    capture_output=False,
                )
                if ret.returncode == 0:
                    print(f"🎉 第 {attempt} 次优化已达标！")
                    break
                else:
                    print(f"第 {attempt} 次优化未达标（退出码 {ret.returncode}），继续下一轮...")
                    trials = int(trials * 1.3)
        else:
            print(f"⚠️ 跳过优化脚本：{optimize_script} 不存在")
        run_historical_backtest(conn, rebuild=True, max_issues=120)
        generate_predictions(conn)
        print(f"Reset+auto done. total={total}, inserted={inserted}, updated={updated}")
        print(f"Mined config: {json.dumps(mined_cfg, ensure_ascii=False)}")
    finally:
        conn.close()

def cmd_sync_recent(args: argparse.Namespace) -> None:
    conn = connect_db(args.db)
    try:
        init_db(conn)
        records = fetch_hk_mark_six_recent_records(
            limit=args.limit,
            timeout=args.api_timeout,
            retries=args.api_retries,
        )
        total, inserted, updated = sync_from_records(conn, records, source="hk_mark_six_api_recent")
        print(f"Recent sync done. limit={args.limit}, total={total}, inserted={inserted}, updated={updated}")
    finally:
        conn.close()

def cmd_predict(args: argparse.Namespace) -> None:
    conn = connect_db(args.db)
    try:
        init_db(conn)
        issue = generate_predictions(conn, issue_no=args.issue)
        patched = backfill_missing_special_picks(conn)
        print(f"Predictions generated for {issue}")
        if patched > 0:
            print(f"Patched missing special picks: {patched}")
    finally:
        conn.close()

def cmd_review(args: argparse.Namespace) -> None:
    conn = connect_db(args.db)
    try:
        init_db(conn)
        reviewed = review_issue(conn, args.issue) if args.issue else review_latest(conn)
        print(f"Reviewed runs: {reviewed}")
    finally:
        conn.close()

def cmd_show(args: argparse.Namespace) -> None:
    conn = connect_db(args.db)
    try:
        init_db(conn)
        backfill_missing_special_picks(conn)
        reviewed_count = conn.execute(
            "SELECT COUNT(*) FROM prediction_runs WHERE status='REVIEWED'"
        ).fetchone()[0]
        if reviewed_count < 10:
            print("检测到复盘数据不足，自动执行 sync --with-backtest ...")
            records = fetch_hk_mark_six_records(timeout=args.api_timeout, retries=args.api_retries)
            sync_from_records(conn, records, source="hk_mark_six_api")
            run_historical_backtest(conn, rebuild=False, max_issues=20)
            generate_predictions(conn)
            print("自动同步与回测完成。")
        xgb_pool20 = None
        model_path_xgb = SCRIPT_DIR / 'xgb_model.pkl'
        if model_path_xgb.exists():
            import pickle
            with open(model_path_xgb, 'rb') as f:
                xgb_predictor = pickle.load(f)
            try:
                xgb_pool20 = xgb_predictor.predict_pool(conn, top_k=20)
                print(f"[XGB] 已加载模型，预测主号池 Top20: {xgb_pool20}")
            except Exception as e:
                print(f"[XGB] 预测失败（将使用原策略融合）: {e}")
                xgb_pool20 = None
        lgb_pool20 = None
        model_path_lgb = SCRIPT_DIR / 'lgb_model.pkl'
        if model_path_lgb.exists():
            import pickle
            try:
                with open(model_path_lgb, 'rb') as f:
                    lgb_predictor = pickle.load(f)
                lgb_pool20 = lgb_predictor.predict_pool(conn, top_k=20)
                print(f"[LGB] 已加载模型，预测主号池 Top20: {lgb_pool20}")
            except Exception as e:
                print(f"[LGB] 加载或预测失败（将仅使用XGB）: {e}")
                lgb_pool20 = None
        merged_pool20 = None
        if xgb_pool20 and lgb_pool20:
            union = []
            seen = set()
            max_len = max(len(xgb_pool20), len(lgb_pool20))
            for i in range(max_len):
                if i < len(xgb_pool20) and xgb_pool20[i] not in seen:
                    union.append(xgb_pool20[i])
                    seen.add(xgb_pool20[i])
                if i < len(lgb_pool20) and lgb_pool20[i] not in seen:
                    union.append(lgb_pool20[i])
                    seen.add(lgb_pool20[i])
            merged_pool20 = union[:20]
            merged_zodiacs = [get_zodiac_by_number(n) for n in merged_pool20]
            print(f"[融合] 按双模型一致性加权 Top20: {merged_pool20}")
            print(f"       生肖对应: {' '.join(merged_zodiacs)}")
        elif xgb_pool20:
            merged_pool20 = xgb_pool20
            merged_zodiacs = [get_zodiac_by_number(n) for n in merged_pool20]
            print(f"[融合] XGB 主号池 Top20: {merged_pool20}")
            print(f"       生肖对应: {' '.join(merged_zodiacs)}")
        elif lgb_pool20:
            merged_pool20 = lgb_pool20
            merged_zodiacs = [get_zodiac_by_number(n) for n in merged_pool20]
            print(f"[融合] LGB 主号池 Top20: {merged_pool20}")
            print(f"       生肖对应: {' '.join(merged_zodiacs)}")
        print_dashboard(conn, xgb_pool20=merged_pool20)
    finally:
        conn.close()

def cmd_backtest(args: argparse.Namespace) -> None:
    conn = connect_db(args.db)
    try:
        init_db(conn)
        mined_cfg = ensure_mined_pattern_config(conn, force=args.remine)
        issues, runs = run_historical_backtest(
            conn,
            min_history=args.min_history,
            rebuild=args.rebuild,
            progress_every=args.progress_every,
            max_issues=args.max_issues if hasattr(args, 'max_issues') else BACKTEST_ISSUES_DEFAULT,
        )
        print(f"Backtest done. issues={issues}, strategy_runs={runs}, rebuild={args.rebuild}")
        print(f"Mined config: {json.dumps(mined_cfg, ensure_ascii=False)}")
    finally:
        conn.close()

def cmd_mine(args: argparse.Namespace) -> None:
    conn = connect_db(args.db)
    try:
        init_db(conn)
        cfg = ensure_mined_pattern_config(conn, force=True)
        print(f"Mine done. config={json.dumps(cfg, ensure_ascii=False)}")
    finally:
        conn.close()

def cmd_backfill_special(args: argparse.Namespace) -> None:
    conn = connect_db(args.db)
    try:
        init_db(conn)
        count = backfill_special_picks_log(conn, max_issues=100)
        print(f"已回溯并写入 {count} 期精选特别号记录。")
    finally:
        conn.close()

def cmd_auto_optimize(args: argparse.Namespace) -> None:
    conn = connect_db(args.db)
    try:
        init_db(conn)
        count = conn.execute("SELECT COUNT(*) FROM draws").fetchone()[0]
        print(f"当前数据库共 {count} 期开奖记录")
        if count < 50:
            print("数据不足50期，正在自动同步最新150期...")
            try:
                records = fetch_hk_mark_six_recent_records(limit=150, timeout=args.api_timeout, retries=args.api_retries)
                total, inserted, updated = sync_from_records(conn, records, source="auto_sync")
                print(f"同步完成: 总计 {total} 期, 新增 {inserted}, 更新 {updated}")
                count = conn.execute("SELECT COUNT(*) FROM draws").fetchone()[0]
                if count < 50:
                    print(f"同步后数据仍不足50期（实际{count}期），优化结果可能不准确。")
            except Exception as e:
                print(f"自动同步失败: {e}")
                print("请手动运行: python hk_predict.py sync --with-backtest")
                return
        if count < 30:
            print(f"数据严重不足（{count}期），无法进行有效优化，请先同步数据。")
            return
        best = auto_optimize_loop(
            conn,
            target_hit_rate=args.target_hit_rate,
            target_max_miss=args.target_max_miss,
            timeout_hours=args.timeout_hours,
            base_trials=args.base_trials
        )
        if best:
            run_historical_backtest(conn, rebuild=True, max_issues=30)
            generate_predictions(conn)
            lgb_zodiac_model.train(conn)
            transformer_path = SCRIPT_DIR / "transformer_sp.pth"
            train_transformer(conn, transformer_path)
            backfill_special_picks_log(conn, max_issues=100)
            print_dashboard(conn)
            online_adjuster.conn = conn
            online_adjuster.load_state()
            online_adjuster.adjust()
    finally:
        conn.close()

def cmd_check_data(args: argparse.Namespace) -> None:
    conn = connect_db(args.db)
    try:
        init_db(conn)
        rows = conn.execute(
            "SELECT issue_no, draw_date, numbers_json, special_number FROM draws ORDER BY draw_date ASC, issue_no ASC"
        ).fetchall()
        if not rows:
            print("数据校验结果：数据库中没有开奖记录。")
            return
        issues = [str(r["issue_no"]) for r in rows]
        duplicate_issues = sorted({x for x in issues if issues.count(x) > 1})
        parsed_rows = []
        invalid_rows = []
        for r in rows:
            try:
                nums = json.loads(r["numbers_json"])
            except Exception:
                invalid_rows.append((r["issue_no"], "numbers_json 无法解析"))
                continue
            sp = r["special_number"]
            if not isinstance(nums, list) or len(nums) != 6:
                invalid_rows.append((r["issue_no"], f"主号数量异常: {len(nums) if isinstance(nums, list) else 'N/A'}"))
            bad_nums = [n for n in nums if not isinstance(n, int) or not (1 <= int(n) <= 49)]
            if bad_nums:
                invalid_rows.append((r["issue_no"], f"主号存在非法号码: {bad_nums}"))
            if not isinstance(sp, int) or not (1 <= int(sp) <= 49):
                invalid_rows.append((r["issue_no"], f"特别号非法: {sp}"))
            parsed_rows.append(r)
        issue_keys = [issue_sort_key(x) for x in issues if issue_sort_key(x) is not None]
        missing_issues = []
        if issue_keys:
            width = 3
            first_parsed = parse_issue(issues[0])
            if first_parsed:
                _, _, width = first_parsed
            year_s, seq, _ = parse_issue(issues[0]) or ("00", 0, width)
            current_year = int(year_s)
            current_seq = seq
            expected = set(issues)
            for _ in range(len(issues) + 5):
                current_seq += 1
                if current_seq > 366:
                    current_year += 1
                    current_seq = 1
                candidate = build_issue(str(current_year).zfill(len(year_s)), current_seq, width)
                if candidate not in expected and conn.execute("SELECT 1 FROM draws WHERE issue_no=?", (candidate,)).fetchone() is None:
                    missing_issues.append(candidate)
                    if len(missing_issues) >= 20:
                        break
        date_errors = []
        for r in rows:
            d = str(r["draw_date"])
            if _parse_date(d) is None:
                date_errors.append(r["issue_no"])
        print("数据校验结果")
        print(f"  - 总期数: {len(rows)}")
        print(f"  - 重复期号: {len(duplicate_issues)}")
        if duplicate_issues:
            print(f"    {', '.join(duplicate_issues[:10])}")
        print(f"  - 缺失期号(估算): {len(missing_issues)}")
        if missing_issues:
            print(f"    {', '.join(missing_issues[:10])}")
        print(f"  - 非法记录: {len(invalid_rows)}")
        if invalid_rows:
            for issue_no, msg in invalid_rows[:10]:
                print(f"    {issue_no}: {msg}")
        print(f"  - 日期异常: {len(date_errors)}")
        if date_errors:
            print(f"    {', '.join(date_errors[:10])}")
        if duplicate_issues or missing_issues or invalid_rows or date_errors:
            print("结论：当前数据存在需要清理或核验的问题。")
        else:
            print("结论：未发现明显数据结构异常。")
    finally:
        conn.close()

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="香港六合彩预测工具 - v4全面优化版")
    p.add_argument("--db", default=DB_PATH_DEFAULT, help=f"SQLite db path (default: {DB_PATH_DEFAULT})")
    p.add_argument("--update", action="store_true", help="Quick sync from API (same as sync)")
    p.add_argument("--remine", action="store_true", help="Re-mine pattern config before sync/backtest")
    p.add_argument("--retrain", action="store_true", help="Force retrain XGB model before running")
    p.add_argument("--force-retrain", action="store_true", help="重训所有 ML 模型")
    p.add_argument("--tail-backtest", action="store_true", help="Run tail backtest and print report")
    p.add_argument("--api-timeout", type=int, default=API_TIMEOUT_DEFAULT, help="API timeout seconds per request")
    p.add_argument("--api-retries", type=int, default=API_RETRIES_DEFAULT, help="API retry attempts when network timeout/error occurs")
    p.add_argument("--require-continuity", action="store_true", default=True, help="Fail update when issue sequence has gaps")
    p.add_argument("--no-require-continuity", dest="require_continuity", action="store_false", help="Allow gaps")
    p.add_argument("--with-backtest", action="store_true", help=f"Run incremental backtest after sync (default last {BACKTEST_ISSUES_DEFAULT} issues)")
    sub = p.add_subparsers(dest="command", required=False)

    p_boot = sub.add_parser("bootstrap", help="Initial import from API and generate next issue predictions")
    p_boot.set_defaults(func=cmd_bootstrap)

    p_sync = sub.add_parser("sync", help="Sync draws from API, review latest, generate next prediction")
    p_sync.add_argument("--with-backtest", action="store_true", help=f"Run incremental backtest after sync (default last {BACKTEST_ISSUES_DEFAULT} issues)")
    p_sync.set_defaults(func=cmd_sync)

    p_recent = sub.add_parser("recent", help="Fetch and store only the latest N draws from API")
    p_recent.add_argument("--limit", type=int, default=120, help="Number of recent issues to fetch")
    p_recent.set_defaults(func=cmd_sync_recent)

    p_predict = sub.add_parser("predict", help="Generate predictions for next or specified issue")
    p_predict.add_argument("--issue", help="Target issue, e.g. 26/023")
    p_predict.set_defaults(func=cmd_predict)

    p_review = sub.add_parser("review", help="Review pending runs for latest or specified issue")
    p_review.add_argument("--issue", help="Issue to review, e.g. 26/022")
    p_review.set_defaults(func=cmd_review)

    p_show = sub.add_parser("show", help="Show local dashboard summary")
    p_show.set_defaults(func=cmd_show)

    p_backtest = sub.add_parser("backtest", help="Run historical backtest for all draw issues")
    p_backtest.add_argument("--min-history", type=int, default=3, help="Min history window before first backtest issue")
    p_backtest.add_argument("--rebuild", action="store_true", help="Rebuild reviewed backtest runs from scratch")
    p_backtest.add_argument("--remine", action="store_true", help="Re-mine pattern config before backtest")
    p_backtest.add_argument("--max-issues", type=int, default=BACKTEST_ISSUES_DEFAULT, help="只回测最近 N 期（0=全部）")
    p_backtest.add_argument("--progress-every", type=int, default=20, help="Print backtest progress every N processed issues (0 to disable)")
    p_backtest.set_defaults(func=cmd_backtest)

    p_mine = sub.add_parser("mine", help="Mine best pattern parameters from history")
    p_mine.set_defaults(func=cmd_mine)

    p_train_xgb = sub.add_parser("train-xgb", help="Train XGBoost model for main numbers")
    p_train_xgb.set_defaults(func=cmd_train_xgb)

    p_train_lgb = sub.add_parser("train-lgb", help="Train LightGBM model for main numbers")
    p_train_lgb.set_defaults(func=cmd_train_lgb)

    p_force = sub.add_parser("force-retrain", help="Re-train all ML models")
    p_force.set_defaults(func=cmd_force_retrain)

    p_backfill_special = sub.add_parser("backfill-special", help="回溯历史精选特别号记录")
    p_backfill_special.set_defaults(func=cmd_backfill_special)

    p_auto = sub.add_parser("auto-optimize", help="全自动优化生肖及特别号参数（目标90%命中率，最大连空1）")
    p_auto.add_argument("--target-hit-rate", type=float, default=0.90, help="目标命中率 (0~1)")
    p_auto.add_argument("--target-max-miss", type=int, default=1, help="目标最大连空")
    p_auto.add_argument("--timeout-hours", type=float, default=3, help="最大运行小时数（默认5）")
    p_auto.add_argument("--base-trials", type=int, default=1800, help="首轮试验次数")
    p_auto.set_defaults(func=cmd_auto_optimize)

    p_auto_pilot = sub.add_parser("auto", help="全自动运行：同步→评审→按需优化→预测→推送")
    p_auto_pilot.set_defaults(func=cmd_auto_pilot)

    p_reset = sub.add_parser("reset-and-auto", help="全自动重置→获取120期→优化近10期→预测")
    p_reset.add_argument("--trials", type=int, default=1000, help="Optuna trials count")
    p_reset.add_argument("--debug", action="store_true", help="打印每期调试信息")
    p_reset.set_defaults(func=cmd_reset_and_auto)

    p_check = sub.add_parser("check-data", help="校验数据库开奖记录是否完整且合法")
    p_check.set_defaults(func=cmd_check_data)

    return p

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, 'force_retrain', False):
        cmd_force_retrain(args)
        return
    if hasattr(args, 'retrain') and args.retrain:
        model_path = SCRIPT_DIR / "xgb_ensemble_model.pkl"
        if model_path.exists():
            model_path.unlink()
            print("[XGB] 旧模型已删除，将重新训练")
    if args.update:
        cmd_sync(args)
        return
    if args.tail_backtest:
        conn = connect_db(args.db)
        try:
            init_db(conn)
            hit_rate, samples, max_miss = backtest_tail(conn)
            print(f"Tail backtest: hit_rate={hit_rate*100:.1f}% samples={samples} max_miss={max_miss}")
        finally:
            conn.close()
    args.func(args)

def _auto_log(msg: str, error: bool = False):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    level = "ERROR" if error else "INFO"
    line = f"[{timestamp}] {level} - {msg}"
    print(line, flush=True)
    log_path = SCRIPT_DIR / "auto_pilot.log"
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _has_consecutive_failure(conn, threshold: int = 5) -> bool:
    rows = conn.execute("""
        SELECT hit_count FROM prediction_runs
        WHERE strategy='ensemble_v2' AND status='REVIEWED'
        ORDER BY reviewed_at DESC LIMIT ?
    """, (threshold,)).fetchall()
    if len(rows) < threshold:
        return False
    return all(int(r["hit_count"] or 0) == 0 for r in rows)


def _train_all_ml_models(conn):
    try:
        xgb = XGBoostPredictor()
        xgb.train(conn)
        import pickle
        with open(SCRIPT_DIR / "xgb_model.pkl", "wb") as f:
            pickle.dump(xgb, f)
        _auto_log("XGBoost 模型已更新")
    except Exception as e:
        _auto_log(f"XGBoost 训练失败: {e}", error=True)
    try:
        lgb_main = LightGBMPredictor()
        lgb_main.train(conn)
        import pickle
        with open(SCRIPT_DIR / "lgb_model.pkl", "wb") as f:
            pickle.dump(lgb_main, f)
        _auto_log("LightGBM 主号模型已更新")
    except Exception as e:
        _auto_log(f"LightGBM 主号训练失败: {e}", error=True)
    try:
        lgb_zodiac_model.train(conn)
        _auto_log("LightGBM 生肖模型已更新")
    except Exception as e:
        _auto_log(f"LightGBM 生肖训练失败: {e}", error=True)
    try:
        transformer_path = SCRIPT_DIR / "transformer_sp.pth"
        train_transformer(conn, transformer_path)
        _auto_log("Transformer 特别号模型已更新")
    except Exception as e:
        _auto_log(f"Transformer 训练失败: {e}", error=True)


def _push_recommendation(conn):
    try:
        rec = get_final_recommendation(conn)
        if not rec:
            _auto_log("无有效推荐数据，跳过推送")
            return
        (issue_no, main6, special, _, _, _, _,
         _, _, zodiac_single, zodiac_two, _, _, _, _, _) = rec
        title = f"香港六合彩预测 {issue_no}"
        content = (
            f"【{issue_no}期推荐】\n"
            f"一生肖：{zodiac_single}\n"
            f"二生肖：{'、'.join(zodiac_two)}\n"
            f"特别号主推：{special:02d}\n"
        )
        send_pushplus_notification(title, content)
        _auto_log("推荐推送成功")
    except Exception as e:
        _auto_log(f"推送失败: {e}", error=True)
        send_pushplus_notification("香港六合彩预测脚本出错", f"推送生成失败: {e}")


def cmd_auto_pilot(args: argparse.Namespace) -> None:
    conn = connect_db(args.db)
    try:
        init_db(conn)
        _auto_log("开始同步开奖数据...")
        records = fetch_hk_mark_six_records(timeout=args.api_timeout, retries=args.api_retries)
        total, inserted, updated = sync_from_records(conn, records, source="auto_pilot")
        _auto_log(f"同步完成: 总{total} 新{inserted} 更新{updated}")
        reviewed = review_latest(conn)
        _auto_log(f"评审完成: 更新了 {reviewed} 条记录")
        _auto_log("执行增量回测 (最近20期)...")
        run_historical_backtest(conn, rebuild=False, max_issues=20)
        generate_predictions(conn)
        _auto_log("增量回测与预测刷新完成")
        if _has_consecutive_failure(conn, 5):
            _auto_log("检测到连续5期未命中，启动在线调整...")
        online_adjuster.conn = conn
        online_adjuster.load_state()
        online_adjuster.adjust()
        draw_count = conn.execute("SELECT COUNT(*) FROM draws").fetchone()[0]
        if draw_count % 20 == 0:
            _auto_log("触发定期模型重训...")
            _train_all_ml_models(conn)
        _auto_log("生成推荐并推送...")
        _push_recommendation(conn)
        _auto_log("全自动任务执行完毕")
    except Exception:
        _auto_log(f"严重异常:\n{traceback.format_exc()}", error=True)
        send_pushplus_notification("香港六合彩预测脚本错误", f"auto 执行失败:\n{traceback.format_exc()}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
