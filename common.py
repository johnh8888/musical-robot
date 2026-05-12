#!/usr/bin/env python3
# common.py - 香港六合彩公共模块（含完整生肖映射）

import json
import re
import time
import urllib.request
import subprocess
import os
import csv
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timezone
from collections import Counter

# ---------- 常量定义 ----------
ALL_NUMBERS = list(range(1, 50))

# 正确完整的生肖映射（12个）
ZODIAC_MAP = {
    "马": [1, 13, 25, 37, 49],
    "蛇": [2, 14, 26, 38],
    "龙": [3, 15, 27, 39],
    "兔": [4, 16, 28, 40],
    "虎": [5, 17, 29, 41],
    "牛": [6, 18, 30, 42],
    "鼠": [7, 19, 31, 43],
    "猪": [8, 20, 32, 44],
    "狗": [9, 21, 33, 45],
    "鸡": [10, 22, 34, 46],
    "猴": [11, 23, 35, 47],
    "羊": [12, 24, 36, 48],
}

# 六合彩对肖配对（相冲）
ZODIAC_PAIR = {
    "鼠": "牛", "牛": "鼠",
    "虎": "猪", "猪": "虎",
    "兔": "狗", "狗": "兔",
    "龙": "鸡", "鸡": "龙",
    "蛇": "猴", "猴": "蛇",
    "马": "羊", "羊": "马",
}

# API 配置
HK_API_URL = "https://marksix6.net/index.php?api=1"
API_TIMEOUT_DEFAULT = 20
API_RETRIES_DEFAULT = 4
API_RETRY_BACKOFF_SECONDS = 2.0

SCRIPT_DIR = Path(__file__).resolve().parent

# ---------- 工具函数 ----------
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def get_zodiac_by_number(number: int) -> str:
    """根据号码返回对应的生肖"""
    for zodiac, nums in ZODIAC_MAP.items():
        if number in nums:
            return zodiac
    return "马"  # fallback

def next_issue(issue_no: str) -> str:
    """根据当前期号生成下一期期号，如 26/050 -> 26/051"""
    try:
        year, seq = issue_no.split('/')
        return f"{year}/{str(int(seq)+1).zfill(3)}"
    except:
        return issue_no

# ---------- 本地 CSV 加载（适配 Mark_Six.csv） ----------
def load_local_history_csv(csv_path="Mark_Six.csv"):
    """
    读取本地 CSV 文件，自动识别列名（期數、日期、中獎號碼 1~6、特別號碼）
    返回记录列表，按开奖日期降序（最新在前）
    """
    records = []
    if not Path(csv_path).exists():
        print(f"⚠️ 本地 CSV 不存在: {csv_path}")
        return records

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return records

        fieldnames = reader.fieldnames
        # 识别列名
        issue_col = next((c for c in fieldnames if "期" in c), None)
        date_col = next((c for c in fieldnames if "日期" in c), None)
        # 中獎號碼列: 可能是 "中獎號碼 1","中獎號碼 2"...
        num_cols = []
        for i in range(1, 7):
            cand = f"中獎號碼 {i}"
            if cand in fieldnames:
                num_cols.append(cand)
            else:
                fallback = str(i)
                if fallback in fieldnames:
                    num_cols.append(fallback)
        special_col = next((c for c in fieldnames if "特別" in c), None)

        if not (issue_col and date_col and len(num_cols) == 6 and special_col):
            print("❌ CSV 列名不匹配，期望: 期數, 日期, 中獎號碼 1~6, 特別號碼")
            return records

        for row in reader:
            try:
                numbers = []
                for col in num_cols:
                    val = row.get(col, "").strip()
                    if val:
                        numbers.append(int(val))
                if len(numbers) != 6:
                    continue
                special = int(row.get(special_col, 0))
                if special == 0:
                    continue

                issue = row[issue_col].strip()
                draw_date = row[date_col].strip()

                records.append({
                    "issue_no": issue,
                    "draw_date": draw_date,
                    "numbers": numbers,
                    "special_number": special,
                })
            except Exception:
                continue

    # 按期数降序排列（最新在前）
    records.sort(key=lambda x: x["issue_no"], reverse=True)
    print(f"✅ 从本地 CSV 加载到 {len(records)} 期历史数据（已按降序排列）")
    return records

# ---------- API 数据获取（备用） ----------
def _parse_numbers_internal(value: str) -> List[int]:
    out = []
    for token in value.replace("，", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            n = int(token)
        except:
            continue
        if 1 <= n <= 49:
            out.append(n)
    return out

def fetch_hk_records(limit: int = 600) -> List[Dict]:
    req = urllib.request.Request(
        HK_API_URL,
        headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    )
    for attempt in range(1, API_RETRIES_DEFAULT + 1):
        try:
            with urllib.request.urlopen(req, timeout=API_TIMEOUT_DEFAULT) as resp:
                raw = resp.read().decode("utf-8-sig")
                payload = json.loads(raw)
                return _parse_hk_payload(payload, limit)
        except Exception as e:
            if attempt == API_RETRIES_DEFAULT:
                raise
            time.sleep(API_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)))
    return []

def _parse_hk_payload(payload: dict, limit: int) -> List[Dict]:
    records = []
    for item in payload.get("lottery_data", []):
        if item.get("name") in ("香港六合彩", "香港彩", "HK六合彩", "HK Mark Six"):
            for line in item.get("history", []):
                m = re.match(r"(\d{7})\s*期[：:]\s*([\d,]+)", line)
                if not m:
                    continue
                nums = _parse_numbers_internal(m.group(2))
                if len(nums) < 7:
                    continue
                issue_raw = m.group(1)
                if len(issue_raw) >= 7:
                    year = issue_raw[2:4]
                    seq = str(int(issue_raw[4:]))
                    issue_no = f"{year}/{seq.zfill(3)}"
                else:
                    issue_no = issue_raw
                draw_date = item.get("openTime", "").split()[0] or "2026-01-01"
                records.append({
                    "issue_no": issue_no,
                    "draw_date": draw_date,
                    "numbers": nums[:6],
                    "special_number": nums[6],
                })
            break
    dedup = {}
    for r in records:
        dedup[r["issue_no"]] = r
    sorted_records = sorted(dedup.values(), key=lambda x: x["issue_no"], reverse=True)
    if limit > 0:
        sorted_records = sorted_records[:limit]
    print(f"从API获取到 {len(sorted_records)} 期，最新: {sorted_records[0]['issue_no'] if sorted_records else '无'}")
    return sorted_records

def fetch_hk_records_merged(limit: int = None, prefer_local: bool = True) -> List[Dict]:
    """
    获取历史记录，默认 prefer_local=True 只使用本地 CSV（推荐）
    若 prefer_local=False，则尝试 API 并与本地合并（备用）
    """
    all_records = load_local_history_csv("Mark_Six.csv")
    if prefer_local:
        if limit is not None and limit > 0:
            all_records = all_records[:limit]
        print(f"📀 使用本地数据，共 {len(all_records)} 期")
        return all_records

    # 若不 prefer_local，则尝试 API 补充（极少使用）
    try:
        online = fetch_hk_records(limit=600)
        existing = {r["issue_no"] for r in all_records}
        for r in online:
            if r["issue_no"] not in existing:
                all_records.append(r)
    except Exception as e:
        print(f"在线获取失败: {e}")
    all_records.sort(key=lambda x: x["issue_no"], reverse=True)
    if limit and limit > 0:
        all_records = all_records[:limit]
    return all_records

# ---------- Git 参数管理 ----------
def load_params(file_name: str) -> dict:
    path = Path(file_name)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_params(file_name: str, params: dict):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

def commit_and_push_params(file_name: str, message: str = "Auto update params"):
    try:
        subprocess.run(["git", "config", "user.name", "github-actions[bot]"], check=True)
        subprocess.run(["git", "config", "user.email", "github-actions[bot]@users.noreply.github.com"], check=True)
        subprocess.run(["git", "add", file_name], check=True)
        status = subprocess.run(["git", "status", "--porcelain", file_name], capture_output=True, text=True)
        if status.stdout.strip():
            subprocess.run(["git", "commit", "-m", message], check=True)
            token = os.environ.get("GITHUB_TOKEN")
            if token:
                remote_url = subprocess.run(["git", "config", "--get", "remote.origin.url"], capture_output=True,
                                            text=True).stdout.strip()
                if remote_url.startswith("https://"):
                    new_url = remote_url.replace("https://", f"https://{token}@")
                    subprocess.run(["git", "remote", "set-url", "origin", new_url], check=True)
            subprocess.run(["git", "push"], check=True)
            print(f"✅ 参数文件 {file_name} 已推送到仓库")
        else:
            print(f"ℹ️ 参数文件 {file_name} 无变化")
    except Exception as e:
        print(f"⚠️ Git 推送失败: {e}")