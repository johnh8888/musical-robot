#!/usr/bin/env python3
# common.py - 香港六合彩公共模块（正确生肖映射）

import json
import re
import time
import urllib.request
import subprocess
import os
import csv
from pathlib import Path
from typing import List, Dict
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

# 六合彩对肖配对
ZODIAC_PAIR = {
    "鼠": "牛", "牛": "鼠",
    "虎": "猪", "猪": "虎",
    "兔": "狗", "狗": "兔",
    "龙": "鸡", "鸡": "龙",
    "蛇": "猴", "猴": "蛇",
    "马": "羊", "羊": "马",
}

HK_API_URL = "https://marksix6.net/index.php?api=1"
API_TIMEOUT_DEFAULT = 20
API_RETRIES_DEFAULT = 4
API_RETRY_BACKOFF_SECONDS = 2.0

SCRIPT_DIR = Path(__file__).resolve().parent

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def get_zodiac_by_number(number: int) -> str:
    for zodiac, nums in ZODIAC_MAP.items():
        if number in nums:
            return zodiac
    return "马"

def next_issue(issue_no: str) -> str:
    try:
        year, seq = issue_no.split('/')
        return f"{year}/{str(int(seq)+1).zfill(3)}"
    except:
        return issue_no

# ---------- 本地 CSV 加载 ----------
def load_local_history_csv(csv_path="Mark_Six.csv"):
    records = []
    if not Path(csv_path).exists():
        print(f"⚠️ 本地 CSV 不存在: {csv_path}")
        return records

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return records
        fieldnames = reader.fieldnames
        issue_col = next((c for c in fieldnames if "期" in c), None)
        date_col = next((c for c in fieldnames if "日期" in c), None)
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
        if not (issue_col and date_col and len(num_cols)==6 and special_col):
            print("❌ CSV 列名不匹配")
            return records

        for row in reader:
            try:
                numbers = [int(row[col].strip()) for col in num_cols if row.get(col, '').strip()]
                if len(numbers) != 6:
                    continue
                special = int(row.get(special_col, 0))
                if special == 0:
                    continue
                records.append({
                    "issue_no": row[issue_col].strip(),
                    "draw_date": row[date_col].strip(),
                    "numbers": numbers,
                    "special_number": special,
                })
            except:
                continue

    records.sort(key=lambda x: x["issue_no"], reverse=True)
    print(f"✅ 从本地 CSV 加载到 {len(records)} 期历史数据")
    return records

# ---------- API 获取（备用）----------
def fetch_hk_records_merged(limit=None, prefer_local=True):
    if prefer_local:
        records = load_local_history_csv("Mark_Six.csv")
        print(f"📀 使用本地数据，共 {len(records)} 期")
        return records[:limit] if limit else records
    # 否则调用API（省略，基本不会用到）
    return []

# ---------- Git 推送辅助 ----------
def commit_and_push_params(file_name: str, message: str = "Auto update"):
    try:
        subprocess.run(["git", "config", "user.name", "github-actions[bot]"], check=True)
        subprocess.run(["git", "config", "user.email", "github-actions[bot]@users.noreply.github.com"], check=True)
        subprocess.run(["git", "add", file_name], check=True)
        status = subprocess.run(["git", "status", "--porcelain", file_name], capture_output=True, text=True)
        if status.stdout.strip():
            subprocess.run(["git", "commit", "-m", message], check=True)
            token = os.environ.get("GITHUB_TOKEN")
            if token:
                remote_url = subprocess.run(["git", "config", "--get", "remote.origin.url"], capture_output=True, text=True).stdout.strip()
                if remote_url.startswith("https://"):
                    new_url = remote_url.replace("https://", f"https://{token}@")
                    subprocess.run(["git", "remote", "set-url", "origin", new_url], check=True)
            subprocess.run(["git", "push"], check=True)
            print(f"✅ {file_name} 已推送")
        else:
            print(f"ℹ️ {file_name} 无变化")
    except Exception as e:
        print(f"⚠️ Git 推送失败: {e}")