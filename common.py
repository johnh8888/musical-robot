#!/usr/bin/env python3
# common.py - 香港六合彩公共模块（支持本地 CSV 合并）

import json
import re
import socket
import time
import urllib.request
import subprocess
import os
import math
import csv
from urllib.error import URLError
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone
from collections import Counter

ALL_NUMBERS = list(range(1, 50))
ZODIAC_MAP = {
    "马": [1,13,25,37,49], "蛇": [2,14,26,38], "龙": [3,15,27,39],
    "兔": [4,16,28,40], "虎": [5,17,29,41], "牛": [6,18,30,42],
    "鼠": [7,19,31,43], "猪": [8,20,32,44], "狗": [9,21,33,45],
    "鸡": [10,22,34,46], "猴": [11,23,35,47], "羊": [12,24,36,48],
}
ZODIAC_PAIR = {
    "鼠":"牛","牛":"鼠","虎":"猪","猪":"虎","兔":"狗","狗":"兔",
    "龙":"鸡","鸡":"龙","蛇":"猴","猴":"蛇","马":"羊","羊":"马"
}
SCRIPT_DIR = Path(__file__).resolve().parent
HK_API_URL = "https://marksix6.net/index.php?api=1"
API_TIMEOUT_DEFAULT = 20
API_RETRIES_DEFAULT = 4
API_RETRY_BACKOFF_SECONDS = 2.0

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def get_zodiac_by_number(number: int) -> str:
    for zodiac, nums in ZODIAC_MAP.items():
        if number in nums:
            return zodiac
    return "马"

def _fmt_num(n: int) -> str:
    return str(n).zfill(2)

def _normalize(score_map: Dict[int, float]) -> Dict[int, float]:
    values = list(score_map.values())
    mn, mx = min(values), max(values)
    if mx == mn:
        return {k:0.0 for k in score_map}
    return {k:(v-mn)/(mx-mn) for k,v in score_map.items()}

def next_issue(issue_no: str) -> str:
    try:
        year, seq = issue_no.split('/')
        return f"{year}/{str(int(seq)+1).zfill(3)}"
    except:
        return issue_no

# ========== 本地 CSV 加载 ==========
def load_local_history_csv(csv_path="hk_full_history.csv"):
    records = []
    if not Path(csv_path).exists():
        return records
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 9:
                continue
            try:
                records.append({
                    "issue_no": row[0].strip(),
                    "draw_date": row[1].strip(),
                    "numbers": [int(x) for x in row[2:8]],
                    "special_number": int(row[8]),
                })
            except:
                continue
    print(f"从本地 CSV 加载到 {len(records)} 期历史数据")
    return records

# ========== API 数据获取（香港六合彩） ==========
def _parse_numbers_internal(value: str) -> List[int]:
    out = []
    for token in value.replace("，",",").split(","):
        token = token.strip()
        if not token: continue
        try:
            n = int(token)
        except:
            continue
        if 1<=n<=49: out.append(n)
    return out

def fetch_hk_records(limit: int = 600) -> List[Dict]:
    req = urllib.request.Request(
        HK_API_URL,
        headers={"User-Agent":"Mozilla/5.0","Accept":"application/json"}
    )
    for attempt in range(1, API_RETRIES_DEFAULT+1):
        try:
            with urllib.request.urlopen(req, timeout=API_TIMEOUT_DEFAULT) as resp:
                raw = resp.read().decode("utf-8-sig")
                payload = json.loads(raw)
                return _parse_hk_payload(payload, limit)
        except Exception as e:
            if attempt == API_RETRIES_DEFAULT:
                raise
            time.sleep(API_RETRY_BACKOFF_SECONDS * (2**(attempt-1)))
    return []

def _parse_hk_payload(payload: dict, limit: int) -> List[Dict]:
    records = []
    for item in payload.get("lottery_data", []):
        if item.get("name") in ("香港六合彩","香港彩","HK六合彩","HK Mark Six"):
            for line in item.get("history", []):
                m = re.match(r"(\d{7})\s*期[：:]\s*([\d,]+)", line)
                if not m: continue
                nums = _parse_numbers_internal(m.group(2))
                if len(nums) < 7: continue
                issue_raw = m.group(1)
                if len(issue_raw)>=7:
                    year = issue_raw[2:4]
                    seq = str(int(issue_raw[4:]))
                    issue_no = f"{year}/{seq.zfill(3)}"
                else:
                    issue_no = issue_raw
                draw_date = item.get("openTime","").split()[0] or "2026-01-01"
                records.append({
                    "issue_no": issue_no,
                    "draw_date": draw_date,
                    "numbers": nums[:6],
                    "special_number": nums[6],
                })
            break
    dedup = {}
    for r in records: dedup[r["issue_no"]] = r
    sorted_records = sorted(dedup.values(), key=lambda x: x["issue_no"], reverse=True)
    if limit>0: sorted_records = sorted_records[:limit]
    print(f"从API获取到 {len(sorted_records)} 期，最新: {sorted_records[0]['issue_no'] if sorted_records else '无'}")
    return sorted_records

def fetch_hk_records_merged(limit: int = 1200) -> List[Dict]:
    all_records = load_local_history_csv()
    try:
        online = fetch_hk_records(limit=600)
        existing = {r["issue_no"] for r in all_records}
        for r in online:
            if r["issue_no"] not in existing:
                all_records.append(r)
    except Exception as e:
        print(f"在线获取失败: {e}")
    all_records.sort(key=lambda x: x["issue_no"], reverse=True)
    if limit>0: all_records = all_records[:limit]
    print(f"合并后共 {len(all_records)} 期，最新: {all_records[0]['issue_no'] if all_records else '无'}")
    return all_records


# ========== 参数管理与 Git 推送 ==========
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
                remote_url = subprocess.run(["git", "config", "--get", "remote.origin.url"], capture_output=True, text=True).stdout.strip()
                if remote_url.startswith("https://"):
                    new_url = remote_url.replace("https://", f"https://{token}@")
                    subprocess.run(["git", "remote", "set-url", "origin", new_url], check=True)
            subprocess.run(["git", "push"], check=True)
            print(f"✅ 参数文件 {file_name} 已推送到仓库")
        else:
            print(f"ℹ️ 参数文件 {file_name} 无变化")
    except Exception as e:
        print(f"⚠️ Git 推送失败: {e}")
