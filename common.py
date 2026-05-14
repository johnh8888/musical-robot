#!/usr/bin/env python3
# common.py - 香港六合彩公共模块（纯在线 API 版，不依赖本地 CSV）

import json
import re
import time
import urllib.request
import subprocess
import os
from pathlib import Path
from typing import List, Dict
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
    "龙":"鸡","鸡":"龙","蛇":"猴","猴":"蛇","马":"羊","羊":"马",
}

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

def next_issue(issue_no: str) -> str:
    try:
        year, seq = issue_no.split('/')
        return f"{year}/{str(int(seq)+1).zfill(3)}"
    except:
        return issue_no

# ---------- API 数据获取（纯在线） ----------
def _parse_numbers_internal(value: str) -> List[int]:
    out = []
    for token in value.replace("，", ",").split(","):
        token = token.strip()
        if not token: continue
        try:
            n = int(token)
        except:
            continue
        if 1<=n<=49: out.append(n)
    return out

def fetch_hk_records(limit: int = 600) -> List[Dict]:
    """从 API 获取历史记录，返回列表（降序，最新在前）"""
    req = urllib.request.Request(
        HK_API_URL,
        headers={"User-Agent":"Mozilla/5.0","Accept":"application/json"}
    )
    for attempt in range(1, API_RETRIES_DEFAULT+1):
        try:
            with urllib.request.urlopen(req, timeout=API_TIMEOUT_DEFAULT) as resp:
                raw = resp.read().decode("utf-8-sig")
                payload = json.loads(raw)
                records = _parse_hk_payload(payload)
                if limit and limit > 0:
                    records = records[:limit]
                print(f"从 API 获取到 {len(records)} 期历史数据，最新: {records[0]['issue_no'] if records else '无'}")
                return records
        except Exception as e:
            if attempt == API_RETRIES_DEFAULT:
                raise
            time.sleep(API_RETRY_BACKOFF_SECONDS * (2**(attempt-1)))
    return []

def _parse_hk_payload(payload: dict) -> List[Dict]:
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
                draw_date = item.get("openTime","").split()[0] or datetime.now().strftime("%Y-%m-%d")
                records.append({
                    "issue_no": issue_no,
                    "draw_date": draw_date,
                    "numbers": nums[:6],
                    "special_number": nums[6],
                })
            break
    # 去重并按期数降序排列
    dedup = {}
    for r in records:
        dedup[r["issue_no"]] = r
    sorted_records = sorted(dedup.values(), key=lambda x: x["issue_no"], reverse=True)
    return sorted_records

def fetch_hk_records_merged(limit: int = 100, prefer_local: bool = False) -> List[Dict]:
    """
    获取历史记录，强制从 API 获取最新数据。
    limit: 获取期数（默认100）
    prefer_local: 忽略，始终使用 API
    """
    return fetch_hk_records(limit=limit)

# ---------- Git 参数管理（保持不变） ----------
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