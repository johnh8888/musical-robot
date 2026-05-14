#!/usr/bin/env python3
# zodiac_main.py - 一二三肖 v6（最强最终版）

import gzip
import json
import re
import time
import urllib.request
from collections import Counter

API_URL = "https://marksix6.net/index.php?api=1"

ZODIAC_MAP = {
    "鼠": [7,19,31,43], "牛": [8,20,32,44], "虎": [9,21,33,45],
    "兔": [10,22,34,46], "龙": [11,23,35,47], "蛇": [12,24,36,48],
    "马": [1,13,25,37,49], "羊": [2,14,26,38], "猴": [3,15,27,39],
    "鸡": [4,16,28,40], "狗": [5,17,29,41], "猪": [6,18,30,42],
}

ZODIAC_LIST = list(ZODIAC_MAP.keys())

COLOR_MAP = {
    "红": [1,2,7,8,12,13,18,19,23,24,29,30,34,35,40,41,45,46],
    "蓝": [3,4,9,10,14,15,20,21,25,26,31,32,36,37,42,43,47,48],
    "绿": [5,6,11,16,17,22,27,28,33,38,39,44,49]
}

DECAY_ALPHA = 0.84

def get_zodiac(n):
    for z, nums in ZODIAC_MAP.items():
        if n in nums: return z
    return "马"

def get_color(n):
    for c, nums in COLOR_MAP.items():
        if n in nums: return c
    return "红"

def is_big(n): return n >= 25
def is_odd(n): return n % 2 == 1
def decay(idx): return DECAY_ALPHA ** idx

def omission_map(rows):
    om = {z: 0 for z in ZODIAC_LIST}
    for r in reversed(rows):
        appeared = {get_zodiac(n) for n in r.get("numbers", []) + [r.get("special_number")]}
        for z in ZODIAC_LIST:
            om[z] = 0 if z in appeared else om[z] + 1
    return om

def parse_nums(value):