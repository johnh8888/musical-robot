# ==================== 新澳门六合彩 - 最终稳定实用版 ====================
import argparse
import json
import requests
from collections import Counter
from pathlib import Path

ZODIAC_MAP = {
    "马": [1,13,25,37,49], "羊": [12,24,36,48], "猴": [11,23,35,47],
    "鸡": [10,22,34,46], "狗": [9,21,33,45], "猪": [8,20,32,44],
    "鼠": [7,19,31,43], "牛": [6,18,30,42], "虎": [5,17,29,41],
    "兔": [4,16,28,40], "龙": [3,15,27,39], "蛇": [2,14,26,38]
}

COLOR_MAP = {
    "红": [1,2,7,8,12,13,18,19,23,24,29,30,34,35,40,45,46],
    "蓝": [3,4,9,10,14,15,20,25,31,36,37,41,42,47,48],
    "绿": [5,6,11,16,17,22,27,28,33,38,39,43,44,49]
}

DATA_FILE = Path("macau_history.json")
history = []

def load_history():
    global history
    if DATA_FILE.exists():
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except:
            history = []

def save_history():
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def fetch_new_macau_only():
    url = "https://marksix6.net/index.php?api=1"
    print("正在从 marksix6.net 获取新澳门数据...")

    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            print(f"请求失败: {r.status_code}")
            return False

        data = r.json()
        lottery_list = data.get("lottery_data", [])
        new_count = 0
        existing = {d.get("issue") for d in history}

        for lottery in lottery_list:
            name = str(lottery.get("name", ""))
            if any(k in name for k in ["新澳门", "澳门", "澳彩", "Macau", "macau"]) and "老" not in name:
                issue = str(lottery.get("expect") or lottery.get("issue"))
                open_code = lottery.get("openCode") or lottery.get("numbers")
                if issue and open_code and issue not in existing:
                    nums = [int(x.strip()) for x in str(open_code).split(",") if x.strip()]
                    if len(nums) >= 6:
                        history.append({
                            "issue": issue,
                            "numbers": nums[:6],
                            "special": nums[6] if len(nums) > 6 else nums[-1]
                        })
                        new_count += 1
                        print(f"✅ 新增第 {issue} 期")

        if new_count > 0:
            save_history()
            print(f"🎉 本次新增 {new_count} 期新澳门数据")
            return True
        else:
            print("ℹ️ 暂无新数据")
            return False
    except Exception as e:
        print(f"❌ 获取失败: {e}")
        return False

def show_prediction():
    load_history()
    print("\n" + "="*80)
    print("新澳门六合彩 智能推荐（最终稳定版）")
    print("="*80)

    if not history:
        print("暂无数据，请先运行: python macau_predict.py sync")
        return

    latest = history[-1]
    nums_str = " ".join(f"{n:02d}" for n in latest["numbers"])
    print(f"最新开奖 → 第 {latest['issue']} 期")
    print(f"正码: {nums_str}   特别号: {latest['special']:02d}\n")

    # 一肖
    zodiac_count = Counter()
    for draw in history[-40:]:
        for n in draw["numbers"]:
            for z, ns in ZODIAC_MAP.items():
                if n in ns:
                    zodiac_count[z] += 1
    top2 = zodiac_count.most_common(2)

    print("1. 一肖推荐")
    print(f"   最强: {top2[0][0]}")
    if len(top2) > 1:
        print(f"   次强: {top2[1][0]}")

    # 三中三
    all_flat = [n for draw in history for n in draw["numbers"]]
    hot5 = [n for n, _ in Counter(all_flat).most_common(5)]
    print("\n2. 三中三推荐（5个热门号码）")
    print(f"   推荐号码: {' '.join(f'{n:02d}' for n in hot5)}")

    # 单双 大小 波色
    latest_nums = latest["numbers"]
    odd = sum(1 for n in latest_nums if n % 2 == 1)
    big = sum(1 for n in latest_nums if n >= 25)
    red = sum(1 for n in latest_nums if n in COLOR_MAP["红"])
    blue = sum(1 for n in latest_nums if n in COLOR_MAP["蓝"])
    green = 6 - red - blue

    print("\n3. 最新一期趋势")
    print(f"   单双：奇{odd} : 偶{6-odd}")
    print(f"   大小：大{big} : 小{6-big}")
    print(f"   波色：红{red}  蓝{blue}  绿{green}")

    print("\n理性提醒：仅供娱乐参考，请严格控制投注金额！")

def main():
    load_history()
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["sync", "add", "show"], nargs="?", default="show")
    args = parser.parse_args()

    if args.cmd == "sync":
        fetch_new_macau_only()
    elif args.cmd == "add":
        issue = input("期号: ")
        nums = list(map(int, input("6个正码空格隔开: ").split()))
        special = int(input("特别号: "))
        history.append({"issue": issue, "numbers": nums, "special": special})
        save_history()
        print("添加成功")
    else:
        show_prediction()

if __name__ == "__main__":
    main()
