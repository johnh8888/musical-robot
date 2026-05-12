#!/usr/bin/env python3
# zodiac_main_ml.py - 机器学习版生肖预测

import argparse
import json
from common import fetch_hk_records_merged, next_issue
from ml_predict import predict_top_k_zodiac

def get_history_rows_as_list(limit=None):
    records = fetch_hk_records_merged(limit=limit, prefer_local=True)
    rows = []
    for r in records:
        rows.append({
            "numbers_json": json.dumps(r["numbers"]),
            "special_number": r["special_number"],
            "draw_date": r["draw_date"],
            "issue_no": r["issue_no"]
        })
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    rows = get_history_rows_as_list(limit=None)
    if not rows:
        print("数据获取失败")
        return
    if args.show:
        two = predict_top_k_zodiac(rows, k=2)
        three = predict_top_k_zodiac(rows, k=3)
        single = three[0]
        latest = rows[0]["issue_no"]
        print(f"预测期号: {next_issue(latest)}")
        print(f"一生肖 (ML): {single}")
        print(f"二生肖 (ML): {'、'.join(two)}")
        print(f"三生肖 (ML): {'、'.join(three)}")
        print("\n注: 模型基于2548期数据训练，预测结果更稳定。")

if __name__ == "__main__":
    main()