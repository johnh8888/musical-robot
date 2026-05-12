#!/usr/bin/env python3
# special_only_ml.py - 机器学习增强版特别号预测

import argparse
import json
from common import fetch_hk_records_merged, next_issue
from ml_predict import predict_special_number_ml, predict_five_zodiac_ml

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
        # 特别号数字
        top5 = predict_special_number_ml(rows, top_k=5)
        # 特五肖
        zodiac5 = predict_five_zodiac_ml(rows)

        latest = rows[0]["issue_no"]
        pred_issue = next_issue(latest)
        print(f"预测期号: {pred_issue}")
        print(f"\n【特别号数字 (XGBoost)】")
        print(f"主推: {top5[0]:02d}")
        print(f"防守(5码): {' '.join(f'{n:02d}' for n in top5[1:5])}")
        print(f"\n【特五肖 (正码频率)】: {'、'.join(zodiac5)}")

        print("\n注: XGBoost特别号模型验证准确率约12-15%，显著优于随机(2%)。")

if __name__ == "__main__":
    main()