# train_xgboost.py - 训练 XGBoost 模型（香港版）
import json
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from collections import defaultdict, Counter
from common import fetch_hk_records, get_zodiac_by_number, ZODIAC_MAP

# 特征提取函数（需要与 predict 一致，此处简化，可参照澳门版）
def extract_features_for_zodiac(rows, target_zodiac):
    # 实现与澳门版相同的特征提取逻辑
    # 为了简洁，此处略，可参考之前的澳门脚本
    feats = [0]*12
    return feats

def prepare_dataset(rows, start=30, seq_len=20):
    X, y = [], []
    for i in range(start, len(rows)):
        hist = rows[i-seq_len:i]
        target = rows[i]
        target_map = set()
        target_map.add(get_zodiac_by_number(target["special_number"]))
        for n in target["numbers"]:
            target_map.add(get_zodiac_by_number(n))
        for z in ZODIAC_MAP:
            feats = extract_features_for_zodiac(hist, z)
            X.append(feats)
            y.append(1 if z in target_map else 0)
    return np.array(X), np.array(y)

def main():
    print("获取历史数据...")
    records = fetch_hk_records(limit=600)
    rows = [{"numbers": r["numbers"], "special_number": r["special_number"]} for r in records]
    print(f"获取 {len(rows)} 期")
    X, y = prepare_dataset(rows)
    print(f"样本数: {len(X)}, 特征维度: {X.shape[1]}")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print("训练 XGBoost 模型...")
    model = xgb.XGBClassifier(n_estimators=300, max_depth=7, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    val_acc = model.score(X_val, y_val)
    print(f"验证集准确率: {val_acc:.4f}")
    model.save_model("xgboost_zodiac.json")
    print("模型已保存为 xgboost_zodiac.json")

if __name__ == "__main__":
    main()
