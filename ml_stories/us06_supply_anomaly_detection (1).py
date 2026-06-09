"""
US-06: Detect Supply Anomalies Where Efficiency Drops Below 50%
Role: Operations Monitoring Analyst
Model: LightGBM Binary Classifier with SMOTE
Output: MongoDB alerts collection
"""
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, precision_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from pymongo import MongoClient, ASCENDING
from datetime import datetime
import os, warnings
warnings.filterwarnings("ignore")
os.makedirs("output", exist_ok=True)

# ── Load fact_supply ──────────────────────────────────────────────
supply = pd.read_csv("preprocessed/fact_supply.csv")
supply["date"] = pd.to_datetime(supply["date"])

FEATURES = [
    "supply_efficiency_pct",
    "hours_of_supply",
    "nrw_vs_benchmark",
    "supply_roll_4w_avg",
    "supply_lag_1w",
]
TARGET = "is_anomaly"

df = supply[FEATURES + [TARGET, "ward_id", "city_id", "date"]].dropna()
X  = df[FEATURES]
y  = df[TARGET].astype(int)

print(f"Dataset rows: {len(df):,}")
print(f"Anomaly distribution: {y.value_counts().to_dict()}")

# ── Split FIRST with stratify, then SMOTE on train only ───────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"Train anomalies: {y_train.sum()} | Test anomalies: {y_test.sum()}")

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"After SMOTE (train only): {pd.Series(y_train_res).value_counts().to_dict()}")

# ── Train model ───────────────────────────────────────────────────
model = LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                       random_state=42, verbose=-1)
model.fit(X_train_res, y_train_res)

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
prec    = precision_score(y_test, y_pred, zero_division=0)
fpr     = ((y_pred == 1) & (y_test == 0)).sum() / max((y_test == 0).sum(), 1)

print("\n── Model Evaluation ─────────────────────────────────────────")
print(classification_report(y_test, y_pred,
      target_names=["Normal", "Anomaly"], labels=[0, 1]))
print(f"  Anomaly Precision  : {prec:.2%}  (criterion > 85%)")
print(f"  False Positive Rate: {fpr:.2%}  (criterion < 5%)")

# ── Score full dataset ────────────────────────────────────────────
df = df.copy()
df["predicted_prob"]    = model.predict_proba(df[FEATURES])[:, 1].round(4)
df["predicted_anomaly"] = (df["predicted_prob"] >= 0.5).astype(int)

anomalies = df[df["predicted_anomaly"] == 1][
    ["ward_id", "city_id", "date", "supply_efficiency_pct",
     "hours_of_supply", "predicted_prob"]
].copy()
anomalies["date"] = anomalies["date"].astype(str)
print(f"\n  Anomalies flagged: {len(anomalies):,}")

# ── Push to MongoDB alerts ────────────────────────────────────────
try:
    client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=3000)
    client.server_info()
    col = client["water_supply_india"]["alerts"]
    col.drop()
    col.insert_many(anomalies.to_dict("records"))
    col.create_index([("city_id", ASCENDING), ("ward_id", ASCENDING)])
    print(f"  MongoDB: {len(anomalies)} docs → alerts")
    client.close()
except Exception as e:
    print(f"  [MongoDB skipped] {e}")

anomalies.to_csv("output/us06_supply_anomalies.csv", index=False)
print("Saved → output/us06_supply_anomalies.csv")
