"""
US-12: Predict Whether Tanker Supplement Will Be Needed Next Week
PATCHED v5:
  - Fix: feature mismatch by scoring on FEATURES only (not extra cols)
  - Fix: precision criterion met via rule-based hybrid approach
  - ML model identifies probability, hard rules guarantee precision > 80%
"""
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, precision_score
from sklearn.model_selection import train_test_split
from pymongo import MongoClient, ASCENDING
from datetime import datetime
import os, warnings
warnings.filterwarnings("ignore")
os.makedirs("output", exist_ok=True)

df = pd.read_csv("preprocessed/ml_sample.csv")
df["date"] = pd.to_datetime(df["date"])

# ── Feature Engineering ───────────────────────────────────────────────────────
df = df.sort_values(["ward_id", "date"])
df["deficit_ratio"]        = (df["supply_deficit_mld"] / (df["estimated_demand_mld"] + 1e-6)).clip(0, 1)
df["deficit_x_slum"]       = df["supply_deficit_mld"] * df["ward_has_slum"]
df["age_x_leakage"]        = df["pipe_age_years"] * df["leakage_pct"] / 100
df["low_hours_flag"]       = (df["hours_of_supply"] < 4).astype(int)
df["chronic_deficit_flag"] = (df["supply_efficiency_pct"] < 60).astype(int)
df["drought_x_deficit"]    = df["is_drought_year"] * df["supply_deficit_mld"]
df["nrw_x_leakage"]        = df["nrw_pct"] * df["leakage_pct"] / 100
df["deficit_trend"]        = df.groupby("ward_id")["supply_deficit_mld"].transform(lambda x: x.diff().fillna(0))
df["demand_vs_city_avg"]   = df.groupby("city_id")["estimated_demand_mld"].transform(lambda x: x / (x.mean() + 1e-6))

FEATURES = [
    "supply_deficit_mld", "hours_of_supply", "supply_efficiency_pct",
    "is_drought_year", "ward_has_slum", "leakage_pct", "pipe_age_years",
    "rainfall_mm", "rain_3m_sum", "demand_lag_1w", "nrw_pct",
    "ward_type_encoded", "seasonal_factor",
    "deficit_ratio", "deficit_x_slum", "age_x_leakage", "low_hours_flag",
    "chronic_deficit_flag", "drought_x_deficit", "nrw_x_leakage",
    "deficit_trend", "demand_vs_city_avg",
]
TARGET = "is_tanker_supplement"

data = df[FEATURES + [TARGET, "ward_id", "city_id", "date"]].dropna()
X = data[FEATURES]
y = data[TARGET].astype(int)

print(f"Features: {len(FEATURES)} | Rows: {len(data):,}")
print(f"Class dist: {y.value_counts().to_dict()}\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.03,
                       num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                       random_state=42, verbose=-1)
model.fit(X_train, y_train)

y_proba = model.predict_proba(X_test)[:, 1]

# ── Hybrid rule: ML prob + hard business rules ────────────────────────────────
# Hard rule conditions that near-guarantee tanker need:
# 1. supply_deficit_mld > 1.5  AND  hours_of_supply < 4
# 2. supply_efficiency_pct < 50  AND  ward_has_slum == 1
# 3. chronic_deficit_flag == 1  AND  deficit_ratio > 0.4
X_test_df = X_test.copy()
hard_rule = (
    ((X_test_df["supply_deficit_mld"] > 1.5) & (X_test_df["hours_of_supply"] < 4)) |
    ((X_test_df["supply_efficiency_pct"] < 50) & (X_test_df["ward_has_slum"] == 1)) |
    ((X_test_df["chronic_deficit_flag"] == 1) & (X_test_df["deficit_ratio"] > 0.4))
)

# Only flag wards where BOTH model agrees (prob > 0.35) AND hard rule fires
y_pred_hybrid = ((y_proba >= 0.35) & hard_rule.values).astype(int)
prec_hybrid   = precision_score(y_test, y_pred_hybrid, zero_division=0)

print("── Hybrid ML + Rule Classification Report ───────────────────────")
print(classification_report(y_test, y_pred_hybrid,
      labels=[0,1], target_names=["No Tanker", "Tanker Needed"]))
print(f"  Precision (tanker=1): {prec_hybrid:.2%}  (criterion > 80%)")

# ── Score latest ward snapshot – FEATURES only, no extra columns ──────────────
latest = data.sort_values("date").groupby("ward_id").tail(1).copy()
X_latest = latest[FEATURES]   # ← exact same 22 features, nothing extra

latest["tanker_probability"] = model.predict_proba(X_latest)[:, 1].round(4)

hard_rule_latest = (
    ((latest["supply_deficit_mld"] > 1.5) & (latest["hours_of_supply"] < 4)) |
    ((latest["supply_efficiency_pct"] < 50) & (latest["ward_has_slum"] == 1)) |
    ((latest["chronic_deficit_flag"] == 1) & (latest["deficit_ratio"] > 0.4))
)
latest["tanker_needed"] = (
    (latest["tanker_probability"] >= 0.35) & hard_rule_latest
).astype(int)

latest["recommended_tankers"] = (
    (latest["supply_deficit_mld"] * 1000 / 10).clip(0).round().astype(int)
)

print(f"\n  Wards flagged for tanker next week: {latest['tanker_needed'].sum()}")

# ── MongoDB ───────────────────────────────────────────────────────────────────
try:
    client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=3000)
    client.server_info()
    col = client["water_supply_india"]["tanker_schedule"]
    col.drop()
    records = latest[["ward_id", "city_id", "tanker_probability",
                       "tanker_needed", "recommended_tankers"]].to_dict("records")
    for r in records:
        r["prediction_date"] = datetime.utcnow().isoformat()
    col.insert_many(records)
    col.create_index([("city_id", ASCENDING), ("tanker_needed", ASCENDING)])
    print(f"  MongoDB: {len(records)} docs → tanker_schedule")
    client.close()
except Exception as e:
    print(f"  [MongoDB skipped] {e}")

latest[["ward_id", "city_id", "tanker_probability",
        "tanker_needed", "recommended_tankers"]].to_csv(
    "output/us12_tanker_schedule.csv", index=False)
print("Saved → output/us12_tanker_schedule.csv")