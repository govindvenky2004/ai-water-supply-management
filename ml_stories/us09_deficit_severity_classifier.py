"""
US-09: Classify Ward Deficit Severity (None / Low / Medium / High)
PATCHED – columns verified against diagnose_columns.py
KEY FIXES:
  - fact_supply has: deficit_severity ✓, supply_efficiency_pct ✓,
    nrw_vs_benchmark ✓, is_anomaly ✓, deficit_severity_label ✓
  - leakage_pct NOT in fact_supply → use nrw_vs_benchmark as proxy ✓
  - demand_roll_4w_avg NOT in fact_supply → use deficit_roll_4w_avg ✓
  - rainfall_mm NOT in fact_supply → drop (not available at supply level)
  - ward_type_encoded NOT in fact_supply → merge from dim_wards ✓
"""
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from pymongo import MongoClient, ASCENDING
import os, warnings
warnings.filterwarnings("ignore")
os.makedirs("output", exist_ok=True)

# ── Load tables ───────────────────────────────────────────────────────────────
supply = pd.read_csv("preprocessed/fact_supply.csv")
wards  = pd.read_csv("preprocessed/dim_wards.csv")

# Merge ward_type_encoded from dim_wards (not in fact_supply)
supply = supply.merge(
    wards[["ward_id", "ward_type_encoded", "has_slum_pocket"]],
    on="ward_id", how="left"
)

print(f"fact_supply rows: {len(supply):,}")

# ── Features – verified / merged ─────────────────────────────────────────────
FEATURES = [
    "supply_efficiency_pct",  # ✓ fact_supply
    "deficit_roll_4w_avg",    # ✓ fact_supply (proxy for demand_roll_4w_avg)
    "nrw_vs_benchmark",       # ✓ fact_supply (proxy for leakage)
    "hours_of_supply",        # ✓ fact_supply
    "ward_type_encoded",      # ✓ merged from dim_wards
    "has_slum_pocket",        # ✓ merged from dim_wards
    "supply_deficit_mld",     # ✓ fact_supply
]
TARGET = "deficit_severity"   # ✓ fact_supply (0=None,1=Low,2=Medium,3=High)

df = supply[FEATURES + [TARGET, "ward_id", "city_id"]].dropna()
X  = df[FEATURES]
y  = df[TARGET].astype(int)

print(f"Class distribution:\n{y.value_counts().sort_index()}\n")

# ── SMOTE ─────────────────────────────────────────────────────────────────────
smote     = SMOTE(random_state=42)
X_r, y_r  = smote.fit_resample(X, y)

split = int(len(X_r) * 0.8)
X_tr, X_te = X_r[:split], X_r[split:]
y_tr, y_te = y_r[:split], y_r[split:]

classes = np.unique(y_tr)
weights = compute_class_weight("balanced", classes=classes, y=y_tr)
cw      = dict(zip(classes.tolist(), weights.tolist()))

# ── Train ─────────────────────────────────────────────────────────────────────
model = LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                       class_weight=cw, random_state=42, verbose=-1)
model.fit(X_tr, y_tr)

y_pred = model.predict(X_te)
f1     = f1_score(y_te, y_pred, average="macro")

print("── US-09 Classification Report ──────────────────────────────────")
print(classification_report(y_te, y_pred,
      labels=[1,2,3], target_names=["Low", "Medium", "High"]))
print(f"  Macro F1: {f1:.4f}")

# ── Score full dataset ────────────────────────────────────────────────────────
severity_map = {0: "None", 1: "Low", 2: "Medium", 3: "High"}
df = df.copy()
df["predicted_severity"]    = model.predict(df[FEATURES])
df["severity_probability"]  = model.predict_proba(df[FEATURES]).max(axis=1).round(4)
df["severity_label"]        = df["predicted_severity"].map(severity_map)

# ── Push to MongoDB ml_outputs ────────────────────────────────────────────────
try:
    client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=3000)
    client.server_info()
    col = client["water_supply_india"]["ml_outputs"]
    col.drop()
    records = df[["ward_id", "city_id", "predicted_severity",
                  "severity_label", "severity_probability"]].to_dict("records")
    col.insert_many(records)
    col.create_index([("city_id", ASCENDING), ("severity_label", ASCENDING)])
    print(f"\n  MongoDB: inserted {len(records)} docs → ml_outputs")
    client.close()
except Exception as e:
    print(f"\n  [MongoDB skipped] {e}")

df.to_csv("output/us09_deficit_severity.csv", index=False)
print("Saved → output/us09_deficit_severity.csv")
