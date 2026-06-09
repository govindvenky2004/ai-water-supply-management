"""
US-05: Predict Pipeline Burst Risk by Ward Using Infrastructure Age
PATCHED v3 – fix: split BEFORE SMOTE (SMOTE only on train set)
"""
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from pymongo import MongoClient, ASCENDING
from datetime import datetime
import os, warnings
warnings.filterwarnings("ignore")
os.makedirs("output", exist_ok=True)

infra   = pd.read_csv("preprocessed/dim_infrastructure.csv")
disrupt = pd.read_csv("preprocessed/fact_disruptions.csv")

print(f"Infrastructure wards : {len(infra):,}")
print(f"Burst wards (label=1): {infra['ward_id'].isin(set(disrupt[disrupt['cause']=='Pipeline Burst']['ward_id'])).sum()}")

burst_wards = set(disrupt[disrupt["cause"] == "Pipeline Burst"]["ward_id"].unique())
infra["burst_label"] = infra["ward_id"].isin(burst_wards).astype(int)

MAT_COLS = [c for c in infra.columns if c.startswith("mat_")]
FEATURES = ["pipe_age_years", "pipeline_condition_encoded",
            "estimated_leakage_pct", "last_major_repair_year"] + MAT_COLS

for f in FEATURES:
    infra[f] = infra[f].fillna(infra[f].median())

X = infra[FEATURES]
y = infra["burst_label"]

# ── Split FIRST, then SMOTE only on train ─────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")
print(f"Train class dist: {y_train.value_counts().to_dict()}")

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"After SMOTE (train only): {pd.Series(y_train_res).value_counts().to_dict()}")

model = LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                       random_state=42, verbose=-1)
model.fit(X_train_res, y_train_res)

y_proba = model.predict_proba(X_test)[:, 1]
y_pred  = (y_proba >= 0.35).astype(int)
auc     = roc_auc_score(y_test, y_proba)

print("\n── Classification Report (threshold=0.35) ───────────────────────")
print(classification_report(y_test, y_pred, target_names=["No Burst", "Burst"]))
print(f"  AUC-ROC: {auc:.4f}")

# ── Score all wards with plain string labels (not Categorical) ────────────────
infra["risk_score"] = model.predict_proba(infra[FEATURES])[:, 1].round(4)
infra["risk_label"] = infra["risk_score"].apply(
    lambda s: "High" if s >= 0.6 else ("Medium" if s >= 0.35 else "Low")
)

# Hard rule override
hard_rule = (infra["pipe_age_years"] > 40) & (infra["pipeline_condition"] == "Poor")
infra.loc[hard_rule, "risk_label"] = "Critical"
print(f"\n  Hard-rule Critical: {hard_rule.sum()} wards")
print(f"  Risk distribution:\n{infra['risk_label'].value_counts()}")

# ── MongoDB ───────────────────────────────────────────────────────────────────
try:
    client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=3000)
    client.server_info()
    col = client["water_supply_india"]["risk_scores"]
    col.drop()
    records = infra[["ward_id", "city_id", "pipe_age_years", "pipeline_condition",
                      "estimated_leakage_pct", "pipeline_material",
                      "risk_score", "risk_label"]].copy()
    records["timestamp"] = datetime.utcnow().isoformat()
    col.insert_many(records.to_dict("records"))
    col.create_index([("city_id", ASCENDING), ("risk_label", ASCENDING)])
    print(f"  MongoDB: {len(records)} docs → risk_scores")
    client.close()
except Exception as e:
    print(f"  [MongoDB skipped] {e}")

infra[["ward_id", "city_id", "pipe_age_years", "pipeline_condition",
       "estimated_leakage_pct", "risk_score", "risk_label"]].to_csv(
    "output/us05_pipeline_burst_risk.csv", index=False)
print("Saved → output/us05_pipeline_burst_risk.csv")