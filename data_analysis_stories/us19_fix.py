"""
US-19 FIX: Improve R² above 0.70 with feature engineering
Run this standalone to replace the US-19 section output
"""
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import os, warnings
warnings.filterwarnings("ignore")
os.makedirs("output", exist_ok=True)

disrupt = pd.read_csv("preprocessed/fact_disruptions.csv")
disrupt["start_date"] = pd.to_datetime(disrupt["start_date"])
disrupt["month"] = disrupt["start_date"].dt.month
disrupt["year"]  = disrupt["start_date"].dt.year

# ── Feature engineering ───────────────────────────────────────────────────────
# 1. Duration buckets (short/medium/long)
disrupt["duration_bucket"] = pd.cut(
    disrupt["duration_hours"],
    bins=[0, 24, 72, 168, 9999],
    labels=[0, 1, 2, 3]
).astype(int)

# 2. Per-capita complaint pressure
disrupt["complaints_per_ward"] = (
    disrupt["complaint_count"] / (disrupt["num_wards_affected"] + 1)
).fillna(0)

# 3. Supply loss per ward
disrupt["loss_per_ward"] = (
    disrupt["estimated_supply_loss_mld"] / (disrupt["num_wards_affected"] + 1)
).fillna(0)

# 4. Season (monsoon = high complaints)
disrupt["is_monsoon"] = disrupt["month"].isin([6, 7, 8, 9]).astype(int)

# 5. Severity encoded
sev_map = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
disrupt["severity_encoded"] = disrupt["severity"].map(sev_map).fillna(1)

# 6. Cause one-hot
cause_dummies = pd.get_dummies(disrupt["cause"], prefix="cause")

# ── Features ──────────────────────────────────────────────────────────────────
X_base = disrupt[[
    "duration_hours",
    "population_affected",
    "estimated_supply_loss_mld",
    "num_wards_affected",
    "duration_bucket",
    "loss_per_ward",
    "is_monsoon",
    "severity_encoded",
    "month",
    "year",
]].fillna(0)

X = pd.concat([X_base, cause_dummies], axis=1)
y_raw = disrupt["complaint_count"].fillna(0)
y_log = np.log1p(y_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

model = LGBMRegressor(
    n_estimators=500, max_depth=6,
    learning_rate=0.03, num_leaves=31,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbose=-1
)
model.fit(X_train, y_train)

y_pred_log = model.predict(X_test)
y_pred     = np.expm1(y_pred_log)
y_actual   = np.expm1(y_test)

r2   = r2_score(y_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))

print(f"── US-19 Improved Results ────────────────────────────────────────")
print(f"  R²  : {r2:.4f}  (criterion > 0.70)  {'✓ PASS' if r2 >= 0.70 else '✗ FAIL'}")
print(f"  RMSE: {rmse:.1f} complaints (original scale)")

# Feature importance
importance = pd.Series(model.feature_importances_, index=X.columns)
print(f"\n  Top 5 features:")
for feat, val in importance.nlargest(5).items():
    print(f"    {feat:<35} {val:>5}")

# Save predictions
disrupt["predicted_complaints"] = np.expm1(model.predict(X)).round(0)
disrupt[["ward_id","city_id","cause","duration_hours",
         "complaint_count","predicted_complaints"]].to_csv(
    "output/us19_complaint_predictions.csv", index=False)
print(f"\nSaved → output/us19_complaint_predictions.csv (updated)")

# Push to MongoDB
try:
    from pymongo import MongoClient
    client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=3000)
    client.server_info()
    col = client["water_supply_india"]["complaint_predictions"]
    col.drop()
    col.insert_many(
        disrupt[["ward_id","city_id","cause",
                 "predicted_complaints"]].to_dict("records")
    )
    print(f"  MongoDB: {len(disrupt)} docs → complaint_predictions (updated)")
    client.close()
except Exception as e:
    print(f"  [MongoDB skipped] {e}")
