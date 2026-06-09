"""
US-17: Power BI Slum Ward Equity Report — data prep
US-18: Complaint Hotspot Analysis
US-19: Predict Disruption Complaint Volume
PATCHED – uses real columns verified from diagnose_columns.py
"""
import pandas as pd
import numpy as np
from scipy import stats
import os, warnings
warnings.filterwarnings("ignore")
os.makedirs("output", exist_ok=True)

supply  = pd.read_csv("preprocessed/fact_supply.csv")
wards   = pd.read_csv("preprocessed/dim_wards.csv")
cities  = pd.read_csv("preprocessed/dim_cities.csv")
disrupt = pd.read_csv("preprocessed/fact_disruptions.csv")
disrupt["start_date"] = pd.to_datetime(disrupt["start_date"])

# ════════════════════════════════════════════════════════════════
# US-17: Slum Ward Equity – Power BI data prep
# ════════════════════════════════════════════════════════════════
print("── US-17: Slum Ward Equity Data Prep ───────────────────────────")

# fact_supply has: supply_efficiency_pct, nrw_pct, hours_of_supply
# dim_wards has: ward_type, piped_connection_coverage_pct,
#                metered_connections_pct, has_slum_pocket, wt_Slum/Informal

# Ward-level averages from fact_supply
ward_supply = (
    supply.groupby(["ward_id", "city_id"])
    .agg(
        supply_efficiency_pct =("supply_efficiency_pct", "mean"),
        hours_of_supply       =("hours_of_supply",       "mean"),
        nrw_pct               =("nrw_pct",               "mean"),
    )
    .reset_index()
)

# Merge ward attributes
equity = ward_supply.merge(
    wards[["ward_id", "ward_type", "piped_connection_coverage_pct",
           "metered_connections_pct", "has_slum_pocket",
           "ward_type_encoded", "wt_Slum/Informal"]],
    on="ward_id", how="left"
).merge(
    cities[["city_id", "city_name", "zone", "state"]],
    on="city_id", how="left"
)

# Merge lpcd_used from ml_sample
try:
    ml = pd.read_csv("preprocessed/ml_sample.csv",
                     usecols=["ward_id", "lpcd_used"]).groupby("ward_id").mean().reset_index()
    equity = equity.merge(ml, on="ward_id", how="left")
except:
    equity["lpcd_used"] = np.nan

equity.to_csv("output/us17_slum_equity.csv", index=False)
print(f"  Saved {len(equity)} rows → output/us17_slum_equity.csv")

# Quick equity summary
slum_eff = equity[equity["wt_Slum/Informal"]==1]["supply_efficiency_pct"].mean()
res_eff  = equity[equity["ward_type"]=="Residential"]["supply_efficiency_pct"].mean()
print(f"  Slum avg efficiency  : {slum_eff:.1f}%")
print(f"  Residential avg eff  : {res_eff:.1f}%")

print("""
── US-17 Power BI Setup ─────────────────────────────────────────
Load: us17_slum_equity.csv
1. Side-by-side bar: piped_connection_coverage_pct + metered_connections_pct
   Axis: ward_type | Slicer: city_name
2. Scatter: has_slum_pocket (x) vs supply_efficiency_pct (y)
3. Box plot: lpcd_used by ward_type (use R visual in Power BI)
4. Slicer: zone → worst regional equity gap
────────────────────────────────────────────────────────────────
""")

# ════════════════════════════════════════════════════════════════
# US-18: Complaint Hotspot Analysis
# ════════════════════════════════════════════════════════════════
print("── US-18: Complaint Hotspot Analysis ────────────────────────────")

# fact_disruptions confirmed columns: complaint_count, duration_hours,
# population_affected, ward_id, month, year, cause, severity
ward_complaints = (
    disrupt.groupby("ward_id")["complaint_count"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
    .rename(columns={"complaint_count": "total_complaints"})
)
print(f"\n  Top 5 complaint wards:")
print(ward_complaints.head(5).to_string(index=False))

# Pearson correlations
valid = disrupt[["complaint_count","duration_hours","population_affected"]].dropna()
r_dur, p_dur = stats.pearsonr(valid["duration_hours"], valid["complaint_count"])
r_pop, p_pop = stats.pearsonr(valid["population_affected"], valid["complaint_count"])
print(f"\n  Complaint vs duration_hours:       r={r_dur:.3f}  p={p_dur:.4f}")
print(f"  Complaint vs population_affected:  r={r_pop:.3f}  p={p_pop:.4f}")
print(f"  → {'duration_hours' if abs(r_dur)>abs(r_pop) else 'population_affected'} "
      f"drives complaints more")

# Monthly heatmap
disrupt["month"] = disrupt["start_date"].dt.month
disrupt["year"]  = disrupt["start_date"].dt.year
monthly_heat = (
    disrupt.groupby(["month","year"])["complaint_count"]
    .sum().reset_index()
    .sort_values("complaint_count", ascending=False)
)
print(f"\n  Peak complaint months:")
print(monthly_heat.head(5).to_string(index=False))

ward_complaints.to_csv("output/us18_ward_complaints.csv", index=False)
monthly_heat.to_csv("output/us18_monthly_heatmap.csv", index=False)
print(f"\nSaved → output/us18_ward_complaints.csv")
print(f"Saved → output/us18_monthly_heatmap.csv")

try:
    from sqlalchemy import create_engine
    engine = create_engine("mysql+pymysql://root:root@localhost/water_supply_india")
    ward_complaints.to_sql("complaint_analysis", engine, if_exists="replace", index=False)
    print("✓ complaint_analysis saved to MySQL")
except Exception as e:
    print(f"[MySQL skipped] {e}")

# ════════════════════════════════════════════════════════════════
# US-19: Predict Complaint Volume (Regression)
# ════════════════════════════════════════════════════════════════
print("\n── US-19: Predict Complaint Volume ──────────────────────────────")
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# One-hot encode cause
cause_dummies = pd.get_dummies(disrupt["cause"], prefix="cause")
X_base = disrupt[["duration_hours", "population_affected",
                   "estimated_supply_loss_mld",
                   "num_wards_affected"]].fillna(0)
X = pd.concat([X_base, cause_dummies], axis=1)

y_raw = disrupt["complaint_count"].fillna(0)
y_log = np.log1p(y_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

model = LGBMRegressor(n_estimators=200, max_depth=4,
                      learning_rate=0.05, random_state=42, verbose=-1)
model.fit(X_train, y_train)

y_pred_log = model.predict(X_test)
y_pred     = np.expm1(y_pred_log)
y_actual   = np.expm1(y_test)

r2   = r2_score(y_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))

print(f"  R²  : {r2:.4f}  (criterion > 0.70)")
print(f"  RMSE: {rmse:.1f} complaints (original scale)")

disrupt["predicted_complaints"] = np.expm1(model.predict(X)).round(0)
disrupt[["ward_id","city_id","cause","duration_hours",
         "complaint_count","predicted_complaints"]].to_csv(
    "output/us19_complaint_predictions.csv", index=False)
print(f"Saved → output/us19_complaint_predictions.csv")

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
    print(f"  MongoDB: {len(disrupt)} docs → complaint_predictions")
    client.close()
except Exception as e:
    print(f"  [MongoDB skipped] {e}")
