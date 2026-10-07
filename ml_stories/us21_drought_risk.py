"""
US-21: Predict Drought Risk for Next Season Using Lagged Rainfall
Per-climate-zone LightGBM classifier. Time-based split (2019-2023 train, 2024 test).
Decision threshold is chosen on TRAIN data only (so the test set is not tuned on).
Edit RAIN_CSV if your rainfall table has a different name.
"""
import os, warnings
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import recall_score, precision_score, roc_auc_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
warnings.filterwarnings("ignore"); os.makedirs("output", exist_ok=True)

RAIN_CSV   = "preprocessed/fact_rainfall.csv"
CITIES_CSV = "preprocessed/dim_cities.csv"
FEATURES = ["rain_lag_1m", "rain_3m_sum", "rain_6m_sum", "rainy_days",
            "departure_from_normal_pct", "annual_rainfall_mm"]
TARGET = "is_drought_year"

df = pd.read_csv(RAIN_CSV)
if "climate_zone" not in df.columns:
    cities = pd.read_csv(CITIES_CSV)
    df = df.merge(cities[["city_id", "climate_zone"]], on="city_id", how="left")
missing = [c for c in FEATURES + [TARGET, "climate_zone", "city_id"] if c not in df.columns]
assert not missing, f"Columns not found: {missing}\nAvailable: {list(df.columns)}"
if "year" not in df.columns:
    df["year"] = pd.to_datetime(df["date"]).dt.year
df = df.dropna(subset=FEATURES + [TARGET, "climate_zone"])
df[TARGET] = df[TARGET].astype(int)

rows, latest_preds = [], []
print(f"Rows: {len(df):,} | drought rate: {df[TARGET].mean():.2%}\n")
for zone, z in df.groupby("climate_zone"):
    tr, te = z[z.year <= 2023], z[z.year == 2024]
    if tr[TARGET].nunique() < 2 or len(te) == 0 or te[TARGET].sum() == 0:
        print(f"  {zone:<18} skipped (not enough drought examples in train/test)"); continue
    model = LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                           class_weight="balanced", random_state=42, verbose=-1)
    # threshold from out-of-fold TRAIN probabilities: highest threshold with recall >= 0.80
    oof = cross_val_predict(model, tr[FEATURES], tr[TARGET], method="predict_proba",
                            cv=StratifiedKFold(5, shuffle=True, random_state=42))[:, 1]
    thr = 0.5
    for t in np.arange(0.9, 0.04, -0.05):
        if recall_score(tr[TARGET], oof >= t) >= 0.80: thr = float(t); break
    model.fit(tr[FEATURES], tr[TARGET])
    p = model.predict_proba(te[FEATURES])[:, 1]
    pred = (p >= thr).astype(int)
    rec = recall_score(te[TARGET], pred); prec = precision_score(te[TARGET], pred, zero_division=0)
    auc = roc_auc_score(te[TARGET], p) if te[TARGET].nunique() > 1 else float("nan")
    ok = "meets" if rec > 0.80 else "does NOT meet"
    print(f"  {zone:<18} thr={thr:.2f} recall={rec:.2%} precision={prec:.2%} AUC={auc:.3f} ({ok} recall>80%)")
    rows.append(dict(climate_zone=zone, threshold=thr, recall=rec, precision=prec, auc=auc,
                     n_test=len(te), n_test_drought=int(te[TARGET].sum())))
    last = z.sort_values("year").groupby("city_id").tail(1).copy()
    last["drought_probability"] = model.predict_proba(last[FEATURES])[:, 1].round(4)
    last["drought_risk_flag"] = (last["drought_probability"] >= thr).astype(int)
    latest_preds.append(last[["city_id", "climate_zone", "drought_probability", "drought_risk_flag"]])

pd.DataFrame(rows).to_csv("output/us21_drought_metrics.csv", index=False)
forecast = pd.concat(latest_preds) if latest_preds else pd.DataFrame()
forecast.to_csv("output/us21_drought_risk_forecast.csv", index=False)
print("\nSaved -> output/us21_drought_metrics.csv, output/us21_drought_risk_forecast.csv")

try:
    from sqlalchemy import create_engine
    pw = os.getenv("MYSQL_PASSWORD", "password")
    eng = create_engine(f"mysql+pymysql://root:{pw}@localhost/water_supply_india")
    forecast.to_sql("drought_risk_forecast", eng, if_exists="replace", index=False)
    print("drought_risk_forecast saved to MySQL")
except Exception as e:
    print(f"[MySQL skipped] {e}")
