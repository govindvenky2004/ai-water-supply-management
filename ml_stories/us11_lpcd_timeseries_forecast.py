"""
US-11: Time-Series Forecasting of LPCD Trends by Ward Type
PATCHED – columns verified against diagnose_columns.py
KEY FIXES:
  - Use fact_demand (has lpcd_used, ward_type, is_festival_month,
    covid_adjustment, season, rainfall_mm, demand_lag_1w) ✓
  - fact_demand confirmed to have ALL required columns
  - ward_type is already in fact_demand ✓
  - season is int (1-4) in fact_demand, use as-is ✓
"""
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sqlalchemy import create_engine
import os, warnings
warnings.filterwarnings("ignore")
os.makedirs("output", exist_ok=True)

# ── Load fact_demand ──────────────────────────────────────────────────────────
demand = pd.read_csv("preprocessed/fact_demand.csv")
demand["date"] = pd.to_datetime(demand["date"])
demand = demand.sort_values("date")

print(f"fact_demand rows: {len(demand):,}")
print(f"Ward types: {demand['ward_type'].unique()}")

# ── Features – all verified in fact_demand ────────────────────────────────────
FEATURES = [
    "demand_lag_1w",      # ✓
    "demand_lag_2w",      # ✓
    "demand_roll_4w_avg", # ✓
    "is_festival_month",  # ✓
    "covid_adjustment",   # ✓
    "season",             # ✓ (int 1-4)
    "rainfall_mm",        # ✓
    "seasonal_factor",    # ✓
]
TARGET = "lpcd_used"  # ✓ in fact_demand

SLUM_TYPE = "Slum/Informal"
RES_TYPE  = "Residential"

ward_types    = demand["ward_type"].dropna().unique()
all_forecasts = []
equity_alerts = []

print(f"\n── Training one model per ward type ────────────────────────────")
for wt in ward_types:
    wt_df = demand[demand["ward_type"] == wt].copy()
    train = wt_df[wt_df["year"] <= 2023]
    test  = wt_df[wt_df["year"] == 2024]

    X_tr = train[FEATURES].fillna(0)
    y_tr = train[TARGET].fillna(train[TARGET].median())

    if len(X_tr) < 50:
        print(f"  {wt}: skipped (only {len(X_tr)} rows)")
        continue

    model = LGBMRegressor(n_estimators=200, max_depth=4,
                          learning_rate=0.05, random_state=42, verbose=-1)
    model.fit(X_tr, y_tr)

    # Test MAPE
    if len(test) > 0:
        X_te = test[FEATURES].fillna(0)
        y_te = test[TARGET].fillna(test[TARGET].median())
        mape = mean_absolute_percentage_error(y_te, model.predict(X_te)) * 100
        print(f"  {wt:<20}: MAPE={mape:.2f}%  rows={len(X_tr):,}")

    # 12-week rolling forecast
    cur = wt_df.sort_values("date").tail(1).copy()
    for w in range(1, 13):
        pred = model.predict(cur[FEATURES].fillna(0))[0]
        all_forecasts.append({"ward_type": wt, "forecast_week": w,
                               "lpcd_forecast": round(pred, 2)})
        cur["demand_lag_2w"] = cur["demand_lag_1w"].values[0]
        cur["demand_lag_1w"] = pred

forecast_df = pd.DataFrame(all_forecasts)

# ── Equity check: Slum vs Residential gap ────────────────────────────────────
print(f"\n── LPCD Equity Check (Slum vs Residential) ──────────────────────")
for w in range(1, 13):
    week_data  = forecast_df[forecast_df["forecast_week"] == w]
    slum_lpcd  = week_data[week_data["ward_type"] == SLUM_TYPE]["lpcd_forecast"].mean()
    res_lpcd   = week_data[week_data["ward_type"] == RES_TYPE]["lpcd_forecast"].mean()
    if pd.isna(slum_lpcd) or pd.isna(res_lpcd):
        continue
    gap   = res_lpcd - slum_lpcd
    alert = "⚠ ALERT" if gap > 80 else ""
    print(f"  Wk {w:02d}: Slum={slum_lpcd:.1f}  Residential={res_lpcd:.1f}  "
          f"Gap={gap:.1f} LPCD  {alert}")
    if gap > 80:
        equity_alerts.append({"week": w, "slum_lpcd": round(slum_lpcd, 1),
                               "residential_lpcd": round(res_lpcd, 1), "gap": round(gap, 1)})

# ── Save to MySQL ─────────────────────────────────────────────────────────────
try:
    engine = create_engine("mysql+pymysql://root:password@localhost/water_supply_india")
    forecast_df.to_sql("lpcd_forecast", engine, if_exists="replace", index=False)
    print("\n✓ lpcd_forecast saved to MySQL")
except Exception as e:
    print(f"\n[MySQL skipped] {e}")
    forecast_df.to_csv("output/us11_lpcd_forecast_mysql.csv", index=False)

forecast_df.to_csv("output/us11_lpcd_forecast.csv", index=False)
if equity_alerts:
    pd.DataFrame(equity_alerts).to_csv("output/us11_equity_alerts.csv", index=False)
    print(f"  ⚠ {len(equity_alerts)} equity alert weeks saved → output/us11_equity_alerts.csv")
print("Saved → output/us11_lpcd_forecast.csv")
