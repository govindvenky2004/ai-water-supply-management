"""
US-20: Power BI Monsoon vs Supply Efficiency — data prep
US-21: Predict Drought Risk per Climate Zone
US-22: Flood Year Impact on Disruption Frequency
PATCHED – dim_rainfall confirmed columns:
rain_id, city_id, year, month, month_name, rainfall_mm, rainy_days,
departure_from_normal_pct, is_drought_year, is_flood_year, source, date,
rain_3m_sum, rain_6m_sum, rain_lag_1m, is_monsoon_month
NOTE: annual_rainfall_mm NOT in dim_rainfall — use dim_cities.annual_rainfall_mm
"""
import pandas as pd
import numpy as np
from scipy import stats
from lightgbm import LGBMClassifier
from sklearn.metrics import recall_score, classification_report
from sklearn.model_selection import train_test_split
import os, warnings
warnings.filterwarnings("ignore")
os.makedirs("output", exist_ok=True)

rainfall = pd.read_csv("preprocessed/dim_rainfall.csv")
supply   = pd.read_csv("preprocessed/fact_supply.csv")
disrupt  = pd.read_csv("preprocessed/fact_disruptions.csv")
cities   = pd.read_csv("preprocessed/dim_cities.csv")

supply["date"]  = pd.to_datetime(supply["date"])
supply["year"]  = supply["date"].dt.year
supply["month"] = supply["date"].dt.month
disrupt["start_date"] = pd.to_datetime(disrupt["start_date"])
disrupt["year"] = disrupt["start_date"].dt.year

print(f"dim_rainfall rows : {len(rainfall):,}")
print(f"fact_supply rows  : {len(supply):,}")
print(f"fact_disruptions  : {len(disrupt):,}")

# ════════════════════════════════════════════════════════════════
# US-20: Monsoon Intensity vs Supply Efficiency – Power BI prep
# ════════════════════════════════════════════════════════════════
print("\n── US-20: Monsoon vs Supply Efficiency ──────────────────────────")

monthly_supply = (
    supply.groupby(["city_id", "year", "month"])
    .agg(supply_efficiency_pct=("supply_efficiency_pct", "mean"),
         supply_deficit_mld   =("supply_deficit_mld",    "mean"))
    .reset_index()
)

rain_supply = rainfall.merge(
    monthly_supply, on=["city_id", "year", "month"], how="inner"
).merge(
    cities[["city_id", "city_name", "climate_zone"]], on="city_id", how="left"
)

# Pearson r: departure_from_normal vs supply_deficit
valid = rain_supply[["departure_from_normal_pct","supply_deficit_mld"]].dropna()
r, p  = stats.pearsonr(valid["departure_from_normal_pct"],
                        valid["supply_deficit_mld"])
print(f"  Pearson r (departure_from_normal vs supply_deficit): {r:.3f} (p={p:.4f})")

rain_supply.to_csv("output/us20_monsoon_supply.csv", index=False)
print(f"  Saved {len(rain_supply)} rows → output/us20_monsoon_supply.csv")

print("""
── US-20 Power BI Setup ─────────────────────────────────────────
Load: us20_monsoon_supply.csv
1. Dual-axis: rainfall_mm (bar) vs supply_efficiency_pct (line)
   X-axis: month | Legend: city_name
2. Scatter: departure_from_normal_pct (x) vs supply_deficit_mld (y)
   Show Pearson r as text box annotation
3. Slicer: climate_zone, year
4. Conditional formatting: is_drought_year=1 → red background
────────────────────────────────────────────────────────────────
""")

# ════════════════════════════════════════════════════════════════
# US-21: Predict Drought Risk per Climate Zone
# ════════════════════════════════════════════════════════════════
print("── US-21: Drought Risk Prediction ───────────────────────────────")

# Features confirmed in dim_rainfall
FEATURES = [
    "rain_3m_sum",              # ✓
    "rain_6m_sum",              # ✓
    "rainy_days",               # ✓
    "rain_lag_1m",              # ✓
    "departure_from_normal_pct",# ✓
    "rainfall_mm",              # ✓ (use monthly as proxy for annual)
]
TARGET = "is_drought_year"  # ✓

# climate_zone is in dim_cities not dim_rainfall — merge it in
rainfall_ext = rainfall.merge(cities[["city_id","climate_zone"]], on="city_id", how="left")
rain_df = rainfall_ext[FEATURES + [TARGET, "city_id", "climate_zone", "year"]].dropna()
all_forecasts = []

print(f"\n  Threshold: Recall > 80% on drought class")
for zone in rain_df["climate_zone"].unique():
    zone_df = rain_df[rain_df["climate_zone"] == zone]
    if len(zone_df) < 30 or zone_df[TARGET].sum() == 0:
        continue

    X = zone_df[FEATURES]
    y = zone_df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = LGBMClassifier(n_estimators=200, max_depth=4,
                           class_weight="balanced",
                           random_state=42, verbose=-1)
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    recall  = recall_score(y_test, y_pred, zero_division=0)
    status  = "✓ PASS" if recall >= 0.80 else "✗ needs more data"
    print(f"  {zone:<25}: Recall={recall:.2%}  n={len(zone_df)}  {status}")

    zone_df = zone_df.copy()
    zone_df["drought_probability"] = model.predict_proba(X)[:, 1].round(4)
    all_forecasts.append(
        zone_df[["city_id","climate_zone","year",
                 "drought_probability", TARGET]]
    )

if all_forecasts:
    drought_df = pd.concat(all_forecasts, ignore_index=True)
    drought_df.to_csv("output/us21_drought_risk.csv", index=False)
    print(f"\n  Saved → output/us21_drought_risk.csv")

    try:
        from sqlalchemy import create_engine
        engine = create_engine("mysql+pymysql://root:root@localhost/water_supply_india")
        drought_df.to_sql("drought_risk_forecast", engine,
                          if_exists="replace", index=False)
        print("  ✓ drought_risk_forecast saved to MySQL")
    except Exception as e:
        print(f"  [MySQL skipped] {e}")

# ════════════════════════════════════════════════════════════════
# US-22: Flood Year Impact on Disruption
# ════════════════════════════════════════════════════════════════
print("\n── US-22: Flood Year Impact Analysis ────────────────────────────")

# Join is_flood_year from dim_rainfall (year + city_id)
rain_year = (
    rainfall.groupby(["city_id","year"])["is_flood_year"]
    .max().reset_index()
)
disrupt_rain = disrupt.merge(rain_year, on=["city_id","year"], how="left")
disrupt_rain["is_flood_year"] = disrupt_rain["is_flood_year"].fillna(0).astype(int)

FLOOD_YEARS = [2017, 2020, 2022]
disrupt_rain["is_flood_year_flag"] = disrupt_rain["year"].isin(FLOOD_YEARS).astype(int)

flood_agg = (
    disrupt_rain.groupby("is_flood_year_flag")
    .agg(
        avg_duration_hours       =("duration_hours",            "mean"),
        avg_wards_affected       =("num_wards_affected",        "mean"),
        avg_supply_loss_mld      =("estimated_supply_loss_mld", "mean"),
        total_events             =("cause",                     "count"),
    )
    .reset_index()
)
flood_agg["is_flood_year_flag"] = flood_agg["is_flood_year_flag"].map(
    {0: "Non-Flood Year", 1: "Flood Year"}
)

print(flood_agg.to_string(index=False))

# Monsoon flooding in flood vs non-flood years
monsoon = disrupt_rain[disrupt_rain["cause"] == "Monsoon Flooding"]
if len(monsoon) > 0:
    monsoon_split = monsoon.groupby("is_flood_year_flag").size().reset_index(name="events")
    print(f"\n  Monsoon Flooding events by flood year flag:")
    print(monsoon_split.to_string(index=False))

# T-test
flood_dur   = disrupt_rain[disrupt_rain["is_flood_year_flag"]==1]["duration_hours"].dropna()
nflood_dur  = disrupt_rain[disrupt_rain["is_flood_year_flag"]==0]["duration_hours"].dropna()
t, p        = stats.ttest_ind(flood_dur, nflood_dur)
print(f"\n  T-test (duration_hours flood vs non-flood): t={t:.3f}, p={p:.4f}")
print(f"  {'Statistically significant' if p < 0.05 else 'Not significant'} (α=0.05)")

flood_agg.to_csv("output/us22_flood_impact.csv", index=False)
disrupt_rain.to_csv("output/us22_disruption_flood_joined.csv", index=False)
print(f"\nSaved → output/us22_flood_impact.csv")
print(f"Saved → output/us22_disruption_flood_joined.csv")
