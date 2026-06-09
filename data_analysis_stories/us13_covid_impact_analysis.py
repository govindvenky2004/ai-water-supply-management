"""
US-13: Analyse COVID-19 Impact on Water Demand Across Ward Types
Role: Government Policy Analyst
PATCHED – uses fact_demand which has: covid_adjustment, estimated_demand_mld,
ward_type, year, month, hours_of_supply NOT in fact_demand →
use fact_supply for hours_of_supply and nrw_pct
"""
import pandas as pd
import numpy as np
import os
os.makedirs("output", exist_ok=True)

demand = pd.read_csv("preprocessed/fact_demand.csv")
supply = pd.read_csv("preprocessed/fact_supply.csv")
cities = pd.read_csv("preprocessed/dim_cities.csv")

print(f"fact_demand rows: {len(demand):,}")

# ── COVID period: Apr-Jun 2020 (covid_adjustment == 1) ───────────────────────
# fact_demand has covid_adjustment ✓, estimated_demand_mld ✓, ward_type ✓
covid = demand[(demand["year"] == 2020) & (demand["covid_adjustment"] == 1)]
pre   = demand[(demand["year"] == 2019) & (demand["month"].isin([4, 5, 6]))]
post  = demand[(demand["year"] == 2021) & (demand["month"].isin([4, 5, 6]))]

print(f"COVID rows  : {len(covid):,}")
print(f"Pre rows    : {len(pre):,}")
print(f"Post rows   : {len(post):,}")

def agg_demand(df, period):
    return (
        df.groupby(["city_id", "ward_type"])
        .agg(avg_demand=("estimated_demand_mld", "mean"))
        .assign(period=period)
        .reset_index()
    )

covid_agg = agg_demand(covid, "COVID Apr-Jun 2020")
pre_agg   = agg_demand(pre,   "Pre Apr-Jun 2019")
post_agg  = agg_demand(post,  "Post Apr-Jun 2021")

# % change: COVID vs Pre-2019
merged = covid_agg.merge(
    pre_agg[["city_id", "ward_type", "avg_demand"]],
    on=["city_id", "ward_type"], suffixes=("_covid", "_pre")
)
merged["demand_pct_change"] = (
    (merged["avg_demand_covid"] - merged["avg_demand_pre"])
    / merged["avg_demand_pre"] * 100
).round(2)

print("\n── COVID Demand Impact by Ward Type ─────────────────────────────")
wt_summary = merged.groupby("ward_type")["demand_pct_change"].mean().round(2)
print(wt_summary)
print("\n(Negative = demand dropped during COVID lockdown)")

# ── Secondary metrics from fact_supply ────────────────────────────────────────
# fact_supply has nrw_pct ✓ and hours_of_supply ✓
supply_covid = supply[
    (supply["date"].str[:7].isin(["2020-04","2020-05","2020-06"]))
] if "date" in supply.columns else pd.DataFrame()

supply_pre = supply[
    (supply["date"].str[:7].isin(["2019-04","2019-05","2019-06"]))
] if "date" in supply.columns else pd.DataFrame()

secondary = {}
if len(supply_covid) > 0 and len(supply_pre) > 0:
    secondary = {
        "avg_hours_covid": round(supply_covid["hours_of_supply"].mean(), 2),
        "avg_hours_pre":   round(supply_pre["hours_of_supply"].mean(), 2),
        "avg_nrw_covid":   round(supply_covid["nrw_pct"].mean(), 2),
        "avg_nrw_pre":     round(supply_pre["nrw_pct"].mean(), 2),
    }
    print(f"\n── Secondary Metrics ────────────────────────────────────────────")
    print(f"  Hours supply: {secondary['avg_hours_pre']}h (pre) → "
          f"{secondary['avg_hours_covid']}h (COVID)")
    print(f"  NRW pct:      {secondary['avg_nrw_pre']}% (pre) → "
          f"{secondary['avg_nrw_covid']}% (COVID)")

# ── Build policy report ───────────────────────────────────────────────────────
policy_report = merged[["city_id", "ward_type",
                         "avg_demand_covid", "avg_demand_pre",
                         "demand_pct_change"]].copy()
policy_report["period"] = "COVID Apr-Jun 2020 vs 2019 baseline"
policy_report = policy_report.merge(
    cities[["city_id", "city_name", "state"]], on="city_id", how="left"
)

# ── Save ──────────────────────────────────────────────────────────────────────
policy_report.to_csv("output/us13_covid_impact.csv", index=False)
pd.concat([covid_agg, pre_agg, post_agg]).to_csv(
    "output/us13_all_periods.csv", index=False)
print(f"\nSaved → output/us13_covid_impact.csv")
print(f"Saved → output/us13_all_periods.csv")

try:
    from sqlalchemy import create_engine
    engine = create_engine("mysql+pymysql://root:root@localhost/water_supply_india")
    policy_report.to_sql("policy_reports", engine, if_exists="replace", index=False)
    print("✓ policy_reports saved to MySQL")
except Exception as e:
    print(f"[MySQL skipped] {e}")
