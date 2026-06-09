"""
US-02: Power BI Supply vs Demand Gap Dashboard - Data Prep
Generates clean CSVs for Power BI using real column names
"""
import pandas as pd
import os
os.makedirs("output", exist_ok=True)

supply = pd.read_csv("preprocessed/fact_supply.csv")
cities = pd.read_csv("preprocessed/dim_cities.csv")
supply["date"] = pd.to_datetime(supply["date"])
supply["year"]  = supply["date"].dt.year
supply["month"] = supply["date"].dt.month

# ── City-level aggregation ────────────────────────────────────────
city_agg = (
    supply.groupby(["city_id", "year"])
    .agg(
        actual_supply_mld        =("actual_supply_mld",       "mean"),
        demand_mld               =("demand_mld",              "mean"),
        supply_deficit_mld       =("supply_deficit_mld",      "mean"),
        avg_hours_supply_per_day =("hours_of_supply",         "mean"),
        nrw_pct                  =("nrw_pct",                 "mean"),
        metering_pct             =("nrw_vs_benchmark",        "mean"),
        supply_efficiency_pct    =("supply_efficiency_pct",   "mean"),
    )
    .reset_index()
    .merge(cities[["city_id","city_name","state","zone","climate_zone"]],
           on="city_id", how="left")
)
city_agg["supply_deficit_flag"] = (city_agg["supply_deficit_mld"] > 0).astype(int)
city_agg = city_agg.round(2)
city_agg.to_csv("output/us02_powerbi_supply_demand.csv", index=False)
print(f"Saved {len(city_agg)} rows → output/us02_powerbi_supply_demand.csv")
print(f"Columns: {list(city_agg.columns)}")

# ── NRW heatmap (city x month) ────────────────────────────────────
nrw = (
    supply.groupby(["city_id","year","month"])
    ["nrw_pct"].mean().reset_index()
    .merge(cities[["city_id","city_name"]], on="city_id", how="left")
    .round(2)
)
nrw.to_csv("output/us02_nrw_heatmap.csv", index=False)
print(f"Saved {len(nrw)} rows → output/us02_nrw_heatmap.csv")
