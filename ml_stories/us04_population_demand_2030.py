"""
US-04: Predict 2030 Population-Driven Water Demand by City
PATCHED – columns verified against diagnose_columns.py
KEY FIXES:
  - actual_lpcd is in dim_cities (not dim_population)
  - dim_population has no actual_lpcd column
  - dim_population has: city_id, year, population, annual_growth_rate_pct,
                        est_daily_water_demand_mld, projected, data_source
"""
import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings("ignore")
os.makedirs("output", exist_ok=True)

# ── Load tables ───────────────────────────────────────────────────────────────
pop     = pd.read_csv("preprocessed/dim_population.csv")
sources = pd.read_csv("preprocessed/dim_sources.csv")
cities  = pd.read_csv("preprocessed/dim_cities.csv")

# ── Base population per city (most recent year available) ─────────────────────
base_pop = (
    pop.sort_values("year")
       .groupby("city_id")
       .last()
       .reset_index()[["city_id", "population", "annual_growth_rate_pct"]]
)

# actual_lpcd lives in dim_cities ✓
base_pop = base_pop.merge(
    cities[["city_id", "actual_lpcd", "city_name", "state"]],
    on="city_id", how="left"
)
base_pop["actual_lpcd"] = base_pop["actual_lpcd"].fillna(135)  # BIS minimum fallback

BASE_YEAR = 2024

# ── Project 2025–2030 ─────────────────────────────────────────────────────────
rows = []
for _, row in base_pop.iterrows():
    for yr in range(2025, 2031):
        n        = yr - BASE_YEAR
        proj_pop = row["population"] * (1 + row["annual_growth_rate_pct"] / 100) ** n
        demand   = (proj_pop * row["actual_lpcd"]) / 1_000_000  # → MLD
        rows.append({
            "city_id":                    row["city_id"],
            "city_name":                  row["city_name"],
            "state":                      row["state"],
            "year":                       yr,
            "population":                 round(proj_pop),
            "annual_growth_rate_pct":     row["annual_growth_rate_pct"],
            "actual_lpcd":                row["actual_lpcd"],
            "est_daily_water_demand_mld": round(demand, 2),
            "projected":                  True,
        })

proj_df = pd.DataFrame(rows)

# ── Historical rows (projected=False already in dim_population) ───────────────
hist = pop[["city_id", "year", "population",
            "annual_growth_rate_pct", "est_daily_water_demand_mld",
            "projected"]].copy()
hist = hist.merge(cities[["city_id", "actual_lpcd", "city_name", "state"]],
                  on="city_id", how="left")

final_df = pd.concat([hist, proj_df], ignore_index=True)

# ── Capacity check: projected demand vs active source capacity ────────────────
# dim_sources: installed_capacity_mld, is_active ✓
active_cap = (
    sources[sources["is_active"] == 1]
    .groupby("city_id")["installed_capacity_mld"]
    .sum()
    .reset_index()
    .rename(columns={"installed_capacity_mld": "total_active_capacity_mld"})
)

proj_check = proj_df.merge(active_cap, on="city_id", how="left")
proj_check["capacity_deficit_mld"] = (
    proj_check["est_daily_water_demand_mld"] - proj_check["total_active_capacity_mld"]
)
proj_check["exceeds_capacity"] = proj_check["capacity_deficit_mld"] > 0

print("── Cities Projected to Exceed Supply Capacity (2025-2030) ──────")
exceed = proj_check[proj_check["exceeds_capacity"]].sort_values(["city_id", "year"])
if len(exceed) == 0:
    print("  No cities exceed capacity through 2030 with current sources.")
else:
    print(exceed[["city_name", "state", "year",
                  "est_daily_water_demand_mld",
                  "total_active_capacity_mld",
                  "capacity_deficit_mld"]].to_string(index=False))

# ── Save outputs ──────────────────────────────────────────────────────────────
final_df.to_csv("output/us04_population_demand_projection.csv", index=False)
proj_check.to_csv("output/us04_capacity_gap_check.csv", index=False)

# Power BI line chart data
pbi = proj_check[["city_id", "city_name", "year",
                  "est_daily_water_demand_mld",
                  "total_active_capacity_mld"]]
pbi.to_csv("output/us04_powerbi_demand_vs_capacity.csv", index=False)

print(f"\nSaved → output/us04_population_demand_projection.csv")
print(f"Saved → output/us04_capacity_gap_check.csv")
print(f"Saved → output/us04_powerbi_demand_vs_capacity.csv")

print("""
── Power BI Line Chart Setup ────────────────────────────────────
  Load: us04_powerbi_demand_vs_capacity.csv
  Visual: Line Chart
    X-axis : year
    Values : est_daily_water_demand_mld  (solid)
             total_active_capacity_mld   (dashed reference)
    Legend : city_name
  Filter  : projected = True  (shows 2025-2030 only)
────────────────────────────────────────────────────────────────
""")
