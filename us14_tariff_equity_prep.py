"""
US-14: Power BI Tariff Equity Analysis — Data Prep
Patched against real columns:
dim_tariff: tariff_id, city_id, slab_number, consumption_lower_kl,
consumption_upper_kl, rate_per_kl_inr, fixed_monthly_charge_inr,
connection_charge_inr, effective_from_year, category, water_board
dim_cities: actual_lpcd, population_density_per_km2, city_name, state
"""
import pandas as pd
import os
os.makedirs("output", exist_ok=True)

tariff = pd.read_csv("preprocessed/dim_tariff.csv")
cities = pd.read_csv("preprocessed/dim_cities.csv")
supply = pd.read_csv("preprocessed/fact_supply.csv")

print(f"dim_tariff rows : {len(tariff)}")
print(f"Columns: {list(tariff.columns)}")

# ── Merge city info into tariff ───────────────────────────────────
tariff_full = tariff.merge(
    cities[["city_id","city_name","state","actual_lpcd",
            "population_density_per_km2","zone","climate_zone"]],
    on="city_id", how="left"
)

# ── Average lpcd from fact_supply (actual usage) ──────────────────
avg_lpcd = (
    supply.groupby("city_id")["nrw_pct"]
    .mean().reset_index()
    .rename(columns={"nrw_pct": "avg_nrw_pct"})
)
tariff_full = tariff_full.merge(avg_lpcd, on="city_id", how="left")

# ── Affordability burden ──────────────────────────────────────────
# fixed_monthly_charge as % of minimum wage (approx Rs 500/day = Rs 15000/month)
MIN_WAGE_MONTHLY = 15000
tariff_full["affordability_burden_pct"] = (
    tariff_full["fixed_monthly_charge_inr"] / MIN_WAGE_MONTHLY * 100
).round(2)

# ── Slab 1 only (for scatter plot) ───────────────────────────────
slab1 = tariff_full[tariff_full["slab_number"] == 1].copy()
slab1["is_zero_rate"] = (slab1["rate_per_kl_inr"] == 0).astype(int)
slab1["equity_flag"] = (
    (slab1["actual_lpcd"] < 100) &
    (slab1["rate_per_kl_inr"] > 2)
).astype(int)  # Low LPCD + high rate = equity red flag

# ── Progressive vs Regressive classification ─────────────────────
city_tariff = (
    tariff_full.groupby("city_id")
    .apply(lambda x: "Progressive" if x.sort_values("slab_number")["rate_per_kl_inr"].is_monotonic_increasing
           else "Regressive")
    .reset_index()
    .rename(columns={0: "tariff_structure"})
)
tariff_full = tariff_full.merge(city_tariff, on="city_id", how="left")

# ── Save outputs ──────────────────────────────────────────────────
tariff_full.to_csv("output/us14_tariff_equity.csv", index=False)
slab1.to_csv("output/us14_slab1_scatter.csv", index=False)

print(f"\nSaved {len(tariff_full)} rows → output/us14_tariff_equity.csv")
print(f"Saved {len(slab1)} rows     → output/us14_slab1_scatter.csv")

print("\n── Quick Stats ──────────────────────────────────────────────")
print(f"  Cities with zero Slab 1 rate: {(slab1['rate_per_kl_inr']==0).sum()}")
print(f"  Equity red flag cities       : {slab1['equity_flag'].sum()}")
print(f"\n── Slab 1 Rates by City ─────────────────────────────────────")
print(slab1[["city_name","rate_per_kl_inr","actual_lpcd",
             "fixed_monthly_charge_inr","affordability_burden_pct",
             "tariff_structure"]].to_string(index=False))
