"""
US-07: Power BI Infrastructure Health Map - Data Prep
Patched against real columns from diagnose_columns.py
dim_infrastructure: infra_id, ward_id, city_id, pipeline_length_km,
pipeline_material, pipe_age_years, pipeline_condition, num_pumping_stations,
storage_tank_capacity_kl, last_major_repair_year, estimated_leakage_pct,
pipeline_condition_encoded, mat_AC, mat_CI etc, has_scada_monitoring
"""
import pandas as pd
import os
os.makedirs("output", exist_ok=True)

infra  = pd.read_csv("preprocessed/dim_infrastructure.csv")
wards  = pd.read_csv("preprocessed/dim_wards.csv")
cities = pd.read_csv("preprocessed/dim_cities.csv")

print(f"Infrastructure wards: {len(infra):,}")

# ── Full map dataset ──────────────────────────────────────────────
# dim_infrastructure has city_id directly ✓ (confirmed in diagnose output)
# but merge wards first to get ward_name and ward_type
map_df = infra.merge(
    wards[["ward_id","ward_name","ward_type"]],
    on="ward_id", how="left"
).merge(
    cities[["city_id","city_name","state","zone","climate_zone"]],
    on="city_id", how="left"
)

# ── Risk score column for bubble size ────────────────────────────
map_df["risk_score"] = (
    (map_df["pipe_age_years"] / 60) * 0.4 +
    (map_df["estimated_leakage_pct"] / 100) * 0.4 +
    (map_df["pipeline_condition_encoded"] / 2) * 0.2
).round(3)

# ── Condition label for color coding ─────────────────────────────
# Good=green, Fair=yellow, Poor=red
map_df["condition_color"] = map_df["pipeline_condition"].map({
    "Good": "Green",
    "Fair": "Amber",
    "Poor": "Red"
})

# ── Age category ─────────────────────────────────────────────────
map_df["age_category"] = pd.cut(
    map_df["pipe_age_years"],
    bins=[0, 10, 25, 40, 100],
    labels=["New (0-10y)", "Mid (10-25y)", "Aging (25-40y)", "Old (40y+)"]
)

# ── High risk wards: pipe_age > 40, sorted by leakage desc ───────
high_risk = (
    map_df[map_df["pipe_age_years"] > 40]
    .sort_values("estimated_leakage_pct", ascending=False)
    [["ward_id","ward_name","city_name","state","zone",
      "pipe_age_years","pipeline_condition","estimated_leakage_pct",
      "pipeline_material","last_major_repair_year","risk_score"]]
)

# ── City-level summary for overview chart ────────────────────────
city_summary = (
    map_df.groupby(["city_id","city_name","state"])
    .agg(
        avg_pipe_age          =("pipe_age_years",       "mean"),
        avg_leakage_pct       =("estimated_leakage_pct","mean"),
        pct_poor_condition    =("pipeline_condition",
                                lambda x: (x=="Poor").mean()*100),
        pct_old_pipes         =("pipe_age_years",
                                lambda x: (x>40).mean()*100),
        total_wards           =("ward_id",              "count"),
        critical_wards        =("pipe_age_years",
                                lambda x: ((x>40)).sum()),
    )
    .reset_index()
    .round(2)
)

# ── Pipeline material distribution ───────────────────────────────
material_dist = (
    map_df.groupby(["city_name","pipeline_material"])
    .size()
    .reset_index(name="ward_count")
)

# ── Condition distribution ────────────────────────────────────────
condition_dist = (
    map_df.groupby(["city_name","pipeline_condition"])
    .size()
    .reset_index(name="ward_count")
)

# ── Save all outputs ──────────────────────────────────────────────
map_df.to_csv("output/us07_infrastructure_map.csv", index=False)
high_risk.to_csv("output/us07_high_risk_wards.csv", index=False)
city_summary.to_csv("output/us07_city_infra_summary.csv", index=False)
material_dist.to_csv("output/us07_material_distribution.csv", index=False)
condition_dist.to_csv("output/us07_condition_distribution.csv", index=False)

print(f"Saved {len(map_df)} rows     → output/us07_infrastructure_map.csv")
print(f"Saved {len(high_risk)} rows  → output/us07_high_risk_wards.csv")
print(f"Saved {len(city_summary)} rows → output/us07_city_infra_summary.csv")
print(f"Saved {len(material_dist)} rows → output/us07_material_distribution.csv")
print(f"Saved {len(condition_dist)} rows → output/us07_condition_distribution.csv")

print("\n── Quick Stats ──────────────────────────────────────────────")
print(f"  Total wards          : {len(map_df):,}")
print(f"  High risk (age>40)   : {len(high_risk):,}")
print(f"  Poor condition wards : {(map_df['pipeline_condition']=='Poor').sum():,}")
print(f"  Avg pipe age         : {map_df['pipe_age_years'].mean():.1f} years")
print(f"  Avg leakage          : {map_df['estimated_leakage_pct'].mean():.1f}%")
print(f"\n  Pipeline condition breakdown:")
print(map_df["pipeline_condition"].value_counts())
print(f"\n  Pipeline material breakdown:")
print(map_df["pipeline_material"].value_counts())