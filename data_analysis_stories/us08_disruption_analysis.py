"""
US-08: Analyse Disruption Patterns and Resolution Time by Cause
Role: Operations Control Room Manager
PATCHED – columns verified: cause, duration_hours, severity, population_affected,
complaint_count, resolved, resolution_action, num_wards_affected
fact_disruptions has: disruption_id, city_id, ward_id, num_wards_affected,
start_date, duration_hours, cause, severity, estimated_supply_loss_mld,
population_affected, complaint_count, resolved, resolution_action, year, month
"""
import pandas as pd
import numpy as np
import os
os.makedirs("output", exist_ok=True)

disrupt = pd.read_csv("preprocessed/fact_disruptions.csv")
cities  = pd.read_csv("preprocessed/dim_cities.csv")
print(f"Disruption events: {len(disrupt):,}")

# ── 1. Group by cause ─────────────────────────────────────────────────────────
by_cause = (
    disrupt.groupby("cause")
    .agg(
        avg_duration_hours       =("duration_hours",            "mean"),
        total_population_affected=("population_affected",       "sum"),
        sum_complaint_count      =("complaint_count",           "sum"),
        total_events             =("cause",                     "count"),
        resolved_count           =("resolved",                  "sum"),
        avg_supply_loss_mld      =("estimated_supply_loss_mld", "mean"),
    )
    .reset_index()
)
by_cause["resolved_rate_pct"] = (
    by_cause["resolved_count"] / by_cause["total_events"] * 100
).round(2)
by_cause["avg_duration_hours"] = by_cause["avg_duration_hours"].round(2)

print("\n── Disruption Summary by Cause ───────────────────────────────────")
print(by_cause.sort_values("avg_duration_hours", ascending=False).to_string(index=False))

# ── 2. Severity distribution per city per cause ────────────────────────────────
severity_dist = (
    disrupt.groupby(["city_id", "cause", "severity"])
    .size()
    .reset_index(name="event_count")
    .merge(cities[["city_id", "city_name"]], on="city_id", how="left")
)

# ── 3. Most common resolution action per cause ────────────────────────────────
common_res = (
    disrupt.groupby(["cause", "resolution_action"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
    .groupby("cause")
    .first()
    .reset_index()[["cause", "resolution_action", "count"]]
    .rename(columns={"resolution_action": "best_practice_action",
                     "count": "times_used"})
)
print("\n── Best Practice Resolution per Cause ────────────────────────────")
print(common_res.to_string(index=False))

# ── 4. Monthly aggregation ────────────────────────────────────────────────────
monthly = (
    disrupt.groupby(["city_id", "year", "month", "cause"])
    .agg(
        avg_duration_hours       =("duration_hours",      "mean"),
        total_population_affected=("population_affected", "sum"),
        sum_complaint_count      =("complaint_count",     "sum"),
        event_count              =("cause",               "count"),
        resolved_rate            =("resolved",            "mean"),
    )
    .reset_index()
)

# ── 5. Save ───────────────────────────────────────────────────────────────────
by_cause.to_csv("output/us08_disruption_by_cause.csv", index=False)
severity_dist.to_csv("output/us08_severity_distribution.csv", index=False)
common_res.to_csv("output/us08_best_practice_resolutions.csv", index=False)
monthly.to_csv("output/us08_disruption_summary_mysql.csv", index=False)

print("\nSaved → output/us08_disruption_by_cause.csv")
print("Saved → output/us08_severity_distribution.csv")
print("Saved → output/us08_best_practice_resolutions.csv")
print("Saved → output/us08_disruption_summary_mysql.csv")

# ── MySQL (optional) ──────────────────────────────────────────────────────────
try:
    from sqlalchemy import create_engine
    engine = create_engine("mysql+pymysql://root:root@localhost/water_supply_india")
    monthly.to_sql("disruption_summary", engine, if_exists="replace", index=False)
    print("✓ disruption_summary saved to MySQL")
except Exception as e:
    print(f"[MySQL skipped] {e}")
