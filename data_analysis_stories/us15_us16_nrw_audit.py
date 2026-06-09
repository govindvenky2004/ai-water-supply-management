"""
US-15: Track NRW Reduction Progress and Flag Chronic Offender Wards
US-16: Store All Disruption Audit Logs in MongoDB for CAG Compliance
PATCHED v2 – city_nrw_benchmark is in fact_supply NOT dim_cities
"""
import pandas as pd
import numpy as np
import os
os.makedirs("output", exist_ok=True)

supply  = pd.read_csv("preprocessed/fact_supply.csv")
supply["date"] = pd.to_datetime(supply["date"])
supply["year"]  = supply["date"].dt.year
supply["month"] = supply["date"].dt.month
cities  = pd.read_csv("preprocessed/dim_cities.csv")
infra   = pd.read_csv("preprocessed/dim_infrastructure.csv")

print(f"fact_supply rows: {len(supply):,}")

# ── US-15 Part 1: City NRW trend ──────────────────────────────────────────────
nrw_trend = (
    supply.groupby(["city_id", "year"])
    .agg(nrw_pct        =("nrw_pct",         "mean"),
         nrw_vs_benchmark=("nrw_vs_benchmark","mean"))
    .reset_index()
    .merge(cities[["city_id","city_name"]], on="city_id", how="left")
)

# city_nrw_benchmark is in fact_supply ✓
if "city_nrw_benchmark" in supply.columns:
    bench = supply.groupby("city_id")["city_nrw_benchmark"].first().reset_index()
    nrw_trend = nrw_trend.merge(bench, on="city_id", how="left")

nrw_trend["nrw_pct"] = nrw_trend["nrw_pct"].round(2)

print("\n── City NRW Trend (mean per year) ────────────────────────────────")
pivot = nrw_trend.pivot(index="city_name", columns="year", values="nrw_pct").round(1)
print(pivot.to_string())

# ── US-15 Part 2: Chronic offender wards ─────────────────────────────────────
ward_monthly = (
    supply.groupby(["ward_id","city_id","year","month"])
    .agg(nrw_pct        =("nrw_pct",         "mean"),
         nrw_vs_benchmark=("nrw_vs_benchmark","mean"))
    .reset_index()
    .sort_values(["ward_id","year","month"])
)

ward_monthly["over_threshold"] = (ward_monthly["nrw_vs_benchmark"] > 10).astype(int)

def consec(s):
    result, count = [], 0
    for v in s:
        count = count + 1 if v == 1 else 0
        result.append(count)
    return result

ward_monthly["consec"] = (
    ward_monthly.groupby("ward_id")["over_threshold"]
    .transform(lambda x: consec(x.tolist()))
)

chronic_wards = ward_monthly[ward_monthly["consec"] >= 3]["ward_id"].unique()
print(f"\n  Chronic offender wards (3+ consecutive months > 10pp): {len(chronic_wards)}")

# ── US-15 Part 3: Correlation ─────────────────────────────────────────────────
supply_infra = supply.merge(
    infra[["ward_id","estimated_leakage_pct","pipe_age_years"]],
    on="ward_id", how="left"
)
corr_cols = ["nrw_pct","nrw_vs_benchmark","estimated_leakage_pct","pipe_age_years"]
avail     = [c for c in corr_cols if c in supply_infra.columns]
print(f"\n── Correlation Matrix ───────────────────────────────────────────")
print(supply_infra[avail].corr().round(3))

# ── Save US-15 ────────────────────────────────────────────────────────────────
nrw_trend.to_csv("output/us15_nrw_trend.csv", index=False)
ward_monthly.to_csv("output/us15_ward_nrw_monthly.csv", index=False)
pd.DataFrame({"ward_id": chronic_wards}).to_csv(
    "output/us15_chronic_offender_wards.csv", index=False)
print(f"\nSaved → output/us15_nrw_trend.csv")
print(f"Saved → output/us15_ward_nrw_monthly.csv")
print(f"Saved → output/us15_chronic_offender_wards.csv")

try:
    from sqlalchemy import create_engine
    engine = create_engine("mysql+pymysql://root:root@localhost/water_supply_india")
    ward_monthly.to_sql("nrw_analysis", engine, if_exists="replace", index=False)
    print("✓ nrw_analysis saved to MySQL")
except Exception as e:
    print(f"[MySQL skipped] {e}")

# ════════════════════════════════════════════════════════════════
# US-16: MongoDB Disruption Audit Log
# ════════════════════════════════════════════════════════════════
print("\n── US-16: MongoDB Disruption Audit Log ──────────────────────────")
from datetime import datetime

disrupt = pd.read_csv("preprocessed/fact_disruptions.csv")
disrupt["start_date"] = pd.to_datetime(disrupt["start_date"]).astype(str)

REQUIRED = ["cause","severity","resolved","resolution_action",
            "complaint_count","population_affected",
            "estimated_supply_loss_mld","start_date","city_id","year"]
avail_cols = [c for c in REQUIRED if c in disrupt.columns]
records    = disrupt[avail_cols].to_dict("records")

for rec in records:
    rec["version"]    = 1
    rec["immutable"]  = True
    rec["ingested_at"]= datetime.utcnow().isoformat()

try:
    from pymongo import MongoClient, ASCENDING
    client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=3000)
    client.server_info()
    col = client["water_supply_india"]["disruption_audit"]
    col.drop()
    col.insert_many(records)
    col.create_index([("start_date", ASCENDING), ("city_id", ASCENDING)])
    unresolved = col.count_documents({"resolved": 0})
    print(f"  Inserted {len(records)} docs → disruption_audit")
    print(f"  Unresolved (RTI query): {unresolved}")
    client.close()
except Exception as e:
    print(f"  [MongoDB skipped] {e}")

disrupt[avail_cols].to_csv("output/us16_disruption_audit.csv", index=False)
print(f"Saved → output/us16_disruption_audit.csv")
