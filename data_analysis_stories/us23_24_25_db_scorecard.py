"""
US-23: MySQL Star Schema Design and Population
US-24: MongoDB Raw Archive with ML Outputs
US-25: Executive Scorecard — Water Health per City
PATCHED – all column names verified against diagnose_columns.py
"""
import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings("ignore")
os.makedirs("output", exist_ok=True)

# ════════════════════════════════════════════════════════════════
# US-23: MySQL Star Schema
# ════════════════════════════════════════════════════════════════
print("── US-23: MySQL Star Schema ─────────────────────────────────────")

STAR_SCHEMA_DDL = """
-- ── DIMENSION TABLES ────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS dim_cities (
    city_id VARCHAR(10) PRIMARY KEY,
    city_name VARCHAR(100), state VARCHAR(100), zone VARCHAR(50),
    climate_zone VARCHAR(50), actual_lpcd INT, nrw_pct INT,
    daily_supply_mld INT, daily_demand_mld INT, supply_deficit_mld INT,
    avg_hours_supply_per_day INT, metering_pct INT,
    population_density_per_km2 FLOAT, water_board VARCHAR(200),
    annual_rainfall_mm INT, city_nrw_benchmark INT
);

CREATE TABLE IF NOT EXISTS dim_wards (
    ward_id VARCHAR(10) PRIMARY KEY,
    city_id VARCHAR(10), ward_name VARCHAR(100), ward_type VARCHAR(50),
    population_2019 INT, piped_connection_coverage_pct FLOAT,
    metered_connections_pct FLOAT, has_slum_pocket TINYINT,
    ward_type_encoded INT,
    FOREIGN KEY (city_id) REFERENCES dim_cities(city_id)
);

CREATE TABLE IF NOT EXISTS dim_infrastructure (
    infra_id VARCHAR(20) PRIMARY KEY,
    ward_id VARCHAR(10), city_id VARCHAR(10),
    pipeline_material VARCHAR(50), pipe_age_years INT,
    pipeline_condition VARCHAR(20), estimated_leakage_pct FLOAT,
    last_major_repair_year INT, num_pumping_stations INT,
    storage_tank_capacity_kl INT, pipeline_condition_encoded INT,
    mat_AC INT, mat_CI INT, mat_DI INT, mat_GI INT,
    mat_HDPE INT, mat_MS INT, mat_PVC INT,
    has_scada_monitoring INT,
    FOREIGN KEY (ward_id) REFERENCES dim_wards(ward_id)
);

CREATE TABLE IF NOT EXISTS dim_sources (
    source_id VARCHAR(10) PRIMARY KEY,
    city_id VARCHAR(10), source_name VARCHAR(100),
    source_type VARCHAR(100), installed_capacity_mld FLOAT,
    current_capacity_mld FLOAT, distance_from_city_km FLOAT,
    year_commissioned INT, last_upgraded_year INT,
    is_active TINYINT, treatment_plant_present TINYINT,
    FOREIGN KEY (city_id) REFERENCES dim_cities(city_id)
);

CREATE TABLE IF NOT EXISTS dim_rainfall (
    rain_id VARCHAR(10) PRIMARY KEY,
    city_id VARCHAR(10), year INT, month INT,
    rainfall_mm FLOAT, rainy_days INT,
    departure_from_normal_pct FLOAT,
    is_drought_year TINYINT, is_flood_year TINYINT,
    rain_3m_sum FLOAT, rain_6m_sum FLOAT, rain_lag_1m FLOAT,
    is_monsoon_month TINYINT,
    FOREIGN KEY (city_id) REFERENCES dim_cities(city_id)
);

CREATE TABLE IF NOT EXISTS dim_population (
    city_id VARCHAR(10), year INT,
    population BIGINT, annual_growth_rate_pct FLOAT,
    est_daily_water_demand_mld FLOAT, projected TINYINT,
    PRIMARY KEY (city_id, year),
    FOREIGN KEY (city_id) REFERENCES dim_cities(city_id)
);

CREATE TABLE IF NOT EXISTS dim_tariff (
    tariff_id VARCHAR(10) PRIMARY KEY,
    city_id VARCHAR(10), slab_number INT,
    consumption_lower_kl INT, consumption_upper_kl INT,
    rate_per_kl_inr FLOAT, fixed_monthly_charge_inr INT,
    FOREIGN KEY (city_id) REFERENCES dim_cities(city_id)
);

-- ── FACT TABLES ──────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS fact_demand (
    demand_id VARCHAR(20) PRIMARY KEY,
    ward_id VARCHAR(10), city_id VARCHAR(10),
    date DATE, year INT, month INT,
    estimated_demand_mld FLOAT, lpcd_used FLOAT,
    seasonal_factor FLOAT, covid_adjustment TINYINT,
    demand_lag_1w FLOAT, demand_lag_2w FLOAT,
    demand_roll_12w_avg FLOAT, rainfall_mm FLOAT,
    rain_3m_sum FLOAT, is_drought_year TINYINT,
    ward_type VARCHAR(50), ward_type_encoded INT,
    FOREIGN KEY (city_id) REFERENCES dim_cities(city_id),
    FOREIGN KEY (ward_id) REFERENCES dim_wards(ward_id)
);

CREATE TABLE IF NOT EXISTS fact_supply (
    supply_id VARCHAR(20) PRIMARY KEY,
    demand_id VARCHAR(20), ward_id VARCHAR(10), city_id VARCHAR(10),
    date DATE, actual_supply_mld FLOAT, demand_mld FLOAT,
    supply_deficit_mld FLOAT, supply_efficiency_pct FLOAT,
    hours_of_supply FLOAT, nrw_pct FLOAT,
    is_tanker_supplement TINYINT, has_deficit TINYINT,
    is_anomaly TINYINT, deficit_severity TINYINT,
    deficit_severity_label VARCHAR(20),
    supply_lag_1w FLOAT, supply_roll_4w_avg FLOAT,
    deficit_roll_4w_avg FLOAT, city_nrw_benchmark INT,
    nrw_vs_benchmark FLOAT,
    FOREIGN KEY (city_id) REFERENCES dim_cities(city_id),
    FOREIGN KEY (ward_id) REFERENCES dim_wards(ward_id)
);

CREATE TABLE IF NOT EXISTS fact_disruptions (
    disruption_id VARCHAR(20) PRIMARY KEY,
    city_id VARCHAR(10), ward_id VARCHAR(10),
    num_wards_affected INT, start_date DATE,
    duration_hours INT, cause VARCHAR(100),
    severity VARCHAR(20), estimated_supply_loss_mld FLOAT,
    population_affected INT, complaint_count INT,
    resolved TINYINT, resolution_action VARCHAR(200),
    year INT, month INT,
    FOREIGN KEY (city_id) REFERENCES dim_cities(city_id)
);

-- ── INDEXES for Power BI DirectQuery performance ──────────────────

CREATE INDEX IF NOT EXISTS idx_supply_city_date  ON fact_supply(city_id, date);
CREATE INDEX IF NOT EXISTS idx_supply_ward_date  ON fact_supply(ward_id, date);
CREATE INDEX IF NOT EXISTS idx_demand_city_date  ON fact_demand(city_id, date);
CREATE INDEX IF NOT EXISTS idx_disrupt_city      ON fact_disruptions(city_id, start_date);
"""

print(STAR_SCHEMA_DDL)
with open("output/us23_star_schema_ddl.sql", "w", encoding="utf-8") as f:
    f.write(STAR_SCHEMA_DDL.encode('utf-8', errors='replace').decode('utf-8'))
print("Saved → output/us23_star_schema_ddl.sql")

try:
    from sqlalchemy import create_engine, text
    engine = create_engine("mysql+pymysql://root:root@localhost/water_supply_india")

    dim_tables = {
        "dim_cities":         "preprocessed/dim_cities.csv",
        "dim_wards":          "preprocessed/dim_wards.csv",
        "dim_infrastructure": "preprocessed/dim_infrastructure.csv",
        "dim_sources":        "preprocessed/dim_sources.csv",
        "dim_rainfall":       "preprocessed/dim_rainfall.csv",
        "dim_population":     "preprocessed/dim_population.csv",
        "dim_tariff":         "preprocessed/dim_tariff.csv",
    }
    for table, path in dim_tables.items():
        df = pd.read_csv(path)
        df.to_sql(table, engine, if_exists="replace", index=False)
        print(f"  ✓ {table}: {len(df):,} rows")

    for table, path in {
        "fact_demand":      "preprocessed/fact_demand.csv",
        "fact_supply":      "preprocessed/fact_supply.csv",
        "fact_disruptions": "preprocessed/fact_disruptions.csv",
    }.items():
        df = pd.read_csv(path)
        df.to_sql(table, engine, if_exists="replace", index=False, chunksize=5000)
        print(f"  ✓ {table}: {len(df):,} rows")

    print("✓ Star schema fully loaded into MySQL")
except Exception as e:
    print(f"[MySQL skipped – not connected]: {e}")
    print("  → Run output/us23_star_schema_ddl.sql in MySQL Workbench manually")
    print("  → Then connect Power BI via Get Data > MySQL Database > localhost")

# ════════════════════════════════════════════════════════════════
# US-24: MongoDB Raw Archive
# ════════════════════════════════════════════════════════════════
print("\n── US-24: MongoDB Raw Archive ───────────────────────────────────")

try:
    from pymongo import MongoClient, ASCENDING
    client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=3000)
    client.server_info()
    db = client["water_supply_india"]

    # raw_demand: denormalized with ward_type and city_name
    demand = pd.read_csv("preprocessed/fact_demand.csv")
    col    = db["raw_demand"]
    col.drop()
    col.insert_many(demand.to_dict("records"))
    col.create_index([("city_id", ASCENDING), ("date", ASCENDING)])
    print(f"  raw_demand: {len(demand):,} docs inserted")

    # raw_supply: with ML output fields embedded
    supply = pd.read_csv("preprocessed/fact_supply.csv")
    # Embed ML severity labels if available
    try:
        sev = pd.read_csv("output/us09_deficit_severity.csv")[
            ["ward_id","severity_label","severity_probability"]
        ]
        supply = supply.merge(sev, on="ward_id", how="left")
    except:
        pass

    col2 = db["raw_supply"]
    col2.drop()
    col2.insert_many(supply.to_dict("records"))
    col2.create_index([("city_id", ASCENDING), ("ward_id", ASCENDING)])
    print(f"  raw_supply: {len(supply):,} docs inserted (with ML fields)")
    client.close()
    print("✓ MongoDB archive complete")
except Exception as e:
    print(f"[MongoDB skipped] {e}")

# ════════════════════════════════════════════════════════════════
# US-25: Executive Scorecard — City Water Health
# ════════════════════════════════════════════════════════════════
print("\n── US-25: Executive Scorecard ────────────────────────────────────")

supply  = pd.read_csv("preprocessed/fact_supply.csv")
disrupt = pd.read_csv("preprocessed/fact_disruptions.csv")
cities  = pd.read_csv("preprocessed/dim_cities.csv")

# City-level KPI aggregation from fact_supply
city_kpi = (
    supply.groupby("city_id")
    .agg(
        supply_efficiency_pct    =("supply_efficiency_pct",  "mean"),
        avg_hours_supply_per_day =("hours_of_supply",        "mean"),
        nrw_pct                  =("nrw_pct",                "mean"),
        supply_deficit_mld       =("supply_deficit_mld",     "mean"),
        actual_supply_mld        =("actual_supply_mld",      "mean"),
        estimated_demand_mld     =("demand_mld",             "mean"),
    )
    .reset_index()
    .merge(cities[["city_id","city_name","state",
                   "actual_lpcd","metering_pct"]], on="city_id", how="left")
)

# Disruption summary per city
disrupt_agg = (
    disrupt.groupby("city_id")
    .agg(
        total_events    =("cause",         "count"),
        avg_duration_hrs=("duration_hours","mean"),
        unresolved_count=("resolved", lambda x: (x == 0).sum()),
    )
    .reset_index()
)
top_cause = (
    disrupt.groupby(["city_id","cause"]).size()
    .reset_index(name="n")
    .sort_values("n", ascending=False)
    .groupby("city_id").first()
    .reset_index()[["city_id","cause"]]
    .rename(columns={"cause":"top_cause"})
)
disrupt_agg = disrupt_agg.merge(top_cause, on="city_id", how="left")

scorecard = city_kpi.merge(disrupt_agg, on="city_id", how="left")

# ── Composite Water Health Score ──────────────────────────────────────────────
# LPCD 30% | Deficit 30% | NRW 20% | Hours 20%
scorecard["lpcd_score"]    = (scorecard["actual_lpcd"] / 135).clip(0, 1) * 100
scorecard["deficit_score"] = (
    1 - (scorecard["supply_deficit_mld"] /
         (scorecard["estimated_demand_mld"] + 1e-6)).clip(0, 1)
) * 100
scorecard["nrw_score"]     = (1 - (scorecard["nrw_pct"] / 60).clip(0, 1)) * 100
scorecard["hours_score"]   = (scorecard["avg_hours_supply_per_day"] / 24).clip(0, 1) * 100

scorecard["water_health_score"] = (
    scorecard["lpcd_score"]    * 0.30 +
    scorecard["deficit_score"] * 0.30 +
    scorecard["nrw_score"]     * 0.20 +
    scorecard["hours_score"]   * 0.20
).round(2)

scorecard = scorecard.sort_values("water_health_score", ascending=False).reset_index(drop=True)
scorecard["rank"]       = scorecard.index + 1
scorecard["rag_status"] = scorecard["water_health_score"].apply(
    lambda s: "GREEN" if s >= 70 else ("AMBER" if s >= 50 else "RED")
)

print("\n── City Water Health Scorecard ──────────────────────────────────")
print(scorecard[["rank","city_name","state","water_health_score",
                 "rag_status","actual_lpcd","nrw_pct",
                 "avg_hours_supply_per_day"]].to_string(index=False))

scorecard.to_csv("output/us25_city_health_scorecard.csv", index=False)
print(f"\nSaved → output/us25_city_health_scorecard.csv")

try:
    from sqlalchemy import create_engine
    engine = create_engine("mysql+pymysql://root:root@localhost/water_supply_india")
    scorecard.to_sql("city_health_scorecard", engine, if_exists="replace", index=False)
    print("✓ city_health_scorecard saved to MySQL")
except Exception as e:
    print(f"[MySQL skipped] {e}")

print("""
── US-25 Power BI Scorecard Setup ────────────────────────────────
Load: us25_city_health_scorecard.csv
1. KPI Cards (5): actual_lpcd | nrw_pct | supply_deficit_mld |
   avg_hours_supply_per_day | metering_pct
   → Conditional format: GREEN/AMBER/RED via rag_status column
2. Rank Table: rank | city_name | water_health_score | rag_status
   Sort: water_health_score DESC
3. Trend Lines (from fact_supply): estimated_demand_mld vs actual_supply_mld
   X: date | Legend: city_name
4. Disruption Box: total_events | avg_duration_hrs | top_cause | unresolved
5. DAX measure:
   Water Health Score =
     [LPCD Score]*0.30 + [Deficit Score]*0.30 +
     [NRW Score]*0.20  + [Hours Score]*0.20
────────────────────────────────────────────────────────────────
""")
