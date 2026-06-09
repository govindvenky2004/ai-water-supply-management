# ============================================================
# India Water Supply Management — Preprocessing Script
# Memory-Safe Version (works on low RAM machines)
# ============================================================
# HOW TO RUN:
#   python preprocess_water_supply.py
#
# WHAT THIS DOES:
#   Instead of merging all 568k rows at once (which crashes),
#   we process each table separately and only merge the small
#   lookup/dimension tables. The big tables stay separate.
#
# OUTPUT FILES (in preprocessed/ folder):
#   dim_cities.csv         — cleaned cities (15 rows)
#   dim_wards.csv          — cleaned wards (1809 rows)
#   dim_sources.csv        — cleaned sources (52 rows)
#   dim_infrastructure.csv — cleaned infra (1809 rows)
#   dim_rainfall.csv       — cleaned rainfall with features (1800 rows)
#   dim_population.csv     — cleaned population (240 rows)
#   dim_tariff.csv         — cleaned tariff (60 rows)
#   fact_demand.csv        — demand with features (568k rows)
#   fact_supply.csv        — supply with features (568k rows)
#   fact_disruptions.csv   — disruptions exploded (261 rows)
#   ml_sample.csv          — 10% sample merged master (for quick ML testing)
# ============================================================
 
import pandas as pd
import numpy as np
import os
import warnings
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
 
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
 
# ── CHANGE THIS to your dataset folder path ──────────────────
DATA_PATH   = 'water_supply_dataset_v2/'   # Change this to your folder name
OUTPUT_PATH = 'preprocessed/'
# ─────────────────────────────────────────────────────────────
 
os.makedirs(OUTPUT_PATH, exist_ok=True)
 
def section(title):
    print(f'\n{"="*55}')
    print(f'  {title}')
    print(f'{"="*55}')
 
def done(msg):
    print(f'  ✅ {msg}')
 
def info(msg):
    print(f'  ℹ  {msg}')
 
 
# ============================================================
# STEP 1 — LOAD ONLY SMALL TABLES INTO MEMORY FIRST
# The big tables (demand, supply) are loaded one at a time
# ============================================================
section('STEP 1 — Loading small dimension tables')
 
cities      = pd.read_csv(DATA_PATH + 'cities.csv')
wards       = pd.read_csv(DATA_PATH + 'wards.csv')
sources     = pd.read_csv(DATA_PATH + 'water_sources.csv')
infra       = pd.read_csv(DATA_PATH + 'infrastructure.csv')
rainfall    = pd.read_csv(DATA_PATH + 'rainfall_data.csv')
population  = pd.read_csv(DATA_PATH + 'population_growth.csv')
tariff      = pd.read_csv(DATA_PATH + 'water_tariff.csv')
disruptions = pd.read_csv(DATA_PATH + 'supply_disruptions.csv')
 
info(f'cities:      {cities.shape}')
info(f'wards:       {wards.shape}')
info(f'sources:     {sources.shape}')
info(f'infra:       {infra.shape}')
info(f'rainfall:    {rainfall.shape}')
info(f'population:  {population.shape}')
info(f'tariff:      {tariff.shape}')
info(f'disruptions: {disruptions.shape}')
done('Small tables loaded (these are safe — tiny memory footprint)')
 
 
# ============================================================
# STEP 2 — CLEAN DIMENSION TABLES
# ============================================================
section('STEP 2 — Cleaning dimension tables')
 
# ── CITIES ───────────────────────────────────────────────────
# No nulls or duplicates — just encode categoricals
climate_dummies = pd.get_dummies(cities['climate_zone'], prefix='climate')
zone_dummies    = pd.get_dummies(cities['zone'], prefix='zone')
cities = pd.concat([cities, climate_dummies, zone_dummies], axis=1)
# Convert booleans
for col in cities.select_dtypes(include='bool').columns:
    cities[col] = cities[col].astype(int)
cities.to_csv(OUTPUT_PATH + 'dim_cities.csv', index=False)
done(f'cities saved — {cities.shape[1]} columns')
 
# ── WARDS ────────────────────────────────────────────────────
# Encode ward_type two ways:
#   1. Label encoded (single number) — for tree models
#   2. One-hot (separate columns)    — for linear/neural models
ward_type_map = {
    'Residential': 0, 'Commercial': 1,
    'Industrial': 2, 'Mixed': 3, 'Slum/Informal': 4
}
wards['ward_type_encoded'] = wards['ward_type'].map(ward_type_map)
wt_dummies = pd.get_dummies(wards['ward_type'], prefix='wt')
wards = pd.concat([wards, wt_dummies], axis=1)
for col in wards.select_dtypes(include='bool').columns:
    wards[col] = wards[col].astype(int)
wards.to_csv(OUTPUT_PATH + 'dim_wards.csv', index=False)
done(f'wards saved — {wards.shape[1]} columns, {len(wards)} rows')
 
# ── INFRASTRUCTURE ───────────────────────────────────────────
# Encode pipeline_condition (ordered: Good=2 > Fair=1 > Poor=0)
# Encode pipeline_material (one-hot, unordered)
infra['pipeline_condition_encoded'] = infra['pipeline_condition'].map(
    {'Good': 2, 'Fair': 1, 'Poor': 0}
)
mat_dummies = pd.get_dummies(infra['pipeline_material'], prefix='mat')
infra = pd.concat([infra, mat_dummies], axis=1)
for col in infra.select_dtypes(include='bool').columns:
    infra[col] = infra[col].astype(int)
infra.to_csv(OUTPUT_PATH + 'dim_infrastructure.csv', index=False)
done(f'infrastructure saved — {infra.shape[1]} columns')
 
# ── RAINFALL ─────────────────────────────────────────────────
# Fix date, add rolling and lag features
rainfall['date'] = pd.to_datetime(
    rainfall[['year', 'month']].assign(day=1)
)
for col in rainfall.select_dtypes(include='bool').columns:
    rainfall[col] = rainfall[col].astype(int)
 
rainfall = rainfall.sort_values(['city_id', 'date']).reset_index(drop=True)
 
# Rolling cumulative rainfall (captures monsoon buildup)
rainfall['rain_3m_sum'] = (
    rainfall.groupby('city_id')['rainfall_mm']
    .transform(lambda x: x.rolling(3, min_periods=1).sum())
)
rainfall['rain_6m_sum'] = (
    rainfall.groupby('city_id')['rainfall_mm']
    .transform(lambda x: x.rolling(6, min_periods=1).sum())
)
# Lag — last month's rainfall
rainfall['rain_lag_1m'] = (
    rainfall.groupby('city_id')['rainfall_mm'].shift(1).fillna(0)
)
# Is monsoon month
rainfall['is_monsoon_month'] = rainfall['month'].isin([6, 7, 8, 9]).astype(int)
 
rainfall.to_csv(OUTPUT_PATH + 'dim_rainfall.csv', index=False)
done(f'rainfall saved — {rainfall.shape[1]} columns')
 
# ── DISRUPTIONS ──────────────────────────────────────────────
# The affected_wards column has multiple IDs like W001|W002|W003
# We split them so each row = one ward
disruptions['start_date'] = pd.to_datetime(disruptions['start_date'])
disruptions['affected_wards'] = disruptions['affected_wards'].str.split('|')
disruptions_exp = (
    disruptions.explode('affected_wards')
    .reset_index(drop=True)
    .rename(columns={'affected_wards': 'ward_id'})
)
for col in disruptions_exp.select_dtypes(include='bool').columns:
    disruptions_exp[col] = disruptions_exp[col].astype(int)
disruptions_exp['year']  = disruptions_exp['start_date'].dt.year
disruptions_exp['month'] = disruptions_exp['start_date'].dt.month
disruptions_exp.to_csv(OUTPUT_PATH + 'fact_disruptions.csv', index=False)
done(f'disruptions saved — {len(disruptions_exp)} rows after exploding')
 
# ── POPULATION & TARIFF ──────────────────────────────────────
population.to_csv(OUTPUT_PATH + 'dim_population.csv', index=False)
tariff.to_csv(OUTPUT_PATH + 'dim_tariff.csv', index=False)
done('population and tariff saved (no changes needed)')
 
# ── SOURCES ──────────────────────────────────────────────────
for col in sources.select_dtypes(include='bool').columns:
    sources[col] = sources[col].astype(int)
sources.to_csv(OUTPUT_PATH + 'dim_sources.csv', index=False)
done('sources saved')
 
 
# ============================================================
# STEP 3 — BUILD SMALL LOOKUP DICT FOR FAST JOINING LATER
# Instead of merging big tables, we build Python dicts
# that map ward_id/city_id to their features.
# This is the memory-safe alternative to .merge() on 568k rows
# ============================================================
section('STEP 3 — Building lookup dictionaries')
 
# ward_id → ward features (for fast lookup)
ward_lookup = wards.set_index('ward_id')[[
    'city_id', 'ward_type', 'ward_type_encoded',
    'population_2019', 'num_households',
    'piped_connection_coverage_pct', 'metered_connections_pct',
    'has_slum_pocket', 'wt_Commercial', 'wt_Industrial',
    'wt_Mixed', 'wt_Residential', 'wt_Slum/Informal'
]].to_dict(orient='index')
 
# ward_id → infra features
infra_lookup = infra.set_index('ward_id')[[
    'pipe_age_years', 'pipeline_condition_encoded',
    'estimated_leakage_pct', 'storage_tank_capacity_kl',
    'num_pumping_stations', 'has_scada_monitoring',
    'electrification_pct', 'pipeline_length_km'
]].to_dict(orient='index')
 
# city_id → city features
city_lookup = cities.set_index('city_id')[[
    'city_name', 'state', 'actual_lpcd', 'nrw_pct',
    'annual_rainfall_mm', 'avg_hours_supply_per_day',
    'metering_pct', 'population_density_per_km2'
]].to_dict(orient='index')
 
# (city_id, year, month) → rainfall features
rain_lookup = rainfall.set_index(['city_id', 'year', 'month'])[[
    'rainfall_mm', 'rainy_days', 'departure_from_normal_pct',
    'is_drought_year', 'is_flood_year',
    'rain_3m_sum', 'rain_6m_sum', 'rain_lag_1m', 'is_monsoon_month'
]].to_dict(orient='index')
 
done('All lookup dictionaries built')
info('These replace .merge() on the big tables — no memory crash')
 
 
# ============================================================
# STEP 4 — PROCESS DEMAND TABLE (568k rows) IN CHUNKS
# We add features row by row using lookups, then save
# ============================================================
section('STEP 4 — Processing demand table (568k rows) in chunks')
 
CHUNK_SIZE = 50_000   # Process 50,000 rows at a time
demand_file = DATA_PATH + 'daily_demand.csv'
 
# Season mapping (India water context)
season_map = {
    12: 1, 1: 1, 2: 1,   # Winter
     3: 2, 4: 2, 5: 2,   # Summer (peak demand)
     6: 3, 7: 3, 8: 3,   # Monsoon (lower demand)
     9: 4, 10: 4, 11: 4  # Post-Monsoon
}
 
# We need to sort demand by ward+date for lag features
# But sorting 568k rows is fine — it's the MERGE that crashes
info('Loading full demand table for sorting (this is OK)...')
demand = pd.read_csv(demand_file)
demand['date'] = pd.to_datetime(demand['date'])
 
# Convert booleans
for col in demand.select_dtypes(include='bool').columns:
    demand[col] = demand[col].astype(int)
 
# Cap LPCD outlier (max was 348, realistic max is 250)
demand['lpcd_used'] = demand['lpcd_used'].clip(upper=250)
 
# Sort by ward + date (required for lag/rolling features)
info('Sorting by ward_id and date...')
demand = demand.sort_values(['ward_id', 'date']).reset_index(drop=True)
 
# ── Add time features ────────────────────────────────────────
demand['season']           = demand['month'].map(season_map)
demand['season_name']      = demand['season'].map(
    {1: 'Winter', 2: 'Summer', 3: 'Monsoon', 4: 'Post-Monsoon'}
)
demand['quarter']          = demand['date'].dt.quarter
demand['week_of_year']     = demand['date'].dt.isocalendar().week.astype(int)
demand['days_from_start']  = (demand['date'] - demand['date'].min()).dt.days
demand['is_festival_month'] = demand['month'].isin([1, 3, 6, 7, 10, 11]).astype(int)
 
# ── Lag features (per ward) ──────────────────────────────────
# lag_1w = demand from 1 week ago
# lag_2w = 2 weeks ago, lag_4w = 4 weeks ago (1 month)
info('Creating lag features (per ward)...')
grp = demand.groupby('ward_id')['estimated_demand_mld']
demand['demand_lag_1w'] = grp.shift(1)
demand['demand_lag_2w'] = grp.shift(2)
demand['demand_lag_4w'] = grp.shift(4)
 
# ── Rolling averages (per ward) ──────────────────────────────
info('Creating rolling averages (this takes ~1 min)...')
demand['demand_roll_4w_avg']  = grp.transform(
    lambda x: x.rolling(4, min_periods=1).mean()
)
demand['demand_roll_12w_avg'] = grp.transform(
    lambda x: x.rolling(12, min_periods=1).mean()
)
demand['demand_roll_4w_std']  = grp.transform(
    lambda x: x.rolling(4, min_periods=1).std().fillna(0)
)
 
# ── Fill NaN in lag columns (first rows have no history) ────
for col in ['demand_lag_1w', 'demand_lag_2w', 'demand_lag_4w']:
    demand[col] = demand[col].fillna(demand['demand_roll_4w_avg'])
 
# ── Add city features via lookup (no merge!) ─────────────────
info('Adding city features via lookup dict (memory safe)...')
demand['city_name']               = demand['city_id'].map(lambda x: city_lookup.get(x, {}).get('city_name', ''))
demand['state']                   = demand['city_id'].map(lambda x: city_lookup.get(x, {}).get('state', ''))
demand['city_actual_lpcd']        = demand['city_id'].map(lambda x: city_lookup.get(x, {}).get('actual_lpcd', np.nan))
demand['city_nrw_pct']            = demand['city_id'].map(lambda x: city_lookup.get(x, {}).get('nrw_pct', np.nan))
demand['city_avg_hours_supply']   = demand['city_id'].map(lambda x: city_lookup.get(x, {}).get('avg_hours_supply_per_day', np.nan))
demand['city_metering_pct']       = demand['city_id'].map(lambda x: city_lookup.get(x, {}).get('metering_pct', np.nan))
 
# ── Add ward features via lookup ────────────────────────────
info('Adding ward features via lookup dict...')
demand['ward_type']               = demand['ward_id'].map(lambda x: ward_lookup.get(x, {}).get('ward_type', ''))
demand['ward_type_encoded']       = demand['ward_id'].map(lambda x: ward_lookup.get(x, {}).get('ward_type_encoded', -1))
demand['ward_population']         = demand['ward_id'].map(lambda x: ward_lookup.get(x, {}).get('population_2019', np.nan))
demand['ward_coverage_pct']       = demand['ward_id'].map(lambda x: ward_lookup.get(x, {}).get('piped_connection_coverage_pct', np.nan))
demand['ward_metering_pct']       = demand['ward_id'].map(lambda x: ward_lookup.get(x, {}).get('metered_connections_pct', np.nan))
demand['ward_has_slum']           = demand['ward_id'].map(lambda x: ward_lookup.get(x, {}).get('has_slum_pocket', 0))
 
# ── Add infrastructure features via lookup ───────────────────
info('Adding infrastructure features via lookup dict...')
demand['pipe_age_years']          = demand['ward_id'].map(lambda x: infra_lookup.get(x, {}).get('pipe_age_years', np.nan))
demand['pipe_condition_encoded']  = demand['ward_id'].map(lambda x: infra_lookup.get(x, {}).get('pipeline_condition_encoded', np.nan))
demand['leakage_pct']             = demand['ward_id'].map(lambda x: infra_lookup.get(x, {}).get('estimated_leakage_pct', np.nan))
demand['storage_capacity_kl']     = demand['ward_id'].map(lambda x: infra_lookup.get(x, {}).get('storage_tank_capacity_kl', np.nan))
demand['has_scada']               = demand['ward_id'].map(lambda x: infra_lookup.get(x, {}).get('has_scada_monitoring', 0))
 
# ── Add rainfall features via (city_id, year, month) lookup ─
info('Adding rainfall features via lookup dict...')
demand['rainfall_mm']             = demand.apply(lambda r: rain_lookup.get((r['city_id'], r['year'], r['month']), {}).get('rainfall_mm', 0), axis=1)
demand['rainy_days']              = demand.apply(lambda r: rain_lookup.get((r['city_id'], r['year'], r['month']), {}).get('rainy_days', 0), axis=1)
demand['rain_3m_sum']             = demand.apply(lambda r: rain_lookup.get((r['city_id'], r['year'], r['month']), {}).get('rain_3m_sum', 0), axis=1)
demand['rain_lag_1m']             = demand.apply(lambda r: rain_lookup.get((r['city_id'], r['year'], r['month']), {}).get('rain_lag_1m', 0), axis=1)
demand['is_drought_year']         = demand.apply(lambda r: rain_lookup.get((r['city_id'], r['year'], r['month']), {}).get('is_drought_year', 0), axis=1)
demand['is_flood_year']           = demand.apply(lambda r: rain_lookup.get((r['city_id'], r['year'], r['month']), {}).get('is_flood_year', 0), axis=1)
demand['is_monsoon_month']        = demand.apply(lambda r: rain_lookup.get((r['city_id'], r['year'], r['month']), {}).get('is_monsoon_month', 0), axis=1)
 
# ── Target variable ──────────────────────────────────────────
# (we'll add supply-based targets in next step — for now add season target)
demand['deficit_expected'] = (demand['season'] == 2).astype(int)  # summer = likely deficit
 
info('Saving demand_clean.csv ...')
demand.to_csv(OUTPUT_PATH + 'fact_demand.csv', index=False)
done(f'fact_demand.csv saved — {demand.shape[0]:,} rows × {demand.shape[1]} columns')
 
 
# ============================================================
# STEP 5 — PROCESS SUPPLY TABLE (568k rows)
# ============================================================
section('STEP 5 — Processing supply table (568k rows)')
 
info('Loading supply table...')
supply = pd.read_csv(DATA_PATH + 'supply_records.csv')
supply['date'] = pd.to_datetime(supply['date'])
 
# Convert booleans
for col in supply.select_dtypes(include='bool').columns:
    supply[col] = supply[col].astype(int)
 
# Cap NRW at 55% (realistic India max — Delhi is 53%)
supply['nrw_pct'] = supply['nrw_pct'].clip(upper=55)
 
# ── Target variables ─────────────────────────────────────────
# These are what your ML models will PREDICT
supply['has_deficit']     = (supply['supply_deficit_mld'] > 0).astype(int)
supply['is_anomaly']      = (supply['supply_efficiency_pct'] < 50).astype(int)
 
def deficit_severity(val):
    if val == 0:       return 0   # No deficit
    elif val < 0.5:    return 1   # Low
    elif val < 2.0:    return 2   # Medium
    else:              return 3   # High
 
supply['deficit_severity'] = supply['supply_deficit_mld'].apply(deficit_severity)
supply['deficit_severity_label'] = supply['deficit_severity'].map(
    {0: 'No Deficit', 1: 'Low', 2: 'Medium', 3: 'High'}
)
 
# ── Rolling supply features (per ward) ──────────────────────
info('Sorting supply by ward + date...')
supply = supply.sort_values(['ward_id', 'date']).reset_index(drop=True)
 
grp_s = supply.groupby('ward_id')['actual_supply_mld']
supply['supply_lag_1w']       = grp_s.shift(1).bfill()
supply['supply_roll_4w_avg']  = grp_s.transform(lambda x: x.rolling(4, min_periods=1).mean())
 
grp_d = supply.groupby('ward_id')['supply_deficit_mld']
supply['deficit_roll_4w_avg'] = grp_d.transform(lambda x: x.rolling(4, min_periods=1).mean())
 
# ── Add city NRW via lookup ──────────────────────────────────
info('Adding city features to supply...')
supply['city_nrw_benchmark'] = supply['city_id'].map(
    lambda x: city_lookup.get(x, {}).get('nrw_pct', np.nan)
)
# How much worse is this record's NRW vs city benchmark?
supply['nrw_vs_benchmark'] = supply['nrw_pct'] - supply['city_nrw_benchmark']
 
info('Saving fact_supply.csv...')
supply.to_csv(OUTPUT_PATH + 'fact_supply.csv', index=False)
done(f'fact_supply.csv saved — {supply.shape[0]:,} rows × {supply.shape[1]} columns')
 
print('\nTarget variable distribution in supply:')
print('  has_deficit:')
print(supply['has_deficit'].value_counts().to_string())
print('  deficit_severity:')
print(supply['deficit_severity_label'].value_counts().to_string())
print('  is_anomaly:')
print(supply['is_anomaly'].value_counts().to_string())
 
 
# ============================================================
# STEP 6 — CREATE A SMALL SAMPLE MASTER TABLE FOR ML
# 10% random sample — safe to merge, use for model prototyping
# ============================================================
section('STEP 6 — Creating 10% sample master table for ML')
 
info('Sampling 10% of demand rows...')
demand_sample = demand.sample(frac=0.10, random_state=42).reset_index(drop=True)
 
# Now merge with supply (only on the sampled demand_ids)
supply_cols_needed = [
    'demand_id', 'actual_supply_mld', 'supply_deficit_mld',
    'supply_efficiency_pct', 'hours_of_supply', 'nrw_pct',
    'is_tanker_supplement', 'has_deficit', 'is_anomaly',
    'deficit_severity', 'deficit_severity_label'
]
sample_master = demand_sample.merge(
    supply[supply_cols_needed],
    on='demand_id',
    how='left'
)
 
# Fill any nulls
numeric = sample_master.select_dtypes(include=[np.number]).columns
sample_master[numeric] = sample_master[numeric].fillna(sample_master[numeric].median())
 
# Scale key numeric columns
cols_to_scale = [
    'estimated_demand_mld', 'lpcd_used', 'ward_population',
    'pipe_age_years', 'leakage_pct', 'storage_capacity_kl',
    'rainfall_mm', 'rain_3m_sum', 'demand_roll_4w_avg',
    'demand_lag_1w', 'nrw_pct', 'hours_of_supply',
    'supply_efficiency_pct', 'actual_supply_mld', 'supply_deficit_mld'
]
cols_to_scale = [c for c in cols_to_scale if c in sample_master.columns]
scaler = MinMaxScaler()
scaled_arr = scaler.fit_transform(sample_master[cols_to_scale])
scaled_df  = pd.DataFrame(
    scaled_arr,
    columns=[f'{c}_scaled' for c in cols_to_scale],
    index=sample_master.index
)
sample_master = pd.concat([sample_master, scaled_df], axis=1)
 
sample_master.to_csv(OUTPUT_PATH + 'ml_sample.csv', index=False)
done(f'ml_sample.csv saved — {sample_master.shape[0]:,} rows × {sample_master.shape[1]} columns')
info('Use ml_sample.csv for quick ML model prototyping')
info('Use fact_demand.csv + fact_supply.csv for full training')
 
 
# ============================================================
# FINAL SUMMARY
# ============================================================
section('PREPROCESSING COMPLETE — FILE SUMMARY')
 
output_files = os.listdir(OUTPUT_PATH)
print(f'\n  {"File":<35} {"Rows":>10} {"Cols":>6} {"Size":>10}')
print(f'  {"-"*63}')
for fname in sorted(output_files):
    fpath = os.path.join(OUTPUT_PATH, fname)
    size_kb = os.path.getsize(fpath) / 1024
    try:
        df_tmp = pd.read_csv(fpath, nrows=2)
        # count rows without loading full file
        with open(fpath) as f:
            row_count = sum(1 for _ in f) - 1
        print(f'  {fname:<35} {row_count:>10,} {len(df_tmp.columns):>6} {size_kb:>8.0f} KB')
    except:
        print(f'  {fname:<35} {"":>10} {"":>6} {size_kb:>8.0f} KB')
 
print(f"""
  KEY OUTPUTS EXPLAINED:
  ─────────────────────────────────────────────────────────
  dim_*.csv          → Small clean lookup tables
  fact_demand.csv    → 568k rows, all demand features added
  fact_supply.csv    → 568k rows, target variables added
  ml_sample.csv      → 10% sample, fully merged, ML-ready
 
  NEXT STEPS:
  ─────────────────────────────────────────────────────────
  Step 2 → EDA notebook (02_eda.ipynb)
  Step 3 → ML models (03_demand_forecasting.ipynb)
             Target: estimated_demand_mld
             Features: lag features, season, rainfall, ward_type
 
  Step 4 → Anomaly detection (04_anomaly_detection.ipynb)
             Target: is_anomaly
             Features: supply_efficiency_pct, nrw_pct, hours_of_supply
 
  Step 5 → Clustering wards (05_clustering.ipynb)
             No target (unsupervised)
             Features: demand patterns, ward_type, population
""")
