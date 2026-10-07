# India AI-Enhanced Water Supply Management System

A data-engineering and machine-learning pipeline for urban water supply across 15 Indian cities and 1,809 wards, built by a five-member team during Infosys training. **All data are synthetic** (generated and calibrated to public benchmarks), about 568,000 demand records. Nothing here was deployed or used by a water authority.

## Project Scale
- 15 Indian cities, 1,809 wards
- ~568K records (2019–2024)
- 25 user stories across 7 stakeholder roles
- 10 database tables (Star Schema)

## Tech Stack
- Python + PySpark (ML pipeline)
- MySQL (Star Schema – fact + dim tables)
- MongoDB (alerts, audit logs, ML outputs)
- Power BI (6 dashboards)
- LightGBM, scikit-learn, SHAP
- Gemini API (AI-generated field alerts)

## ML Models Built

| Story | What it does | Method |
|---|---|---|
| US-01 | 12-week city demand forecast | LightGBM, time split (train 2019-2023, test 2024), SHAP |
| US-04 | 2030 demand by city | Compound population growth x per-capita use (projection, not trained) |
| US-05 | Pipeline burst risk by ward | LightGBM classifier, SMOTE on train split only |
| US-06 | Supply anomaly flags | LightGBM classifier |
| US-09 | Deficit severity classes | LightGBM, split before SMOTE |
| US-10 | Ward access clusters | K-Means, K chosen by silhouette (3 to 6) |
| US-11 | LPCD forecast by ward type | One LightGBM model per ward type |
| US-12 | Tanker need next week | LightGBM plus business rules (hybrid) |
| US-19 | Complaint volume per disruption | Log-target LightGBM regression |
| US-21 | Drought risk by climate zone | LightGBM classifier per climate zone |

## Results (as printed by the scripts)
- US-01: MAPE 5.63% on the 2024 holdout, against 5.84% for a "same as last week" baseline
- US-11: MAPE 15.7% to 19.1% by ward type
- [Add the printed results for US-05, US-09, US-10, US-12, US-19 and US-21 after running them]

## Power BI Dashboards

Six dashboards built on exported star-schema tables:

- Supply vs Demand gap (15 cities)
- Infrastructure health map (1,809 wards)
- Tariff equity analysis
- Slum ward water access equity
- Monsoon vs supply efficiency
- Executive city health scorecard

## Analysis & Storage Stories
- **Analysis:** US-08 (disruption patterns), US-13 (COVID impact), US-15 (NRW trend), US-17 (slum equity), US-18 (complaint hotspots), US-20 (monsoon vs supply), US-22 (flood-year impact)
- **Storage:** US-23 (MySQL schema), US-24 (MongoDB archive), US-16 (MongoDB audit log), US-25 (city health scorecard)

## Gemini Summaries (Optional)

`ml_stories/gemini_summary.py` reads output CSVs for a city and asks Gemini to write a briefing. Gemini does not make predictions.

```bash
# Set your API key
export GEMINI_API_KEY=your_key_here

# Run for a city
python ml_stories/gemini_summary.py --city <city_id>

# Preview the prompt without calling API
python ml_stories/gemini_summary.py --city <city_id> --dry-run
```

## Run

```bash
python ml_stories/us01_demand_forecast.py
python ml_stories/us05_pipeline_burst.py
python ml_stories/us09_deficit_severity_classifier.py
python ml_stories/us10_ward_clusters_kmeans.py
python ml_stories/us11_lpcd_timeseries_forecast.py
python ml_stories/us19_complaint_volume.py
python ml_stories/us21_drought_risk.py
# ... and other usXX scripts
```

## Database Design

Star schema with 3 fact tables (fact_demand, fact_supply, fact_disruptions) and 7 dimension tables optimized for Power BI DirectQuery.

## Limitations
- Data are synthetic, so results show the pipeline works, not performance on real utility data
- US-06: the `is_anomaly` label is derived from `supply_efficiency_pct`, also a model feature; treat as rule-learning check
- US-12: final flag combines model with business rules
- US-04 is arithmetic projection, not a trained model
- Multi-step forecasts flatten after first few weeks
- US-25: health score weights (30/30/20/20) are heuristic
- MySQL and MongoDB writes skip if databases unavailable; credentials read from `MYSQL_PASSWORD` environment variable

## Documentation

See `user_stories_v2.pdf` for full acceptance criteria across all 25 user stories.