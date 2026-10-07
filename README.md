# India AI-Enhanced Water Supply Management System

A data-engineering and machine-learning pipeline for urban water supply across 15 Indian cities and 1,809 wards, built by a five-member team during Infosys training. **All data are synthetic** (generated and calibrated to public benchmarks), about 568,000 demand records. Nothing here was deployed or used by a water authority.

**Stack:** PySpark, MySQL star schema, MongoDB, Power BI, Python (LightGBM, scikit-learn, SHAP, SciPy). An optional Gemini API layer writes plain-language summaries of model outputs.

## Scope
25 Agile user stories across 7 stakeholder roles (`user_stories_v2.pdf`), covering ML prediction, data analysis, Power BI dashboards and MySQL/MongoDB storage. Scripts are in `ml_stories/`. Inputs are CSV exports of the star schema in `preprocessed/`.

## ML stories
| Story | What it does | Method |
|---|---|---|
| US-01 | 12-week city demand forecast | LightGBM, time split (train 2019-2023, test 2024), SHAP |
| US-04 | 2030 demand by city | Compound population growth x per-capita use (a projection, not a trained model) |
| US-05 | Pipeline burst risk by ward | LightGBM classifier, SMOTE on the train split only |
| US-06 | Supply anomaly flags | LightGBM classifier (see limitations) |
| US-09 | Deficit severity classes | LightGBM, split before SMOTE |
| US-10 | Ward access clusters | K-Means, K chosen by silhouette (3 to 6) |
| US-11 | LPCD forecast by ward type | One LightGBM model per ward type |
| US-12 | Tanker need next week | LightGBM plus business rules (hybrid) |
| US-19 | Complaint volume per disruption | Log-target LightGBM regression |
| US-21 | Drought risk by climate zone | LightGBM classifier per climate zone |

## Analysis, storage and dashboard stories
- **Analysis:** US-08 (disruption patterns), US-13 (COVID impact), US-15 (NRW trend and chronic wards), US-17 (slum equity), US-18 (complaint hotspots), US-20 (monsoon vs supply), US-22 (flood-year impact)
- **Storage:** US-23 (MySQL schema and load), US-24 (MongoDB archive), US-16 (MongoDB disruption audit log), US-25 (city health scorecard)
- **Power BI:** six dashboards built on the exported tables (US-02, US-07, US-14, US-17, US-20, US-25) [add screenshots to `docs/`]

## Results (as printed by the scripts)
- US-01: MAPE 5.63% on the 2024 holdout, against 5.84% for a "same as last week" baseline. The lag features carry most of the signal, so this is a small improvement over the baseline.
- US-11: MAPE 15.7% to 19.1% by ward type.
- [Add the printed results for US-05, US-09, US-10, US-12, US-19 and US-21 after running them. Report them exactly as printed.]

## Limitations
- Data are synthetic, so results show that the pipeline works, not how it would perform on real utility data. Effects such as the COVID dip in US-13 reflect how the data were generated.
- US-06: the `is_anomaly` label is defined from `supply_efficiency_pct`, which is also a model feature. Treat the scores as a rule-learning check, not independent detection accuracy.
- US-12: the final flag combines the model with business rules, so its precision reflects the rules as well as the model.
- US-04 is an arithmetic projection and is not counted as a trained model.
- Multi-step forecasts (US-01, US-11) become nearly flat after the first few weeks.
- US-25: the health score weights (30/30/20/20) are a heuristic, not a validated index.
- US-16: documents carry an `immutable` flag, but MongoDB does not enforce it.
- MySQL and MongoDB writes are skipped automatically if the databases are not running. Database credentials are read from the `MYSQL_PASSWORD` environment variable.

## Gemini summaries (optional)
`ml_stories/gemini_summary.py` reads the output CSVs for a city and asks Gemini to write a short briefing. Gemini does not make predictions. Set `GEMINI_API_KEY`, then run `python ml_stories/gemini_summary.py --city <id>`. Use `--dry-run` to see the prompt without calling the API.

## Run
```
python ml_stories/us01_demand_forecast.py     # and the other usXX scripts
```