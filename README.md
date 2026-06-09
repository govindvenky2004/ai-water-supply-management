# India AI-Enhanced Water Supply Management System

ML pipeline for demand forecasting, anomaly detection, 
and leakage prediction across 15 Indian cities — 
built during Infosys Systems Engineer Trainee Internship.

## Project Scale
- 15 Indian cities, 1,809 wards
- ~568K records (2019–2024)
- 25 user stories across 7 stakeholder roles
- 10 database tables (Star Schema)

## Tech Stack
- Python + PySpark (ML pipeline)
- MySQL (Star Schema — fact + dim tables)
- MongoDB (alerts, audit logs, ML outputs)
- Power BI (6 dashboards)
- LightGBM, Random Forest, sklearn
- Gemini API (AI-generated field alerts)

## ML Models Built
- US-01: City-wide water demand forecast (12 weeks)
- US-05: Pipeline burst risk classifier by ward
- US-06: Supply anomaly detection (SMOTE)
- US-09: Deficit severity classifier (4-class)
- US-11: LPCD time-series forecast by ward type
- US-12: Tanker supplement predictor
- US-19: Complaint volume regression (R²=0.90)
- US-21: Drought risk predictor by climate zone

## Power BI Dashboards
- Supply vs Demand gap (15 cities)
- Infrastructure health map (1,809 wards)
- Tariff equity analysis
- Slum ward water access equity
- Monsoon vs supply efficiency
- Executive city health scorecard

## Database Design
Star schema with 3 fact tables (fact_demand, 
fact_supply, fact_disruptions) and 7 dimension 
tables optimized for Power BI DirectQuery.

## Project Documentation
See user_stories_v2.pdf for full acceptance 
criteria across all 25 user stories.
