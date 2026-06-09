"""
US-01: Predict City-Wide Daily Water Demand for Next 12 Weeks
PATCHED – columns verified against diagnose_columns.py output
"""
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
import shap, os, warnings
warnings.filterwarnings("ignore")
os.makedirs("output", exist_ok=True)

# ── Load ml_sample ────────────────────────────────────────────────────────────
df = pd.read_csv("preprocessed/ml_sample.csv")
df["date"] = pd.to_datetime(df["date"])

# ── Features (all verified present) ──────────────────────────────────────────
FEATURES = [
    "demand_lag_1w",       # ✓ ml_sample
    "demand_lag_2w",       # ✓ ml_sample
    "demand_roll_12w_avg", # ✓ ml_sample
    "seasonal_factor",     # ✓ ml_sample
    "rainfall_mm",         # ✓ ml_sample
    "rain_3m_sum",         # ✓ ml_sample
    "is_drought_year",     # ✓ ml_sample
]
TARGET = "estimated_demand_mld"   # ✓

# ── Time-based split ──────────────────────────────────────────────────────────
train = df[df["year"] <= 2023]
test  = df[df["year"] == 2024]
X_train, y_train = train[FEATURES], train[TARGET]
X_test,  y_test  = test[FEATURES],  test[TARGET]
print(f"Train: {len(X_train):,} rows | Test (2024 holdout): {len(X_test):,} rows")

# ── Train LightGBM ────────────────────────────────────────────────────────────
model = LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                      num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                      random_state=42, verbose=-1)
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
print(f"\n── Metrics ──────────────────────────────────────────────────────")
print(f"  MAPE : {mape:.2f}%  (criterion < 10%)")
print(f"  RMSE : {rmse:.4f} MLD")
print(f"  R²   : {r2:.4f}")
assert mape < 10, f"MAPE {mape:.2f}% exceeds 10% acceptance criterion!"

# ── 12-Week rolling city forecast ─────────────────────────────────────────────
latest  = df.sort_values("date").groupby("ward_id").tail(1).copy()
current = latest.copy()
results = []
print(f"\n── 12-Week City Forecast ────────────────────────────────────────")
for w in range(1, 13):
    current["predicted_demand"] = model.predict(current[FEATURES])
    city_pred = current.groupby("city_id")["predicted_demand"].sum().round(2)
    city_pred.name = f"week_{w}"
    results.append(city_pred)
    print(f"  Week {w:02d}: {city_pred.sum():.2f} MLD total")
    current["demand_lag_2w"] = current["demand_lag_1w"]
    current["demand_lag_1w"] = current["predicted_demand"]

forecast_df = pd.concat(results, axis=1).reset_index()
forecast_df.to_csv("output/us01_12week_city_forecast.csv", index=False)
print(f"Saved → output/us01_12week_city_forecast.csv")

# ── SHAP values ───────────────────────────────────────────────────────────────
explainer = shap.TreeExplainer(model)
shap_vals = explainer.shap_values(X_test.head(500))
mean_shap = pd.DataFrame(shap_vals, columns=FEATURES).abs().mean().sort_values(ascending=False)
print(f"\n── SHAP Feature Importance ──────────────────────────────────────")
for feat, val in mean_shap.items():
    print(f"  {feat:<30} {val:.4f}")
mean_shap.to_frame("mean_abs_shap").to_csv("output/us01_shap_importance.csv")
print(f"Saved → output/us01_shap_importance.csv")
