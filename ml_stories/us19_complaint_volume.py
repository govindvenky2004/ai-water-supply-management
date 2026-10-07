"""
US-19: Predict Disruption Complaint Volume (log-target regression + 90% interval)
Time-based split if a date column exists, otherwise a random 80/20 split.
Report R2 and RMSE as printed; the criterion R2>0.70 is checked, not assumed.
"""
import os, warnings
import numpy as np, pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore"); os.makedirs("output", exist_ok=True)

d = pd.read_csv("preprocessed/fact_disruptions.csv")
NUM = ["duration_hours", "population_affected", "num_wards_affected", "estimated_supply_loss_mld"]
TARGET = "complaint_count"
missing = [c for c in NUM + [TARGET, "cause"] if c not in d.columns]
assert not missing, f"Columns not found: {missing}\nAvailable: {list(d.columns)}"
d = d.dropna(subset=NUM + [TARGET, "cause"]).reset_index(drop=True)
X = pd.concat([d[NUM], pd.get_dummies(d["cause"], prefix="cause")], axis=1)
y = np.log1p(d[TARGET])

date_col = next((c for c in ("date", "start_date", "disruption_date") if c in d.columns), None)
if date_col:
    order = pd.to_datetime(d[date_col]).argsort().values
    cut = int(len(d) * 0.8)
    tr_idx, te_idx = order[:cut], order[cut:]
    print(f"Time-based split on '{date_col}'")
else:
    tr_idx, te_idx = train_test_split(np.arange(len(d)), test_size=0.2, random_state=42)
    print("No date column found: random 80/20 split")
Xtr, Xte, ytr, yte = X.iloc[tr_idx], X.iloc[te_idx], y.iloc[tr_idx], y.iloc[te_idx]

base = dict(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1)
mid = LGBMRegressor(**base).fit(Xtr, ytr)
lo  = LGBMRegressor(objective="quantile", alpha=0.05, **base).fit(Xtr, ytr)
hi  = LGBMRegressor(objective="quantile", alpha=0.95, **base).fit(Xtr, ytr)

p = mid.predict(Xte)
r2_log = r2_score(yte, p)
actual = np.expm1(yte); pred = np.expm1(p)
rmse = float(np.sqrt(mean_squared_error(actual, pred)))
lo_c, hi_c = np.expm1(lo.predict(Xte)), np.expm1(hi.predict(Xte))
coverage = float(((actual >= np.minimum(lo_c, hi_c)) & (actual <= np.maximum(lo_c, hi_c))).mean())
print(f"R2 (log scale): {r2_log:.4f}  (criterion > 0.70: {'met' if r2_log > 0.70 else 'NOT met'})")
print(f"R2 (original units): {r2_score(actual, pred):.4f}")
print(f"RMSE (original complaint units): {rmse:.2f}")
print(f"90% interval empirical coverage on test: {coverage:.2%}")

res = d.loc[te_idx, ["cause"] + NUM].copy()
res["actual"] = actual.values; res["predicted"] = pred.round(1)
res["lower_90"] = np.maximum(0, np.minimum(lo_c, hi_c)).round(1)
res["upper_90"] = np.maximum(lo_c, hi_c).round(1)
res.to_csv("output/us19_complaint_predictions_test.csv", index=False)
print("Saved -> output/us19_complaint_predictions_test.csv")
