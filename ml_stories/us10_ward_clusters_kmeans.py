"""
US-10: Cluster Wards by Water Access Profile (K-Means)
Standardised features, K chosen by silhouette over K=3..6, labels written to ward_clusters.
"""
import os, warnings
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
warnings.filterwarnings("ignore"); os.makedirs("output", exist_ok=True)

wards  = pd.read_csv("preprocessed/dim_wards.csv")
supply = pd.read_csv("preprocessed/fact_supply.csv")
agg = (supply.groupby("ward_id")[["supply_efficiency_pct", "hours_of_supply"]].mean()
             .rename(columns={"supply_efficiency_pct": "avg_supply_efficiency_pct",
                              "hours_of_supply": "avg_hours_supply"}).reset_index())
df = wards.merge(agg, on="ward_id", how="inner")

FEATURES = ["piped_connection_coverage_pct", "metered_connections_pct", "has_slum_pocket",
            "ward_type_encoded", "avg_supply_efficiency_pct", "avg_hours_supply"]
missing = [c for c in FEATURES if c not in df.columns]
assert not missing, f"Columns not found: {missing}\nAvailable: {list(df.columns)}"
df = df.dropna(subset=FEATURES).reset_index(drop=True)
Xs = StandardScaler().fit_transform(df[FEATURES])

print(f"Wards clustered: {len(df):,}")
scores = {}
for k in range(3, 7):
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(Xs)
    scores[k] = silhouette_score(Xs, labels, sample_size=min(10000, len(df)), random_state=42)
    print(f"  K={k}: silhouette={scores[k]:.4f}")
best_k = max(scores, key=scores.get)
print(f"Best K by silhouette: {best_k}")

km = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(Xs)
df["cluster_id"] = km.labels_
prof = df.groupby("cluster_id")[FEATURES].mean()
prof["n_wards"] = df.groupby("cluster_id").size()

# Descriptive names derived from the cluster means (review them before using in reports)
access = (prof["piped_connection_coverage_pct"].rank() + prof["avg_supply_efficiency_pct"].rank()
          + prof["avg_hours_supply"].rank())
order = access.sort_values().index.tolist()
names = {}
for i, c in enumerate(order):
    names[c] = "Underserved" if i == 0 else ("Well-Served" if i == len(order) - 1 else f"Moderately Served {i}")
slum_c = prof["has_slum_pocket"].idxmax()
if slum_c != order[-1] and prof.loc[slum_c, "has_slum_pocket"] > 0.5:
    names[slum_c] = "Slum-Heavy"
df["cluster_name"] = df["cluster_id"].map(names)
prof["cluster_name"] = prof.index.map(names)
print("\nCluster profiles (means):\n", prof.round(2).to_string())

out = df[["ward_id", "cluster_id", "cluster_name"]]
out.to_csv("output/us10_ward_clusters.csv", index=False)
prof.round(3).to_csv("output/us10_cluster_profiles.csv")
print("\nSaved -> output/us10_ward_clusters.csv, output/us10_cluster_profiles.csv")
try:
    from sqlalchemy import create_engine
    pw = os.getenv("MYSQL_PASSWORD", "password")
    eng = create_engine(f"mysql+pymysql://root:{pw}@localhost/water_supply_india")
    out.to_sql("ward_clusters", eng, if_exists="replace", index=False)
    print("ward_clusters saved to MySQL")
except Exception as e:
    print(f"[MySQL skipped] {e}")
