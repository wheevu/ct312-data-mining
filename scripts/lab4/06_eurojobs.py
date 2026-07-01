#!/usr/bin/env python3
"""Bài 6: EuroJobs clustering.

Eurojobs.csv: 26 countries × 9 employment sector percentages.
Tasks:
  a) Find which country has max Finance sector.
  b) Scale and cluster countries with hierarchical clustering (dendrogram).
  c) Summarize cluster profiles by sector means.
  d) KMeans clustering for comparison.
"""

from common import RESULTS_DIR, cache_data, ensure_dirs, prepare_data, save_metrics, savefig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans

SEED = 42


def main() -> None:
    ensure_dirs()
    df = pd.read_csv(cache_data("Eurojobs.csv"))
    print(f"=== Bài 6: Eurojobs === shape={df.shape}")
    print(df.head())

    # Column structure: Country, Agr, Min, Man, PS, Con, SI, Fin, SPS, TC
    sector_cols = ["Agr", "Min", "Man", "PS", "Con", "SI", "Fin", "SPS", "TC"]
    countries = df["Country"].tolist()

    # (a) Country with max Finance
    fin = df.loc[df["Fin"].idxmax()]
    print(f"\n(a) Country with max Finance: {fin['Country']} ({fin['Fin']}%)")

    # (b) Scale features, hierarchical clustering (Ward)
    X_scaled, coords = prepare_data(df, sector_cols)
    Z = linkage(X_scaled, method="ward")

    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(Z, labels=countries, leaf_rotation=90, leaf_font_size=8,
               color_threshold=0, above_threshold_color="#444444", ax=ax)
    ax.set_title("Bài 6 — Eurojobs dendrogram (Ward linkage)")
    ax.set_ylabel("Distance")
    savefig("bai_06_dendrogram.png")

    # Cut tree at k=4
    labels_hc = fcluster(Z, t=4, criterion="maxclust")
    df_hc = df.copy()
    df_hc["cluster"] = labels_hc

    # Cluster profiles
    profiles = df_hc.groupby("cluster")[sector_cols].mean().round(1)
    profiles["n_countries"] = df_hc.groupby("cluster").size()
    print("\n(c) Cluster profiles (hierarchical, k=4):")
    print(profiles.to_string())
    profiles.to_csv(RESULTS_DIR / "bai_06_hc_profiles.csv")

    # Bar chart of sector means per cluster
    fig, ax = plt.subplots(figsize=(10, 5))
    profiles[sector_cols].T.plot(kind="bar", ax=ax)
    ax.set_title("Bài 6 — Sector means by cluster")
    ax.set_ylabel("% workforce")
    ax.legend(title="Cluster")
    savefig("bai_06_sector_profiles.png")

    # Country-to-cluster mapping
    country_cluster = df_hc[["Country", "cluster"]].sort_values(["cluster", "Country"])
    print("\nCountry → cluster:")
    print(country_cluster.to_string(index=False))
    country_cluster.to_csv(RESULTS_DIR / "bai_06_country_clusters.csv", index=False)

    # (d) KMeans comparison
    km = KMeans(n_clusters=4, n_init=10, random_state=SEED)
    labels_km = km.fit_predict(X_scaled)
    df_km = df.copy()
    df_km["cluster"] = labels_km
    km_profiles = df_km.groupby("cluster")[sector_cols].mean().round(1)
    km_profiles["n_countries"] = df_km.groupby("cluster").size()
    km_profiles.to_csv(RESULTS_DIR / "bai_06_km_profiles.csv")

    # PCA scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=labels_hc, cmap="Set2", s=80, edgecolors="k")
    for i, c in enumerate(countries):
        ax.text(coords[i, 0] + 0.05, coords[i, 1] + 0.05, c, fontsize=7)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("Bài 6 — Eurojobs hierarchical clustering (PCA view)")
    savefig("bai_06_pca_clusters.png")

    save_metrics([
        {"dataset": "Eurojobs", "n_countries": len(df),
         "max_finance_country": fin["Country"], "max_finance_pct": fin["Fin"],
         "hc_clusters": 4, "km_clusters": 4},
    ], "bai_06_metrics.csv")


if __name__ == "__main__":
    main()
