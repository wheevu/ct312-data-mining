#!/usr/bin/env python3
"""Bài 3: ABC Customers — hierarchical clustering & algorithm comparison.

a) Dendrogram with Ward linkage.
b) Agglomerative clustering k=5.
c) 2×2 comparison: KMeans / Agglomerative / DBSCAN / MeanShift.
"""

from common import (DATA_DIR, RESULTS_DIR, cache_data, ensure_dirs, labeled_points,
                    prepare_data, safe_silhouette, save_metrics, savefig)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans, MeanShift, estimate_bandwidth

SEED = 42


def main() -> None:
    ensure_dirs()
    df = pd.read_csv(cache_data("ABC_Customers.csv"))
    feats = ["Annual Income (k$)", "Spending Score (1-100)"]
    X_scaled, coords = prepare_data(df, feats)
    print(f"=== Bài 3: ABC Customers — hierarchical & comparison ===")

    # (a) Dendrogram
    Z = linkage(X_scaled, method="ward")
    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=90,
               leaf_font_size=8, ax=ax)
    ax.set_title("Bài 3a — Dendrogram (Ward linkage)")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Distance")
    savefig("bai_03_dendrogram.png")

    # (b) Agglomerative k=5
    agg = AgglomerativeClustering(n_clusters=5, linkage="ward")
    labels_agg = agg.fit_predict(X_scaled)
    labs_agg = labeled_points(df, labels_agg, coords)
    sil_agg = safe_silhouette(X_scaled, labels_agg)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(labs_agg["Annual Income (k$)"], labs_agg["Spending Score (1-100)"],
               c=labs_agg["cluster"], cmap="Set2", s=40, edgecolors="k")
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1-100)")
    ax.set_title("Bài 3b — Agglomerative clustering k=5")
    savefig("bai_03_agglomerative.png")

    agg_summary = labs_agg.groupby("cluster")[feats].mean().round(1)
    print("\nAgglomerative segment profiles:")
    print(agg_summary.to_string())
    agg_summary.to_csv(RESULTS_DIR / "bai_03_agg_segments.csv")

    # (c) 2×2 comparison
    kmeans = KMeans(n_clusters=5, n_init=10, random_state=SEED)
    labels_km = kmeans.fit_predict(X_scaled)
    sil_km = safe_silhouette(X_scaled, labels_km)

    bandwidth = estimate_bandwidth(X_scaled, quantile=0.3)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    labels_ms = ms.fit_predict(X_scaled)
    sil_ms = safe_silhouette(X_scaled, labels_ms) if len(set(labels_ms.tolist())) > 1 else None

    db = DBSCAN(eps=0.3, min_samples=5)
    labels_db = db.fit_predict(X_scaled)
    n_clusters_db = len(set(labels_db.tolist()) - {-1})
    sil_db = safe_silhouette(X_scaled, labels_db) if n_clusters_db >= 2 and (labels_db != -1).sum() >= 3 else None

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    plots = [
        ("KMeans", labels_km, kmeans.cluster_centers_),
        ("Agglomerative", labels_agg, None),
        ("DBSCAN", labels_db, None),
        ("MeanShift", labels_ms, ms.cluster_centers_),
    ]
    for ax, (title, labs, centers) in zip(axs.flat, plots):
        ax.scatter(coords[:, 0], coords[:, 1], c=labs, cmap="Set2", s=30, edgecolors="k")
        if centers is not None:
            ax.scatter(centers[:, 0], centers[:, 1], c="red", marker="X", s=150, edgecolors="k")
        ax.set_title(title)
        ax.set_xlabel("Income (scaled)" if title != "MeanShift" else "PCA 1")
        ax.set_ylabel("Spending (scaled)" if title != "MeanShift" else "PCA 2")
    savefig("bai_03_algorithm_comparison.png")

    save_metrics([
        {"algorithm": "KMeans", "k_or_eps": 5, "silhouette": round(sil_km or 0, 4)},
        {"algorithm": "Agglomerative", "k_or_eps": 5, "silhouette": round(sil_agg or 0, 4)},
        {"algorithm": "DBSCAN", "k_or_eps": "eps=0.3,min=5",
         "n_clusters": n_clusters_db, "silhouette": round(sil_db or 0, 4)},
        {"algorithm": "MeanShift", "k_or_eps": f"bw={bandwidth:.3f}",
         "n_clusters": len(ms.cluster_centers_), "silhouette": round(sil_ms or 0, 4)},
    ], "bai_03_metrics.csv")


if __name__ == "__main__":
    main()
