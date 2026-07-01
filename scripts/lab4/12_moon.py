#!/usr/bin/env python3
"""Bài 12: Moon dataset — DBSCAN vs KMeans comparison.

moon_dataset.csv: 200 points, 2 features, non-convex shape (two interleaving
half circles).  KMeans cannot separate them correctly because the clusters
are not convex.  DBSCAN should recover the two crescents.
"""

from common import RESULTS_DIR, cache_data, ensure_dirs, prepare_data, safe_silhouette, save_metrics, savefig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score

SEED = 42


def main() -> None:
    ensure_dirs()
    df = pd.read_csv(cache_data("moon_dataset.csv"))
    print(f"=== Bài 12: Moon dataset === shape={df.shape}")
    print(df.head())

    feats = ["Feature 1", "Feature 2"]
    X = df[feats].values.astype(float)

    # --- KMeans (expects failure) ---
    km = KMeans(n_clusters=2, n_init=10, random_state=SEED)
    labels_km = km.fit_predict(X)
    sil_km = safe_silhouette(X, labels_km)

    # --- DBSCAN (expects success) ---
    # Needs parameter tuning: the two crescents have different densities
    # Use a moderate eps
    db = DBSCAN(eps=0.2, min_samples=5)
    labels_db = db.fit_predict(X)
    n_clusters_db = len(set(labels_db.tolist()) - {-1})
    n_noise = int((labels_db == -1).sum())
    sil_db = safe_silhouette(X, labels_db) if n_clusters_db >= 2 else None

    print(f"\nKMeans: silhouette={sil_km:.4f}")
    print(f"DBSCAN: {n_clusters_db} clusters, {n_noise} noise, silhouette={sil_db}")

    # 2×2 comparison plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # KMeans
    axs[0].scatter(df["Feature 1"], df["Feature 2"],
                   c=labels_km, cmap="Set1", s=40, edgecolors="k")
    axs[0].scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                   c="black", marker="X", s=200, edgecolors="k")
    axs[0].set_title(f"KMeans (sil={sil_km:.4f})")
    axs[0].set_xlabel("Feature 1")
    axs[0].set_ylabel("Feature 2")

    # DBSCAN
    unique_db = np.unique(labels_db)
    # Make noise points (label -1) gray
    color_map = {c: plt.cm.Set1(i / max(len(unique_db) - 1, 1))
                 for i, c in enumerate(unique_db) if c != -1}
    color_map[-1] = (0.5, 0.5, 0.5, 0.5)  # gray for noise
    colors_db = [color_map[l] for l in labels_db]
    axs[1].scatter(df["Feature 1"], df["Feature 2"],
                   c=colors_db, s=40, edgecolors="k")
    axs[1].set_title(f"DBSCAN eps=0.2 ({n_clusters_db} clusters, {n_noise} noise, sil={sil_db:.4f})" if sil_db else
                     f"DBSCAN eps=0.2 ({n_clusters_db} clusters, {n_noise} noise)")
    axs[1].set_xlabel("Feature 1")
    axs[1].set_ylabel("Feature 2")

    savefig("bai_12_moon_comparison.png")

    # Also try a few eps values to show parameter sensitivity
    eps_values = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
    eps_rows = []
    for eps in eps_values:
        m = DBSCAN(eps=eps, min_samples=5)
        lbl = m.fit_predict(X)
        nc = len(set(lbl.tolist()) - {-1})
        nn = int((lbl == -1).sum())
        s = safe_silhouette(X, lbl)
        eps_rows.append({"eps": eps, "n_clusters": nc, "noise": nn,
                         "silhouette": round(s or 0, 4)})
        # Compute ARI vs KMeans (not ground truth)
    eps_df = pd.DataFrame(eps_rows)
    print("\nDBSCAN parameter sensitivity:")
    print(eps_df.to_string(index=False))
    eps_df.to_csv(RESULTS_DIR / "bai_12_dbscan_eps_sweep.csv", index=False)

    # Multi-eps plot
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    for ax, eps in zip(axs.flat, eps_values):
        m = DBSCAN(eps=eps, min_samples=5)
        lbl = m.fit_predict(X)
        u = np.unique(lbl)
        cm = {c: plt.cm.Set1(i / max(len(u) - 1, 1)) for i, c in enumerate(u) if c != -1}
        cm[-1] = (0.5, 0.5, 0.5, 0.5)
        colors = [cm[l] for l in lbl]
        ax.scatter(df["Feature 1"], df["Feature 2"], c=colors, s=15, edgecolors="k")
        ax.set_title(f"eps={eps}")
    savefig("bai_12_dbscan_eps_sweep.png")

    save_metrics([
        {"dataset": "moon", "algorithm": "KMeans", "k": 2,
         "silhouette": round(sil_km, 4)},
        {"dataset": "moon", "algorithm": "DBSCAN", "k_or_eps": "eps=0.2,min=5",
         "n_clusters": n_clusters_db, "noise": n_noise,
         "silhouette": round(sil_db or 0, 4)},
        *eps_rows,
    ], "bai_12_metrics.csv")


if __name__ == "__main__":
    main()
