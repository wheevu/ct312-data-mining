#!/usr/bin/env python3
"""Bài 7: Flowers clustering.

flowers.csv contains 150 rows with petal/sepal measurements (like Iris but
without species labels).  We cluster with KMeans k=3 and compare silhouette
for different k values.
"""

from common import (DATA_DIR, cache_data, elbow_data, ensure_dirs,
                    labeled_points, prepare_data, safe_silhouette, save_metrics,
                    savefig)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

SEED = 42


def main() -> None:
    ensure_dirs()
    df = pd.read_csv(cache_data("flowers.csv"))
    print(f"=== Bài 7: Flowers === shape={df.shape}")
    print(df.head())

    feats = ["PetalLength", "PetalWidth", "SepalLength", "SepalWidth"]
    X_scaled, coords = prepare_data(df, feats)

    # Elbow
    elbow = elbow_data(X_scaled, k_max=8)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(elbow["k"], elbow["inertia"], marker="o", color="royalblue")
    ax.axvline(3, color="red", ls="--", alpha=0.5, label="k=3 (natural)")
    ax.set_xlabel("k")
    ax.set_ylabel("Inertia")
    ax.set_title("Bài 7 — Flowers elbow method")
    ax.legend()
    savefig("bai_07_elbow.png")

    # KMeans k=3
    km = KMeans(n_clusters=3, n_init=10, random_state=SEED)
    labels = km.fit_predict(X_scaled)
    labs = labeled_points(df, labels, coords)
    sil = safe_silhouette(X_scaled, labels)

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(labs["plot_x"], labs["plot_y"],
                         c=labs["cluster"], cmap="Set2", s=40, edgecolors="k")
    ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
               c="red", marker="X", s=200, edgecolors="k", label="Centroids")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title(f"Bài 7 — Flowers KMeans k=3 (silhouette={sil:.4f})")
    ax.legend()
    savefig("bai_07_k3_clusters.png")

    # Pairplot colored by cluster
    fig, axs = plt.subplots(4, 4, figsize=(12, 12))
    for i, feat_i in enumerate(feats):
        for j, feat_j in enumerate(feats):
            ax = axs[i, j]
            if i == j:
                ax.hist(labs[feat_i], bins=10, color="gray", alpha=0.5)
            else:
                for cid in range(3):
                    mask = labs["cluster"] == cid
                    ax.scatter(labs[feat_j][mask], labs[feat_i][mask],
                               s=8, alpha=0.6, c=plt.cm.Set2(cid / 2))
            if i == 3:
                ax.set_xlabel(feat_j, fontsize=6)
            if j == 0:
                ax.set_ylabel(feat_i, fontsize=6)
            ax.tick_params(labelsize=5)
    savefig("bai_07_pairplot.png")
    plt.close("all")

    # Silhouette for different k
    sil_scores = []
    for k in range(2, 7):
        m = KMeans(n_clusters=k, n_init=10, random_state=SEED)
        lbl = m.fit_predict(X_scaled)
        sil_scores.append({"k": k, "silhouette": round(safe_silhouette(X_scaled, lbl) or 0, 4)})
    sil_df = pd.DataFrame(sil_scores)
    print(f"\nSilhouette scores:\n{sil_df.to_string()}")

    save_metrics([
        {"dataset": "flowers", "n": len(df), "features": len(feats),
         "k": 3, "inertia": round(km.inertia_, 2), "silhouette": round(sil or 0, 4)},
        *sil_scores,
    ], "bai_07_metrics.csv")


if __name__ == "__main__":
    main()
