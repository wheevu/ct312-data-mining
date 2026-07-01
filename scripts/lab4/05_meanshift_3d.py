#!/usr/bin/env python3
"""Bài 5: MeanShift on 3D synthetic dataset.

Loads MeanShift-3D.csv (X, Y, Z), estimates bandwidth, runs MeanShift,
plots original 3D scatter and clustered 3D scatter with centroids marked.
"""

from common import cache_data, ensure_dirs, safe_silhouette, save_metrics, savefig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth

SEED = 42


def main() -> None:
    ensure_dirs()
    df = pd.read_csv(cache_data("MeanShift-3D.csv"), sep="\t")
    print(f"=== Bài 5: MeanShift 3D === shape={df.shape}")
    print(df.head())

    X = df[["X", "Y", "Z"]].values.astype(float)

    bandwidth = estimate_bandwidth(X, quantile=0.3, random_state=SEED)
    print(f"Estimated bandwidth: {bandwidth:.4f}")

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True)
    labels = ms.fit_predict(X)
    centers = ms.cluster_centers_
    n_clusters = len(centers)
    unique_labels = np.unique(labels)
    print(f"Number of clusters found: {n_clusters}")
    print(f"Cluster sizes: {pd.Series(labels).value_counts().sort_index().to_dict()}")

    # 3D plot — original
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(df["X"], df["Y"], df["Z"], c="royalblue", s=25, alpha=0.6)
    ax1.set_title("Bài 5 — Original 3D data")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # 3D plot — clustered
    ax2 = fig.add_subplot(122, projection="3d")
    colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))
    for i in range(n_clusters):
        mask = labels == i
        ax2.scatter(df["X"][mask], df["Y"][mask], df["Z"][mask],
                    c=[colors[i]], s=25, alpha=0.6, label=f"Cluster {i}")
    ax2.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                c="black", marker="X", s=200, label="Centroids")
    ax2.set_title(f"Bài 5 — MeanShift (bandwidth={bandwidth:.3f}, {n_clusters} clusters)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.legend()
    savefig("bai_05_meanshift_3d.png")

    # Metrics
    sil = safe_silhouette(X, labels)
    save_metrics([{
        "dataset": "MeanShift-3D",
        "n_points": len(df),
        "bandwidth": round(bandwidth, 4),
        "n_clusters": n_clusters,
        "silhouette": round(sil or 0, 4),
        "centroids_n": len(centers),
    }], "bai_05_metrics.csv")


if __name__ == "__main__":
    main()
