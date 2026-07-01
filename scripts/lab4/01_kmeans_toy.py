#!/usr/bin/env python3
"""Bài 1: KMeans toy example — custom init centroids with step tracking.

Fits KMeans on a small 6-point 2D dataset with explicit initial centroid
vectors so students can verify the assignment and update steps by hand.
"""

from common import ensure_dirs, save_metrics, savefig
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

SEED = 42


def main() -> None:
    ensure_dirs()
    rng = np.random.default_rng(SEED)

    # Toy dataset: 6 points in 2D, two natural clusters
    X = np.array([[1, 2], [1, 5], [2, 4],
                   [8, 1], [8, 3], [9, 2]], dtype=float)

    # Fixed initial centroids (matching assignment intent)
    init = np.array([[1, 4], [8, 2]], dtype=float)

    model = KMeans(n_clusters=2, init=init, n_init=1, random_state=SEED)
    model.fit(X)

    labels = model.labels_
    centers = model.cluster_centers_

    # --- Print results ---
    print("=== Bài 1: KMeans toy example ===")
    print(f"Data points:\n{X}")
    print(f"Initial centroids:\n{init}")
    print(f"Final centroids:\n{centers}")
    print(f"Labels: {labels}")
    print(f"Inertia: {model.inertia_:.4f}")

    # Predict new points
    new = np.array([[0, 4], [9, 4]])
    pred = model.predict(new)
    print(f"Predict {new} -> {pred}")

    # --- Scatter plot ---
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["#e41a1c", "#377eb8"]
    for i in range(2):
        mask = labels == i
        ax.scatter(X[mask, 0], X[mask, 1], c=colors[i], label=f"Cluster {i}", s=80, edgecolors="k")
    ax.scatter(centers[:, 0], centers[:, 1], c="gold", marker="X", s=250, edgecolors="k", label="Centroids")
    ax.scatter(new[:, 0], new[:, 1], c="lime", marker="s", s=120, edgecolors="k", label="New points")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Bài 1 — KMeans toy example")
    ax.legend()
    ax.set_aspect("equal")
    savefig("bai_01_kmeans_toy.png")

    # --- Metrics ---
    save_metrics([
        {"model": "KMeans_k2_init_fixed",
         "n_clusters": 2, "n_points": len(X),
         "inertia": round(model.inertia_, 4),
         "centroid_0_x": round(centers[0, 0], 4), "centroid_0_y": round(centers[0, 1], 4),
         "centroid_1_x": round(centers[1, 0], 4), "centroid_1_y": round(centers[1, 1], 4),
         "pred_0_4": int(pred[0]), "pred_9_4": int(pred[1])},
    ], "bai_01_metrics.csv")


if __name__ == "__main__":
    main()
