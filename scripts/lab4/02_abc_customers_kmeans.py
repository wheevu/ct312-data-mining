#!/usr/bin/env python3
"""Bài 2: ABC Customers — elbow method & KMeans segmentation.

Exercises:
  a) Elbow method for Annual Income + Spending Score.
  b) KMeans k=5 on Income + Spending, interpret segments.
  c) KMeans on Age + Spending.
  d) KMeans on Age + Income + Spending (PCA plot for 3D).
"""

from common import (DATA_DIR, RESULTS_DIR, cache_data, elbow_data, ensure_dirs,
                    labeled_points, prepare_data, safe_silhouette, save_metrics,
                    savefig)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

SEED = 42


def main() -> None:
    ensure_dirs()
    df = pd.read_csv(cache_data("ABC_Customers.csv"))
    print(f"=== Bài 2: ABC Customers KMeans === shape={df.shape}")

    # --------------------------------
    # (a) Elbow: Income + Spending
    # --------------------------------
    feats_inc_spend = ["Annual Income (k$)", "Spending Score (1-100)"]
    X_inc_spend, coords_inc_spend = prepare_data(df, feats_inc_spend)
    elbow = elbow_data(X_inc_spend, k_max=10)
    elbow.to_csv(RESULTS_DIR / "bai_02_elbow.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(elbow["k"], elbow["inertia"], marker="o", color="royalblue")
    ax.axvline(5, color="red", ls="--", alpha=0.5, label="chosen k=5")
    ax.set_xlabel("k")
    ax.set_ylabel("Inertia")
    ax.set_title("Bài 2a — Elbow method (Income + Spending)")
    ax.legend()
    savefig("bai_02_elbow.png")

    # --------------------------------
    # (b) KMeans k=5 on Income + Spending
    # --------------------------------
    k5 = KMeans(n_clusters=5, n_init=10, random_state=SEED)
    labels_b = k5.fit_predict(X_inc_spend)
    labs_b = labeled_points(df, labels_b, coords_inc_spend)

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(labs_b["Annual Income (k$)"], labs_b["Spending Score (1-100)"],
                         c=labs_b["cluster"], cmap="Set2", s=40, edgecolors="k")
    ax.scatter(k5.cluster_centers_[:, 0], k5.cluster_centers_[:, 1],
               c="red", marker="X", s=200, edgecolors="k", label="Centroids")
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1-100)")
    ax.set_title("Bài 2b — KMeans k=5 (Income + Spending)")
    ax.legend()
    savefig("bai_02_k5_income_spending.png")

    # Segment interpretation
    segs = labs_b.groupby("cluster")[feats_inc_spend].mean().round(1)
    print("\nSegment profiles (Income + Spending):")
    print(segs.to_string())
    segs.to_csv(RESULTS_DIR / "bai_02_segment_profiles_income_spending.csv")

    # --------------------------------
    # (c) Age + Spending
    # --------------------------------
    feats_age_spend = ["Age", "Spending Score (1-100)"]
    X_age_spend, coords_age_spend = prepare_data(df, feats_age_spend)
    k5_age = KMeans(n_clusters=5, n_init=10, random_state=SEED)
    labels_c = k5_age.fit_predict(X_age_spend)
    labs_c = labeled_points(df, labels_c, coords_age_spend)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(labs_c["Age"], labs_c["Spending Score (1-100)"],
               c=labs_c["cluster"], cmap="Set2", s=40, edgecolors="k")
    ax.scatter(k5_age.cluster_centers_[:, 0], k5_age.cluster_centers_[:, 1],
               c="red", marker="X", s=200, edgecolors="k")
    ax.set_xlabel("Age")
    ax.set_ylabel("Spending Score (1-100)")
    ax.set_title("Bài 2c — KMeans k=5 (Age + Spending)")
    savefig("bai_02_k5_age_spending.png")

    segs_c = labs_c.groupby("cluster")[feats_age_spend].mean().round(1)
    print("\nSegment profiles (Age + Spending):")
    print(segs_c.to_string())
    segs_c.to_csv(RESULTS_DIR / "bai_02_segment_profiles_age_spending.csv")

    # --------------------------------
    # (d) Age + Income + Spending (3D -> PCA plot)
    # --------------------------------
    feats_3d = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    X_3d, coords_3d = prepare_data(df, feats_3d)
    k5_3d = KMeans(n_clusters=5, n_init=10, random_state=SEED)
    labels_d = k5_3d.fit_predict(X_3d)
    labs_d = labeled_points(df, labels_d, coords_3d)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(labs_d["plot_x"], labs_d["plot_y"],
               c=labs_d["cluster"], cmap="Set2", s=40, edgecolors="k")
    ax.scatter(k5_3d.cluster_centers_[:, 0], k5_3d.cluster_centers_[:, 1],
               c="red", marker="X", s=200, edgecolors="k")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("Bài 2d — KMeans k=5 (Age + Income + Spending, PCA view)")
    savefig("bai_02_k5_age_income_spending_pca.png")

    # --- Summary metrics ---
    save_metrics([
        {"features": "Income + Spending", "k": 5,
         "inertia": round(k5.inertia_, 2),
         "silhouette": round(safe_silhouette(X_inc_spend, labels_b) or 0, 4)},
        {"features": "Age + Spending", "k": 5,
         "inertia": round(k5_age.inertia_, 2),
         "silhouette": round(safe_silhouette(X_age_spend, labels_c) or 0, 4)},
        {"features": "Age + Income + Spending", "k": 5,
         "inertia": round(k5_3d.inertia_, 2),
         "silhouette": round(safe_silhouette(X_3d, labels_d) or 0, 4)},
    ], "bai_02_metrics.csv")


if __name__ == "__main__":
    main()
