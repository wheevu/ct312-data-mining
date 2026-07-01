#!/usr/bin/env python3
"""Bài 9: RFM customer segmentation.

dataCustomerRFM.csv contains order-level data.  We aggregate per customer:
  - Recency: days since last order (reference = max date + 1)
  - Frequency: number of orders
  - Monetary: total amount spent
Then scale RFM and run KMeans k=4 segmentation.
"""

from common import (DATA_DIR, RESULTS_DIR, cache_data, elbow_data, ensure_dirs,
                    prepare_data, safe_silhouette, save_metrics, savefig)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

SEED = 42


def main() -> None:
    ensure_dirs()
    df = pd.read_csv(cache_data("dataCustomerRFM.csv"))
    print(f"=== Bài 9: RFM segmentation === shape={df.shape}")

    # Clean columns
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Parse dates
    df["OrderDate"] = pd.to_datetime(df["OrderDate"], errors="coerce")
    df = df.dropna(subset=["OrderDate", "CustomerID"])

    # Compute RFM
    ref_date = df["OrderDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency=("OrderDate", lambda x: (ref_date - x.max()).days),
        Frequency=("OrderID", "nunique"),
        Monetary=("Amount", "sum"),
    ).reset_index()
    print(f"RFM shape: {rfm.shape}")
    print(rfm.describe())

    # Remove outliers — clip Monetary at 99th percentile
    m99 = rfm["Monetary"].quantile(0.99)
    rfm["Monetary"] = rfm["Monetary"].clip(upper=m99)

    # Save RFM table
    rfm.to_csv(RESULTS_DIR / "bai_09_rfm_values.csv", index=False)

    # Prepare for clustering
    feats = ["Recency", "Frequency", "Monetary"]
    X_scaled, coords = prepare_data(rfm, feats)

    # Elbow
    elbow = elbow_data(X_scaled, k_max=8)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(elbow["k"], elbow["inertia"], marker="o", color="royalblue")
    ax.axvline(4, color="red", ls="--", alpha=0.5, label="k=4")
    ax.set_xlabel("k")
    ax.set_ylabel("Inertia")
    ax.set_title("Bài 9 — RFM elbow method")
    ax.legend()
    savefig("bai_09_elbow.png")

    # KMeans k=4
    km = KMeans(n_clusters=4, n_init=10, random_state=SEED)
    labels = km.fit_predict(X_scaled)
    sil = safe_silhouette(X_scaled, labels)

    rfm["cluster"] = labels.astype(int)
    rfm["plot_x"] = coords[:, 0]
    rfm["plot_y"] = coords[:, 1]

    # PCA scatter
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(rfm["plot_x"], rfm["plot_y"], c=rfm["cluster"],
               cmap="Set2", s=10, edgecolors="k", alpha=0.6)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title(f"Bài 9 — RFM segments k=4 (silhouette={sil:.4f})")
    savefig("bai_09_rfm_clusters.png")

    # Segment profiles
    profiles = rfm.groupby("cluster")[feats].mean().round(1)
    profiles["n_customers"] = rfm.groupby("cluster").size()
    print("\nRFM segment profiles:")
    print(profiles.to_string())
    profiles.to_csv(RESULTS_DIR / "bai_09_rfm_profiles.csv")

    # Interpretation
    print("\nSegment interpretation (higher Recency = less recent):")
    for cid, row in profiles.iterrows():
        print(f"  Cluster {cid}: {int(row['n_customers'])} customers, "
              f"R={row['Recency']:.0f}d, F={row['Frequency']:.1f}, M={row['Monetary']:.0f}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    profiles[feats].T.plot(kind="bar", ax=ax)
    ax.set_title("Bài 9 — RFM means by cluster")
    ax.legend(title="Cluster")
    savefig("bai_09_rfm_profiles.png")

    save_metrics([{
        "dataset": "dataCustomerRFM",
        "n_customers": len(rfm), "n_orders": len(df),
        "k": 4, "inertia": round(km.inertia_, 2),
        "silhouette": round(sil or 0, 4),
        "cluster_0_n": int((labels == 0).sum()),
        "cluster_1_n": int((labels == 1).sum()),
        "cluster_2_n": int((labels == 2).sum()),
        "cluster_3_n": int((labels == 3).sum()),
    }], "bai_09_metrics.csv")


if __name__ == "__main__":
    main()
