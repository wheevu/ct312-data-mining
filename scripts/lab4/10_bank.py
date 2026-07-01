#!/usr/bin/env python3
"""Bài 10: Bank customer clustering.

bank-data.csv: 600 customers with demographics and product holdings.
We encode categorical fields, scale, cluster with KMeans, and interpret
the resulting segments.
"""

from common import RESULTS_DIR, cache_data, elbow_data, ensure_dirs, prepare_data, safe_silhouette, save_metrics, savefig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

SEED = 42


def main() -> None:
    ensure_dirs()
    df = pd.read_csv(cache_data("bank-data.csv"))
    print(f"=== Bài 10: Bank customer clustering === shape={df.shape}")
    print(df.head())

    # Drop ID column
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Encode categoricals
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    feats = df.columns.tolist()
    print(f"Features used: {feats}")

    X_scaled, coords = prepare_data(df, feats)

    # Elbow
    elbow = elbow_data(X_scaled, k_max=8)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(elbow["k"], elbow["inertia"], marker="o", color="royalblue")
    ax.axvline(4, color="red", ls="--", alpha=0.5, label="k=4")
    ax.set_xlabel("k")
    ax.set_ylabel("Inertia")
    ax.set_title("Bài 10 — Bank data elbow method")
    ax.legend()
    savefig("bai_10_elbow.png")

    # KMeans k=4
    km = KMeans(n_clusters=4, n_init=10, random_state=SEED)
    labels = km.fit_predict(X_scaled)
    sil = safe_silhouette(X_scaled, labels)

    df_out = df.copy()
    df_out["cluster"] = labels.astype(int)
    df_out["plot_x"] = coords[:, 0]
    df_out["plot_y"] = coords[:, 1]

    # PCA scatter
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df_out["plot_x"], df_out["plot_y"],
               c=df_out["cluster"], cmap="Set2", s=30, edgecolors="k", alpha=0.7)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title(f"Bài 10 — Bank customers k=4 (silhouette={sil:.4f})")
    savefig("bai_10_bank_clusters.png")

    # Cluster profiles (mean per feature per cluster)
    profiles = df_out.groupby("cluster")[feats].mean().round(2)
    profiles["n_customers"] = df_out.groupby("cluster").size()
    print("\nBank segment profiles (mean values):")
    print(profiles.to_string())
    profiles.to_csv(RESULTS_DIR / "bai_10_bank_profiles.csv")

    # Customer-cluster mapping (first 10 per cluster)
    mapping = df_out[["cluster"] + feats].copy()
    mapping.to_csv(RESULTS_DIR / "bai_10_customer_clusters.csv", index=False)

    # Bar chart of key features per cluster
    key_feats = ["age", "income", "children"]
    key_avail = [f for f in key_feats if f in profiles.columns]
    if key_avail:
        fig, ax = plt.subplots(figsize=(7, 4))
        profiles[key_avail].T.plot(kind="bar", ax=ax)
        ax.set_title("Bài 10 — Key features by cluster")
        ax.legend(title="Cluster")
        savefig("bai_10_key_features.png")

    save_metrics([{
        "dataset": "bank-data", "n_customers": len(df),
        "n_features": len(feats), "n_categorical_encoded": len(cat_cols),
        "k": 4, "inertia": round(km.inertia_, 2),
        "silhouette": round(sil or 0, 4),
    }], "bai_10_metrics.csv")


if __name__ == "__main__":
    main()
