#!/usr/bin/env python3
"""Bài 8: USArrests — KMeans clustering with scaling.

USArrests.csv: 50 states × (Murder, Assault, UrbanPop, Rape).
Scale features before KMeans; plot PCA with state labels; interpret clusters.
"""

from common import (DATA_DIR, RESULTS_DIR, cache_data, ensure_dirs, prepare_data,
                    safe_silhouette, save_metrics, savefig)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

SEED = 42


def main() -> None:
    ensure_dirs()
    df = pd.read_csv(cache_data("USArrests.csv"))
    print(f"=== Bài 8: USArrests === shape={df.shape}")
    print(df.head())

    # Rename unnamed index column
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "State"})

    states = df["State"].tolist()
    feats = ["Murder", "Assault", "UrbanPop", "Rape"]
    X_scaled, coords = prepare_data(df, feats)

    # KMeans k=4
    km = KMeans(n_clusters=4, n_init=10, random_state=SEED)
    labels = km.fit_predict(X_scaled)
    sil = safe_silhouette(X_scaled, labels)

    df_lab = df.copy()
    df_lab["cluster"] = labels.astype(int)
    df_lab["plot_x"] = coords[:, 0]
    df_lab["plot_y"] = coords[:, 1]

    # PCA plot with state labels
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    for cid in range(4):
        mask = df_lab["cluster"] == cid
        ax.scatter(df_lab["plot_x"][mask], df_lab["plot_y"][mask],
                   c=colors[cid], s=80, edgecolors="k", label=f"Cluster {cid}")
        for _, row in df_lab[mask].iterrows():
            ax.text(row["plot_x"] + 0.05, row["plot_y"] + 0.05,
                    row["State"], fontsize=7)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title(f"Bài 8 — USArrests KMeans k=4 (silhouette={sil:.4f})")
    ax.legend()
    savefig("bai_08_usarrests_pca.png")

    # Cluster profiles
    profiles = df_lab.groupby("cluster")[feats].mean().round(1)
    profiles["n_states"] = df_lab.groupby("cluster").size()
    print("\nCluster profiles (crime means):")
    print(profiles.to_string())
    profiles.to_csv(RESULTS_DIR / "bai_08_cluster_profiles.csv")

    # State-to-cluster mapping
    state_cluster = df_lab[["State", "cluster"]].sort_values(["cluster", "State"])
    state_cluster.to_csv(RESULTS_DIR / "bai_08_state_clusters.csv", index=False)
    print("\nState → cluster (first 10):")
    print(state_cluster.head(10).to_string(index=False))

    # Bar chart comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    profiles[feats].T.plot(kind="bar", ax=ax)
    ax.set_title("Bài 8 — Crime means by cluster")
    ax.set_ylabel("Value (scaled feature mean)")
    ax.legend(title="Cluster")
    savefig("bai_08_cluster_profiles.png")

    save_metrics([{
        "dataset": "USArrests", "n_states": len(df), "features": len(feats),
        "k": 4, "inertia": round(km.inertia_, 2), "silhouette": round(sil or 0, 4),
    }], "bai_08_metrics.csv")


if __name__ == "__main__":
    main()
