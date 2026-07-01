#!/usr/bin/env python3
"""Bài 11: E-commerce customer spending clustering.

ABC_customerSpending.csv: large transaction dataset (92k rows) with
product category, province, order cost, customer ID.
We aggregate per customer by total spend, order count, and category diversity,
then cluster segments.
"""

from common import RESULTS_DIR, cache_data, ensure_dirs, prepare_data, safe_silhouette, save_metrics, savefig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

SEED = 42


def main() -> None:
    ensure_dirs()
    df = pd.read_csv(cache_data("ABC_customerSpending.csv"))
    print(f"=== Bài 11: E-commerce customer spending === shape={df.shape}")

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Parse order date
    df["ORDER_DATE"] = pd.to_datetime(df["ORDER_DATE"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["CUST_ID"])
    print(f"Unique customers: {df['CUST_ID'].nunique()}")
    print(f"Unique categories: {df['PRODUCT_CATE'].nunique()}")
    print(f"Unique provinces: {df['PROVINCE'].nunique()}")

    # Aggregate per customer
    cust = df.groupby("CUST_ID").agg(
        total_spend=("ORDER_COST", "sum"),
        order_count=("ORDER_ID", "nunique"),
        category_count=("PRODUCT_CATE", "nunique"),
        province_count=("PROVINCE", "nunique"),
        avg_order_value=("ORDER_COST", "mean"),
    ).reset_index()

    # Drop extreme outliers (top 1% of total_spend)
    cap = cust["total_spend"].quantile(0.99)
    cust["total_spend"] = cust["total_spend"].clip(upper=cap)
    cap_avg = cust["avg_order_value"].quantile(0.99)
    cust["avg_order_value"] = cust["avg_order_value"].clip(upper=cap_avg)

    print(f"\nCustomer-level features:\n{cust.describe()}")

    feats = ["total_spend", "order_count", "category_count", "province_count", "avg_order_value"]
    X_scaled, coords = prepare_data(cust, feats)

    # KMeans k=4
    km = KMeans(n_clusters=4, n_init=10, random_state=SEED)
    labels = km.fit_predict(X_scaled)
    sil = safe_silhouette(X_scaled, labels)

    cust["cluster"] = labels.astype(int)
    cust["plot_x"] = coords[:, 0]
    cust["plot_y"] = coords[:, 1]

    # PCA scatter
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(cust["plot_x"], cust["plot_y"],
               c=cust["cluster"], cmap="Set2", s=5, edgecolors="k", alpha=0.5)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title(f"Bài 11 — Customer spending segments k=4 (sil={sil:.4f})")
    savefig("bai_11_customer_segments.png")

    # Profiles
    profiles = cust.groupby("cluster")[feats].mean().round(1)
    profiles["n_customers"] = cust.groupby("cluster").size()
    print("\nCustomer segment profiles:")
    print(profiles.to_string())
    profiles.to_csv(RESULTS_DIR / "bai_11_segment_profiles.csv")

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    profiles[feats].T.plot(kind="bar", ax=ax)
    ax.set_title("Bài 11 — Spending means by cluster")
    ax.legend(title="Cluster")
    savefig("bai_11_profiles.png")

    save_metrics([{
        "dataset": "ABC_customerSpending",
        "n_transactions": len(df),
        "n_customers": len(cust),
        "k": 4,
        "inertia": round(km.inertia_, 2),
        "silhouette": round(sil or 0, 4),
    }], "bai_11_metrics.csv")


if __name__ == "__main__":
    main()
