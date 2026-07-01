#!/usr/bin/env python3
"""Shared helpers for CT312 Lab 4 clustering exercises."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "lab4"
OUTPUT_DIR = ROOT / "outputs" / "lab4"
PLOTS_DIR = OUTPUT_DIR / "plots"
RESULTS_DIR = OUTPUT_DIR / "results"

DATA_URLS: dict[str, str] = {
    "ABC_Customers.csv": "https://raw.githubusercontent.com/ltdaovn/dataset/master/ABC_Customers.csv",
    "MeanShift-3D.csv": "https://raw.githubusercontent.com/ltdaovn/dataset/master/MeanShift-3D.csv",
    "Eurojobs.csv": "https://raw.githubusercontent.com/ltdaovn/dataset/master/Eurojobs.csv",
    "flowers.csv": "https://raw.githubusercontent.com/ltdaovn/dataset/master/flowers.csv",
    "USArrests.csv": "https://raw.githubusercontent.com/ltdaovn/dataset/master/USArrests.csv",
    "dataCustomerRFM.csv": "https://raw.githubusercontent.com/ltdaovn/dataset/master/dataCustomerRFM.csv",
    "bank-data.csv": "https://raw.githubusercontent.com/ltdaovn/dataset/master/bank-data.csv",
    "ABC_customerSpending.csv": "https://raw.githubusercontent.com/ltdaovn/dataset/master/ABC_customerSpending.csv",
    "moon_dataset.csv": "https://raw.githubusercontent.com/ltdaovn/dataset/master/moon_dataset.csv",
}


def ensure_dirs() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def cache_data(filename: str) -> Path:
    """Download a remote dataset to data/lab4/ if not already cached."""
    path = DATA_DIR / filename
    if path.exists():
        return path
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    url = DATA_URLS.get(filename)
    if url is None:
        raise ValueError(f"No URL for {filename!r}")
    print(f"Downloading {url} -> {path}")
    urllib.request.urlretrieve(url, path)
    return path


def prepare_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    scale: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Impute missing, optionally scale, return (X_scaled, 2D-coords)."""
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)
    if scale:
        X_scaled = StandardScaler().fit_transform(X_imp)
    else:
        X_scaled = X_imp
    n = X_scaled.shape[1]
    if n == 2:
        coords = X_scaled
    elif n == 1:
        coords = np.column_stack([X_scaled[:, 0], np.zeros(len(X_scaled))])
    else:
        coords = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    return X_scaled, coords


def labeled_points(frame: pd.DataFrame, labels: np.ndarray, coords: np.ndarray) -> pd.DataFrame:
    """Return frame with cluster, plot_x, plot_y columns appended."""
    out = frame.copy()
    out["cluster"] = labels.astype(int)
    out["plot_x"] = coords[:, 0]
    out["plot_y"] = coords[:, 1]
    return out


def cluster_summary(labels: np.ndarray, *, true_labels: pd.Series | None = None) -> pd.DataFrame:
    rows = []
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        row: dict[str, object] = {
            "cluster": int(label),
            "n_rows": int(mask.sum()),
            "share_pct": round(float(mask.mean() * 100), 2),
        }
        if true_labels is not None:
            vc = true_labels[mask].astype(str).value_counts()
            if len(vc):
                row["dominant_true"] = vc.index[0]
                row["dominant_count"] = int(vc.iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def safe_silhouette(X: np.ndarray, labels: np.ndarray) -> float | None:
    if len(set(labels.tolist())) < 2:
        return None
    if len(labels) < 3:
        return None
    return float(silhouette_score(X, labels))


def save_metrics(rows: list[dict], filename: str) -> pd.DataFrame:
    ensure_dirs()
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / filename, index=False)
    print(df.to_string(index=False))
    print(f"Saved {RESULTS_DIR / filename}")
    return df


def savefig(filename: str) -> None:
    ensure_dirs()
    path = PLOTS_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def elbow_data(X: np.ndarray, k_max: int = 10, random_state: int = 42) -> pd.DataFrame:
    """Compute inertia for k=1..k_max for elbow plot."""
    from sklearn.cluster import KMeans

    rows = []
    for k in range(1, k_max + 1):
        m = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        m.fit(X)
        rows.append({"k": k, "inertia": float(m.inertia_)})
    return pd.DataFrame(rows)
