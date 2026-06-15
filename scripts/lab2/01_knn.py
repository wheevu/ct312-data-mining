#!/usr/bin/env python3
"""K‑Nearest Neighbors — lab2-Knn.pdf

Evaluates KNN (k = 1–5) on Breast Cancer, Wine, and Optical Digits
using 5‑fold stratified cross‑validation with StandardScaler.
"""

from common import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ensure_dirs()

datasets = {
    'Breast Cancer':  load_breast_cancer(),
    'Wine':            load_wine(),
    'Optical Digits':  load_optical_digits(),
}

k_values = [1, 2, 3, 4, 5]
results = []

print("=" * 60)
print("K‑Nearest Neighbors — Cross-validation Results")
print("=" * 60)

for dset_name, (X, y, _) in datasets.items():
    print(f"\n── {dset_name} ──")
    for k in k_values:
        pipe = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=k),
        )
        scores = evaluate_cross_val(pipe, X, y)
        mean_acc = scores.mean()
        std_acc = scores.std()
        results.append({
            'dataset': dset_name,
            'k': k,
            'mean_accuracy': f"{mean_acc:.4f}",
            'std_accuracy': f"{std_acc:.4f}",
        })
        print(f"  k={k}:  accuracy = {mean_acc:.4f} ± {std_acc:.4f}")

save_results_csv(results, 'knn_results.csv')

# ── Best k per dataset ──
print("\n── Best k per dataset ──")
for dset_name in datasets:
    subset = [r for r in results if r['dataset'] == dset_name]
    best = max(subset, key=lambda r: float(r['mean_accuracy']))
    print(f"  {dset_name}: best k={best['k']}  "
          f"({best['mean_accuracy']} ± {best['std_accuracy']})")

print("\nDone.")
