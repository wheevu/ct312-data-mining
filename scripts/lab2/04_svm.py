#!/usr/bin/env python3
"""Support Vector Machine — ThucHanh-SVM.pdf

Part 1: Iris with 5‑fold CV (linear kernel).
Part 2: Breast Cancer, Wine, Optical Digits with linear / rbf / poly
        kernels and varying hyperparameters.
"""

from common import *
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ensure_dirs()

# ── Part 1: Iris example with 5-fold CV ─────────────────────────
print("=" * 60)
print("Part 1: SVM — Iris with 5-fold Cross-validation")
print("=" * 60)

X_iris, y_iris, iris_classes = load_iris()
pipe_iris = make_pipeline(
    StandardScaler(),
    SVC(kernel='linear', C=1, random_state=42),
)
scores_iris = evaluate_cross_val(pipe_iris, X_iris, y_iris)
print(f"\nSVC(kernel=linear, C=1) on Iris:")
print(f"  Accuracy: {scores_iris.mean():.4f} ± {scores_iris.std():.4f}")

# ── Part 2: grid search on larger datasets ──────────────────────
print("\n" + "=" * 60)
print("Part 2: SVM — Grid Search (5-fold CV)")
print("=" * 60)

datasets = {
    'Breast Cancer':  load_breast_cancer(),
    'Wine':            load_wine(),
    'Optical Digits':  load_optical_digits(),
}

# (kernel, param_dict) pairs — explicit, easy to read
param_combos = [
    # linear
    {'kernel': 'linear', 'C': 0.1},
    {'kernel': 'linear', 'C': 1},
    {'kernel': 'linear', 'C': 10},
    # rbf
    {'kernel': 'rbf', 'C': 1,   'gamma': 'scale'},
    {'kernel': 'rbf', 'C': 1,   'gamma': 0.01},
    {'kernel': 'rbf', 'C': 1,   'gamma': 0.1},
    {'kernel': 'rbf', 'C': 10,  'gamma': 'scale'},
    {'kernel': 'rbf', 'C': 10,  'gamma': 0.01},
    {'kernel': 'rbf', 'C': 10,  'gamma': 0.1},
    # poly
    {'kernel': 'poly', 'degree': 2, 'C': 1},
    {'kernel': 'poly', 'degree': 3, 'C': 1},
]

results = []

for dset_name, (X, y, _) in datasets.items():
    print(f"\n── {dset_name} ──")
    for kwargs in param_combos:
        pipe = make_pipeline(
            StandardScaler(),
            SVC(**kwargs, random_state=42),
        )
        scores = evaluate_cross_val(pipe, X, y)
        mean_acc = scores.mean()
        std_acc = scores.std()

        param_str = ', '.join(f'{k}={v}' for k, v in kwargs.items())
        results.append({
            'dataset': dset_name,
            'params': param_str,
            'mean_accuracy': f"{mean_acc:.4f}",
            'std_accuracy': f"{std_acc:.4f}",
        })
        print(f"  {param_str:45s}  →  {mean_acc:.4f} ± {std_acc:.4f}")

save_results_csv(results, 'svm_results.csv')

# ── Best configuration per dataset ──
print("\n── Best configuration per dataset ──")
for dset_name in datasets:
    subset = [r for r in results if r['dataset'] == dset_name]
    best = max(subset, key=lambda r: float(r['mean_accuracy']))
    print(f"  {dset_name}:  {best['params']}  "
          f"({best['mean_accuracy']} ± {best['std_accuracy']})")

print("\nDone.")
