#!/usr/bin/env python3
"""Ensemble Methods — lab2-Ens.pdf

  5.1 Bagging (DecisionTree × 50)
  5.2 AdaBoost (DecisionTree, depth 1–3, 50 estimators)
  5.3 Random Forest (100 trees) with feature‑importance plots
"""

from common import *
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

ensure_dirs()

# sklearn >= 1.8: use `estimator=` (not deprecated `base_estimator=`)

datasets = {
    'Breast Cancer':  load_breast_cancer(),
    'Wine':            load_wine(),
    'Optical Digits':  load_optical_digits(),
}

# ── 5.1  Bagging ────────────────────────────────────────────────
print("=" * 60)
print("5.1  Bagging (DecisionTree × 50)")
print("=" * 60)

bagging_results = []

for dset_name, (X, y, _) in datasets.items():
    print(f"\n── {dset_name} ──")
    bag = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=42),
        n_estimators=50,
        random_state=42,
        n_jobs=-1,
    )
    scores = evaluate_cross_val(bag, X, y)
    mean_acc = scores.mean()
    std_acc = scores.std()
    bagging_results.append({
        'dataset': dset_name,
        'n_estimators': 50,
        'mean_accuracy': f"{mean_acc:.4f}",
        'std_accuracy': f"{std_acc:.4f}",
    })
    print(f"  Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

save_results_csv(bagging_results, 'bagging_results.csv')

# ── 5.2  AdaBoost ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("5.2  AdaBoost (DecisionTree × 50, depths 1–3)")
print("=" * 60)

adaboost_results = []
depths = [1, 2, 3]

for dset_name, (X, y, _) in datasets.items():
    print(f"\n── {dset_name} ──")
    for depth in depths:
        ada = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=depth, random_state=42),
            n_estimators=50,
            random_state=42,
        )
        scores = evaluate_cross_val(ada, X, y)
        mean_acc = scores.mean()
        std_acc = scores.std()
        adaboost_results.append({
            'dataset': dset_name,
            'max_depth': depth,
            'n_estimators': 50,
            'mean_accuracy': f"{mean_acc:.4f}",
            'std_accuracy': f"{std_acc:.4f}",
        })
        print(f"  max_depth={depth}:  {mean_acc:.4f} ± {std_acc:.4f}")

save_results_csv(adaboost_results, 'adaboost_results.csv')

# ── 5.3  Random Forest ──────────────────────────────────────────
print("\n" + "=" * 60)
print("5.3  Random Forest (100 trees, feature importance)")
print("=" * 60)

rf_results = []

for dset_name, (X, y, _) in datasets.items():
    print(f"\n── {dset_name} ──")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    scores = evaluate_cross_val(rf, X, y)
    mean_acc = scores.mean()
    std_acc = scores.std()
    rf_results.append({
        'dataset': dset_name,
        'n_estimators': 100,
        'mean_accuracy': f"{mean_acc:.4f}",
        'std_accuracy': f"{std_acc:.4f}",
    })
    print(f"  CV accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

    # Feature importance (fit on full data)
    rf.fit(X, y)
    importances = rf.feature_importances_
    feature_names = [f'f{i+1}' for i in range(X.shape[1])]

    if dset_name == 'Optical Digits':
        save_feature_importance(
            importances, feature_names,
            f"random_forest_importance_{dset_name.lower().replace(' ', '_')}_top20.png",
            title=f"Random Forest — {dset_name} (Top 20)",
            top_n=20,
        )
    else:
        save_feature_importance(
            importances, feature_names,
            f"random_forest_importance_{dset_name.lower().replace(' ', '_')}.png",
            title=f"Random Forest — {dset_name}",
        )

save_results_csv(rf_results, 'random_forest_results.csv')
print("\nDone.")
