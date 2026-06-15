#!/usr/bin/env python3
"""Decision Tree — lab2-dt.pdf

Evaluates DecisionTreeClassifier with varying criterion and
min_samples_split on Breast Cancer, Wine, and Optical Digits
using 5‑fold stratified cross‑validation.

Also exports a sample tree plot for Wine (depth‑limited for readability).
"""

from common import *
from sklearn.tree import DecisionTreeClassifier, plot_tree

ensure_dirs()

datasets = {
    'Breast Cancer':  load_breast_cancer(),
    'Wine':            load_wine(),
    'Optical Digits':  load_optical_digits(),
}

criterions = ['gini', 'entropy']
min_samples_splits = [2, 5, 10, 20]
results = []

print("=" * 60)
print("Decision Tree — Cross-validation Results")
print("=" * 60)

for dset_name, (X, y, _) in datasets.items():
    print(f"\n── {dset_name} ──")
    for criterion in criterions:
        for mss in min_samples_splits:
            dt = DecisionTreeClassifier(
                criterion=criterion,
                min_samples_split=mss,
                random_state=42,
            )
            scores = evaluate_cross_val(dt, X, y)
            mean_acc = scores.mean()
            std_acc = scores.std()
            results.append({
                'dataset': dset_name,
                'criterion': criterion,
                'min_samples_split': mss,
                'mean_accuracy': f"{mean_acc:.4f}",
                'std_accuracy': f"{std_acc:.4f}",
            })
            print(f"  criterion={criterion:7s}  "
                  f"min_samples_split={mss:2d}  →  "
                  f"{mean_acc:.4f} ± {std_acc:.4f}")

# ── Sample tree plot (Wine, depth limited) ──
print("\n── Plotting sample decision tree (Wine) ──")
X_wine, y_wine, _ = load_wine()
dt_viz = DecisionTreeClassifier(
    criterion='gini', min_samples_split=10,
    max_depth=3, random_state=42,
)
dt_viz.fit(X_wine, y_wine)

fig, ax = plt.subplots(figsize=(14, 8))
plot_tree(
    dt_viz,
    feature_names=[f'f{i+1}' for i in range(13)],
    class_names=['1', '2', '3'],
    filled=True, ax=ax, fontsize=9, rounded=True,
)
ax.set_title("Decision Tree (Wine, max_depth=3 for readability)", fontsize=14)
plt.savefig(os.path.join(PLOTS_DIR, 'decision_tree_wine_sample.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  → Saved decision_tree_wine_sample.png")

save_results_csv(results, 'decision_tree_results.csv')
print("\nDone.")
