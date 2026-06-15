#!/usr/bin/env python3
"""Naive Bayes — lab2-naiveBayes.pdf

Three parts:
  1. Iris single‑sample prediction with class probabilities.
  2. Iris 10‑run hold‑out (70/30 stratified) — average accuracy.
  3. Breast Cancer, Wine, Optical Digits hold‑out with confusion matrices.
"""

from common import *
from sklearn.naive_bayes import GaussianNB
import json

ensure_dirs()

# ── Part 1: Iris single‑sample prediction ───────────────────────
print("=" * 60)
print("Part 1: Iris — Single Sample Prediction")
print("=" * 60)

X_iris, y_iris, iris_classes = load_iris()
nb_iris = GaussianNB()
nb_iris.fit(X_iris, y_iris)

sample = np.array([[5.0, 3.5, 1.5, 0.2]])
pred = nb_iris.predict(sample)[0]
probs = nb_iris.predict_proba(sample)[0]

print(f"\nInput features: [5.0, 3.5, 1.5, 0.2]")
print(f"Predicted class: {iris_classes[pred]}  (encoded = {pred})")
print("Class probabilities:")
for i, p in enumerate(probs):
    print(f"  {iris_classes[i]:12s}  {p:.6f}")

# ── Part 2: Iris 10‑run hold-out ────────────────────────────────
print("\n" + "=" * 60)
print("Part 2: Iris — 10-Run Hold-out (70/30 stratified)")
print("=" * 60)

ten_accs = []
for seed in range(10):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_iris, y_iris, test_size=0.3, stratify=y_iris, random_state=seed,
    )
    nb = GaussianNB()
    nb.fit(X_tr, y_tr)
    acc = nb.score(X_te, y_te)
    ten_accs.append(acc)
    print(f"  Run {seed+1:2d}:  accuracy = {acc:.4f}")

avg_acc = np.mean(ten_accs)
print(f"\n  Average accuracy over 10 runs: {avg_acc:.4f}")

# ── Part 3: larger datasets with confusion matrices ─────────────
print("\n" + "=" * 60)
print("Part 3: Naive Bayes — Hold-out with Confusion Matrices")
print("=" * 60)

datasets = {
    'Breast Cancer':  load_breast_cancer(),
    'Wine':            load_wine(),
    'Optical Digits':  load_optical_digits(),
}

results = []

for dset_name, (X, y, classes) in datasets.items():
    print(f"\n── {dset_name} ──")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42,
    )
    nb = GaussianNB()
    nb.fit(X_tr, y_tr)
    y_pred = nb.predict(X_te)
    acc = nb.score(X_te, y_te)

    print(f"  Accuracy: {acc:.4f}")
    print(f"  Classification report:")
    print(classification_report(y_te, y_pred, target_names=classes,
                                zero_division=0, digits=4))

    results.append({
        'dataset': dset_name,
        'accuracy': f"{acc:.4f}",
    })

    fname = f"naive_bayes_confusion_{dset_name.lower().replace(' ', '_')}.png"
    save_confusion_matrix(y_te, y_pred, classes, fname,
                          title=f"Naive Bayes — {dset_name}")

save_results_csv(results, 'naive_bayes_results.csv')

# ── Save Iris detail as JSON ──
iris_detail = {
    'single_sample_input': [5.0, 3.5, 1.5, 0.2],
    'predicted_class': iris_classes[int(pred)],
    'predicted_class_encoded': int(pred),
    'probabilities': {iris_classes[i]: float(p) for i, p in enumerate(probs)},
    'ten_run_accuracies': [float(a) for a in ten_accs],
    'ten_run_average': float(avg_acc),
}
iris_path = os.path.join(RESULTS_DIR, 'naive_bayes_iris_details.json')
with open(iris_path, 'w') as f:
    json.dump(iris_detail, f, indent=2)
print(f"  → Saved {iris_path}")

print("\nDone.")
