# Week 6 — Classification Algorithms

Evaluated 5 algorithm families on 4 UCI datasets. Standard protocol: 5-fold stratified cross-validation, accuracy ± std.

## Datasets

| Dataset | Instances | Features | Classes | Source |
|---|---|---|---|---|
| Iris | 150 | 4 | 3 | `data/lab2/iris/` |
| Wine | 178 | 13 | 3 | `data/lab2/wine/` |
| Breast Cancer (WDBC) | 569 | 30 | 2 | `data/lab2/breast+cancer+wisconsin+diagnostic/` |
| Optical Digits | 5620 | 64 | 10 | `data/lab2/optical+recognition+of+handwritten+digits/` |

## Results Summary

### KNN (k = 1–5, scaled)

| Dataset | Best k | Accuracy |
|---|---|---|
| Breast Cancer | 3 | 0.9666 ± 0.0171 |
| Wine | 5 | 0.9717 ± 0.0181 |
| Optical Digits | 1 | 0.9792 ± 0.0029 |

- Scaling via `StandardScaler` is required — KNN is distance-based.
- Small k (1) works best for digits (many classes, large data).
- Larger k (5) helps Wine (smaller, cleaner separation).

### Decision Tree (criterion × min_samples_split)

| Dataset | Best config | Accuracy |
|---|---|---|
| Breast Cancer | entropy, split=2 | 0.9384 ± 0.0217 |
| Wine | entropy, split=20 | 0.9210 ± 0.0489 |
| Optical Digits | entropy, split=2 | 0.9050 ± 0.0112 |

- Entropy consistently outperformed gini.
- Trees overfit on Breast Cancer at small split values — larger splits help slightly.
- Maximum depth limit not tuned; full trees produce readable plots (see `outputs/lab2/plots/decision_tree_wine_sample.png`).

### Gaussian Naive Bayes

| Dataset | Accuracy (70/30 holdout) |
|---|---|
| Breast Cancer | 0.9357 |
| Wine | 1.0000 |
| Optical Digits | 0.7942 |

- Perfect score on Wine — features are conditionally independent enough.
- Poor on digits (64 pixel features are correlated) — NB's independence assumption hurts.
- Iris single-sample prediction correctly classified `[5.0, 3.5, 1.5, 0.2]` as Iris-setosa (p = 1.0).
- Iris 10-run holdout average: 0.9489.

### SVM (linear/rbf/poly kernels, scaled)

| Dataset | Best config | Accuracy |
|---|---|---|
| Breast Cancer | rbf, C=10, γ=0.01 | 0.9807 ± 0.0179 |
| Wine | rbf, C=1, γ=0.01 | 0.9887 ± 0.0138 |
| Optical Digits | rbf, C=10, γ=0.01 | 0.9895 ± 0.0009 |

- SVM + RBF kernel is the strongest overall model across all datasets.
- Poly kernel underperforms at low C=1.
- Scaling is critical.

### Ensemble Methods

#### Bagging (DecisionTree × 50)

| Dataset | Accuracy |
|---|---|
| Breast Cancer | 0.9578 ± 0.0187 |
| Wine | 0.9551 ± 0.0339 |
| Optical Digits | 0.9619 ± 0.0087 |

#### AdaBoost (DecisionTree, depths 1–3, 50 estimators)

| Dataset | Best depth | Accuracy |
|---|---|---|
| Breast Cancer | 3 | 0.9684 ± 0.0181 |
| Wine | 2 | 0.9605 ± 0.0427 |
| Optical Digits | 3 | 0.9331 ± 0.0060 |

- Shallower trees (depth 1) underfit digits (0.79) but work decently on smaller datasets.
- Depth 2–3 is the sweet spot.

#### Random Forest (100 trees)

| Dataset | Accuracy |
|---|---|
| Breast Cancer | 0.9543 ± 0.0128 |
| Wine | 0.9775 ± 0.0213 |
| Optical Digits | 0.9843 ± 0.0059 |

- Strong default choice — near SVM performance without tuning.
- Feature importance plots saved in `outputs/lab2/plots/`.

## Key Takeaways

1. **SVM (RBF)** is the best single algorithm when data is scaled.
2. **Random Forest** is the best default — robust, no scaling needed, built-in importance.
3. **Naive Bayes** only wins if features are independent (Wine). Fails on correlated pixel data.
4. **KNN** needs careful k selection and always benefits from scaling.
5. **Decision Trees** are interpretable but weaker alone — ensemble methods fix this.

## Files

- `scripts/lab2/` — all 5 assignment scripts + shared `common.py`
- `data/lab2/` — UCI datasets
- `outputs/lab2/results/` — CSV tables for every experiment
- `outputs/lab2/plots/` — confusion matrices, feature importances, tree diagram
