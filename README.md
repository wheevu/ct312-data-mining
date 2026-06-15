# CT312 Data Mining

This repo stores weekly notes, experiments, and project work for the CT312 Data Mining course.

Current project direction: mining GitHub issues and categorizing them using data mining and machine learning techniques.

---

## Overview

| Week | Topic / Focus | Project Work | Status |
|------|--------------|--------------|--------|
| 2 | GitHub Issue Mining Setup | Repo structure, data collection, preprocessing notes, dataset evaluation | Complete |
| 6 | Classification Algorithms | KNN, Decision Tree, Naive Bayes, SVM, Ensemble methods on UCI datasets | Complete |

---

## Weekly Notes

<details>
<summary>Week 2 -- GitHub Issue Classification</summary>

**Goal.** Collect GitHub issues from three repos, normalize labels into categories, build feature vectors, and train classifiers — KNN as the main Week 2 model, plus baselines for comparison.

**Data.** 600 issues (200 each from scikit-learn, pandas, numpy), PRs filtered out. Raw JSON in `data/raw/`, processed CSV in `data/processed/issues_labeled.csv`.

| Metric | Value |
|--------|-------|
| Total issues | 600 |
| Closed / Open | 324 / 276 |
| Issues with labels | 554 |
| Avg labels per issue | 1.8 |
| Avg body length | 2,477 chars |
| Avg comments | 5.1 |
| Date range | Oct 2013 -- May 2026 |

**Targets.** Labels mapped to 5 categories using keyword priority: bug → documentation → feature → question → other. Process labels (Needs Triage, Needs Discussion, priority, stale) ignored. `question` relied on text heuristics since direct labels were rare.

| Category | Issues |
|----------|-------:|
| Bug | 227 |
| Other | 174 |
| Feature | 92 |
| Documentation | 84 |
| Question | 23 |

**Results.**

| Model | Features | Acc | Macro F1 | W. F1 |
|-------|----------|-----|----------|-------|
| Multinomial Naive Bayes | TF-IDF cleaned | 0.700 | 0.538 | 0.674 |
| Logistic Regression | TF-IDF cleaned | 0.683 | 0.543 | 0.674 |
| KNN, k=5, cosine | TF-IDF cleaned | 0.667 | 0.512 | 0.643 |
| KNN, k=7, Minkowski | Manual scaled | 0.658 | 0.506 | 0.629 |

NB and LR tied for best. KNN lags slightly but demonstrates Week 2 concepts (distance metrics, scaling, k selection). `question` had 0 F1 across all models — 23 noisy examples aren't enough. Text cleaning gave a small boost. Scaling manual features made a huge difference for KNN (0.399 → 0.629).

**Lessons.** Labels need preprocessing. Text must be vectorized. KNN needs scaling — features with larger ranges dominate distances otherwise. Accuracy hides class imbalance; weighted F1 is more informative. Always compare multiple algorithms.

**Run it.**

```bash
python scripts/build_dataset.py
python scripts/run_knn.py
python scripts/compare_models.py
```

**Notes.** Small dataset, noisy labels, repo-specific conventions. The professor's reminder: compare algorithms and justify your choice, don't just report numbers.</details>

<details>
<summary>Week 6 -- Classification Algorithms</summary>

**Goal.** Implement and evaluate 5 classification algorithms on 4 standard UCI datasets using cross-validation. Compare performance and understand each algorithm's behavior.

**Data.**

| Dataset | Instances | Features | Classes |
|---|---|---|---|
| Iris | 150 | 4 | 3 |
| Wine | 178 | 13 | 3 |
| Breast Cancer (WDBC) | 569 | 30 | 2 |
| Optical Digits | 5620 | 64 | 10 |

**Algorithms & best accuracy.**

| Algorithm | Breast Cancer | Wine | Optical Digits | Notebook/File |
|---|---|---|---|---|
| KNN | 0.9666 (k=3) | 0.9717 (k=5) | 0.9792 (k=1) | `scripts/lab2/01_knn.py` |
| Decision Tree | 0.9384 (entropy, s=2) | 0.9210 (entropy, s=20) | 0.9050 (entropy, s=2) | `scripts/lab2/02_decision_tree.py` |
| Gaussian NB | 0.9357 | 1.0000 | 0.7942 | `scripts/lab2/03_naive_bayes.py` |
| SVM | 0.9807 (rbf, C=10, γ=0.01) | 0.9887 (rbf, C=1, γ=0.01) | 0.9895 (rbf, C=10, γ=0.01) | `scripts/lab2/04_svm.py` |
| Bagging (50) | 0.9578 | 0.9551 | 0.9619 | `scripts/lab2/05_ensemble.py` |
| AdaBoost (d=3) | 0.9684 | 0.9605 (d=2) | 0.9331 (d=3) | `scripts/lab2/05_ensemble.py` |
| Random Forest (100) | 0.9543 | 0.9775 | 0.9843 | `scripts/lab2/05_ensemble.py` |

**Key findings.**

- **SVM with RBF kernel** was the strongest overall — consistently best or near-best across all 3 larger datasets.
- **Naive Bayes** hit 100% on Wine (separability) but struggled with Optical Digits (features aren't independent).
- **Decision Tree** performance improved with entropy over gini on most datasets.
- **AdaBoost** depth mattered more for complex data: Optical Digits needed depth 3, while Wine peaked at depth 2.
- **Random Forest** matched SVM on digits without hyperparameter tuning — strong default choice.
- Scaling (StandardScaler) was essential for KNN and SVM.

**Run it.**

```bash
python scripts/lab2/01_knn.py
python scripts/lab2/02_decision_tree.py
python scripts/lab2/03_naive_bayes.py
python scripts/lab2/04_svm.py
python scripts/lab2/05_ensemble.py
```

**Datasets** in `data/lab2/`, outputs (results CSVs, plots) in `outputs/lab2/`.

**Lessons.** No single algorithm dominates — choice depends on data size, feature independence, and scaling requirements. Ensemble methods (RF, Bagging) provide robustness with minimal tuning. SVM is worth the tuning effort for clean, well-scaled data.</details>
