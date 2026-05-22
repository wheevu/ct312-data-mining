# CT312 Data Mining

This repo stores weekly notes, experiments, and project work for the CT312 Data Mining course.

Current project direction: mining GitHub issues and categorizing them using data mining and machine learning techniques.

---

## Overview

| Week | Topic / Focus | Project Work | Status |
|------|--------------|--------------|--------|
| 2 | GitHub Issue Mining Setup | Repo structure, data collection, preprocessing notes, dataset evaluation | Complete |

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
