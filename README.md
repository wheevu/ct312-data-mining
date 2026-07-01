# CT312 Data Mining

This repo stores weekly notes, experiments, and project work for the CT312 Data Mining course.

Current project direction: Topic 2 — a learning website for KMeans, hierarchical clustering, and DBSCAN. This repo owns the dataset choices, preprocessing, algorithm outputs, and visualization-ready handoff files.

---

## Overview

| Week | Topic / Focus | Project Work | Status |
|------|--------------|--------------|--------|
| 2 | GitHub Issue Mining Setup | Repo structure, data collection, preprocessing notes, dataset evaluation | Complete |
| 6 | Classification Algorithms | KNN, Decision Tree, Naive Bayes, SVM, Ensemble methods on UCI datasets | Complete |
| 7 | Regression | Linear regression, EDA, outlier handling, model evaluation | Complete |
| 7 | Clustering (Lab 4) | KMeans, hierarchical, DBSCAN, MeanShift on 9 real-world datasets | Complete |
| 7 (extra) | Single Linkage practice | Hand-traced single linkage on Italian cities and 2D points, with dendrograms | Practice only |
| Project Topic 2 | Clustering learning website | Curated datasets, preprocessing, algorithm demos, visualization handoff | Active |

---

## Weekly Notes

<details open>
<summary>Project Topic 2 -- Clustering Learning Website</summary>

**Goal.** Support a website that helps students learn KMeans, hierarchical clustering, and DBSCAN. The app teammate handles UI/deployment; this repo provides clean demo datasets and visualization-ready algorithm outputs.

**Selected datasets.** Seeds for KMeans, User Knowledge Modeling for hierarchical clustering, and Spiral for DBSCAN.

**Run.**

```bash
python scripts/topic2/01_prepare_demo_data.py
python scripts/topic2/02_validate_handoff.py
```

**Outputs.** Example CSVs live in `data/topic2/examples/`. The main handoff file is `outputs/topic2/handoff/topic2_visualization_contract.json`; result tables and plots are under `outputs/topic2/results/` and `outputs/topic2/plots/`. See [`topic2_README.md`](./topic2_README.md) and `docs/topic2_dataset_notes.md`.</details>

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
<summary>Topic 6 -- Drug Review Sentiment Analysis and Prediction</summary>

**Goal.** Predict patient satisfaction class from a free-text drug review (UCI Drugs.com dataset, 161k train / 54k test rows, official split).

**Tasks.**
- 3-class satisfaction classification (negative / neutral / positive)
- Numeric rating regression (secondary)
- Review clustering (TF-IDF + SVD + KMeans)
- Deployed prediction app

**Layout.** Source under `src/drug_review_mining/`, scripts under `scripts/topic6/`, outputs under `outputs/topic6/`. See [`topic6_README.md`](./topic6_README.md) for full details.

**Run.**

```bash
python scripts/topic6/01_eda.py
python scripts/topic6/02_classification.py
python scripts/topic6/03_regression.py
python scripts/topic6/04_clustering.py
python scripts/topic6/05_interpretability.py
python scripts/topic6/06_export_app.py
python outputs/topic6/app_assets/predict_demo.py "It works, but the side effects are not worth it."
```

**Notes.** 59.92% of test rows have an exact-review duplicate in train. We evaluate on the no-overlap subset as a more honest measure. Best classifier is selected by no-overlap macro-F1 (Logistic Regression with TF-IDF unigram+bigram). Useful as a research demo, **not** as a clinical decision aid.</details>

<details>
<summary>Week 7 -- Lab 3 Regression</summary>

**Goal.** Practice regression workflows: plotting data, fitting linear models, detecting outliers, reading correlations, evaluating predictions, and comparing linear/polynomial fits.

**Layout.** Data under `data/lab3/`, scripts under `scripts/lab3/`, outputs under `outputs/lab3/`. See [`lab3_README.md`](./lab3_README.md) for details.

**Run.**

```bash
python scripts/lab3/run_all.py
```

**Notes.** Bài 2 shows how one outlier can distort a linear model. Bài 6 uses the selected `x6 + x7` model from the source article. Bài 8 demonstrates why polynomial regression is more appropriate when the relationship is curved.</details>

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

<details>
<summary>Week 7 -- Lab 4 Clustering</summary>

**Goal.** Implement clustering algorithms: KMeans, hierarchical clustering, DBSCAN, and MeanShift on diverse datasets. Practice elbow method, silhouette analysis, manual single-linkage, and RFM segmentation.

**Layout.** Data under `data/lab4/`, scripts under `scripts/lab4/`, outputs under `outputs/lab4/`. See [`lab4_README.md`](./lab4_README.md) for details.

**Exercises.**

| Bài | Algorithm | Dataset | Focus |
|-----|-----------|---------|-------|
| 1 | KMeans | Toy 6-point | Fixed init centroids |
| 2 | KMeans | ABC Customers | Elbow, 3 feature combos |
| 3 | KMeans/Agglo/DBSCAN/MeanShift | ABC Customers | Algorithm comparison |
| 4 | Single linkage | 8 students | Manual merge + dendrogram |
| 5 | MeanShift | 3D synthetic | Bandwidth estimation |
| 6 | Hierarchical | Eurojobs | Sector profile clusters |
| 7 | KMeans | Flowers (Iris-like) | Silhouette analysis |
| 8 | KMeans | USArrests | PCA state plot |
| 9 | KMeans (RFM) | dataCustomerRFM | Customer segmentation |
| 10 | KMeans | bank-data | Categorical encoding |
| 11 | KMeans | ABC_customerSpending | E-commerce segments |
| 12 | DBSCAN vs KMeans | Moon dataset | Non-convex clusters |

**Run.**

```bash
python scripts/lab4/run_all.py
```

**Key findings.**

- **KMeans** is fast and interpretable but assumes spherical clusters and requires k.
- **Hierarchical clustering** provides dendrogram visibility; Ward linkage works well for compact groups.
- **DBSCAN** handles non-convex shapes (moon dataset) but is sensitive to eps.
- **MeanShift** finds clusters without specifying k, but bandwidth tuning is critical.
- **Scaling** (StandardScaler) is essential before any distance-based clustering.
- **Silhouette** favors convex clusters — on moon data KMeans scores higher despite wrong assignments.
- RFM segmentation and customer spending clustering show real-world applications.

</details>

<details>
<summary>Week 7 (extra) -- Single Linkage Hierarchical Clustering (practice only)</summary>

Not part of the lab assignment. Two toy datasets clustered by hand so the merge logic is visible step by step:

- Italian cities (precomputed distance matrix) -- six cities, expected first merge `MI + TO` at 138.
- 2D coordinate points A..F (Euclidean) -- expected first merge `D + F` at 0.5.

For each dataset the script prints the initial distance matrix, every merge with the closest cross-pair, the updated matrix after each merge, and the final merge order. Dendrograms and a full walkthrough are saved to `outputs/week7/`. Notes are in `experiments/week07_single_linkage.md`.

**Run it.**

```bash
python scripts/week7/01_single_linkage_practice.py
```

**Notes.** Manual single linkage matches `scipy.cluster.hierarchy.linkage(..., method="single")` for both datasets. Italian cities: 138, 219, 255, 268, 295. Coordinates: 0.5, 0.707, 1.0, 1.414, 2.5.</details>
