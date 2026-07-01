# Lab 4 — Clustering

Clustering exercises for CT312 Week 7+.

## Layout

```text
data/lab4/       input CSV files (downloaded on first run)
scripts/lab4/    one script per exercise + shared helpers
outputs/lab4/    generated plots and result CSV files
```

## Run

From the repository root:

```bash
python scripts/lab4/run_all.py
```

Or run a single exercise:

```bash
python scripts/lab4/03_abc_customers_hierarchical_compare.py
```

## Exercises

| Bài | Script | Focus |
|-----|--------|-------|
| 1 | `01_kmeans_toy.py` | KMeans toy example with custom init centroids |
| 2 | `02_abc_customers_kmeans.py` | ABC customers — elbow, income/spending, age/spending, 3D KMeans |
| 3 | `03_abc_customers_hierarchical_compare.py` | Dendrogram, agglomerative, 4-algo comparison |
| 4 | `04_student_single_linkage.py` | Manual single-linkage for 8 students |
| 5 | `05_meanshift_3d.py` | MeanShift on 3D synthetic data |
| 6 | `06_eurojobs.py` | EuroJobs hierarchical clustering, sector profiles |
| 7 | `07_flowers.py` | Flowers (Iris-like) KMeans, silhouette analysis |
| 8 | `08_usarrests.py` | USArrests KMeans with state labels |
| 9 | `09_rfm.py` | RFM customer segmentation from order data |
| 10 | `10_bank.py` | Bank customer clustering with categorical encoding |
| 11 | `11_ecommerce_spending.py` | E-commerce customer spending segments |
| 12 | `12_moon.py` | DBSCAN vs KMeans on non-convex moon data |

## Datasets

| File | Source | Rows | Used in |
|------|--------|------|---------|
| ABC_Customers.csv | ltdaovn/dataset | 200 | Bài 2, 3 |
| MeanShift-3D.csv | ltdaovn/dataset | 120 | Bài 5 |
| Eurojobs.csv | ltdaovn/dataset | 26 | Bài 6 |
| flowers.csv | ltdaovn/dataset | 150 | Bài 7 |
| USArrests.csv | ltdaovn/dataset | 50 | Bài 8 |
| dataCustomerRFM.csv | ltdaovn/dataset | 303k | Bài 9 |
| bank-data.csv | ltdaovn/dataset | 600 | Bài 10 |
| ABC_customerSpending.csv | ltdaovn/dataset | 92k | Bài 11 |
| moon_dataset.csv | ltdaovn/dataset | 200 | Bài 12 |
