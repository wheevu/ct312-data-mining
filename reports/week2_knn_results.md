# Week 2 KNN Results

## Class distribution before modeling
| target | count |
| --- | --- |
| bug | 227.000 |
| other | 174.000 |
| feature | 92.000 |
| documentation | 84.000 |
| question | 23.000 |

## Summary scores
| model | feature_type | accuracy | macro_f1 | weighted_f1 | notes | k | distance_metric |
| --- | --- | --- | --- | --- | --- | --- | --- |
| KNN | tfidf | 0.667 | 0.512 | 0.643 | TF-IDF text vectors (cleaned) | 5.000 | cosine |
| KNN | tfidf | 0.658 | 0.516 | 0.636 | TF-IDF text vectors (cleaned) | 7.000 | cosine |
| KNN | manual_scaled | 0.658 | 0.506 | 0.629 | StandardScaler used | 7.000 | minkowski |
| KNN | manual_scaled | 0.658 | 0.506 | 0.629 | StandardScaler used | 7.000 | euclidean |
| KNN | manual_scaled | 0.658 | 0.502 | 0.626 | StandardScaler used | 5.000 | euclidean |
| KNN | manual_scaled | 0.658 | 0.502 | 0.626 | StandardScaler used | 5.000 | minkowski |
| KNN | manual_scaled | 0.650 | 0.504 | 0.623 | StandardScaler used | 5.000 | manhattan |
| KNN | manual_scaled | 0.642 | 0.516 | 0.622 | StandardScaler used | 3.000 | manhattan |
| KNN | tfidf | 0.642 | 0.484 | 0.617 | TF-IDF text vectors (cleaned) | 3.000 | cosine |
| KNN | manual_scaled | 0.642 | 0.490 | 0.610 | StandardScaler used | 7.000 | manhattan |
| KNN | manual_scaled | 0.625 | 0.487 | 0.595 | StandardScaler used | 3.000 | euclidean |
| KNN | manual_scaled | 0.625 | 0.487 | 0.595 | StandardScaler used | 3.000 | minkowski |
| KNN | manual_unscaled | 0.425 | 0.282 | 0.399 | No scaling | 5.000 | manhattan |
| KNN | manual_unscaled | 0.417 | 0.271 | 0.386 | No scaling | 7.000 | manhattan |
| KNN | manual_unscaled | 0.433 | 0.259 | 0.382 | No scaling | 3.000 | manhattan |
| KNN | manual_unscaled | 0.392 | 0.259 | 0.364 | No scaling | 5.000 | minkowski |
| KNN | manual_unscaled | 0.392 | 0.259 | 0.364 | No scaling | 5.000 | euclidean |
| KNN | manual_unscaled | 0.408 | 0.236 | 0.358 | No scaling | 3.000 | minkowski |
| KNN | manual_unscaled | 0.408 | 0.236 | 0.358 | No scaling | 3.000 | euclidean |
| KNN | manual_unscaled | 0.392 | 0.237 | 0.355 | No scaling | 7.000 | minkowski |
| KNN | manual_unscaled | 0.392 | 0.237 | 0.355 | No scaling | 7.000 | euclidean |

## Scaling demonstration
KNN uses distances between vectors. Without scaling, large-range features such as body_length and text_length can dominate smaller binary features. StandardScaler was applied only to input features, never to the target labels.

## GridSearchCV
Best weighted F1 CV score: 0.601; best params: {'model__metric': 'manhattan', 'model__n_neighbors': 9, 'model__weights': 'distance'}. GridSearch tests KNN settings systematically.

## Detailed experiment outputs

### KNN (manual_unscaled)
- k: 3
- metric: euclidean
- accuracy: 0.408
- macro F1: 0.236
- weighted F1: 0.358

```text
               precision    recall  f1-score   support

          bug       0.48      0.78      0.59        45
documentation       0.08      0.06      0.07        17
      feature       0.50      0.11      0.18        18
        other       0.37      0.31      0.34        35
     question       0.00      0.00      0.00         5

     accuracy                           0.41       120
    macro avg       0.28      0.25      0.24       120
 weighted avg       0.37      0.41      0.36       120

```

Confusion matrix labels: bug, documentation, feature, other, question

```text
[[35  3  0  7  0]
 [ 6  1  1  9  0]
 [11  4  2  1  0]
 [18  5  1 11  0]
 [ 3  0  0  2  0]]
```

### KNN (manual_unscaled)
- k: 3
- metric: manhattan
- accuracy: 0.433
- macro F1: 0.259
- weighted F1: 0.382

```text
               precision    recall  f1-score   support

          bug       0.49      0.80      0.61        45
documentation       0.15      0.12      0.13        17
      feature       0.50      0.11      0.18        18
        other       0.41      0.34      0.38        35
     question       0.00      0.00      0.00         5

     accuracy                           0.43       120
    macro avg       0.31      0.27      0.26       120
 weighted avg       0.40      0.43      0.38       120

```

Confusion matrix labels: bug, documentation, feature, other, question

```text
[[36  3  0  6  0]
 [ 6  2  1  8  0]
 [11  4  2  1  0]
 [18  4  1 12  0]
 [ 3  0  0  2  0]]
```

### KNN (manual_unscaled)
- k: 3
- metric: minkowski
- accuracy: 0.408
- macro F1: 0.236
- weighted F1: 0.358

```text
               precision    recall  f1-score   support

          bug       0.48      0.78      0.59        45
documentation       0.08      0.06      0.07        17
      feature       0.50      0.11      0.18        18
        other       0.37      0.31      0.34        35
     question       0.00      0.00      0.00         5

     accuracy                           0.41       120
    macro avg       0.28      0.25      0.24       120
 weighted avg       0.37      0.41      0.36       120

```

Confusion matrix labels: bug, documentation, feature, other, question

```text
[[35  3  0  7  0]
 [ 6  1  1  9  0]
 [11  4  2  1  0]
 [18  5  1 11  0]
 [ 3  0  0  2  0]]
```

### KNN (manual_unscaled)
- k: 5
- metric: euclidean
- accuracy: 0.392
- macro F1: 0.259
- weighted F1: 0.364

```text
               precision    recall  f1-score   support

          bug       0.50      0.69      0.58        45
documentation       0.18      0.18      0.18        17
      feature       0.29      0.22      0.25        18
        other       0.33      0.26      0.29        35
     question       0.00      0.00      0.00         5

     accuracy                           0.39       120
    macro avg       0.26      0.27      0.26       120
 weighted avg       0.35      0.39      0.36       120

```

Confusion matrix labels: bug, documentation, feature, other, question

```text
[[31  3  2  9  0]
 [ 5  3  2  7  0]
 [ 9  3  4  2  0]
 [14  7  5  9  0]
 [ 3  1  1  0  0]]
```

### KNN (manual_unscaled)
- k: 5
- metric: manhattan
- accuracy: 0.425
- macro F1: 0.282
- weighted F1: 0.399

```text
               precision    recall  f1-score   support

          bug       0.52      0.71      0.60        45
documentation       0.20      0.18      0.19        17
      feature       0.25      0.22      0.24        18
        other       0.43      0.34      0.38        35
     question       0.00      0.00      0.00         5

     accuracy                           0.42       120
    macro avg       0.28      0.29      0.28       120
 weighted avg       0.39      0.42      0.40       120

```

Confusion matrix labels: bug, documentation, feature, other, question

```text
[[32  2  4  7  0]
 [ 4  3  3  7  0]
 [10  2  4  2  0]
 [13  6  4 12  0]
 [ 2  2  1  0  0]]
```

### KNN (manual_unscaled)
- k: 5
- metric: minkowski
- accuracy: 0.392
- macro F1: 0.259
- weighted F1: 0.364

```text
               precision    recall  f1-score   support

          bug       0.50      0.69      0.58        45
documentation       0.18      0.18      0.18        17
      feature       0.29      0.22      0.25        18
        other       0.33      0.26      0.29        35
     question       0.00      0.00      0.00         5

     accuracy                           0.39       120
    macro avg       0.26      0.27      0.26       120
 weighted avg       0.35      0.39      0.36       120

```

Confusion matrix labels: bug, documentation, feature, other, question

```text
[[31  3  2  9  0]
 [ 5  3  2  7  0]
 [ 9  3  4  2  0]
 [14  7  5  9  0]
 [ 3  1  1  0  0]]
```

### KNN (manual_unscaled)
- k: 7
- metric: euclidean
- accuracy: 0.392
- macro F1: 0.237
- weighted F1: 0.355

```text
               precision    recall  f1-score   support

          bug       0.48      0.71      0.57        45
documentation       0.21      0.18      0.19        17
      feature       0.08      0.06      0.07        18
        other       0.41      0.31      0.35        35
     question       0.00      0.00      0.00         5

     accuracy                           0.39       120
    macro avg       0.24      0.25      0.24       120
 weighted avg       0.34      0.39      0.36       120

```

Confusion matrix labels: bug, documentation, feature, other, question

```text
[[32  2  3  8  0]
 [ 7  3  3  4  0]
 [12  2  1  3  0]
 [13  7  4 11  0]
 [ 3  0  1  1  0]]
```

### KNN (manual_unscaled)
- k: 7
- metric: manhattan
- accuracy: 0.417
- macro F1: 0.271
- weighted F1: 0.386

```text
               precision    recall  f1-score   support

          bug       0.48      0.69      0.57        45
documentation       0.27      0.24      0.25        17
      feature       0.18      0.11      0.14        18
        other       0.43      0.37      0.40        35
     question       0.00      0.00      0.00         5

     accuracy                           0.42       120
    macro avg       0.27      0.28      0.27       120
 weighted avg       0.37      0.42      0.39       120

```

Confusion matrix labels: bug, documentation, feature, other, question

```text
[[31  3  3  8  0]
 [ 6  4  3  4  0]
 [11  1  2  4  0]
 [13  7  2 13  0]
 [ 3  0  1  1  0]]
```

### KNN (manual_unscaled)
- k: 7
- metric: minkowski
- accuracy: 0.392
- macro F1: 0.237
- weighted F1: 0.355

```text
               precision    recall  f1-score   support

          bug       0.48      0.71      0.57        45
documentation       0.21      0.18      0.19        17
      feature       0.08      0.06      0.07        18
        other       0.41      0.31      0.35        35
     question       0.00      0.00      0.00         5

     accuracy                           0.39       120
    macro avg       0.24      0.25      0.24       120
 weighted avg       0.34      0.39      0.36       120

```

Confusion matrix labels: bug, documentation, feature, other, question

```text
[[32  2  3  8  0]
 [ 7  3  3  4  0]
 [12  2  1  3  0]
 [13  7  4 11  0]
 [ 3  0  1  1  0]]
```

### KNN (manual_scaled)
- k: 3
- metric: euclidean
- accuracy: 0.625
- macro F1: 0.487
- weighted F1: 0.595

```text
               precision    recall  f1-score   support

          bug       0.61      0.87      0.72        45
documentation       0.70      0.82      0.76        17
      feature       0.50      0.39      0.44        18
        other       0.68      0.43      0.53        35
     question       0.00      0.00      0.00         5

     accuracy                           0.62       120
    macro avg       0.50      0.50      0.49       120
 weighted avg       0.60      0.62      0.59       120

```

Confusion matrix labels: bug, documentation, feature, other, question

```text
[[39  2  2  2  0]
 [ 3 14  0  0  0]
 [ 6  2  7  3  0]
 [13  2  5 15  0]
 [ 3  0  0  2  0]]
```

### KNN (manual_scaled)
- k: 3
- metric: manhattan
- accuracy: 0.642
- macro F1: 0.516
- weighted F1: 0.622

```text
               precision    recall  f1-score   support

          bug       0.64      0.87      0.74        45
documentation       0.74      0.82      0.78        17
      feature       0.56      0.50      0.53        18
        other       0.71      0.43      0.54        35
     question       0.00      0.00      0.00         5

     accuracy                           0.64       120
    macro avg       0.53      0.52      0.52       120
 weighted avg       0.64      0.64      0.62       120

```

Confusion matrix labels: bug, documentation, feature, other, question

```text
[[39  0  2  3  1]
 [ 3 14  0  0  0]
 [ 5  2  9  2  0]
 [10  3  5 15  2]
 [ 4  0  0  1  0]]
```

### KNN (manual_scaled)
- k: 3
- metric: minkowski
- accuracy: 0.625
- macro F1: 0.487
- weighted F1: 0.595

```text
               precision    recall  f1-score   support

          bug       0.61      0.87      0.72        45
documentation       0.70      0.82      0.76        17
      feature       0.50      0.39      0.44        18
        other       0.68      0.43      0.53        35
     question       0.00      0.00      0.00         5

     accuracy                           0.62       120
    macro avg       0.50      0.50      0.49       120
 weighted avg       0.60      0.62      0.59       120

```

Confusion matrix labels: bug, documentation, feature, other, question

```text
[[39  2  2  2  0]
 [ 3 14  0  0  0]
 [ 6  2  7  3  0]
 [13  2  5 15  0]
 [ 3  0  0  2  0]]
```

### KNN (manual_scaled)
- k: 5
- metric: euclidean
- accuracy: 0.658
- macro F1: 0.502
- weighted F1: 0.626

```text
               precision    recall  f1-score   support

          bug       0.68      0.93      0.79        45
documentation       0.62      0.76      0.68        17
      feature       0.64      0.39      0.48        18
        other       0.65      0.49      0.56        35
     question       0.00      0.00      0.00         5

     accuracy                           0.66       120
    macro avg       0.52      0.51      0.50       120
 weighted avg       0.63      0.66      0.63       120

```

Confusion matrix labels: bug, documentation, feature, other, question

```text
[[42  1  0  2  0]
 [ 3 13  0  1  0]
 [ 4  3  7  4  0]
 [10  4  4 17  0]
 [ 3  0  0  2  0]]
```

### KNN (manual_scaled)
- k: 5
- metric: manhattan
- accuracy: 0.650
- macro F1: 0.504
- weighted F1: 0.623

```text
               precision    recall  f1-score   support

          bug       0.69      0.91      0.79        45
documentation       0.62      0.76      0.68        17
      feature       0.56      0.50      0.53        18
        other       0.65      0.43      0.52        35
     question       0.00      0.00      0.00         5

     accuracy                           0.65       120
    macro avg       0.51      0.52      0.50       120
 weighted avg       0.62      0.65      0.62       120

```

Confusion matrix labels: bug, documentation, feature, other, question

```text
[[41  1  1  2  0]
 [ 3 13  0  1  0]
 [ 3  3  9  3  0]
 [ 9  4  6 15  1]
 [ 3  0  0  2  0]]
```

### KNN (manual_scaled)
- k: 5
- metric: minkowski
- accuracy: 0.658
- macro F1: 0.502
- weighted F1: 0.626

```text
               precision    recall  f1-score   support

          bug       0.68      0.93      0.79        45
documentation       0.62      0.76      0.68        17
      feature       0.64      0.39      0.48        18
        other       0.65      0.49      0.56        35
     question       0.00      0.00      0.00         5

     accuracy                           0.66       120
    macro avg       0.52      0.51      0.50       120
 weighted avg       0.63      0.66      0.63       120

```

Confusion matrix labels: bug, documentation, feature, other, question

```text
[[42  1  0  2  0]
 [ 3 13  0  1  0]
 [ 4  3  7  4  0]
 [10  4  4 17  0]
 [ 3  0  0  2  0]]
```

### KNN (manual_scaled)
- k: 7
- metric: euclidean
- accuracy: 0.658
- macro F1: 0.506
- weighted F1: 0.629

```text
               precision    recall  f1-score   support

          bug       0.66      0.93      0.77        45
documentation       0.57      0.71      0.63        17
      feature       0.73      0.44      0.55        18
        other       0.71      0.49      0.58        35
     question       0.00      0.00      0.00         5

     accuracy                           0.66       120
    macro avg       0.53      0.51      0.51       120
 weighted avg       0.64      0.66      0.63       120

```

Confusion matrix labels: bug, documentation, feature, other, question

```text
[[42  1  0  2  0]
 [ 4 12  0  1  0]
 [ 5  3  8  2  0]
 [10  5  3 17  0]
 [ 3  0  0  2  0]]
```

### KNN (manual_scaled)
- k: 7
- metric: manhattan
- accuracy: 0.642
- macro F1: 0.490
- weighted F1: 0.610

```text
               precision    recall  f1-score   support

          bug       0.67      0.91      0.77        45
documentation       0.62      0.76      0.68        17
      feature       0.58      0.39      0.47        18
        other       0.62      0.46      0.52        35
     question       0.00      0.00      0.00         5

     accuracy                           0.64       120
    macro avg       0.50      0.50      0.49       120
 weighted avg       0.61      0.64      0.61       120

```

Confusion matrix labels: bug, documentation, feature, other, question

```text
[[41  1  0  3  0]
 [ 3 13  0  1  0]
 [ 4  3  7  4  0]
 [10  4  5 16  0]
 [ 3  0  0  2  0]]
```

### KNN (manual_scaled)
- k: 7
- metric: minkowski
- accuracy: 0.658
- macro F1: 0.506
- weighted F1: 0.629

```text
               precision    recall  f1-score   support

          bug       0.66      0.93      0.77        45
documentation       0.57      0.71      0.63        17
      feature       0.73      0.44      0.55        18
        other       0.71      0.49      0.58        35
     question       0.00      0.00      0.00         5

     accuracy                           0.66       120
    macro avg       0.53      0.51      0.51       120
 weighted avg       0.64      0.66      0.63       120

```

Confusion matrix labels: bug, documentation, feature, other, question

```text
[[42  1  0  2  0]
 [ 4 12  0  1  0]
 [ 5  3  8  2  0]
 [10  5  3 17  0]
 [ 3  0  0  2  0]]
```

### KNN (tfidf)
- k: 3
- metric: cosine
- accuracy: 0.642
- macro F1: 0.484
- weighted F1: 0.617

```text
               precision    recall  f1-score   support

          bug       0.68      0.91      0.78        45
documentation       0.50      0.71      0.59        17
      feature       0.58      0.39      0.47        18
        other       0.74      0.49      0.59        35
     question       0.00      0.00      0.00         5

     accuracy                           0.64       120
    macro avg       0.50      0.50      0.48       120
 weighted avg       0.63      0.64      0.62       120

```

Confusion matrix labels: bug, documentation, feature, other, question

```text
[[41  2  1  1  0]
 [ 4 12  0  1  0]
 [ 5  3  7  3  0]
 [ 8  6  3 17  1]
 [ 2  1  1  1  0]]
```

### KNN (tfidf)
- k: 5
- metric: cosine
- accuracy: 0.667
- macro F1: 0.512
- weighted F1: 0.643

```text
               precision    recall  f1-score   support

          bug       0.70      0.89      0.78        45
documentation       0.54      0.76      0.63        17
      feature       0.62      0.44      0.52        18
        other       0.73      0.54      0.62        35
     question       0.00      0.00      0.00         5

     accuracy                           0.67       120
    macro avg       0.52      0.53      0.51       120
 weighted avg       0.65      0.67      0.64       120

```

Confusion matrix labels: bug, documentation, feature, other, question

```text
[[40  2  0  3  0]
 [ 3 13  1  0  0]
 [ 4  2  8  4  0]
 [ 7  6  3 19  0]
 [ 3  1  1  0  0]]
```

### KNN (tfidf)
- k: 7
- metric: cosine
- accuracy: 0.658
- macro F1: 0.516
- weighted F1: 0.636

```text
               precision    recall  f1-score   support

          bug       0.68      0.87      0.76        45
documentation       0.54      0.76      0.63        17
      feature       0.67      0.56      0.61        18
        other       0.71      0.49      0.58        35
     question       0.00      0.00      0.00         5

     accuracy                           0.66       120
    macro avg       0.52      0.53      0.52       120
 weighted avg       0.64      0.66      0.64       120

```

Confusion matrix labels: bug, documentation, feature, other, question

```text
[[39  3  0  3  0]
 [ 3 13  0  1  0]
 [ 4  1 10  3  0]
 [ 8  6  4 17  0]
 [ 3  1  1  0  0]]
```
