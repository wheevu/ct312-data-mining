# Week 2 Model Comparison

This table compares the Week 2 KNN classifier against simple text baselines. The goal is not to claim a universal winner, but to compare outputs across algorithms and choose what works best for this small mapped dataset.

## Comparison table
| model | feature_type | accuracy | macro_f1 | weighted_f1 | k | distance_metric | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | tfidf | 0.683 | 0.543 | 0.674 |  |  | Cleaned text + class balancing |
| Multinomial Naive Bayes | tfidf | 0.700 | 0.538 | 0.674 |  |  | Cleaned text baseline |
| KNN | tfidf | 0.658 | 0.516 | 0.636 | 7 | cosine | Cleaned text + cosine |
| KNN | manual_scaled | 0.658 | 0.506 | 0.629 | 7 | minkowski | Best simple KNN manual setting from KNN run |

## Interpretation
Within this small dataset and target-mapping setup, **Logistic Regression with tfidf features** performed best by weighted F1 (0.674). KNN remains the main Week 2 model because it demonstrates distance-based classification, scaling, k selection, and distance metrics. The baseline models help justify whether KNN is competitive for text-heavy issue classification.

## Detailed reports

### KNN (manual_scaled)
Accuracy: 0.658; Macro F1: 0.506; Weighted F1: 0.629

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
Accuracy: 0.658; Macro F1: 0.516; Weighted F1: 0.636

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

### Multinomial Naive Bayes (tfidf)
Accuracy: 0.700; Macro F1: 0.538; Weighted F1: 0.674

```text
               precision    recall  f1-score   support

          bug       0.71      0.93      0.81        45
documentation       0.91      0.59      0.71        17
      feature       0.70      0.39      0.50        18
        other       0.62      0.71      0.67        35
     question       0.00      0.00      0.00         5

     accuracy                           0.70       120
    macro avg       0.59      0.52      0.54       120
 weighted avg       0.68      0.70      0.67       120

```
Confusion matrix labels: bug, documentation, feature, other, question
```text
[[42  0  0  3  0]
 [ 5 10  0  2  0]
 [ 4  0  7  7  0]
 [ 6  1  3 25  0]
 [ 2  0  0  3  0]]
```

### Logistic Regression (tfidf)
Accuracy: 0.683; Macro F1: 0.543; Weighted F1: 0.674

```text
               precision    recall  f1-score   support

          bug       0.81      0.78      0.80        45
documentation       0.59      0.76      0.67        17
      feature       0.58      0.61      0.59        18
        other       0.66      0.66      0.66        35
     question       0.00      0.00      0.00         5

     accuracy                           0.68       120
    macro avg       0.53      0.56      0.54       120
 weighted avg       0.67      0.68      0.67       120

```
Confusion matrix labels: bug, documentation, feature, other, question
```text
[[35  4  0  6  0]
 [ 2 13  1  1  0]
 [ 1  2 11  4  0]
 [ 3  3  5 23  1]
 [ 2  0  2  1  0]]
```
