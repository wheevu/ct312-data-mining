# Week 2 Baseline Experiment

## Objective

Establish a baseline for the GitHub issue categorization task before any modeling begins.

## Status

No models have been run yet. This file will be updated once we have:

- A collected dataset of GitHub issues.
- A set of extracted features.
- A first pass at a simple classifier (e.g., logistic regression or Naive Bayes).

## Planned Baseline Approach

1. Use title + body text as input features (TF-IDF vectorization).
2. Train a simple classifier (multinomial Naive Bayes).
3. Evaluate using accuracy, precision, recall, and F1-score.
4. Report confusion matrix.

## Notes

- This baseline will give us a lower-bound performance to beat with more sophisticated approaches.
- No hyperparameter tuning yet -- just default scikit-learn settings.
