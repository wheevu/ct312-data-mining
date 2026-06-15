#!/usr/bin/env python3
"""Shared utilities for CT312 Lab 2 assignments.

Provides:
  • Dataset loaders (iris, wine, breast cancer, optical digits)
  • Cross‑validation helper
  • Result / plot exporters
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import LabelEncoder

# ── Paths ────────────────────────────────────────────────────────
# Data relative to ct312-data-mining/data/lab2/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_SCRIPT_DIR, '..', '..', 'data', 'lab2')
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, '..', '..', 'outputs', 'lab2')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')


def ensure_dirs():
    for d in (OUTPUT_DIR, RESULTS_DIR, PLOTS_DIR):
        os.makedirs(d, exist_ok=True)


# ── Dataset loaders ─────────────────────────────────────────────
# Each returns (X: ndarray, y: ndarray, class_names: list[str])

def load_iris():
    path = os.path.join(DATA_DIR, 'iris', 'iris.data')
    cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    df = pd.read_csv(path, header=None, names=cols)
    le = LabelEncoder()
    df['class'] = le.fit_transform(df['class'])
    X = df.drop('class', axis=1).values.astype(np.float64)
    y = df['class'].values.astype(np.int32)
    return X, y, le.classes_.tolist()


def load_wine():
    path = os.path.join(DATA_DIR, 'wine', 'wine.data')
    df = pd.read_csv(path, header=None)
    y = df.iloc[:, 0].values.astype(np.int32) - 1   # original is 1-based
    X = df.iloc[:, 1:].values.astype(np.float64)
    return X, y, ['1', '2', '3']


def load_breast_cancer():
    path = os.path.join(DATA_DIR, 'breast+cancer+wisconsin+diagnostic', 'wdbc.data')
    df = pd.read_csv(path, header=None)
    le = LabelEncoder()
    y = le.fit_transform(df.iloc[:, 1])               # B → 0,  M → 1
    X = df.iloc[:, 2:].values.astype(np.float64)
    return X, y, le.classes_.tolist()                 # ['B', 'M']


def load_optical_digits():
    data_dir = os.path.join(DATA_DIR, 'optical+recognition+of+handwritten+digits')
    train = pd.read_csv(os.path.join(data_dir, 'optdigits.tra'), header=None)
    test = pd.read_csv(os.path.join(data_dir, 'optdigits.tes'), header=None)
    df = pd.concat([train, test], ignore_index=True)
    X = df.iloc[:, :-1].values.astype(np.float64)
    y = df.iloc[:, -1].values.astype(np.int32)
    return X, y, [str(i) for i in range(10)]


# ── Evaluation helpers ───────────────────────────────────────────

def evaluate_cross_val(model, X, y, cv=5, scoring='accuracy'):
    """Return per‑fold scores array for a stratified k‑fold CV."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    return cross_val_score(model, X, y, cv=skf, scoring=scoring)


def save_results_csv(rows, filename):
    """Write a list of dicts to outputs/results/<filename>."""
    ensure_dirs()
    path = os.path.join(RESULTS_DIR, filename)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  → Saved {path}")


def save_confusion_matrix(y_true, y_pred, class_names, filename, title=None):
    """Plot and save a confusion matrix to outputs/plots/<filename>."""
    ensure_dirs()
    path = os.path.join(PLOTS_DIR, filename)
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=class_names, ax=ax,
        cmap='Blues', colorbar=False,
    )
    if title:
        ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved {path}")


def save_feature_importance(importances, feature_names, filename,
                            title=None, top_n=None):
    """Horizontal bar‑chart of feature importances."""
    ensure_dirs()
    path = os.path.join(PLOTS_DIR, filename)

    if top_n is not None:
        idx = np.argsort(importances)[-top_n:]
    else:
        idx = np.argsort(importances)

    labels = [feature_names[i] for i in idx]
    values = importances[idx]

    fig, ax = plt.subplots(figsize=(10, max(5, len(labels) * 0.35)))
    ax.barh(range(len(values)), values, color='steelblue', edgecolor='none')
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    if title:
        ax.set_title(title, fontsize=14)
    ax.set_xlabel('Importance', fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved {path}")
