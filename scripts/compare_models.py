"""Compare KNN with simple baseline algorithms for Week 2."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from features import build_manual_preprocessor, build_tfidf_vectorizer, clean_for_modeling
from run_knn import stratify_or_none


DATA_PATH = Path("data/processed/issues_labeled.csv")
REPORT_DIR = Path("reports")
SCORES_PATH = REPORT_DIR / "week2_model_scores.csv"


def markdown_table(df: pd.DataFrame) -> str:
    """Render a small dataframe as a markdown table without tabulate."""
    table = df.copy()
    for column in table.select_dtypes(include="number").columns:
        table[column] = table[column].map(lambda value: f"{value:.3f}")
    headers = list(table.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in headers) + " |")
    return "\n".join(lines)


def evaluate(name: str, feature_type: str, pipeline: Pipeline, x_train, x_test, y_train, y_test, **meta) -> dict:
    """Fit and score one model."""
    pipeline.fit(x_train, y_train)
    preds = pipeline.predict(x_test)
    labels = sorted(y_test.unique())
    return {
        "model": name,
        "feature_type": feature_type,
        "accuracy": accuracy_score(y_test, preds),
        "macro_f1": f1_score(y_test, preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test, preds, average="weighted", zero_division=0),
        "classification_report": classification_report(y_test, preds, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, preds, labels=labels),
        "labels": labels,
        **meta,
    }


def main() -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    df = clean_for_modeling(pd.read_csv(DATA_PATH), include_other=True)
    x = df.drop(columns=["target"])
    y = df["target"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=stratify_or_none(y)
    )

    experiments = [
        ("KNN", "manual_scaled", Pipeline([("features", build_manual_preprocessor(True)), ("model", KNeighborsClassifier(n_neighbors=7, metric="minkowski"))]), x_train, x_test, {"k": 7, "distance_metric": "minkowski", "notes": "Best simple KNN manual setting from KNN run"}),
        ("KNN", "tfidf", Pipeline([("tfidf", build_tfidf_vectorizer()), ("model", KNeighborsClassifier(n_neighbors=7, metric="cosine"))]), x_train["text_cleaned"], x_test["text_cleaned"], {"k": 7, "distance_metric": "cosine", "notes": "Cleaned text + cosine"}),
        ("Multinomial Naive Bayes", "tfidf", Pipeline([("tfidf", build_tfidf_vectorizer()), ("model", MultinomialNB())]), x_train["text_cleaned"], x_test["text_cleaned"], {"k": "", "distance_metric": "", "notes": "Cleaned text baseline"}),
        ("Logistic Regression", "tfidf", Pipeline([("tfidf", build_tfidf_vectorizer()), ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))]), x_train["text_cleaned"], x_test["text_cleaned"], {"k": "", "distance_metric": "", "notes": "Cleaned text + class balancing"}),
    ]

    results = [evaluate(name, feature_type, pipe, train_x, test_x, y_train, y_test, **meta) for name, feature_type, pipe, train_x, test_x, meta in experiments]
    score_rows = [{k: v for k, v in row.items() if k not in {"classification_report", "confusion_matrix", "labels"}} for row in results]
    scores = pd.DataFrame(score_rows).sort_values("weighted_f1", ascending=False)
    scores.to_csv(SCORES_PATH, index=False)

    best = scores.iloc[0]
    lines = [
        "# Week 2 Model Comparison",
        "",
        "This table compares the Week 2 KNN classifier against simple text baselines. The goal is not to claim a universal winner, but to compare outputs across algorithms and choose what works best for this small mapped dataset.",
        "",
        "## Comparison table",
        markdown_table(scores),
        "",
        "## Interpretation",
        f"Within this small dataset and target-mapping setup, **{best['model']} with {best['feature_type']} features** performed best by weighted F1 ({best['weighted_f1']:.3f}). KNN remains the main Week 2 model because it demonstrates distance-based classification, scaling, k selection, and distance metrics. The baseline models help justify whether KNN is competitive for text-heavy issue classification.",
        "",
        "## Detailed reports",
    ]
    for row in results:
        lines.extend([
            "",
            f"### {row['model']} ({row['feature_type']})",
            f"Accuracy: {row['accuracy']:.3f}; Macro F1: {row['macro_f1']:.3f}; Weighted F1: {row['weighted_f1']:.3f}",
            "",
            "```text",
            row["classification_report"],
            "```",
            "Confusion matrix labels: " + ", ".join(row["labels"]),
            "```text",
            str(row["confusion_matrix"]),
            "```",
        ])

    (REPORT_DIR / "week2_model_comparison.md").write_text("\n".join(lines) + "\n")
    print(scores)
    print(f"Saved comparison report to {REPORT_DIR / 'week2_model_comparison.md'}")


if __name__ == "__main__":
    main()
