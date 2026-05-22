"""Run Week 2 KNN experiments on the processed GitHub issue dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from features import build_manual_preprocessor, build_tfidf_vectorizer, clean_for_modeling


DATA_PATH = Path("data/processed/issues_labeled.csv")
REPORT_DIR = Path("reports")
SCORES_PATH = REPORT_DIR / "week2_model_scores.csv"


def stratify_or_none(y: pd.Series) -> pd.Series | None:
    """Use stratification only when every class has enough examples."""
    return y if y.value_counts().min() >= 2 else None


def evaluate_model(name: str, feature_type: str, pipeline: Pipeline, x_train, x_test, y_train, y_test, **meta) -> dict:
    """Fit a model and return common evaluation metrics plus display objects."""
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
        "notes": "",
        **meta,
    }


def append_scores(rows: list[dict]) -> None:
    """Append KNN score rows to the shared model scores CSV."""
    score_rows = [
        {k: v for k, v in row.items() if k not in {"classification_report", "confusion_matrix", "labels"}}
        for row in rows
    ]
    scores = pd.DataFrame(score_rows)
    REPORT_DIR.mkdir(exist_ok=True)
    scores.to_csv(SCORES_PATH, index=False)


def markdown_table(df: pd.DataFrame) -> str:
    """Render a small dataframe as a markdown table without extra dependencies."""
    table = df.copy()
    for column in table.select_dtypes(include="number").columns:
        table[column] = table[column].map(lambda value: f"{value:.3f}")
    headers = list(table.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in headers) + " |")
    return "\n".join(lines)


def markdown_report(rows: list[dict], distribution: pd.Series, grid_summary: str) -> str:
    """Create a readable markdown report for KNN experiments."""
    summary = pd.DataFrame([
        {k: v for k, v in row.items() if k not in {"classification_report", "confusion_matrix", "labels"}}
        for row in rows
    ]).sort_values("weighted_f1", ascending=False)

    lines = [
        "# Week 2 KNN Results",
        "",
        "## Class distribution before modeling",
        markdown_table(distribution.rename_axis("target").reset_index(name="count")),
        "",
        "## Summary scores",
        markdown_table(summary),
        "",
        "## Scaling demonstration",
        "KNN uses distances between vectors. Without scaling, large-range features such as body_length and text_length can dominate smaller binary features. StandardScaler was applied only to input features, never to the target labels.",
        "",
        "## GridSearchCV",
        grid_summary,
        "",
        "## Detailed experiment outputs",
    ]
    for row in rows:
        lines.extend([
            "",
            f"### {row['model']} ({row['feature_type']})",
            f"- k: {row.get('k', '')}",
            f"- metric: {row.get('distance_metric', '')}",
            f"- accuracy: {row['accuracy']:.3f}",
            f"- macro F1: {row['macro_f1']:.3f}",
            f"- weighted F1: {row['weighted_f1']:.3f}",
            "",
            "```text",
            row["classification_report"],
            "```",
            "",
            "Confusion matrix labels: " + ", ".join(row["labels"]),
            "",
            "```text",
            str(row["confusion_matrix"]),
            "```",
        ])
    return "\n".join(lines) + "\n"


def main() -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    df = clean_for_modeling(pd.read_csv(DATA_PATH), include_other=True)
    distribution = df["target"].value_counts()
    print("Class distribution before modeling:")
    print(distribution)

    x = df.drop(columns=["target"])
    y = df["target"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=stratify_or_none(y)
    )

    results: list[dict] = []
    for scaled in [False, True]:
        for k in [3, 5, 7]:
            for metric in ["euclidean", "manhattan", "minkowski"]:
                pipe = Pipeline([
                    ("features", build_manual_preprocessor(scale_numeric=scaled)),
                    ("model", KNeighborsClassifier(n_neighbors=k, metric=metric)),
                ])
                results.append(evaluate_model(
                    "KNN",
                    "manual_scaled" if scaled else "manual_unscaled",
                    pipe,
                    x_train,
                    x_test,
                    y_train,
                    y_test,
                    k=k,
                    distance_metric=metric,
                    notes="StandardScaler used" if scaled else "No scaling",
                ))

    for k in [3, 5, 7]:
        pipe = Pipeline([
            ("tfidf", build_tfidf_vectorizer()),
            ("model", KNeighborsClassifier(n_neighbors=k, metric="cosine")),
        ])
        results.append(evaluate_model("KNN", "tfidf", pipe, x_train["text_cleaned"], x_test["text_cleaned"], y_train, y_test, k=k, distance_metric="cosine", notes="TF-IDF text vectors (cleaned)"))

    grid = GridSearchCV(
        Pipeline([
            ("features", build_manual_preprocessor(scale_numeric=True)),
            ("model", KNeighborsClassifier()),
        ]),
        param_grid={
            "model__n_neighbors": [3, 5, 7, 9],
            "model__metric": ["euclidean", "manhattan", "minkowski"],
            "model__weights": ["uniform", "distance"],
        },
        scoring="f1_weighted",
        cv=3,
    )
    grid.fit(x_train, y_train)
    grid_summary = f"Best weighted F1 CV score: {grid.best_score_:.3f}; best params: {grid.best_params_}. GridSearch tests KNN settings systematically."

    append_scores(results)
    report = markdown_report(results, distribution, grid_summary)
    (REPORT_DIR / "week2_knn_results.md").write_text(report)

    print("Top KNN results:")
    print(pd.DataFrame([{k: v for k, v in r.items() if k not in {"classification_report", "confusion_matrix", "labels"}} for r in results]).sort_values("weighted_f1", ascending=False).head())
    print(f"Saved report to {REPORT_DIR / 'week2_knn_results.md'}")


if __name__ == "__main__":
    main()
