"""Feature extraction helpers for GitHub issue classification."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "title_length",
    "body_length",
    "text_length",
    "label_count",
    "comments",
    "has_question_mark",
    "has_code_block",
    "has_error_word",
    "has_traceback_word",
    "has_docs_word",
    "has_feature_word",
]

CATEGORICAL_FEATURES = ["repo"]


def get_manual_columns() -> tuple[list[str], list[str]]:
    """Return the numeric and categorical manual feature columns."""
    return NUMERIC_FEATURES, CATEGORICAL_FEATURES


def build_manual_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """Build a transformer for manual numeric features plus repo one-hot encoding."""
    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("repo", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def build_tfidf_vectorizer(max_features: int = 2000) -> TfidfVectorizer:
    """Build a simple TF-IDF vectorizer for issue title + body text."""
    return TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=max_features,
        min_df=2,
        ngram_range=(1, 2),
    )


def clean_for_modeling(df: pd.DataFrame, include_other: bool = True) -> pd.DataFrame:
    """Prepare dataframe values for modeling scripts."""
    model_df = df.copy()
    model_df["text"] = model_df["text"].fillna("")
    model_df["text_cleaned"] = model_df["text_cleaned"].fillna("")
    for column in NUMERIC_FEATURES:
        model_df[column] = model_df[column].fillna(0)
    model_df["repo"] = model_df["repo"].fillna("unknown")
    if not include_other:
        model_df = model_df[model_df["target"] != "other"].copy()
    return model_df
