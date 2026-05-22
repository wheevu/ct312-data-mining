"""Build a processed GitHub issue dataset for Week 2 classification.

The script reads raw JSON files from data/raw/, normalizes repository-specific
labels into a small target set, and writes data/processed/issues_labeled.csv.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from clean_text import clean_issue_text


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
TARGET_PRIORITY = ["bug", "documentation", "feature", "question", "other"]


PROCESS_LABEL_WORDS = (
    "needs triage",
    "needs discussion",
    "good first issue",
    "priority",
    "stale",
    "wontfix",
    "duplicate",
    "invalid",
    "help wanted",
    "easy",
    "blocker",
)


def repo_from_filename(path: Path) -> str:
    """Convert owner_repo_issues.json into owner/repo."""
    stem = path.stem.removesuffix("_issues")
    owner, repo = stem.split("_", 1)
    return f"{owner}/{repo}"


def label_names(issue: dict) -> list[str]:
    """Return label names from a GitHub issue object."""
    labels = issue.get("labels") or []
    names: list[str] = []
    for label in labels:
        if isinstance(label, dict):
            names.append(str(label.get("name", "")))
        else:
            names.append(str(label))
    return [name for name in names if name]


def is_process_label(label: str) -> bool:
    """Detect labels that describe workflow/process instead of content."""
    normalized = label.lower()
    return any(word in normalized for word in PROCESS_LABEL_WORDS)


def category_matches(labels: list[str], title: str, body: str) -> set[str]:
    """Find all target categories suggested by labels and simple text rules."""
    matches: set[str] = set()
    content_labels = [label.lower() for label in labels if not is_process_label(label)]

    for label in content_labels:
        if "bug" in label:
            matches.add("bug")
        if "doc" in label or "documentation" in label or "docs" in label:
            matches.add("documentation")
        if "feature" in label or "enhancement" in label or "request" in label:
            matches.add("feature")
        if "question" in label or "support" in label or "usage" in label:
            matches.add("question")

    title_lower = title.lower().strip()
    text_lower = f"{title} {body}".lower()
    question_starts = ("how", "why", "what")
    if (
        title_lower.startswith(question_starts)
        or "?" in title
        or "how do i" in text_lower
        or "is it possible" in text_lower
        or "can i" in text_lower
    ):
        matches.add("question")

    return matches


def choose_target(labels: list[str], title: str, body: str) -> tuple[str, list[str]]:
    """Choose one target using the documented priority order."""
    matches = category_matches(labels, title, body)
    for category in TARGET_PRIORITY:
        if category == "other" or category in matches:
            return category, sorted(matches)
    return "other", sorted(matches)


def has_word(text: str, pattern: str) -> int:
    """Return 1 if regex pattern appears in text, else 0."""
    return int(re.search(pattern, text, flags=re.IGNORECASE) is not None)


def build_rows(raw_dir: Path = RAW_DIR) -> tuple[list[dict], Counter, Counter]:
    """Load raw JSON files and convert issue objects into flat rows."""
    rows: list[dict] = []
    raw_label_counts: Counter = Counter()
    match_counts: Counter = Counter()

    for path in sorted(raw_dir.glob("*_issues.json")):
        repo = repo_from_filename(path)
        issues = json.loads(path.read_text())
        print(f"Loading {len(issues)} issues from {path}")

        for issue in issues:
            labels = label_names(issue)
            raw_label_counts.update(labels)

            title = str(issue.get("title") or "")
            body = str(issue.get("body") or "")
            text = f"{title}\n\n{body}".strip()
            text_cleaned = clean_issue_text(text)
            target, matched = choose_target(labels, title, body)
            match_counts.update(matched or ["other"])

            text_lower = text.lower()
            row = {
                "repo": repo,
                "number": issue.get("number"),
                "title": title,
                "body": body,
                "text": text,
                "text_cleaned": text_cleaned,
                "labels_raw": "; ".join(labels),
                "target": target,
                "state": issue.get("state"),
                "created_at": issue.get("created_at"),
                "updated_at": issue.get("updated_at"),
                "comments": int(issue.get("comments") or 0),
                "title_length": len(title),
                "body_length": len(body),
                "text_length": len(text),
                "label_count": len(labels),
                "has_question_mark": int("?" in text),
                "has_code_block": int("```" in text),
                "has_error_word": has_word(text, r"\berror\b|\bexception\b|\bfail(?:ed|ure)?\b"),
                "has_traceback_word": has_word(text, r"\btraceback\b"),
                "has_docs_word": has_word(text, r"\bdoc(?:s|umentation)?\b"),
                "has_feature_word": has_word(text, r"\bfeature\b|\benhancement\b|\brequest\b"),
            }
            rows.append(row)

    return rows, raw_label_counts, match_counts


def main() -> None:
    """Build and save the processed dataset and mapping report."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    rows, raw_label_counts, match_counts = build_rows()
    df = pd.DataFrame(rows)
    output_path = PROCESSED_DIR / "issues_labeled.csv"
    df.to_csv(output_path, index=False)

    report = {
        "target_priority": TARGET_PRIORITY,
        "process_labels_ignored": PROCESS_LABEL_WORDS,
        "rows": len(df),
        "target_distribution": df["target"].value_counts().to_dict(),
        "top_raw_labels": dict(raw_label_counts.most_common(30)),
        "category_match_counts_before_priority": dict(match_counts),
    }
    report_path = PROCESSED_DIR / "label_mapping_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print(f"Saved {len(df)} processed issues to {output_path}")
    print("Target distribution:")
    print(df["target"].value_counts())
    print(f"Saved label mapping report to {report_path}")


if __name__ == "__main__":
    main()
