"""Collect issues from the GitHub REST API.

This module handles pagination, rate limiting, and saving raw response
data to the data/raw/ directory for later processing.
"""

import json
import time
from pathlib import Path

import requests


GITHUB_API = "https://api.github.com"


def fetch_issues(
    owner: str,
    repo: str,
    state: str = "all",
    per_page: int = 100,
    max_issues: int | None = None,
    sort: str = "created",
    direction: str = "desc",
) -> list[dict]:
    """Fetch issues from a GitHub repository.

    Parameters
    ----------
    owner : str
        Repository owner (user or org).
    repo : str
        Repository name.
    state : str, default "all"
        Issue state: "open", "closed", or "all".
    per_page : int, default 100
        Results per page (max 100).
    max_issues : int or None, default None
        Maximum number of issues to fetch. None fetches all available.
    sort : str, default "created"
        Sort field: "created", "updated", "comments".
    direction : str, default "desc"
        Sort direction: "asc" or "desc".

    Returns
    -------
    list[dict]
        List of raw issue objects from the API.
    """
    issues: list[dict] = []
    page = 1

    while True:
        url = f"{GITHUB_API}/repos/{owner}/{repo}/issues"
        params = {
            "state": state,
            "per_page": min(per_page, 100),
            "page": page,
            "sort": sort,
            "direction": direction,
        }
        resp = requests.get(url, params=params, headers={"Accept": "application/vnd.github.v3+json"})
        resp.raise_for_status()

        batch = resp.json()
        if not batch:
            break

        # Filter out pull requests (GitHub API includes PRs in /issues)
        batch = [i for i in batch if "pull_request" not in i]
        issues.extend(batch)

        print(f"  Page {page}: got {len(batch)} issues (total {len(issues)})")

        if max_issues and len(issues) >= max_issues:
            issues = issues[:max_issues]
            break

        # Check if there are more pages
        link_header = resp.headers.get("Link", "")
        if 'rel="next"' not in link_header:
            break

        page += 1
        time.sleep(0.5)  # be polite

    return issues


def save_issues(issues: list[dict], owner: str, repo: str, raw_dir: str = "data/raw") -> str:
    """Save fetched issues to a JSON file.

    Parameters
    ----------
    issues : list[dict]
        Issue data to save.
    owner : str
        Repository owner (used in filename).
    repo : str
        Repository name (used in filename).
    raw_dir : str, default "data/raw"
        Directory to save the file in.

    Returns
    -------
    str
        Path to the saved file.
    """
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    filepath = raw_path / f"{owner}_{repo}_issues.json"

    with open(filepath, "w") as f:
        json.dump(issues, f, indent=2)

    print(f"Saved {len(issues)} issues to {filepath}")
    return str(filepath)
