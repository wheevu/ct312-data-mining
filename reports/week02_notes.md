# Week 2 Notes -- GitHub Issue Mining Setup

## What I Did This Week

- Set up the repository structure for CT312 Data Mining.
- Defined the project direction: mining GitHub issues and categorizing them.
- Created placeholder data directories (raw, interim, processed).
- Drafted the initial preprocessing plan.

## Key Decisions

- Using the GitHub REST API for data collection (no auth needed for public repos at low rate limits).
- Storing raw data as JSON, processed data as CSV.
- Focusing on issues from a few large open-source repos first (e.g., pandas, scikit-learn, VS Code).

## Questions / Open Items

- Which repos to mine? Need to pick ones with consistent label conventions.
- How to handle repos with very high issue volumes? May need pagination logic.
- Should we include pull requests? GitHub API includes them in the issues endpoint by default.

## Next Steps

- Write the `fetch_issues.py` script and collect a small sample dataset.
- Begin exploratory analysis in a notebook.
- Look at label distributions to decide which categories to predict.
