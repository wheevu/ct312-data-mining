"""Clean raw GitHub issue text for TF-IDF feature extraction.

Removes noisy markdown formatting, code fences, URLs, and issue template
boilerplate that adds little signal for text classification.
"""

from __future__ import annotations

import re


# Ordered patterns to apply sequentially: (pattern, replacement, flags)
_CLEANUP_PATTERNS: list[tuple[str, str, int]] = [
    # Fenced code blocks: ```...``` (greedy, multiline)
    (r"```.*?```", " ", 0),
    # Inline code backticks: `code`
    (r"`[^`]*`", " ", 0),
    # Markdown images: ![alt](url)
    (r"!\[.*?\]\(.*?\)", " ", 0),
    # Markdown links: [text](url)
    (r"\[([^\]]*)\]\([^)]*\)", r"\1", 0),
    # URLs (http, https, ftp)
    (r"https?://\S+", " ", 0),
    # HTML tags
    (r"<[^>]+>", " ", 0),
    # Horizontal rules
    (r"^-{3,}\s*$", " ", re.MULTILINE),
    (r"^_{3,}\s*$", " ", re.MULTILINE),
    # Issue template checkboxes: - [ ] or - [x]
    (r"- \[[ x]\]", " ", 0),
    # Leading "### " or "## " or "# " from markdown headings (keep the text after)
    (r"^#{1,6}\s+", "", re.MULTILINE),
    # Bold/italic markers
    (r"\*\*", " ", 0),
    (r"__", " ", 0),
    # Leading/trailing whitespace per line
    (r"^[ \t]+|[ \t]+$", "", re.MULTILINE),
    # Multiple consecutive blank lines → single blank line
    (r"\n{3,}", "\n\n", 0),
    # Multiple consecutive spaces → single space
    (r" {2,}", " ", 0),
]


def clean_issue_text(text: str) -> str:
    """Remove noisy markdown, code fences, URLs, and template markup from issue text.

    Parameters
    ----------
    text : str
        Raw issue text (title + body combined).

    Returns
    -------
    str
        Cleaned text, stripped of leading/trailing whitespace.
    """
    cleaned = text
    for pattern, replacement, flags in _CLEANUP_PATTERNS:
        cleaned = re.sub(pattern, replacement, cleaned, flags=flags)
    return cleaned.strip()
