"""
preprocessing.py
-----------------
Text cleaning utilities for the ticket triage pipeline.

Step 2 of the project brief: "Clean text and create labels."
"""

import re

_URL_RE = re.compile(r"http\S+|www\.\S+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s']")
_MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/punctuation, collapse whitespace."""
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _NON_ALNUM_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def clean_series(series):
    return series.astype(str).map(clean_text)
