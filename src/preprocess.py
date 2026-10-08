"""
Text cleaning for the FastAPI Model Inference & Embeddings API.

clean_text() is a plain function so it can be dropped into an
sklearn Pipeline via FunctionTransformer, or called directly by the
API / demo scripts.

Steps:
  1. lowercase
  2. strip URLs, email addresses, @mentions and #hashtags
  3. replace standalone numbers with a <num> placeholder (keeps the
     signal "a number was here" without blowing up the vocabulary)
  4. remove punctuation (keep basic word characters + spaces)
  5. collapse whitespace
"""

from __future__ import annotations

import re

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)*\b")
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _HASHTAG_RE.sub(r"\1", text)  # keep the tag word, drop '#'
    text = _NUMBER_RE.sub(" <num> ", text)
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def clean_batch(texts: list[str]) -> list[str]:
    return [clean_text(t) for t in texts]
