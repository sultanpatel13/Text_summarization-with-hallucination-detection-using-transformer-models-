"""Hierarchical summarizer and relative word-budget length control.

This module implements:
1. Narrative cleaning (stripping timestamps, speaker tags, conversational fillers).
2. Relative word-budget calculation (% of source words) based on presets (short, medium, long).
3. Words-to-token budget mapping for BART and PEGASUS models.
4. Output post-processing and strict word-budget enforcement (sentence boundary truncation).
"""

from __future__ import annotations

import re
from typing import Tuple

try:
    from backend.summarizer import split_sentences
except ModuleNotFoundError:
    from summarizer import split_sentences

# Preset length definitions (% of source words and paragraph caps)
BUDGET_CONFIG = {
    "short": {
        "min_ratio": 0.25,
        "max_ratio": 0.45,
        "paragraph_cap": 250,
    },
    "medium": {
        "min_ratio": 0.45,
        "max_ratio": 0.60,
        "paragraph_cap": 450,
    },
    "long": {
        "min_ratio": 0.75,
        "max_ratio": 0.80,
        "paragraph_cap": 800,
    },
}

DEFAULT_PRESET = "medium"

# Regex patterns for narrative cleaning
TIMESTAMP_PATTERN = re.compile(
    r"(?:\[?\b\d{1,2}:\d{2}(?::\d{2})?\b\]?|\(\b\d{1,2}:\d{2}(?::\d{2})?\b\))",
    re.IGNORECASE,
)
SPEAKER_TAG_PATTERN = re.compile(
    r"^(?:(?:speaker\s*\d+|interviewer|interviewee|host|guest|narrator|[A-Z][a-zA-Z\s]{1,20}))\s*:\s*",
    re.IGNORECASE | re.MULTILINE,
)


def clean_and_filter_narrative(text: str) -> str:
    """Clean narrative text by stripping timestamps, speaker tags, and normalizing whitespace.

    Removes:
    - Audio/video timestamps (e.g. '[00:15]', '12:34', '(01:02:03)')
    - Speaker labels (e.g. 'Speaker 1:', 'Interviewer:', 'John:')
    - Excessive formatting, extra line breaks, and conversational filler
    """
    if not text or not text.strip():
        return ""

    cleaned = text
    # Strip timestamps
    cleaned = TIMESTAMP_PATTERN.sub(" ", cleaned)
    # Strip speaker tags at line starts
    cleaned = SPEAKER_TAG_PATTERN.sub("", cleaned)
    # Strip bullet points and list dashes at line starts
    cleaned = re.sub(r"^\s*[-*•]\s+", "", cleaned, flags=re.MULTILINE)
    # Normalize whitespaces and line endings into a single narrative
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def calculate_word_budget(source_word_count: int, preset: str = "medium") -> Tuple[int, int]:
    """Calculate relative (min_words, max_words) summary budget based on source length and preset.

    Presets:
    - short: max = 45%, min = 25%, paragraph_cap = 250
    - medium (default): max = 60%, min = 45%, paragraph_cap = 450
    - long: max = 80%, min = 75%, paragraph_cap = 800

    Edge cases:
    - <= 20 words: max = source * max_ratio, min = source * min_ratio
    - <= 60 words: max >= 10, min >= 5
    - > 60 words: max = min(cap, source * max_ratio), min = min(max - 3, max(15, source * min_ratio))
    """
    preset_key = str(preset).lower().strip() if preset else DEFAULT_PRESET
    cfg = BUDGET_CONFIG.get(preset_key, BUDGET_CONFIG[DEFAULT_PRESET])
    min_ratio = cfg["min_ratio"]
    max_ratio = cfg["max_ratio"]
    cap = cfg["paragraph_cap"]

    if source_word_count <= 20:
        max_words = max(1, int(round(source_word_count * max_ratio)))
        min_words = max(1, int(round(source_word_count * min_ratio)))
        min_words = min(min_words, max_words)
    elif source_word_count <= 60:
        max_words = max(10, int(round(source_word_count * max_ratio)))
        min_words = max(5, int(round(source_word_count * min_ratio)))
        min_words = min(min_words, max_words)
    else:
        max_words = min(cap, int(round(source_word_count * max_ratio)))
        min_words = min(max_words - 3, max(15, int(round(source_word_count * min_ratio))))
        min_words = max(5, min_words)

    if min_words > max_words:
        min_words = max_words

    return min_words, max_words


def words_to_token_budget(
    min_words: int,
    max_words: int,
    max_token_cap: int = 1024,
) -> Tuple[int, int]:
    """Convert word budget to tokenizer token bounds with standard ~1.35x expansion."""
    max_tokens = min(max_token_cap, max(28, int(round(max_words * 1.35))))
    min_tokens = max(14, int(round(min_words * 1.10)))

    if min_tokens >= max_tokens:
        min_tokens = max(10, max_tokens - 10)

    return min_tokens, max_tokens


def enforce_word_budget(text: str, max_words: int) -> str:
    """Enforce word budget: flatten to single paragraph and truncate at sentence boundary if exceeded.

    If length exceeds max_words:
    - Attempt to truncate at the last complete sentence boundary that fits.
    - If even the first sentence exceeds max_words, hard cut at max_words with a trailing period.
    """
    if not text or not text.strip():
        return ""

    flattened = " ".join(text.split())
    words = flattened.split()

    if len(words) <= max_words:
        return flattened

    sentences = split_sentences(flattened)
    if not sentences:
        return " ".join(words[:max_words]).rstrip(",;:-–—") + "."

    accumulated: list[str] = []
    current_count = 0

    for s in sentences:
        s_words = len(s.split())
        if current_count + s_words <= max_words:
            accumulated.append(s)
            current_count += s_words
        else:
            break

    if accumulated:
        return " ".join(accumulated)

    return " ".join(words[:max_words]).rstrip(",;:-–—") + "."
