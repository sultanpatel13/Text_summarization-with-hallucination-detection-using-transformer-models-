import re


STOPWORDS = {
    "the", "a", "an", "and", "or", "but",
    "is", "are", "was", "were", "be",
    "been", "being", "to", "of", "in",
    "on", "for", "with", "at", "by",
    "from", "as", "that", "this", "it",
    "its", "they", "their", "he", "she",
    "his", "her", "has", "have", "had",
    "will", "would", "could", "should",
    "may", "might", "can", "do", "does",
    "did", "not"
}


def tokenize(text):
    """Extract meaningful words."""

    words = re.findall(
        r"\b[a-zA-Z][a-zA-Z'-]*\b",
        str(text).lower()
    )

    return {
        word
        for word in words
        if word not in STOPWORDS
        and len(word) > 2
    }


def keyword_overlap(source, summary):
    """
    Fraction of summary keywords that
    also occur in the source.
    """

    source_words = tokenize(source)
    summary_words = tokenize(summary)

    if not summary_words:
        return 0.0

    common_words = source_words.intersection(
        summary_words
    )

    return len(common_words) / len(summary_words)


def source_coverage(source, summary):
    """
    Measures how much of the summary's
    meaningful vocabulary is supported by
    the source.
    """

    return keyword_overlap(source, summary)
