import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_text(text):
    """Basic text cleaning."""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tfidf_similarity(source, summary):
    """
    Overall TF-IDF similarity between the complete source
    document and summary.
    """
    source = clean_text(source)
    summary = clean_text(summary)

    if not source or not summary:
        return 0.0

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000
    )

    try:
        vectors = vectorizer.fit_transform([source, summary])
        score = cosine_similarity(
            vectors[0:1],
            vectors[1:2]
        )[0][0]

        return float(score)

    except ValueError:
        return 0.0


def sentence_similarity(source_sentence, summary_sentence):
    """TF-IDF similarity between two individual sentences."""
    return tfidf_similarity(
        source_sentence,
        summary_sentence
    )


def split_sentences(text):
    """
    Simple sentence splitter.
    Avoids requiring heavy NLP processing.
    """
    text = str(text).strip()

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def source_sentence_similarities(source, summary_sentence):
    """
    Compare one summary sentence against every sentence
    in the original document.
    """

    source_sentences = split_sentences(source)

    if not source_sentences or not summary_sentence.strip():
        return []

    summary_sentence = clean_text(summary_sentence)

    documents = [
        clean_text(sentence)
        for sentence in source_sentences
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000
    )

    try:
        vectors = vectorizer.fit_transform(
            documents + [summary_sentence]
        )

        source_vectors = vectors[:-1]
        summary_vector = vectors[-1:]

        scores = cosine_similarity(
            source_vectors,
            summary_vector
        ).flatten()

        return scores.tolist()

    except ValueError:
        return [0.0] * len(source_sentences)


def best_source_sentence_similarity(source, summary_sentence):
    """
    Highest similarity between the summary sentence
    and any source sentence.
    """

    scores = source_sentence_similarities(
        source,
        summary_sentence
    )

    if not scores:
        return 0.0

    return float(max(scores))


def sentence_evidence_features(source, summary_sentence):
    """Compute evidence features with one TF-IDF vectorization pass."""
    scores = source_sentence_similarities(source, summary_sentence)
    if not scores:
        return 0.0, 0.0, 1.0
    best = float(max(scores))
    return best, float(np.mean(scores)), float(1.0 - best)


def average_source_sentence_similarity(source, summary_sentence):
    """
    Average similarity between the summary sentence
    and all source sentences.
    """

    scores = source_sentence_similarities(
        source,
        summary_sentence
    )

    if not scores:
        return 0.0

    return float(np.mean(scores))


def evidence_gap(source, summary_sentence):
    """
    Measures how poorly the summary sentence is supported
    by the original document.

    Higher value = weaker evidence.
    """

    best_score = best_source_sentence_similarity(
        source,
        summary_sentence
    )

    return float(1.0 - best_score)
