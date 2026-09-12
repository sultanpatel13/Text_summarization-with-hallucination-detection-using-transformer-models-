from sentence_transformers import SentenceTransformer
import numpy as np


MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


def semantic_similarity_from_embeddings(
    source_embedding,
    summary_embedding
):
    """
    Cosine similarity between two normalized embeddings.
    """

    return float(
        np.dot(
            source_embedding,
            summary_embedding
        )
    )


def batch_semantic_features(
    sources,
    summaries,
    batch_size=64
):
    """
    Efficiently calculate semantic features
    for multiple source-summary pairs.

    Returns:
        overall_similarity
        best_sentence_similarity
        average_sentence_similarity
        evidence_gap
    """

    import re

    def split_sentences(text):

        text = str(text).strip()

        if not text:
            return []

        return [
            s.strip()
            for s in re.split(
                r"(?<=[.!?])\s+",
                text
            )
            if s.strip()
        ]

    # ---------------------------------------------------------
    # Overall document-summary embeddings
    # ---------------------------------------------------------

    source_embeddings = model.encode(
        [str(x) for x in sources],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    summary_embeddings = model.encode(
        [str(x) for x in summaries],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    overall_scores = np.sum(
        source_embeddings * summary_embeddings,
        axis=1
    )

    # ---------------------------------------------------------
    # Sentence-level evidence
    # ---------------------------------------------------------

    best_scores = []
    average_scores = []

    for source, summary in zip(
        sources,
        summaries
    ):

        source_sentences = split_sentences(
            source
        )

        summary_sentences = split_sentences(
            summary
        )

        if (
            not source_sentences
            or not summary_sentences
        ):
            best_scores.append(0.0)
            average_scores.append(0.0)
            continue

        source_sentence_embeddings = model.encode(
            source_sentences,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        summary_sentence_embeddings = model.encode(
            summary_sentences,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        # cosine similarity because embeddings
        # are normalized
        similarity_matrix = (
            summary_sentence_embeddings
            @ source_sentence_embeddings.T
        )

        # Best source evidence for each
        # summary sentence
        sentence_best = similarity_matrix.max(
            axis=1
        )

        best_score = float(
            np.mean(sentence_best)
        )

        average_score = float(
            np.mean(similarity_matrix)
        )

        best_scores.append(
            best_score
        )

        average_scores.append(
            average_score
        )

    best_scores = np.array(
        best_scores
    )

    average_scores = np.array(
        average_scores
    )

    evidence_gaps = (
        1.0 - best_scores
    )

    return (
        overall_scores.tolist(),
        best_scores.tolist(),
        average_scores.tolist(),
        evidence_gaps.tolist()
    )
