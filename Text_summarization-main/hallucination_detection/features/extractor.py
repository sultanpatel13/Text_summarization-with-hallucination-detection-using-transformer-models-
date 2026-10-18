from hallucination_detection.features.similarity import (
    tfidf_similarity,
    sentence_evidence_features
)

from hallucination_detection.features.lexical import (
    keyword_overlap,
    source_coverage
)

from hallucination_detection.features.entities import entity_match

from hallucination_detection.features.numeric import (
    number_match,
    date_match
)


def extract_features(source, summary):

    source = str(source)
    summary = str(summary)

    source_words = len(source.split())
    summary_words = len(summary.split())

    # Avoid division by zero
    if source_words == 0:
        length_ratio = 0.0
    else:
        length_ratio = summary_words / source_words

    best_similarity, average_similarity, gap = sentence_evidence_features(source, summary)

    features = {

        # Existing feature
        "tfidf_similarity":
            tfidf_similarity(source, summary),

        # Existing lexical features
        "keyword_overlap":
            keyword_overlap(source, summary),

        "source_coverage":
            source_coverage(source, summary),

        # Existing factual consistency features
        "entity_match":
            entity_match(source, summary),

        "number_match":
            number_match(source, summary),

        "date_match":
            date_match(source, summary),

        # Existing length feature
        "length_ratio":
            length_ratio,

        # NEW EVIDENCE FEATURES
        "best_source_sentence_similarity": best_similarity,
        "average_source_sentence_similarity": average_similarity,
        "evidence_gap": gap,
    }

    return features
