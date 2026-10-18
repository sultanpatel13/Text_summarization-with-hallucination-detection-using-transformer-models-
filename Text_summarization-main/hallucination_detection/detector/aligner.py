from hallucination_detection.features.similarity import best_source_sentence_similarity
from detector.sentence_splitter import split_sentences

def find_best_evidence(document, summary_sentence):
    source_sentences = split_sentences(document)

    if not source_sentences:
        return {"evidence": "", "similarity": 0.0}

    best_sentence = ""
    best_score = 0.0

    for source in source_sentences:
        score = best_source_sentence_similarity(
            source,
            summary_sentence
        )

        if score > best_score:
            best_score = score
            best_sentence = source

    return {
        "evidence": best_sentence,
        "similarity": float(best_score)
    }
