import re
import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgboost_final_hallucination_detector.joblib")

DATA = joblib.load(MODEL_PATH)
MODEL = DATA["model"]
FEATURES = DATA["features"]

def split_sentences(text):
    if not text:
        return []
    return [x.strip() for x in re.split(r'(?<=[.!?])\s+', text.strip()) if x.strip()]

def predict_sentence(document, sentence):
    from hallucination_detection.features.extractor import extract_features

    f = extract_features(document, sentence)

    X = pd.DataFrame(
        [[f.get(name, 0.0) for name in FEATURES]],
        columns=FEATURES
    )

    probability = float(MODEL.predict_proba(X)[0][1])
    prediction = int(probability >= 0.5)

    return {
        "label": "Hallucinated" if prediction else "Factual",
        "probability": round(probability * 100, 2),
        "features": f
    }

def find_evidence(document, sentence):
    from hallucination_detection.features.similarity import best_source_sentence_similarity

    sources = split_sentences(document)

    if not sources:
        return "", 0.0

    best = max(
        sources,
        key=lambda s: best_source_sentence_similarity(s, sentence)
    )

    score = best_source_sentence_similarity(best, sentence)

    return best, float(score)

def explain(result):
    f = result["features"]
    reasons = []

    if f.get("number_match", 1) < 1:
        reasons.append("Number mismatch")

    if f.get("date_match", 1) < 1:
        reasons.append("Date mismatch")

    if f.get("entity_match", 1) < 0.5:
        reasons.append("Entity mismatch")

    if f.get("keyword_overlap", 1) < 0.5:
        reasons.append("Low source overlap")

    if f.get("source_coverage", 1) < 0.5:
        reasons.append("Low source coverage")

    if f.get("best_source_sentence_similarity", 1) < 0.5:
        reasons.append("Weak source evidence")

    return "; ".join(reasons) if reasons else "No strong factual inconsistency detected"

def detect_hallucinations(document, summary):
    sentences = split_sentences(summary)
    results = []

    for sentence in sentences:
        prediction = predict_sentence(document, sentence)
        evidence, similarity = find_evidence(document, sentence)

        results.append({
            "sentence": sentence,
            "label": prediction["label"],
            "probability": prediction["probability"],
            "reason": explain(prediction),
            "evidence": evidence,
            "evidence_similarity": round(similarity, 4)
        })

    hallucinated = sum(1 for r in results if r["label"] == "Hallucinated")
    total = len(results)

    # Overall score = average hallucination probability.
    # This keeps even small probabilities visible instead of forcing them to 0%.
    overall_score = (
        round(sum(float(r["probability"]) for r in results) / total, 2)
        if total else 0
    )

    return {
        "total_sentences": total,
        "hallucinated_sentences": hallucinated,
        "factual_sentences": total - hallucinated,
        "hallucination_score": overall_score,
        "results": results
    }
