def explain_prediction(result):
    features = result["features"]
    reasons = []

    if features.get("number_match", 1) < 1:
        reasons.append("Number mismatch detected.")

    if features.get("date_match", 1) < 1:
        reasons.append("Date mismatch detected.")

    if features.get("entity_match", 1) < 0.5:
        reasons.append("Entity mismatch detected.")

    if features.get("keyword_overlap", 1) < 0.5:
        reasons.append("Low keyword overlap with source.")

    if features.get("source_coverage", 1) < 0.5:
        reasons.append("Insufficient source coverage.")

    if features.get("best_source_sentence_similarity", 1) < 0.5:
        reasons.append("Weak source evidence similarity.")

    if not reasons:
        reasons.append("Prediction based on combined factual-consistency features.")

    return " ".join(reasons)
