import os
import joblib
import pandas as pd
from hallucination_detection.features.extractor import extract_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgboost_final_hallucination_detector.joblib")

_model_data = joblib.load(MODEL_PATH)
model = _model_data["model"]
FEATURES = _model_data["features"]

def predict_sentence(document, summary_sentence):
    feature_dict = extract_features(document, summary_sentence)

    feature_vector = pd.DataFrame(
        [[feature_dict.get(f, 0.0) for f in FEATURES]],
        columns=FEATURES
    )

    probability = model.predict_proba(feature_vector)[0][1]
    prediction = 1 if probability >= 0.5 else 0

    return {
        "label": "Hallucinated" if prediction == 1 else "Factual",
        "prediction": prediction,
        "probability": float(probability),
        "features": feature_dict
    }
