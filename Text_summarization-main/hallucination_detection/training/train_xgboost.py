import os
import joblib
import pandas as pd
import xgboost as xgb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_FILE = os.path.join(BASE_DIR, "data", "processed", "train_features.csv")
VAL_FILE = os.path.join(BASE_DIR, "data", "processed", "validation_features.csv")
TEST_FILE = os.path.join(BASE_DIR, "data", "processed", "test_features.csv")

MODEL_FILE = os.path.join(
    BASE_DIR, "models", "xgboost_final_hallucination_detector.joblib"
)

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)
test_df = pd.read_csv(TEST_FILE)

# Automatically use every feature except label
FEATURES = [col for col in train_df.columns if col != "label"]

print("Features:")
print(FEATURES)
print("Number of features:", len(FEATURES))

X_train = train_df[FEATURES]
y_train = train_df["label"]

X_val = val_df[FEATURES]
y_val = val_df["label"]

X_test = test_df[FEATURES]
y_test = test_df["label"]

# ---------------------------------------------------------
# Train XGBoost
# ---------------------------------------------------------
print("\nTraining final XGBoost...")

model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.04,
    min_child_weight=1,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

print("Training completed.")

# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

val_pred = model.predict(X_val)
test_pred = model.predict(X_test)

print("\n" + "=" * 40)
print("FINAL VALIDATION RESULT")
print("=" * 40)

print(f"Accuracy : {accuracy_score(y_val, val_pred) * 100:.2f}%")
print(f"Precision: {precision_score(y_val, val_pred) * 100:.2f}%")
print(f"Recall   : {recall_score(y_val, val_pred) * 100:.2f}%")
print(f"F1 Score : {f1_score(y_val, val_pred) * 100:.2f}%")

print("\n" + "=" * 40)
print("FINAL TEST RESULT")
print("=" * 40)

print(f"Accuracy : {accuracy_score(y_test, test_pred) * 100:.2f}%")
print(f"Precision: {precision_score(y_test, test_pred) * 100:.2f}%")
print(f"Recall   : {recall_score(y_test, test_pred) * 100:.2f}%")
print(f"F1 Score : {f1_score(y_test, test_pred) * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_pred,
        target_names=["Factual", "Hallucinated"]
    )
)

# ---------------------------------------------------------
# Feature importance
# ---------------------------------------------------------
importance = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\n" + "=" * 40)
print("FEATURE IMPORTANCE")
print("=" * 40)
print(importance.to_string(index=False))

# ---------------------------------------------------------
# Save model
# ---------------------------------------------------------
os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)

joblib.dump(
    {
        "model": model,
        "features": FEATURES
    },
    MODEL_FILE
)

print("\nModel saved:")
print(MODEL_FILE)
