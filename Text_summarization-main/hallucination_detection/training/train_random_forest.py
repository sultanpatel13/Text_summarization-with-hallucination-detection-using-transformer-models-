import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

print("Loading feature datasets...")

train = pd.read_csv("data/processed/train_features.csv")
val = pd.read_csv("data/processed/validation_features.csv")
test = pd.read_csv("data/processed/test_features.csv")

X_train = train.drop(columns=["label"])
y_train = train["label"]

X_val = val.drop(columns=["label"])
y_val = val["label"]

X_test = test.drop(columns=["label"])
y_test = test["label"]

print("\nFeatures:")
print(list(X_train.columns))

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ---------------- VALIDATION ----------------

val_pred = model.predict(X_val)

print("\n========================================")
print("VALIDATION RESULT")
print("========================================")

print(f"Accuracy : {accuracy_score(y_val, val_pred) * 100:.2f}%")
print(f"Precision: {precision_score(y_val, val_pred) * 100:.2f}%")
print(f"Recall   : {recall_score(y_val, val_pred) * 100:.2f}%")
print(f"F1 Score : {f1_score(y_val, val_pred) * 100:.2f}%")

# ---------------- TEST ----------------

test_pred = model.predict(X_test)

print("\n========================================")
print("FINAL TEST RESULT")
print("========================================")

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

# ---------------- FEATURE IMPORTANCE ----------------

importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
}).sort_values(
    by="Importance",
    ascending=False
)

print("\n========================================")
print("FEATURE IMPORTANCE")
print("========================================")

print(importance.to_string(index=False))

# ---------------- SAVE MODEL ----------------

joblib.dump(
    model,
    "models/random_forest_hallucination_detector.joblib"
)

print("\nModel saved:")
print("models/random_forest_hallucination_detector.joblib")
