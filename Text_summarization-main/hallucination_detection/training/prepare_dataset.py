import json
import os
import hashlib
import pandas as pd
from sklearn.model_selection import train_test_split


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_FILE = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "HaluEval-main",
    "data",
    "summarization_data.json"
)

PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
SPLIT_DIR = os.path.join(BASE_DIR, "data", "splits")


def document_id(text):
    """Create a stable ID for each source document."""
    return hashlib.md5(
        str(text).encode("utf-8")
    ).hexdigest()


def load_halueval():

    print("Loading HaluEval")
    print("Input file:", RAW_FILE)

    records = []

    with open(RAW_FILE, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            document = item.get("document", "").strip()
            right_summary = item.get("right_summary", "").strip()
            hallucinated_summary = item.get(
                "hallucinated_summary", ""
            ).strip()

            if document and right_summary:
                records.append({
                    "source": document,
                    "summary": right_summary,
                    "label": 0,
                    "document_id": document_id(document)
                })

            if document and hallucinated_summary:
                records.append({
                    "source": document,
                    "summary": hallucinated_summary,
                    "label": 1,
                    "document_id": document_id(document)
                })

    return pd.DataFrame(records)


def main():

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(SPLIT_DIR, exist_ok=True)

    df = load_halueval()

    print("\nTotal examples before cleaning:", len(df))

    df = df.drop_duplicates(
        subset=["source", "summary", "label"]
    ).reset_index(drop=True)

    print("Total examples after cleaning:", len(df))

    print("\nClass distribution:")
    print(df["label"].value_counts())

    # --------------------------------------------------
    # IMPORTANT:
    # Split by DOCUMENT, not individual rows.
    # This prevents the same document appearing in
    # both training and testing.
    # --------------------------------------------------

    unique_documents = df["document_id"].unique()

    train_docs, temp_docs = train_test_split(
        unique_documents,
        test_size=0.20,
        random_state=42
    )

    validation_docs, test_docs = train_test_split(
        temp_docs,
        test_size=0.50,
        random_state=42
    )

    train_df = df[df["document_id"].isin(train_docs)].copy()
    validation_df = df[
        df["document_id"].isin(validation_docs)
    ].copy()
    test_df = df[
        df["document_id"].isin(test_docs)
    ].copy()

    # Remove internal ID from final CSVs
    final_columns = ["source", "summary", "label"]

    train_df[final_columns].to_csv(
        os.path.join(SPLIT_DIR, "train.csv"),
        index=False
    )

    validation_df[final_columns].to_csv(
        os.path.join(SPLIT_DIR, "validation.csv"),
        index=False
    )

    test_df[final_columns].to_csv(
        os.path.join(SPLIT_DIR, "test.csv"),
        index=False
    )

    df[final_columns].to_csv(
        os.path.join(PROCESSED_DIR, "halu_eval.csv"),
        index=False
    )

    print("\nDATASET CREATED SUCCESSFULLY")

    print("Total examples :", len(df))
    print("Training       :", len(train_df))
    print("Validation     :", len(validation_df))
    print("Testing        :", len(test_df))

    print("\nDocument counts:")
    print("Training documents   :", len(train_docs))
    print("Validation documents :", len(validation_docs))
    print("Testing documents    :", len(test_docs))

    print("\nTraining class distribution:")
    print(train_df["label"].value_counts())

    print("\nValidation class distribution:")
    print(validation_df["label"].value_counts())

    print("\nTesting class distribution:")
    print(test_df["label"].value_counts())

    print("\nFiles created:")
    print("data\\processed\\halu_eval.csv")
    print("data\\splits\\train.csv")
    print("data\\splits\\validation.csv")
    print("data\\splits\\test.csv")


if __name__ == "__main__":
    main()
