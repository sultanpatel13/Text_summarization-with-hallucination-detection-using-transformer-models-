import os
import re
import json
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

RAW_FILE = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "HaluEval-main",
    "data",
    "summarization_data.json"
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "all-MiniLM-L6-v2"

BATCH_SIZE = 128


# ============================================================
# SENTENCE SPLITTER
# ============================================================

def split_sentences(text):

    text = str(text).strip()

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        s.strip()
        for s in sentences
        if s.strip()
    ]


# ============================================================
# LOAD HALUEVAL
# ============================================================

def load_dataset():

    print("Loading HaluEval...")

    examples = []

    with open(
        RAW_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            document = item.get(
                "document",
                ""
            )

            right_summary = item.get(
                "right_summary",
                ""
            )

            hallucinated_summary = item.get(
                "hallucinated_summary",
                ""
            )

            if (
                document
                and right_summary
                and hallucinated_summary
            ):

                examples.append({
                    "document": document,
                    "summary": right_summary,
                    "label": 0
                })

                examples.append({
                    "document": document,
                    "summary": hallucinated_summary,
                    "label": 1
                })

    df = pd.DataFrame(examples)

    print(
        "Total examples:",
        len(df)
    )

    return df


# ============================================================
# SAME DOCUMENT-LEVEL SPLIT
# ============================================================

def create_splits(df):

    print(
        "\nCreating document-level splits..."
    )

    documents = df[
        "document"
    ].drop_duplicates().tolist()

    rng = np.random.default_rng(42)

    rng.shuffle(documents)

    total = len(documents)

    train_end = int(
        total * 0.80
    )

    val_end = int(
        total * 0.90
    )

    train_docs = set(
        documents[:train_end]
    )

    val_docs = set(
        documents[train_end:val_end]
    )

    test_docs = set(
        documents[val_end:]
    )

    train_df = df[
        df["document"].isin(
            train_docs
        )
    ].reset_index(drop=True)

    val_df = df[
        df["document"].isin(
            val_docs
        )
    ].reset_index(drop=True)

    test_df = df[
        df["document"].isin(
            test_docs
        )
    ].reset_index(drop=True)

    print(
        "Training:",
        len(train_df)
    )

    print(
        "Validation:",
        len(val_df)
    )

    print(
        "Testing:",
        len(test_df)
    )

    return (
        train_df,
        val_df,
        test_df
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print(
        "\nLoading semantic model..."
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    print(
        "Semantic model loaded."
    )

    return model


# ============================================================
# OVERALL SEMANTIC SIMILARITY
# ============================================================

def calculate_overall_similarity(
    model,
    sources,
    summaries
):

    print(
        "\nEncoding source documents..."
    )

    source_embeddings = model.encode(
        sources,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print(
        "\nEncoding summaries..."
    )

    summary_embeddings = model.encode(
        summaries,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    similarities = np.sum(
        source_embeddings *
        summary_embeddings,
        axis=1
    )

    return similarities


# ============================================================
# SENTENCE-LEVEL SEMANTIC EVIDENCE
# ============================================================

def calculate_sentence_evidence(
    model,
    sources,
    summaries
):

    print(
        "\nCalculating sentence-level evidence..."
    )

    all_source_sentences = []
    all_summary_sentences = []

    source_ranges = []
    summary_ranges = []

    source_position = 0
    summary_position = 0

    # --------------------------------------------------------
    # Prepare sentences
    # --------------------------------------------------------

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

        source_start = source_position

        all_source_sentences.extend(
            source_sentences
        )

        source_position += len(
            source_sentences
        )

        source_ranges.append(
            (
                source_start,
                source_position
            )
        )

        summary_start = summary_position

        all_summary_sentences.extend(
            summary_sentences
        )

        summary_position += len(
            summary_sentences
        )

        summary_ranges.append(
            (
                summary_start,
                summary_position
            )
        )

    print(
        "Total source sentences:",
        len(all_source_sentences)
    )

    print(
        "Total summary sentences:",
        len(all_summary_sentences)
    )

    # --------------------------------------------------------
    # Encode sentences in batches
    # --------------------------------------------------------

    print(
        "\nEncoding source sentences..."
    )

    source_sentence_embeddings = model.encode(
        all_source_sentences,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print(
        "\nEncoding summary sentences..."
    )

    summary_sentence_embeddings = model.encode(
        all_summary_sentences,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    # --------------------------------------------------------
    # Calculate evidence
    # --------------------------------------------------------

    best_scores = []
    average_scores = []

    total = len(sources)

    for i in range(total):

        ss, se = source_ranges[i]
        ms, me = summary_ranges[i]

        source_vectors = (
            source_sentence_embeddings[
                ss:se
            ]
        )

        summary_vectors = (
            summary_sentence_embeddings[
                ms:me
            ]
        )

        if (
            len(source_vectors) == 0
            or len(summary_vectors) == 0
        ):

            best_scores.append(0.0)
            average_scores.append(0.0)

            continue

        similarity_matrix = (
            summary_vectors
            @ source_vectors.T
        )

        # Best source evidence for
        # each summary sentence
        sentence_best = (
            similarity_matrix.max(
                axis=1
            )
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

        if (
            (i + 1) % 500 == 0
            or i + 1 == total
        ):

            print(
                f"Evidence: {i + 1}/{total}"
            )

    best_scores = np.array(
        best_scores
    )

    average_scores = np.array(
        average_scores
    )

    evidence_gap = (
        1.0 - best_scores
    )

    return (
        best_scores,
        average_scores,
        evidence_gap
    )


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(
    model,
    df,
    split_name
):

    print(
        "\n======================================"
    )

    print(
        f"PROCESSING {split_name.upper()}"
    )

    print(
        "======================================"
    )

    sources = df[
        "document"
    ].astype(str).tolist()

    summaries = df[
        "summary"
    ].astype(str).tolist()

    # Overall semantic similarity
    overall = calculate_overall_similarity(
        model,
        sources,
        summaries
    )

    # Sentence-level evidence
    (
        best_similarity,
        average_similarity,
        evidence_gap
    ) = calculate_sentence_evidence(
        model,
        sources,
        summaries
    )

    # Load existing feature CSV
    feature_file = os.path.join(
        PROCESSED_DIR,
        f"{split_name}_features.csv"
    )

    features = pd.read_csv(
        feature_file
    )

    print(
        "\nExisting feature rows:",
        len(features)
    )

    print(
        "Dataset rows:",
        len(df)
    )

    if len(features) != len(df):

        raise ValueError(
            f"Row mismatch in {split_name}: "
            f"features={len(features)}, "
            f"dataset={len(df)}"
        )

    # Add semantic features
    features[
        "semantic_similarity"
    ] = overall

    features[
        "semantic_best_similarity"
    ] = best_similarity

    features[
        "semantic_average_similarity"
    ] = average_similarity

    features[
        "semantic_evidence_gap"
    ] = evidence_gap

    # Save
    features.to_csv(
        feature_file,
        index=False
    )

    print(
        f"\nUpdated: {feature_file}"
    )

    print(
        "Total features:",
        len(features.columns) - 1
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_dataset()

    (
        train_df,
        val_df,
        test_df
    ) = create_splits(df)

    model = load_model()

    process_split(
        model,
        train_df,
        "train"
    )

    process_split(
        model,
        val_df,
        "validation"
    )

    process_split(
        model,
        test_df,
        "test"
    )

    print(
        "\n======================================"
    )

    print(
        "SEMANTIC FEATURES COMPLETED"
    )

    print(
        "======================================"
    )

    print(
        "\nAdded features:"
    )

    print(
        "semantic_similarity"
    )

    print(
        "semantic_best_similarity"
    )

    print(
        "semantic_average_similarity"
    )

    print(
        "semantic_evidence_gap"
    )

    print(
        "\nYour existing features were preserved."
    )


if __name__ == "__main__":
    main()
