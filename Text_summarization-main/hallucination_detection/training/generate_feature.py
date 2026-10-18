import os
import re
import json
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

os.makedirs(PROCESSED_DIR, exist_ok=True)


# ============================================================
# BASIC TEXT FUNCTIONS
# ============================================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text):
    """
    Lightweight sentence splitting.
    """
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


def tokenize(text):
    return set(
        re.findall(
            r"\b[a-zA-Z][a-zA-Z0-9'-]*\b",
            str(text).lower()
        )
    )


# ============================================================
# LEXICAL FEATURES
# ============================================================

def keyword_overlap(source, summary):

    source_tokens = tokenize(source)
    summary_tokens = tokenize(summary)

    if not summary_tokens:
        return 0.0

    return len(
        source_tokens.intersection(summary_tokens)
    ) / len(summary_tokens)


def source_coverage(source, summary):

    source_tokens = tokenize(source)
    summary_tokens = tokenize(summary)

    if not summary_tokens:
        return 0.0

    return len(
        summary_tokens.intersection(source_tokens)
    ) / len(summary_tokens)


# ============================================================
# ENTITY FEATURE
# ============================================================

def extract_entities(text):

    text = str(text)

    patterns = [
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",
        r"\b[A-Z][a-z]{2,}\b"
    ]

    entities = set()

    for pattern in patterns:
        for match in re.findall(pattern, text):
            entities.add(match.lower().strip())

    return entities


def entity_match(source, summary):

    source_entities = extract_entities(source)
    summary_entities = extract_entities(summary)

    if not summary_entities:
        return 1.0

    return len(
        source_entities.intersection(summary_entities)
    ) / len(summary_entities)


# ============================================================
# NUMBER FEATURE
# ============================================================

def extract_numbers(text):

    return set(
        re.findall(
            r"\b\d+(?:\.\d+)?%?\b",
            str(text)
        )
    )


def number_match(source, summary):

    source_numbers = extract_numbers(source)
    summary_numbers = extract_numbers(summary)

    if not summary_numbers:
        return 1.0

    return len(
        source_numbers.intersection(summary_numbers)
    ) / len(summary_numbers)


# ============================================================
# DATE FEATURE
# ============================================================

def extract_dates(text):

    text = str(text)

    patterns = [
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
        r"\b(?:19|20)\d{2}\b"
    ]

    dates = set()

    for pattern in patterns:
        dates.update(
            re.findall(pattern, text)
        )

    return {d.lower() for d in dates}


def date_match(source, summary):

    source_dates = extract_dates(source)
    summary_dates = extract_dates(summary)

    if not summary_dates:
        return 1.0

    return len(
        source_dates.intersection(summary_dates)
    ) / len(summary_dates)


# ============================================================
# SENTENCE EVIDENCE FEATURES
# ============================================================

def sentence_evidence_features(
    source,
    summary,
    vectorizer
):
    """
    Finds how strongly each summary sentence is supported
    by the source document.

    Returns:
        best similarity
        average similarity
        evidence gap
    """

    source_sentences = split_sentences(source)
    summary_sentences = split_sentences(summary)

    if not source_sentences or not summary_sentences:
        return 0.0, 0.0, 1.0

    source_clean = [
        clean_text(s)
        for s in source_sentences
    ]

    summary_clean = [
        clean_text(s)
        for s in summary_sentences
    ]

    try:

        source_vectors = vectorizer.transform(
            source_clean
        )

        summary_vectors = vectorizer.transform(
            summary_clean
        )

        # TF-IDF vectors are L2 normalized by default.
        # Therefore dot product = cosine similarity.
        similarities = summary_vectors.dot(
            source_vectors.T
        )

        best_scores = []

        for i in range(
            similarities.shape[0]
        ):

            row = similarities.getrow(i)

            if row.nnz == 0:
                best_score = 0.0
            else:
                best_score = float(
                    row.data.max()
                )

            best_scores.append(best_score)

        if not best_scores:
            return 0.0, 0.0, 1.0

        best_similarity = float(
            np.mean(best_scores)
        )

        average_similarity = float(
            np.mean(
                [
                    score
                    for row in similarities
                    for score in row.data
                ]
            )
        ) if similarities.nnz > 0 else 0.0

        evidence_gap = 1.0 - best_similarity

        return (
            best_similarity,
            average_similarity,
            evidence_gap
        )

    except Exception:
        return 0.0, 0.0, 1.0


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(
    source,
    summary,
    vectorizer
):

    source = str(source)
    summary = str(summary)

    source_words = len(source.split())
    summary_words = len(summary.split())

    if source_words == 0:
        length_ratio = 0.0
    else:
        length_ratio = (
            summary_words / source_words
        )

    # Overall TF-IDF similarity
    try:

        vectors = vectorizer.transform(
            [
                clean_text(source),
                clean_text(summary)
            ]
        )

        tfidf_similarity = float(
            vectors[0].multiply(
                vectors[1]
            ).sum()
        )

    except Exception:
        tfidf_similarity = 0.0

    # Sentence evidence
    (
        best_sentence_similarity,
        average_sentence_similarity,
        evidence_gap
    ) = sentence_evidence_features(
        source,
        summary,
        vectorizer
    )

    features = {

        "tfidf_similarity":
            tfidf_similarity,

        "keyword_overlap":
            keyword_overlap(
                source,
                summary
            ),

        "source_coverage":
            source_coverage(
                source,
                summary
            ),

        "entity_match":
            entity_match(
                source,
                summary
            ),

        "number_match":
            number_match(
                source,
                summary
            ),

        "date_match":
            date_match(
                source,
                summary
            ),

        "length_ratio":
            length_ratio,

        # NEW
        "best_source_sentence_similarity":
            best_sentence_similarity,

        "average_source_sentence_similarity":
            average_sentence_similarity,

        "evidence_gap":
            evidence_gap
    }

    return features


# ============================================================
# LOAD HALUEVAL
# ============================================================

def load_halueval():

    print("Loading HaluEval...")

    print(
        "Input file:",
        RAW_FILE
    )

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

            factual_summary = item.get(
                "right_summary",
                ""
            )

            hallucinated_summary = item.get(
                "hallucinated_summary",
                ""
            )

            if (
                document
                and factual_summary
                and hallucinated_summary
            ):

                examples.append(
                    {
                        "document": document,
                        "summary": factual_summary,
                        "label": 0
                    }
                )

                examples.append(
                    {
                        "document": document,
                        "summary": hallucinated_summary,
                        "label": 1
                    }
                )

    print(
        "Total examples:",
        len(examples)
    )

    return pd.DataFrame(examples)


# ============================================================
# SPLIT WITHOUT DOCUMENT LEAKAGE
# ============================================================

def create_splits(df):

    print("\nCreating document-level splits...")

    documents = df[
        "document"
    ].drop_duplicates().tolist()

    rng = np.random.default_rng(42)

    rng.shuffle(documents)

    n_documents = len(documents)

    train_end = int(
        n_documents * 0.80
    )

    val_end = int(
        n_documents * 0.90
    )

    train_documents = set(
        documents[:train_end]
    )

    val_documents = set(
        documents[train_end:val_end]
    )

    test_documents = set(
        documents[val_end:]
    )

    train_df = df[
        df["document"].isin(
            train_documents
        )
    ].reset_index(drop=True)

    val_df = df[
        df["document"].isin(
            val_documents
        )
    ].reset_index(drop=True)

    test_df = df[
        df["document"].isin(
            test_documents
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
# BUILD SHARED TF-IDF
# ============================================================

def build_vectorizer(train_df):

    print("\nBuilding shared TF-IDF vectorizer...")

    texts = (
        train_df["document"].tolist()
        +
        train_df["summary"].tolist()
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=30000,
        min_df=2,
        sublinear_tf=True
    )

    vectorizer.fit(texts)

    print(
        "TF-IDF vocabulary:",
        len(vectorizer.vocabulary_)
    )

    return vectorizer


# ============================================================
# GENERATE FEATURES
# ============================================================

def generate_features(
    df,
    vectorizer,
    name
):

    print(
        f"\nGenerating {name} features..."
    )

    rows = []

    total = len(df)

    for index, row in df.iterrows():

        features = extract_features(
            row["document"],
            row["summary"],
            vectorizer
        )

        features["label"] = int(
            row["label"]
        )

        rows.append(features)

        if (
            (index + 1) % 500 == 0
            or index + 1 == total
        ):

            print(
                f"{index + 1}/{total}"
            )

    result = pd.DataFrame(rows)

    output_file = os.path.join(
        PROCESSED_DIR,
        f"{name}_features.csv"
    )

    result.to_csv(
        output_file,
        index=False
    )

    print(
        f"{name} saved:",
        output_file
    )

    print(
        "Features:",
        list(result.columns)
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_halueval()

    train_df, val_df, test_df = (
        create_splits(df)
    )

    vectorizer = build_vectorizer(
        train_df
    )

    generate_features(
        train_df,
        vectorizer,
        "train"
    )

    generate_features(
        val_df,
        vectorizer,
        "validation"
    )

    generate_features(
        test_df,
        vectorizer,
        "test"
    )

    print("\n================================")
    print("ENHANCED FEATURES CREATED")
    print("================================")

    print(
        "\nNew features:"
    )

    print(
        "1. TF-IDF similarity"
    )

    print(
        "2. Keyword overlap"
    )

    print(
        "3. Source coverage"
    )

    print(
        "4. Entity match"
    )

    print(
        "5. Number match"
    )

    print(
        "6. Date match"
    )

    print(
        "7. Length ratio"
    )

    print(
        "8. Best source sentence similarity"
    )

    print(
        "9. Average source sentence similarity"
    )

    print(
        "10. Evidence gap"
    )

    print(
        "\nFeature generation completed."
    )


if __name__ == "__main__":
    main()
