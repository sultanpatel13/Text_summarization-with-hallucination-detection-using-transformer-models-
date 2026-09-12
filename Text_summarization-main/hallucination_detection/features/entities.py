import re


def extract_entities(text):
    """
    Lightweight entity extraction.

    Detects:
    - Proper names
    - Organizations
    - Multi-word capitalized phrases
    """

    text = str(text)

    patterns = [
        # Multi-word proper names:
        # Steve Jobs, New York City, United States
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",

        # Single capitalized words that may represent entities
        r"\b[A-Z][a-z]{2,}\b"
    ]

    entities = set()

    for pattern in patterns:
        matches = re.findall(pattern, text)

        for match in matches:
            entities.add(match.lower().strip())

    return entities


def entity_match(source, summary):
    """
    Calculates how many entities in the summary
    are also present in the source.
    """

    source_entities = extract_entities(source)
    summary_entities = extract_entities(summary)

    if not summary_entities:
        return 1.0

    matched = summary_entities.intersection(source_entities)

    return len(matched) / len(summary_entities)


def entity_match_from_sets(source_entities, summary_entities):
    """
    Calculates entity overlap when entity sets
    are already available.
    """

    if not summary_entities:
        return 1.0

    matched = summary_entities.intersection(source_entities)

    return len(matched) / len(summary_entities)
