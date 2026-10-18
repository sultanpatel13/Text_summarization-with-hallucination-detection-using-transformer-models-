import re


def extract_numbers(text):
    """
    Extract numerical values from text.
    Handles integers, decimals, percentages
    and comma-separated numbers.
    """

    text = str(text)

    numbers = re.findall(
        r"\b\d+(?:,\d{3})*(?:\.\d+)?%?\b",
        text
    )

    return set(numbers)


def number_match(source, summary):
    """
    Calculate the fraction of summary numbers
    that are present in the source.
    """

    source_numbers = extract_numbers(source)
    summary_numbers = extract_numbers(summary)

    if not summary_numbers:
        return 1.0

    matched = summary_numbers.intersection(
        source_numbers
    )

    return len(matched) / len(summary_numbers)


def extract_dates(text):
    """
    Extract common date formats.
    """

    text = str(text)

    patterns = [
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        r"\b\d{1,2}-\d{1,2}-\d{2,4}\b",
        r"\b\d{4}-\d{1,2}-\d{1,2}\b",
        r"\b(?:19|20)\d{2}\b"
    ]

    dates = set()

    for pattern in patterns:
        dates.update(
            re.findall(pattern, text)
        )

    return dates


def date_match(source, summary):
    """
    Calculate the fraction of summary dates
    that occur in the source.
    """

    source_dates = extract_dates(source)
    summary_dates = extract_dates(summary)

    if not summary_dates:
        return 1.0

    matched = summary_dates.intersection(
        source_dates
    )

    return len(matched) / len(summary_dates)
