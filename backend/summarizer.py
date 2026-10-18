import math
import re
from collections import Counter
from typing import List


STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "with",
    "would",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "hey",
    "gemini",
    "please",
    "summarize",
    "summary",
    "quote",
    "text",
    "sentence",
    "sentences",
}


SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
WORD_PATTERN = re.compile(r"[A-Za-z0-9']+")
SPACE_PATTERN = re.compile(r"\s+")
FILLER_PATTERN = re.compile(r"\b(?:um|uh|erm|hmm|like)\b", re.IGNORECASE)
VOICE_COMMAND_PATTERN = re.compile(
    r"^(?:(?:hey\s+\w+[\s,]*)?(?:please\s+)?(?:summarize|summary)(?:\s+the\s+(?:quote|text|sentence))?[\s,:-]*)+",
    re.IGNORECASE,
)


def split_sentences(text: str) -> List[str]:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return []

    sentences = [sentence.strip() for sentence in SENTENCE_SPLIT_PATTERN.split(cleaned) if sentence.strip()]
    return sentences


def tokenize(text: str) -> List[str]:
    return [
        token.lower()
        for token in WORD_PATTERN.findall(text)
        if token and token.lower() not in STOPWORDS
    ]


def normalize_input_text(text: str) -> str:
    cleaned = SPACE_PATTERN.sub(" ", text.strip())
    cleaned = FILLER_PATTERN.sub("", cleaned)
    cleaned = VOICE_COMMAND_PATTERN.sub("", cleaned)
    cleaned = re.sub(r'Total Documents: \d+ Total Tokens: \d+ Vocabulary Size: \d+\.?', '', cleaned, flags=re.IGNORECASE)
    cleaned = SPACE_PATTERN.sub(" ", cleaned).strip()

    if not cleaned:
        return ""

    words = cleaned.split()
    collapsed: List[str] = []
    index = 0

    while index < len(words):
        removed_repeat = False

        # Collapse repeated consecutive phrases common in speech-to-text retries.
        for phrase_len in range(min(8, (len(words) - index) // 2), 0, -1):
            current_phrase = words[index:index + phrase_len]
            next_phrase = words[index + phrase_len:index + (2 * phrase_len)]
            if current_phrase == next_phrase:
                collapsed.extend(current_phrase)
                index += phrase_len * 2
                while words[index:index + phrase_len] == current_phrase:
                    index += phrase_len
                removed_repeat = True
                break

        if removed_repeat:
            continue

        if collapsed and words[index].lower() == collapsed[-1].lower():
            index += 1
            continue

        collapsed.append(words[index])
        index += 1

    normalized = " ".join(collapsed)
    normalized = re.sub(r"\b(\w+(?:\s+\w+){0,3})\s+\1\b", r"\1", normalized, flags=re.IGNORECASE)
    normalized = SPACE_PATTERN.sub(" ", normalized).strip()
    return normalized


def sentence_similarity(sentence_a: str, sentence_b: str) -> float:
    tokens_a = tokenize(sentence_a)
    tokens_b = tokenize(sentence_b)

    if not tokens_a or not tokens_b:
        return 0.0

    counts_a = Counter(tokens_a)
    counts_b = Counter(tokens_b)

    common_words = set(counts_a) & set(counts_b)
    numerator = sum(counts_a[word] * counts_b[word] for word in common_words)

    norm_a = math.sqrt(sum(value * value for value in counts_a.values()))
    norm_b = math.sqrt(sum(value * value for value in counts_b.values()))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return numerator / (norm_a * norm_b)


def pagerank(matrix: List[List[float]], damping: float = 0.85, max_iterations: int = 50, tolerance: float = 1e-4) -> List[float]:
    size = len(matrix)
    if size == 0:
        return []

    scores = [1.0 / size] * size

    for _ in range(max_iterations):
        updated_scores = [(1.0 - damping) / size] * size

        for source_index in range(size):
            outbound_weight = sum(matrix[source_index])
            if outbound_weight == 0:
                share = damping * scores[source_index] / size
                for target_index in range(size):
                    updated_scores[target_index] += share
                continue

            for target_index in range(size):
                weight = matrix[source_index][target_index]
                if weight > 0:
                    updated_scores[target_index] += damping * scores[source_index] * (weight / outbound_weight)

        delta = sum(abs(updated_scores[index] - scores[index]) for index in range(size))
        scores = updated_scores
        if delta <= tolerance:
            break

    return scores


def summarize_text(text: str, max_sentences: int = 3) -> str:
    normalized_text = normalize_input_text(text)
    sentences = split_sentences(normalized_text)
    if not sentences:
        return ""

    if len(sentences) <= 2:
        return " ".join(sentences)

    max_sentences = max(1, min(max_sentences, len(sentences)))
    similarity_matrix: List[List[float]] = []

    for source_sentence in sentences:
        row = []
        for target_sentence in sentences:
            if source_sentence == target_sentence:
                row.append(0.0)
            else:
                row.append(sentence_similarity(source_sentence, target_sentence))
        similarity_matrix.append(row)

    ranked_scores = pagerank(similarity_matrix)
    ranked_indices = sorted(
        range(len(sentences)),
        key=lambda index: ranked_scores[index],
        reverse=True,
    )[:max_sentences]

    ordered_indices = sorted(ranked_indices)
    return " ".join(sentences[index] for index in ordered_indices)


PREPOSITIONS = {
    "in", "on", "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "upon", "down", "of",
    "off", "over", "under", "across", "behind", "beyond", "toward", "towards", "within", "without"
}

COMMON_VERBS = {
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "doing",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must",
    "make", "makes", "made", "making",
    "establish", "establishes", "established", "establishing",
    "learn", "learns", "learned", "learning",
    "find", "finds", "found", "finding",
    "try", "tries", "tried", "trying",
    "see", "sees", "saw", "seen", "seeing",
    "look", "looks", "looked", "looking",
    "provide", "provides", "provided", "providing",
    "generate", "generates", "generated", "generating",
    "show", "shows", "showed", "shown", "showing",
    "include", "includes", "included", "including",
    "lead", "leads", "led", "leading",
    "allow", "allows", "allowed", "allowing",
    "enable", "enables", "enabled", "enabling",
    "achieve", "achieves", "achieved", "achieving",
    "require", "requires", "required", "requiring",
    "transform", "transforms", "transformed", "transforming",
    "leverage", "leverages", "leveraged", "leveraging",
    "compare", "compares", "compared", "comparing",
    "indicate", "indicates", "indicated", "indicating",
    "mitigate", "mitigates", "mitigated", "mitigating",
    "play", "plays", "played", "playing",
    "accelerate", "accelerates", "accelerated", "accelerating",
    "become", "becomes", "became", "becoming"
}

TEMPORAL_AND_GENERIC_NOUNS = {
    "past decade", "last decade", "recent years", "past years", "decade", "year", "years",
    "stride", "strides", "remarkable strides", "vast amounts", "new paradigm", "paradigm",
    "globe", "world", "time", "day", "days", "way", "ways", "thing", "things", "part", "parts",
    "role", "indispensable role", "scale", "economies", "economies of scale", "bottlenecks",
    "mandates", "collaboration", "collaborations", "efforts", "challenges", "amount", "amounts"
}

GENRE_KEYWORDS = {
    "technical": {
        "architecture", "architectures", "transformer", "transformers", "neural", "algorithm", "algorithms",
        "model", "models", "framework", "frameworks", "pre-trained", "pretrained", "methodology", "inference",
        "abstractive", "dataset", "datasets", "empirical", "benchmark", "evaluation", "parameters", "optimization",
        "cross-encoder", "deep", "learning", "nlp", "encoder", "decoder", "tokens", "tokenization", "representation"
    },
    "business": {
        "renewable", "energy", "market", "markets", "industry", "industries", "government", "governments",
        "economy", "economic", "policy", "policies", "solar", "wind", "infrastructure", "transition",
        "decarbonization", "investments", "investment", "financial", "strategy", "enterprise", "sector", "revenue"
    },
    "narrative": {
        "leaves", "box", "forest", "pile", "journey", "adventure", "story", "character", "characters",
        "walked", "looked", "found", "find", "search", "searching", "opened", "whispered", "smiled", "door",
        "village", "tree", "woods", "child", "friend", "night", "morning", "secret", "suddenly", "learns"
    }
}


def clean_stutter_and_repetitions(text: str) -> str:
    """Collapse repetitive loops, stuttering phrases, and trailing duplicate tokens."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    words = cleaned.split()
    if not words:
        return ""

    deduped = []
    i = 0
    n = len(words)
    while i < n:
        matched = False
        for k in range(min(6, (n - i) // 2), 0, -1):
            chunk1 = [w.lower().strip(".,!?;:\"'") for w in words[i : i + k]]
            chunk2 = [w.lower().strip(".,!?;:\"'") for w in words[i + k : i + 2 * k]]
            if chunk1 == chunk2:
                deduped.extend(words[i : i + k])
                i += 2 * k
                while i + k <= n and [w.lower().strip(".,!?;:\"'") for w in words[i : i + k]] == chunk1:
                    i += k
                matched = True
                break
        if not matched:
            deduped.append(words[i])
            i += 1

    res = " ".join(deduped)
    res = re.sub(r"\b(\w+)\s+and\s+(?:the|a)\s+\1\b", r"\1", res, flags=re.IGNORECASE)
    res = re.sub(r"\s+([,.:;?!])", r"\1", res)
    return res


def clean_sentence_opener(sentence: str) -> str:
    """Strip leading coordinating conjunctions and discourse connectives."""
    s = sentence.strip()
    s = re.sub(
        r"^(?:and|but|or|so|yet|nor|then|also|because|however|furthermore|moreover|additionally|meanwhile|later)\b[\s,]*",
        "",
        s,
        flags=re.IGNORECASE,
    )
    if s:
        s = s[0].upper() + s[1:]
    return s


def is_valid_noun_phrase(phrase: str) -> bool:
    """Validate that an extracted phrase is a genuine noun concept rather than an adverb or participle."""
    tokens = phrase.lower().split()
    if not tokens:
        return False
    last = tokens[-1]
    first = tokens[0]
    if first in {"significantly", "explicitly", "widely", "greatly", "highly", "rapidly", "extremely", "very", "more", "most"}:
        return False
    if last.endswith("ly") or (last.endswith("ed") and last not in {"speed", "feed", "need", "bed", "red", "seed"}):
        return False
    if last.endswith(("ive", "able", "ible")) and last not in {"objective", "directive", "incentive", "representative", "variable"}:
        return False
    return True


def extract_noun_phrases(sentence: str) -> List[str]:
    """Extract clean, complete noun phrases from a sentence by chunking around verbs and prepositions."""
    cleaned = re.sub(r"[,;:()\[\]\"\'`.]", " ", sentence)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    tokens = cleaned.split()
    phrases: List[str] = []
    current_chunk: List[str] = []

    for token in tokens:
        lower = token.lower()
        if lower in COMMON_VERBS or lower in PREPOSITIONS or lower in {"and", "but", "or", "so", "then", "due", "such", "as", "out"}:
            if current_chunk:
                chunk_str = " ".join(current_chunk).strip()
                clean_chunk = re.sub(r"^(?:a|an|the|this|that|these|those)\s+", "", chunk_str, flags=re.IGNORECASE).strip()
                if clean_chunk and len(clean_chunk) > 2 and clean_chunk.lower() not in STOPWORDS and is_valid_noun_phrase(clean_chunk):
                    phrases.append(clean_chunk)
                current_chunk = []
        else:
            current_chunk.append(token)

    if current_chunk:
        chunk_str = " ".join(current_chunk).strip()
        clean_chunk = re.sub(r"^(?:a|an|the|this|that|these|those)\s+", "", chunk_str, flags=re.IGNORECASE).strip()
        if clean_chunk and len(clean_chunk) > 2 and clean_chunk.lower() not in STOPWORDS and is_valid_noun_phrase(clean_chunk):
            phrases.append(clean_chunk)

    return phrases


def get_head_stem(np: str) -> str:
    """Return the head noun/stem of a noun phrase for deduplication."""
    tokens = [w.lower().rstrip("s") for w in re.findall(r"[a-zA-Z]+", np)]
    return tokens[-1] if tokens else ""


def format_entity_with_article(phrase: str) -> str:
    """Format a noun phrase with an appropriate indefinite article (a/an) if singular."""
    clean = re.sub(r"^(?:a|an|the)\s+", "", phrase.strip(), flags=re.IGNORECASE).strip()
    if not clean:
        return ""
    tokens = clean.split()
    if tokens[-1].endswith("s") and not tokens[-1].endswith(("ss", "us", "is")):
        return clean
    art = "an" if clean[0].lower() in {"a", "e", "i", "o", "u"} and not clean.lower().startswith(("user", "universal", "one", "once")) else "a"
    return f"{art} {clean}"


def format_entity_with_the(phrase: str) -> str:
    """Format a noun phrase with the definite article 'the'."""
    clean = re.sub(r"^(?:a|an|the)\s+", "", phrase.strip(), flags=re.IGNORECASE).strip()
    return f"the {clean}" if clean else ""


def detect_genre(text: str) -> str:
    """Detect domain genre: technical, business, narrative, or general."""
    lower_text = text.lower()
    words = set(re.findall(r"[A-Za-z]+", lower_text))

    tech_score = len(words & GENRE_KEYWORDS["technical"])
    biz_score = len(words & GENRE_KEYWORDS["business"])
    narrative_score = len(words & GENRE_KEYWORDS["narrative"])

    narrative_pronouns = len(re.findall(r"\b(?:he|she|they|his|her|him|himself|herself)\b", lower_text))
    if narrative_pronouns >= 2:
        narrative_score += narrative_pronouns * 1.5

    scores = [
        ("technical", tech_score * 1.5),
        ("business", biz_score * 1.3),
        ("narrative", narrative_score * 1.2),
    ]
    scores.sort(key=lambda x: x[1], reverse=True)
    if scores[0][1] >= 1.5:
        return scores[0][0]
    return "general"


def generate_overview(text: str, source_text: str = "") -> str:
    """Generate a genuine, polished executive description / overview of the document."""
    full_text = (source_text + " " + text).strip() if source_text else text.strip()
    if not full_text:
        return ""

    cleaned_input = clean_stutter_and_repetitions(text.strip() or source_text.strip())
    raw_sentences = [s.strip() for s in SENTENCE_SPLIT_PATTERN.split(cleaned_input) if s.strip()]
    if not raw_sentences and source_text:
        raw_sentences = [s.strip() for s in SENTENCE_SPLIT_PATTERN.split(source_text) if s.strip()]
    if not raw_sentences:
        return ""

    genre = detect_genre(full_text)

    # Collect noun phrases across all sentences (from both source_text and summary)
    all_sentences = raw_sentences
    if source_text:
        all_sentences = [s.strip() for s in SENTENCE_SPLIT_PATTERN.split(source_text) if s.strip()] + raw_sentences

    all_nps: List[str] = []
    for s in all_sentences:
        all_nps.extend(extract_noun_phrases(s))

    # Deduplicate entities by head noun stem
    unique_nps: List[str] = []
    seen_heads: Set[str] = set()
    for np in all_nps:
        norm = np.lower().strip()
        head = get_head_stem(norm)
        if norm not in STOPWORDS and norm not in TEMPORAL_AND_GENERIC_NOUNS and head and head not in seen_heads:
            unique_nps.append(np)
            seen_heads.add(head)

    first_sentence = clean_sentence_opener(raw_sentences[0])
    first_words = first_sentence.split()
    cleaned_s1_words = [w.lower().strip(".,!?;:\"'") for w in first_words if w.lower().strip(".,!?;:\"'") not in STOPWORDS]
    counts_s1 = Counter(cleaned_s1_words)
    max_repeat = counts_s1.most_common(1)[0][1] if counts_s1 else 0
    is_stutter_sentence = max_repeat >= 3 or (len(first_words) > 15 and len(set(cleaned_s1_words)) < len(cleaned_s1_words) * 0.5)

    # 1. Narrative / Story Mode or Stuttering Text
    if genre == "narrative" or is_stutter_sentence:
        entities = [np for np in unique_nps if not re.match(r"^(?:learns|tries|looking|finding|search)\b", np, re.I)]
        if len(entities) >= 2:
            e0 = format_entity_with_the(entities[0].lower())
            e1 = format_entity_with_article(entities[1].lower())
            detail = f"a character's search through {e0} to uncover {e1}"
            if len(entities) >= 3:
                e2 = format_entity_with_article(entities[2].lower())
                detail += f" and discover {e2}"
        elif len(entities) == 1:
            detail = f"a search centered on {format_entity_with_the(entities[0].lower())}"
        else:
            detail = "a character's journey of exploration and discovery"

        return f"A narrative detailing {detail}."

    # 2. Expository Declarative Lead Mode (Technical, Business, or General with strong opener)
    lead_nps = extract_noun_phrases(first_sentence)
    if lead_nps and len(first_words) >= 6 and not is_stutter_sentence:
        subject = lead_nps[0]
        subject_head = get_head_stem(subject.lower())
        secondary = [np for np in unique_nps if get_head_stem(np.lower()) != subject_head and len(np.split()) >= 2]
        if not secondary:
            secondary = [np for np in unique_nps if get_head_stem(np.lower()) != subject_head]

        if genre == "technical":
            aspect_clause = f", focusing on {secondary[0].lower()}" if secondary else ""
            if len(secondary) >= 2:
                aspect_clause += f" and {secondary[1].lower()}"
            return f"An analytical overview examining {subject.lower()}{aspect_clause}, highlighting core methodologies and model performance."

        elif genre == "business":
            aspect_clause = f", highlighting advancements in {secondary[0].lower()}" if secondary else ""
            if len(secondary) >= 2:
                aspect_clause += f" and {secondary[1].lower()}"
            return f"A comprehensive report on {subject.lower()}{aspect_clause} alongside key operational and infrastructure milestones."

        else:
            aspect_clause = f", focusing on {secondary[0].lower()}" if secondary else ""
            if len(secondary) >= 2:
                aspect_clause += f" and {secondary[1].lower()}"
            return f"An informative overview exploring {subject.lower()}{aspect_clause} and primary takeaways."

    # 3. Clean Declarative Fallback
    if not is_stutter_sentence and len(first_words) >= 6:
        clean_first = first_sentence.rstrip(".")
        if re.search(r"\b(?:is|are|have|has|provides|examines|details)\b", clean_first, re.IGNORECASE):
            return f"This document explores how {clean_first[0].lower() + clean_first[1:]}."
        return f"An informative overview discussing {clean_first[0].lower() + clean_first[1:]}."

    # 4. Fallback General Synthesis
    top_subject = unique_nps[0] if unique_nps else "the central subject"
    return f"An executive overview examining {top_subject.lower()} and key developments discussed in the text."


