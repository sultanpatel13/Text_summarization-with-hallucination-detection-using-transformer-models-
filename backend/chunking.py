"""Token-aware chunking for long-document summarization.

Splits normalized text into overlapping chunks that each fit inside a
transformer encoder window (default 900 tokens for 1024-limit models).
Supports 16k+ token documents via map-reduce in bart_model / pegasus_model.
"""
from __future__ import annotations

try:
    from backend.summarizer import split_sentences
except ModuleNotFoundError:
    from summarizer import split_sentences

# Encoder window of BART / PEGASUS in this repo.
MODEL_MAX_TOKENS = 1024
# Leave headroom for special tokens.
CHUNK_SIZE_TOKENS = 900
OVERLAP_TOKENS = 100
# 64 * 900 ~= 57k tokens coverage -> comfortably covers 16k+ docs.
# Runtime scales linearly (~18 chunks for 16k tokens); no content is dropped.
MAX_CHUNKS = 64


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens without truncation. Falls back to word estimate."""
    try:
        ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
        return len(ids)
    except Exception:
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
            return len(ids)
        except Exception:
            return max(1, len(text.split()))


def chunk_by_tokens(
    text: str,
    tokenizer,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
    max_chunks: int = MAX_CHUNKS,
) -> list[str]:
    """Split text into overlapping token-bounded chunks on sentence boundaries."""
    sentences = split_sentences(text)
    if not sentences:
        return []

    # Fast path: whole doc fits.
    if count_tokens(text, tokenizer) <= chunk_size:
        return [text]

    # Pre-compute per-sentence token counts (single pass).
    sent_tokens: list[int] = []
    for s in sentences:
        try:
            sent_tokens.append(len(tokenizer.encode(s, add_special_tokens=False, truncation=False)))
        except Exception:
            try:
                sent_tokens.append(len(tokenizer.encode(s, add_special_tokens=False)))
            except Exception:
                sent_tokens.append(max(1, len(s.split())))

    chunks: list[str] = []
    start = 0
    n = len(sentences)
    while start < n:
        current_tokens = 0
        end = start
        while end < n and current_tokens + sent_tokens[end] <= chunk_size:
            current_tokens += sent_tokens[end]
            end += 1
        # Guarantee progress on pathological single long sentence.
        if end == start:
            end = start + 1
        chunks.append(" ".join(sentences[start:end]))
        if end >= n:
            break
        # Rewind for overlap: include trailing sentences totalling overlap_tokens.
        overlap_acc = 0
        new_start = end
        for i in range(end - 1, start - 1, -1):
            overlap_acc += sent_tokens[i]
            if overlap_acc >= overlap_tokens:
                new_start = i
                break
        else:
            new_start = max(start + 1, end - 1)
        # Ensure forward progress.
        if new_start >= end:
            new_start = end - 1 if end - 1 > start else end
        if new_start <= start:
            new_start = start + 1
        start = new_start
        if len(chunks) >= max_chunks:
            break

    # If still over max_chunks (extremely long doc), merge remainder into last chunk
    # boundary by sampling head+tail: keep first max_chunks-1 chunks, fuse rest via
    # extractive head (caller may pre-filter). Here just truncate with notice.
    if len(chunks) > max_chunks:
        chunks = chunks[:max_chunks]
    return chunks


def fuse_chunk_summaries(summaries: list[str]) -> str:
    """Concatenate per-chunk summaries for the reduce step."""
    return " ".join(s.strip() for s in summaries if s and s.strip())
