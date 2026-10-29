"""Chunking.

Splits a document into overlapping chunks of about ``chunk_tokens`` tokens
with ``overlap_tokens`` tokens of overlap, so no fact is lost at a boundary
and every chunk can be cited back to its source.

Token counting here is a word-based approximation (1 token ≈ 0.75 words,
the usual rule of thumb for English text) — no paid tokenizer is needed.
Chunk size is therefore "~512 tokens", not exactly 512.
"""

from __future__ import annotations

CHUNK_TOKENS = 512
OVERLAP_TOKENS = 50
_WORDS_PER_TOKEN = 0.75


def _words_for(tokens: int) -> int:
    return max(1, int(tokens * _WORDS_PER_TOKEN))


def chunk_text(
    text: str,
    metadata: dict,
    chunk_tokens: int = CHUNK_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[dict]:
    """Split ``text`` into overlapping chunks, each carrying ``metadata``.

    Returns a list of ``{"text": ..., "metadata": {..., "chunk_id": n}}``.
    """
    words = text.split()
    if not words:
        return []
    chunk_words = _words_for(chunk_tokens)
    overlap_words = _words_for(overlap_tokens)
    step = max(1, chunk_words - overlap_words)

    chunks: list[dict] = []
    start = 0
    chunk_id = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunk_words_text = " ".join(words[start:end])
        meta = dict(metadata)
        meta["chunk_id"] = chunk_id
        chunks.append({"text": chunk_words_text, "metadata": meta})
        chunk_id += 1
        if end >= len(words):
            break
        start += step
    return chunks
