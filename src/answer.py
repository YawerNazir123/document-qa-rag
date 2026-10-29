"""Day 3 — answers with citations.

Usage (from the project root, with the venv active)::

    python -m src.answer "Kashmiri name for Greylag Goose?" [--k 5] [--index index]

Reuses ``Retriever`` from Day 2 (no new index, no re-ingest). The answer
is EXTRACTIVE, not generative — there is no LLM in this project and no
API key. It quotes / paraphrases the top chunk's own text and names the
species it came from. It never adds a fact that is not in a retrieved
chunk.

Abstention (the honest half of Day 3): when the question names a species
that is not in the corpus, or the top chunk shares no words with the
question (top score 0), the CLI prints "I don't know — not in the
documents" instead of guessing. This is the behaviour the Day 2 video
promises ("if it is not in the documents, it says so").

TF-IDF note: on the TF-IDF backend a paraphrase question can still
retrieve the wrong chunk with a non-zero score (words, not meaning) —
e.g. "what do locals call the owl that hunts at night in barns?" picks
Black-crowned Night Heron. That is a wrong answer, recorded honestly in
the evaluation, not an abstention: abstention only fires when the
evidence for an abstention is clear (unknown species, or no overlap).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieve import Retriever  # noqa: E402

ABSTAIN_TEXT = "I don't know — not in the documents"

# Words that carry no meaning for sentence scoring.
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "what", "which", "who",
    "how", "why", "when", "where", "do", "does", "did", "in", "on", "of",
    "for", "to", "by", "called", "call", "locals", "kashmir", "kashmiri",
    "name", "bird", "birds", "that", "it", "its", "this", "long",
}

# Question intent -> chunk sentence label to prefer.
FIELD_LABELS = [
    ("kashmiri name", "Kashmiri name:"),
    ("locals call", "Kashmiri name:"),
    ("called", "Kashmiri name:"),
    ("length", "Length:"),
    ("how long", "Length:"),
    ("wingspan", "Wingspan:"),
    ("weight", "Weight:"),
    ("how heavy", "Weight:"),
    ("family", "Family:"),
    ("region", "Region:"),
    ("migratory", "Migratory status:"),
    ("habitat", "Habitat:"),
    ("conservation", "Conservation status:"),
]


def _sentences(text: str) -> list[str]:
    """Split chunk text into sentences (chunks join fields with '. ')."""
    parts = [s.strip() for s in re.split(r"\.\s+", text.strip())]
    out = []
    for p in parts:
        if not p:
            continue
        if not p.endswith("."):
            p += "."
        out.append(p)
    return out


def _content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z][a-z\-]+", text.lower())
    return {w for w in words if w not in STOPWORDS}


def _corpus_names(retriever: Retriever) -> dict[str, dict]:
    """common_name (lower) -> chunk, for every chunk in the index."""
    names: dict[str, dict] = {}
    for chunk in retriever.store.chunks:
        name = (chunk["metadata"].get("common_name") or "").strip()
        if name:
            names[name.lower()] = chunk
    return names


def _quoted_name(question: str) -> str | None:
    m = re.search(r'"([^"]+)"', question)
    if m:
        return m.group(1).strip()
    m = re.search(r"\bcalled\s+([A-Z][A-Za-z\-]+)\b", question)
    if m:
        return m.group(1).strip()
    return None


def _capitalised_phrases(question: str) -> list[str]:
    """Sequences of 2+ capitalised words, e.g. 'African Penguin'."""
    return re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", question)


def _kashmiri_name_of(chunk: dict) -> str:
    return (chunk["metadata"].get("kashmiri_name") or "").strip()


def _find_by_kashmiri_name(retriever: Retriever, name: str) -> dict | None:
    """Chunk whose Kashmiri name is ``name`` (names may be comma-separated)."""
    wanted = name.strip().lower()
    for chunk in retriever.store.chunks:
        for part in _kashmiri_name_of(chunk).split(","):
            if part.strip().lower() == wanted:
                return chunk
    return None


def _strip_name_prefix(sentence: str, name: str) -> str:
    """Drop a leading "Common Name (Scientific name)" from a sentence —
    the answer already names the species, so it would read twice."""
    if not name:
        return sentence
    prefix = f"{name} ("
    if sentence.startswith(prefix):
        rest = sentence[len(prefix):]
        return rest.split(") ", 1)[-1] if ") " in rest else rest
    return sentence


def _best_sentence(chunk_text: str, question: str) -> str:
    """Pick the sentence that best answers ``question``.

    A sentence carrying the field the question asks about wins first;
    otherwise the sentence sharing the most content words with the
    question wins; the first sentence is the fallback.
    """
    sentences = _sentences(chunk_text)
    if not sentences:
        return chunk_text.strip()
    q_lower = question.lower()
    for intent, label in FIELD_LABELS:
        if intent in q_lower:
            for s in sentences:
                if label in s:
                    return s
    q_words = _content_words(question)
    best, best_score = sentences[0], -1
    for s in sentences:
        score = len(_content_words(s) & q_words)
        if score > best_score:
            best, best_score = s, score
    return best


def answer_question(retriever: Retriever, question: str, k: int = 5) -> dict:
    """Answer ``question`` extractively from the loaded index.

    Returns a dict with ``answer`` (str), ``abstained`` (bool),
    ``citation`` (dict with source/page/species/score, or None) and
    ``evidence`` (the quoted chunk sentence, or None).
    """
    corpus = _corpus_names(retriever)

    # Reverse lookup: "Which bird is called 'Bor' in Kashmir?"
    quoted = _quoted_name(question)
    if quoted:
        chunk = _find_by_kashmiri_name(retriever, quoted)
        if chunk is None:
            return {"answer": ABSTAIN_TEXT, "abstained": True,
                    "citation": None, "evidence": None}
        sentence = _best_sentence(chunk["text"], question)
        meta = chunk["metadata"]
        sentence = _strip_name_prefix(sentence, meta.get("common_name", ""))
        return {
            "answer": f"{meta.get('common_name', '?')}: {sentence}",
            "abstained": False,
            "citation": {"source": meta.get("source", "?"),
                         "page": meta.get("page", "?"),
                         "species": meta.get("common_name", "?"),
                         "score": None},
            "evidence": sentence,
        }

    # The question names a species that is not in the corpus -> abstain.
    for phrase in _capitalised_phrases(question):
        if phrase.lower() not in corpus:
            return {"answer": ABSTAIN_TEXT, "abstained": True,
                    "citation": None, "evidence": None}

    results = retriever.retrieve(question, k=k)
    if not results or results[0]["score"] <= 0:
        return {"answer": ABSTAIN_TEXT, "abstained": True,
                "citation": None, "evidence": None}

    # Prefer a retrieved chunk whose species the question names.
    q_lower = question.lower()
    chosen = results[0]
    for result in results:
        name = (result["chunk"]["metadata"].get("common_name") or "").lower()
        if name and name in q_lower:
            chosen = result
            break

    chunk = chosen["chunk"]
    meta = chunk["metadata"]
    sentence = _best_sentence(chunk["text"], question)
    sentence = _strip_name_prefix(sentence, meta.get("common_name", ""))
    name = meta.get("common_name", "")
    return {
        "answer": f"{name}: {sentence}",
        "abstained": False,
        "citation": {"source": meta.get("source", "?"),
                     "page": meta.get("page", "?"),
                     "species": meta.get("common_name", "?"),
                     "score": round(float(chosen["score"]), 3)},
        "evidence": sentence,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Answer a question from the documents, with citations (Day 3).")
    parser.add_argument("question", help="The question to ask the documents.")
    parser.add_argument("--k", type=int, default=5, help="Number of chunks to retrieve (default: 5).")
    parser.add_argument("--index", type=Path, default=Path("index"), help="Index folder (default: index).")
    args = parser.parse_args()

    retriever = Retriever(args.index)
    print(f"Index    : {args.index} ({retriever.n_chunks} chunks, {retriever.store.backend} store)")
    print(f"Embedding: {retriever.backend}")
    print(f"Question : {args.question!r}\n")
    result = answer_question(retriever, args.question, k=args.k)
    print(f"Answer   : {result['answer']}")
    if result["citation"]:
        c = result["citation"]
        score = f", retrieval score {c['score']}" if c.get("score") is not None else ""
        print(f"Citation : {c['source']} (page {c['page']}){score}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
