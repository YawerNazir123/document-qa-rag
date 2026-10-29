"""Day 2 — retrieval tests, run against the real Day 1 index (744 chunks).

The retrieval test set (expected answers verified against
``data/bird_database/*/info.json``):

1. "Kashmiri name for Greylag Goose?"            -> Greylag Goose, "Ans"
2. "What is the Kashmiri name of the Barn Owl?"  -> Barn Owl, "Rata Mogul"
3. "Kashmiri name for Black-headed Gull?"        -> Black-headed Gull, "Krind"
4. "Which bird is called Ababeel in Kashmir?"   -> Asian House Martin, "Ababeel"
5. "What do locals call the owl that hunts at night in barns?"
   -> Barn Owl, "Rata Mogul" (paraphrase — no shared keywords except
   "barn"/"owl"; TF-IDF is expected to miss this, dense embeddings to
   get it. The test records the honest result either way.)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.retrieve import Retriever

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def retriever() -> Retriever:
    return Retriever(PROJECT_ROOT / "index")


def _top_common_name(retriever: Retriever, question: str, k: int = 5) -> str:
    results = retriever.retrieve(question, k=k)
    assert results, f"no results for {question!r}"
    return results[0]["chunk"]["metadata"].get("common_name", "")


@pytest.mark.parametrize(
    "question,expected",
    [
        ("Kashmiri name for Greylag Goose?", "Greylag Goose"),
        ("What is the Kashmiri name of the Barn Owl?", "Barn Owl"),
        ("Kashmiri name for Black-headed Gull?", "Black-headed Gull"),
    ],
)
def test_exact_name_query_returns_right_species_at_rank_1(retriever, question, expected):
    assert _top_common_name(retriever, question) == expected


def test_reverse_lookup_kashmiri_name_to_bird(retriever):
    # "Ababeel" appears only in the Asian House Martin chunk.
    assert _top_common_name(retriever, "Which bird is called Ababeel in Kashmir?") == "Asian House Martin"


def test_top_k_is_respected(retriever):
    assert len(retriever.retrieve("Kashmiri name for Greylag Goose?", k=1)) == 1
    assert len(retriever.retrieve("Kashmiri name for Greylag Goose?", k=3)) == 3
    assert len(retriever.retrieve("Kashmiri name for Greylag Goose?", k=5)) == 5


def test_every_result_carries_source_and_page(retriever):
    for result in retriever.retrieve("Kashmiri name for Greylag Goose?", k=5):
        meta = result["chunk"]["metadata"]
        assert meta.get("source"), "chunk is missing its source"
        assert "page" in meta, "chunk is missing its page"
        assert isinstance(result["score"], float)


def test_tfidf_refit_reproduces_index(retriever):
    """The TF-IDF vectorizer re-fitted on the chunk texts must reproduce
    the stored vectors — otherwise queries and index live in different
    embedding spaces. Skipped for the dense backend, which needs no re-fit."""
    if retriever.backend != "tfidf":
        pytest.skip("dense backend needs no TF-IDF re-fit")
    chunk = retriever.store.chunks[0]
    vec = retriever.embedder.transform([chunk["text"]])[0]
    stored = retriever.store.vectors[0]
    cosine = float(np.dot(vec, stored) / (np.linalg.norm(vec) * np.linalg.norm(stored)))
    assert cosine > 0.999, f"re-fit vector diverges from stored vector (cosine {cosine:.4f})"


def test_paraphrase_query(retriever):
    """Honest paraphrase result: TF-IDF matches words, not meaning, so it
    is expected to miss this one until dense embeddings land. The test
    xfails on a TF-IDF miss (recorded, suite stays green) and passes
    outright when the backend gets it right."""
    results = retriever.retrieve("What do locals call the owl that hunts at night in barns?", k=5)
    top = results[0]["chunk"]["metadata"].get("common_name", "")
    if retriever.backend == "tfidf" and top != "Barn Owl":
        pytest.xfail(
            "TF-IDF matches words, not meaning — expected miss until "
            f"dense embeddings land (top-1 was {top!r})"
        )
    assert top == "Barn Owl"
