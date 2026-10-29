"""Day 2 — retrieval.

Usage (from the project root, with the venv active)::

    python -m src.retrieve "Kashmiri name for Greylag Goose?" [--k 5] [--index index]

Loads the Day 1 index (``index/``) without re-ingesting, embeds the
question with the same embedder as ingestion, and prints the top-k
chunks ranked by cosine similarity, each with its score, species,
source + page, and text — everything the Day 3 answer step needs to
cite its sources.

TF-IDF note: the index stores vectors but not the fitted vectorizer,
so for the TF-IDF backend the vectorizer is re-fitted on the indexed
chunk texts (same corpus, same parameters → same vocabulary and idf
weights → same embedding space). A test verifies the re-fit reproduces
the stored vectors. The sentence-transformers backend needs no re-fit:
the model embeds the query directly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.embeddings import Embedder  # noqa: E402
from src.store import VectorStore  # noqa: E402


class Retriever:
    """A loaded index plus its embedder, ready to answer questions."""

    def __init__(self, index_dir: str | Path = "index") -> None:
        self.index_dir = Path(index_dir)
        self.store = VectorStore.load(self.index_dir)
        self.embedder = Embedder()
        if self.embedder.backend == "tfidf":
            self.embedder.fit([c["text"] for c in self.store.chunks])

    @property
    def backend(self) -> str:
        return self.embedder.backend

    @property
    def n_chunks(self) -> int:
        return len(self.store.chunks)

    def retrieve(self, question: str, k: int = 5) -> list[dict]:
        """Return the top-``k`` chunks for ``question``, ranked by cosine similarity."""
        query_vec = self.embedder.transform([question])
        if query_vec.shape[1] != self.store.vectors.shape[1]:
            raise ValueError(
                f"Query embedding dim {query_vec.shape[1]} does not match "
                f"index dim {self.store.vectors.shape[1]} — the index was "
                f"built with a different embedder backend; re-run "
                f"python -m src.ingest to rebuild it."
            )
        return self.store.search(np.asarray(query_vec[0]), k=k)


def format_result(rank: int, result: dict) -> str:
    chunk = result["chunk"]
    meta = chunk["metadata"]
    name = meta.get("common_name") or meta.get("source", "?")
    sci = meta.get("scientific_name") or ""
    source = meta.get("source", "?")
    page = meta.get("page", "?")
    header = f"[{rank}] {name}"
    if sci:
        header += f" ({sci})"
    header += f" — score {result['score']:.3f}"
    return (
        f"{header}\n"
        f"    source: {source} (page {page})\n"
        f"    {chunk['text']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieve the top chunks for a question (Day 2).")
    parser.add_argument("question", help="The question to ask the documents.")
    parser.add_argument("--k", type=int, default=5, help="Number of chunks to return (default: 5).")
    parser.add_argument("--index", type=Path, default=Path("index"), help="Index folder (default: index).")
    args = parser.parse_args()

    retriever = Retriever(args.index)
    print(f"Index    : {args.index} ({retriever.n_chunks} chunks, {retriever.store.backend} store)")
    print(f"Embedding: {retriever.backend}")
    print(f"Question : {args.question!r}\n")
    results = retriever.retrieve(args.question, k=args.k)
    if not results:
        print("No chunks found.")
        return 0
    if results[0]["score"] <= 0:
        print("Note: top score is 0 — the question shares no words with the index "
              "(TF-IDF matches words, not meaning).\n")
    for i, result in enumerate(results, start=1):
        print(format_result(i, result))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
