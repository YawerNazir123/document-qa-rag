"""Day 1 — document ingestion.

Usage (from the project root, with the venv active)::

    python -m src.ingest [--bird-db data/bird_database] [--docs data/uploads] [--out index]

Builds the corpus, chunks it, embeds it, saves a local vector index and
runs a keyword spot-check: the query "Kashmiri name for Greylag Goose?"
must retrieve a chunk containing "Ans".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking import chunk_text  # noqa: E402
from src.embeddings import Embedder  # noqa: E402
from src.loaders import load_bird_corpus, load_documents  # noqa: E402
from src.store import VectorStore  # noqa: E402

SPOT_CHECK_QUERY = "Kashmiri name for Greylag Goose?"
SPOT_CHECK_NEEDLE = "Ans"


def ingest(bird_db: Path | None, docs_dir: Path | None, out_dir: Path) -> dict:
    documents = []
    if bird_db is not None and bird_db.exists():
        documents.extend(load_bird_corpus(bird_db))
    if docs_dir is not None and docs_dir.exists():
        documents.extend(load_documents(docs_dir))

    species = {d["metadata"]["common_name"] for d in documents if d["metadata"].get("common_name")}
    chunks: list[dict] = []
    for doc in documents:
        chunks.extend(chunk_text(doc["text"], doc["metadata"]))

    embedder = Embedder()
    vectors = embedder.fit_transform([c["text"] for c in chunks])
    store = VectorStore(vectors.shape[1])
    store.add(vectors, chunks)
    store.save(out_dir)

    # Spot-check: keyword retrieval must find the Greylag Goose chunk.
    query_vec = embedder.transform([SPOT_CHECK_QUERY])
    results = store.search(query_vec[0], k=5)
    retrieved_text = " ".join(r["chunk"]["text"] for r in results)
    spot_check_pass = SPOT_CHECK_NEEDLE in retrieved_text
    spot_check_chunk = next(
        (r["chunk"] for r in results if SPOT_CHECK_NEEDLE in r["chunk"]["text"]), None
    )

    return {
        "n_documents": len(documents),
        "n_species": len(species),
        "n_chunks": len(chunks),
        "embedding_backend": embedder.backend,
        "store_backend": store.backend,
        "spot_check_pass": spot_check_pass,
        "spot_check_chunk": spot_check_chunk,
        "spot_check_top_score": results[0]["score"] if results else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest documents for the RAG system (Day 1).")
    parser.add_argument("--bird-db", type=Path, default=Path("data/bird_database"))
    parser.add_argument("--docs", type=Path, default=Path("data/uploads"))
    parser.add_argument("--out", type=Path, default=Path("index"))
    args = parser.parse_args()

    report = ingest(args.bird_db, args.docs, args.out)
    print(f"Documents ingested : {report['n_documents']}")
    print(f"Species ingested   : {report['n_species']}")
    print(f"Chunks             : {report['n_chunks']}")
    print(f"Embedding backend  : {report['embedding_backend']}")
    print(f"Vector store       : {report['store_backend']}")
    print(f"Spot-check query   : {SPOT_CHECK_QUERY!r}")
    print(f"Spot-check result  : {'PASS' if report['spot_check_pass'] else 'FAIL'}")
    if report["spot_check_chunk"]:
        meta = report["spot_check_chunk"]["metadata"]
        print(f"  retrieved chunk  : {meta.get('common_name')} (chunk {meta.get('chunk_id')})")
        print(f"  retrieved text   : {report['spot_check_chunk']['text'][:300]}")
    return 0 if report["spot_check_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
