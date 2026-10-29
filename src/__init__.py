"""Document Q&A / RAG — Day 1: document ingestion.

Loads documents, chunks them, embeds them and stores them in a local
vector index, so later days can retrieve the right chunks for a question.

Day 1 scope only: ingestion. No retrieval API, no answering, no evaluation
yet — those are Days 2–4.
"""

from .chunking import chunk_text
from .embeddings import build_embeddings
from .loaders import load_bird_corpus, load_documents
from .store import VectorStore

__all__ = [
    "chunk_text",
    "build_embeddings",
    "load_bird_corpus",
    "load_documents",
    "VectorStore",
]
