"""Day 4 — FastAPI service for the document Q&A system.

Run from the project root (with the venv active)::

    uvicorn src.api:app --reload

The index must already exist — build it first with ``python -m src.ingest``
(or let Docker build it; see ``Dockerfile``). The API loads the index
lazily on first use, so ``/`` stays reachable even before ingestion,
and ``/health`` reports honestly whether an index is loaded.

Endpoints
---------
GET  /            Service info
GET  /health      Liveness check + index status (chunks, backend)
GET  /index-info  Index metadata (chunks, embedding backend, store)
POST /ask        {"question": "...", "k": 5} -> extractive answer + citation,
                  or an abstention when the documents do not contain it
POST /search     {"question": "...", "k": 5} -> top-k chunks, ranked, with scores
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.answer import answer_question
from src.retrieve import Retriever

INDEX_DIR = Path(os.environ.get("RAG_INDEX", "index"))

_retriever: Retriever | None = None
_retriever_error: str | None = None


def get_retriever() -> Retriever:
    """Return the loaded index, loading it on first use.

    Raises HTTPException(503) with a plain-language fix when the index
    has not been built yet, instead of a bare traceback.
    """
    global _retriever, _retriever_error
    if _retriever is None and _retriever_error is None:
        try:
            _retriever = Retriever(INDEX_DIR)
        except Exception as exc:  # missing/corrupt index
            _retriever_error = str(exc)
    if _retriever is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No usable index found at "
                f"'{INDEX_DIR}'. Build one first with: python -m src.ingest"
            ),
        )
    return _retriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload so the first request is fast; a missing index is reported
    # by /health and by /ask//search, not by a failed startup.
    try:
        get_retriever()
    except HTTPException:
        pass
    yield
    global _retriever, _retriever_error
    _retriever = None
    _retriever_error = None


app = FastAPI(
    title="Document Q&A / RAG API",
    description=(
        "Ask a question about the indexed documents and get an "
        "extractive answer with a citation — or an honest abstention "
        "when the documents do not contain it. No LLM, no API key."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000,
                          description="The question to ask the documents.")
    k: int = Field(5, ge=1, le=20,
                   description="Number of chunks to retrieve (default: 5).")


@app.get("/")
def root() -> dict:
    return {
        "service": "document-qa-rag",
        "version": "1.0.0",
        "corpus": "Kashmir Birds field guide (744 species) + your own documents",
        "endpoints": ["/health", "/index-info", "/ask", "/search", "/docs"],
    }


@app.get("/health")
def health() -> dict:
    try:
        retriever = get_retriever()
    except HTTPException:
        return {"status": "ok", "index_loaded": False,
                "detail": f"No usable index at '{INDEX_DIR}' — run: python -m src.ingest"}
    return {"status": "ok", "index_loaded": True,
            "n_chunks": retriever.n_chunks, "backend": retriever.backend}


@app.get("/index-info")
def index_info() -> dict:
    retriever = get_retriever()
    return {
        "index_dir": str(INDEX_DIR),
        "n_chunks": retriever.n_chunks,
        "embedding_backend": retriever.backend,
        "vector_store": retriever.store.backend,
    }


@app.post("/ask")
def ask(request: QuestionRequest) -> dict:
    retriever = get_retriever()
    result = answer_question(retriever, request.question, k=request.k)
    return {
        "question": request.question,
        "answer": result["answer"],
        "abstained": result["abstained"],
        "citation": result["citation"],
        "evidence": result["evidence"],
    }


@app.post("/search")
def search(request: QuestionRequest) -> dict:
    retriever = get_retriever()
    results = retriever.retrieve(request.question, k=request.k)
    return {
        "question": request.question,
        "results": [
            {
                "rank": i + 1,
                "score": round(float(r["score"]), 3),
                "species": r["chunk"]["metadata"].get("common_name"),
                "source": r["chunk"]["metadata"].get("source"),
                "page": r["chunk"]["metadata"].get("page"),
                "text": r["chunk"]["text"],
            }
            for i, r in enumerate(results)
        ],
    }
