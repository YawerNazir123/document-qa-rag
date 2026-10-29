"""Local vector store.

FAISS (``IndexFlatIP`` on L2-normalised vectors = cosine similarity) when
it is installed; otherwise a plain NumPy index saved as ``.npz`` plus a
JSON file of chunk texts/metadata. Both are local files — no server, no
paid service.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class VectorStore:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.vectors: np.ndarray | None = None
        self.chunks: list[dict] = []
        self.backend = "numpy"
        self._index = None
        try:
            import faiss

            self._index = faiss.IndexFlatIP(dim)
            self.backend = "faiss"
        except Exception:
            self.backend = "numpy"

    def add(self, vectors: np.ndarray, chunks: list[dict]) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)
        self.vectors = vectors
        self.chunks = chunks
        if self._index is not None:
            self._index.add(vectors)

    def search(self, query_vec: np.ndarray, k: int = 5) -> list[dict]:
        """Return the top-``k`` chunks for a query vector, with scores."""
        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        q = q / max(np.linalg.norm(q), 1e-12)
        if self._index is not None:
            scores, idxs = self._index.search(q, min(k, len(self.chunks)))
            return [
                {"score": float(scores[0][i]), "chunk": self.chunks[int(idxs[0][i])]}
                for i in range(len(idxs[0]))
                if idxs[0][i] >= 0
            ]
        sims = (self.vectors @ q.T).ravel()
        top = np.argsort(-sims)[:k]
        return [{"score": float(sims[i]), "chunk": self.chunks[int(i)]} for i in top]

    def save(self, folder: str | Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / "vectors.npz", vectors=self.vectors)
        (folder / "chunks.json").write_text(
            json.dumps(self.chunks, ensure_ascii=False), encoding="utf-8"
        )
        (folder / "meta.json").write_text(
            json.dumps({"dim": self.dim, "backend": self.backend, "n_chunks": len(self.chunks)}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, folder: str | Path) -> "VectorStore":
        folder = Path(folder)
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        store = cls(meta["dim"])
        store.chunks = json.loads((folder / "chunks.json").read_text(encoding="utf-8"))
        store.vectors = np.load(folder / "vectors.npz")["vectors"]
        if store._index is not None:
            store._index.add(store.vectors)
        return store
