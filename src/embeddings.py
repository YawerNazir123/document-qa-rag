"""Embeddings — free and local only, no paid API key.

Two backends, tried in order:

1. ``sentence-transformers`` (all-MiniLM-L6-v2) if it is installed — real
   dense embeddings, 384 dimensions.
2. TF-IDF (scikit-learn) otherwise — a sparse, offline fallback that
   still embeds and retrieves by cosine similarity. Day 1 ships with
   this backend when sentence-transformers is not installed; see the
   README for which one is actually in use.
"""

from __future__ import annotations

import numpy as np


class Embedder:
    def __init__(self) -> None:
        self.backend = "tfidf"
        self._model = None
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            self.backend = "sentence-transformers"
        except Exception:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
            self.backend = "tfidf"

    @property
    def dim(self) -> int:
        if self.backend == "sentence-transformers":
            return 384
        return 20000

    def fit(self, texts: list[str]) -> None:
        """Fit the TF-IDF vectorizer on ``texts`` (no-op for the dense backend).

        Needed at retrieval time: the Day 1 index stores vectors but not
        the fitted vectorizer, so a query must be embedded with a
        vectorizer fitted on the same corpus with the same parameters —
        which reproduces the same vocabulary and idf weights, and
        therefore the same embedding space as the stored index.
        """
        if self.backend == "sentence-transformers":
            return
        self._vectorizer.fit(texts)

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        if self.backend == "sentence-transformers":
            vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(vecs, dtype=np.float32)
        mat = self._vectorizer.fit_transform(texts)
        return np.asarray(mat.todense(), dtype=np.float32)

    def transform(self, texts: list[str]) -> np.ndarray:
        if self.backend == "sentence-transformers":
            vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(vecs, dtype=np.float32)
        mat = self._vectorizer.transform(texts)
        return np.asarray(mat.todense(), dtype=np.float32)


def build_embeddings(texts: list[str]) -> tuple[Embedder, np.ndarray]:
    """Fit an embedder on ``texts`` and return ``(embedder, vectors)``."""
    embedder = Embedder()
    return embedder, embedder.fit_transform(texts)
