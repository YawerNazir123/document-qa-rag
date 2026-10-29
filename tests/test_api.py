"""Day 4 — API tests (FastAPI TestClient, against the real local index).

These tests need the built index (``python -m src.ingest`` first) and
the API dependencies (``pip install -r requirements.txt``). They use
the same golden questions as Day 3, so an API regression shows up as
a wrong answer, not just a wrong status code.
"""

from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_root_lists_endpoints():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["service"] == "document-qa-rag"
    assert "/ask" in r.json()["endpoints"]


def test_health_reports_loaded_index():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["index_loaded"] is True
    assert body["n_chunks"] >= 700  # 744 species in the built-in corpus


def test_index_info_matches_health():
    r = client.get("/index-info")
    assert r.status_code == 200
    body = r.json()
    assert body["n_chunks"] >= 700
    assert body["embedding_backend"] in ("tfidf", "sentence-transformers")


def test_ask_returns_cited_answer():
    r = client.post("/ask", json={"question": "Kashmiri name for Greylag Goose?"})
    assert r.status_code == 200
    body = r.json()
    assert body["abstained"] is False
    assert "Ans" in body["answer"]
    assert body["citation"]["species"] == "Greylag Goose"
    assert body["citation"]["source"]


def test_ask_abstains_on_unknown_species():
    r = client.post(
        "/ask", json={"question": "What is the Kashmiri name of the African Penguin?"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["abstained"] is True
    assert body["citation"] is None


def test_search_returns_ranked_chunks():
    r = client.post(
        "/search", json={"question": "Kashmiri name for Barn Owl?", "k": 3}
    )
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 3
    assert results[0]["species"] == "Barn Owl"
    scores = [x["score"] for x in results]
    assert scores == sorted(scores, reverse=True)


def test_ask_rejects_empty_question():
    r = client.post("/ask", json={"question": ""})
    assert r.status_code == 422


def test_ask_rejects_huge_k():
    r = client.post("/ask", json={"question": "Kashmiri name for Greylag Goose?", "k": 999})
    assert r.status_code == 422
