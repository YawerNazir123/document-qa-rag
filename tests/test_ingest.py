import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking import chunk_text
from src.loaders import _bird_text, load_bird_corpus


def test_chunking_overlap_and_metadata():
    chunks = chunk_text(" ".join(f"word{i}" for i in range(1000)), {"source": "x", "page": 1})
    assert len(chunks) > 1
    # ~512 tokens ~= 384 words per chunk, ~50 token overlap ~= 37 words
    assert len(chunks[0]["text"].split()) <= 400
    assert chunks[0]["metadata"]["chunk_id"] == 0
    assert chunks[1]["metadata"]["chunk_id"] == 1
    # overlap: the last 37 words of chunk 0 reappear at the start of chunk 1
    # (50-token overlap ≈ 37 words)
    assert chunks[0]["text"].split()[-37:] == chunks[1]["text"].split()[:37]


def test_chunking_empty_text():
    assert chunk_text("", {"source": "x"}) == []


def test_bird_text_contains_kashmiri_name():
    info = {
        "common_name": "Greylag Goose",
        "scientific_name": "Anser anser",
        "local_name_kashmiri": "Ans",
        "family": "Anatidae",
        "length": "74–91 cm",
        "weight": "2,160–4,560 g",
        "migratory_status": "Winter Migratory",
        "region": "Jammu, Kashmir Valley, Ladakh",
        "description": "A winter migratory bird of Kashmir Valley.",
    }
    text = _bird_text(info)
    assert "Greylag Goose" in text
    assert "Anser anser" in text
    assert "Ans" in text
    assert "Anatidae" in text


def test_load_bird_corpus(tmp_path):
    d = tmp_path / "bird_database" / "greylag_goose"
    d.mkdir(parents=True)
    (d / "info.json").write_text(
        '{"common_name": "Greylag Goose", "scientific_name": "Anser anser", '
        '"local_name_kashmiri": "Ans", "family": "Anatidae"}',
        encoding="utf-8",
    )
    docs = load_bird_corpus(tmp_path / "bird_database")
    assert len(docs) == 1
    assert docs[0]["metadata"]["common_name"] == "Greylag Goose"
    assert docs[0]["metadata"]["kashmiri_name"] == "Ans"
    assert "Ans" in docs[0]["text"]
