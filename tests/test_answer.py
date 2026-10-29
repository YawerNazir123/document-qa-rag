"""Day 3 — answer tests, run against the real Day 1 index (744 chunks).

The evaluation set (expected answers verified against
``data/bird_database/*/info.json``) is exercised end-to-end in
``evaluate.py``; these tests pin the contract: exact-name questions
return the right fact with a citation whose source + page match the
retrieved chunk, reverse lookups return the right bird, out-of-corpus
questions abstain instead of guessing, and every answer carries a
citation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.answer import ABSTAIN_TEXT, answer_question
from src.retrieve import Retriever

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def retriever() -> Retriever:
    return Retriever(PROJECT_ROOT / "index")


def test_exact_name_question_returns_right_fact_and_citation(retriever):
    result = answer_question(retriever, "Kashmiri name for Greylag Goose?")
    assert not result["abstained"]
    assert "Ans" in result["answer"]
    assert result["citation"]["species"] == "Greylag Goose"
    assert result["citation"]["source"] == "bird_database/greylag_goose/info.json"
    assert result["citation"]["page"] == 1
    assert result["citation"]["score"] is not None and result["citation"]["score"] > 0


def test_length_question_returns_measurement_with_citation(retriever):
    result = answer_question(retriever, "How long is the Greylag Goose?")
    assert not result["abstained"]
    assert "74–91 cm" in result["answer"]
    assert result["citation"]["species"] == "Greylag Goose"


def test_reverse_lookup_kashmiri_name_to_bird(retriever):
    result = answer_question(retriever, 'Which bird is called "Bor" in Kashmir?')
    assert not result["abstained"]
    assert result["citation"]["species"] == "Black-crowned Night Heron"
    assert "Bor" in result["answer"]


def test_out_of_corpus_question_abstains(retriever):
    result = answer_question(retriever, "What is the Kashmiri name of the African Penguin?")
    assert result["abstained"]
    assert result["answer"] == ABSTAIN_TEXT
    assert result["citation"] is None


def test_unknown_kashmiri_name_abstains(retriever):
    result = answer_question(retriever, 'Which bird is called "Zanzibar" in Kashmir?')
    assert result["abstained"]
    assert result["answer"] == ABSTAIN_TEXT


def test_every_answer_carries_citation_with_source_and_page(retriever):
    for question in [
        "Kashmiri name for Greylag Goose?",
        "What is the Kashmiri name of the Barn Owl?",
        "Kashmiri name for Black-headed Gull?",
        "How long is the Greylag Goose?",
    ]:
        result = answer_question(retriever, question)
        assert not result["abstained"], question
        assert result["citation"]["source"], question
        assert "page" in result["citation"], question
        assert result["evidence"], question
