"""Day 3 — evaluation.

Runs the 10-question golden set from ``rag-day3-prep.md`` through
``src.answer`` and prints fact correctness, citation correctness and
abstention correctness. Ground truth was read directly from the
indexed chunks (``index/chunks.json``, built from
``data/bird_database/*/info.json``).

Usage (from the project root, with the venv active)::

    python evaluate.py [--index index]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.answer import answer_question  # noqa: E402
from src.retrieve import Retriever  # noqa: E402

# (question, expected fact substring or None, expected species or None,
#  abstain?)
GOLDEN = [
    ("Kashmiri name for Greylag Goose?", "Ans", "Greylag Goose", False),
    ("What is the Kashmiri name of the Barn Owl?", "Rata Mogul", "Barn Owl", False),
    ("Kashmiri name for Black-headed Gull?", "Krind", "Black-headed Gull", False),
    ("Which bird is called Ababeel in Kashmir?", "Asian House Martin", "Asian House Martin", False),
    ("What do locals call the owl that hunts at night in barns?", "Rata Mogul", "Barn Owl", False),
    ("Kashmiri name for Alpine Accentor?", "Zanda tsar", "Alpine Accentor", False),
    ("How long is the Greylag Goose?", "74–91 cm", "Greylag Goose", False),
    ("What is the Kashmiri name of the Barn Swallow?", "Katij", "Barn Swallow", False),
    ('Which bird is called "Bor" in Kashmir?', "Black-crowned Night Heron", "Black-crowned Night Heron", False),
    ("What is the Kashmiri name of the African Penguin?", None, None, True),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Day 3 answers on the 10-question golden set.")
    parser.add_argument("--index", type=Path, default=Path("index"))
    args = parser.parse_args()

    retriever = Retriever(args.index)
    print(f"Backend: {retriever.backend} ({retriever.n_chunks} chunks)\n")
    fact_ok = cite_ok = abstain_ok = 0
    for i, (question, fact, species, want_abstain) in enumerate(GOLDEN, start=1):
        result = answer_question(retriever, question)
        if want_abstain:
            ok = result["abstained"]
            abstain_ok += int(ok)
            print(f"Q{i:2d} {'PASS' if ok else 'FAIL'} abstain  | {question} -> {result['answer']}")
            continue
        fact_good = (not result["abstained"] and fact in result["answer"]
                     and result["citation"]["species"] == species)
        fact_ok += int(fact_good)
        cite_good = bool(fact_good and result["citation"]["source"]
                         and "page" in result["citation"])
        cite_ok += int(cite_good)
        print(f"Q{i:2d} {'PASS' if fact_good else 'FAIL'} fact     | {question} -> {result['answer']}")
        if result["citation"]:
            c = result["citation"]
            print(f"     citation: {c['source']} (page {c['page']}) score {c.get('score')}")
    print(f"\nFact accuracy:      {fact_ok}/9")
    print(f"Citation accuracy:  {cite_ok}/9 (cited source+page matches the chunk the fact came from)")
    print(f"Abstention:         {abstain_ok}/1 (African Penguin — not in the documents)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
