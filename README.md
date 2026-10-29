# Document Q&A / RAG — answers with citations, not vibes

A document question-answering system built over four days (26–29 Oct 2026).
Ask a question about your documents and get an answer that points back to
the exact passage it came from.

**Built-in corpus:** the Kashmir Birds field guide — 744 species, with
common name, scientific name, Kashmiri name, family, size, migratory
status, region, habitat and description. You can also upload your own
PDF / text / Markdown documents.

## The four days

| Day | What | Status |
| --- | ---- | ------ |
| 1 | **Document ingestion** — loaders, chunking, embeddings, local vector index | ✅ Done |
| 2 | **Retrieval** — find the right chunks for a question | ✅ Done |
| 3 | **Answers with citations + evaluation** — extractive answers, abstention, 10-question golden set | ✅ Done |
| 4 | **Finished repo + API** — FastAPI service, Docker serving, full test suite | ✅ Done |

## Day 1 — ingestion, what it does

- **Loaders** (`src/loaders.py`): builds one text document per species
  from `bird_database/*/info.json` (fetched from
  [kashmir-birds-ai](https://github.com/YawerNazir123/kashmir-birds-ai),
  images not included), plus `.pdf` (pypdf, one document per page so page
  numbers survive into citations), `.txt` and `.md` from `data/uploads/`.
- **Chunking** (`src/chunking.py`): ~512 tokens per chunk with ~50 token
  overlap, so no fact is lost at a boundary. Token counts are a
  word-based approximation (1 token ≈ 0.75 words) — no paid tokenizer.
  Each chunk carries metadata: source, species, scientific name, page,
  chunk id.
- **Embeddings** (`src/embeddings.py`): free and local only, no paid API
  key. Uses `sentence-transformers` (all-MiniLM-L6-v2, 384-dim) when it is
  installed; otherwise falls back to TF-IDF (scikit-learn). **Day 1 as
  shipped uses TF-IDF** — an install of sentence-transformers was
  attempted and was still downloading torch (~900 MB) after 12+ minutes,
  so it was stopped; the code tries it first and the fallback is
  automatic, and dense embeddings are the planned Day 2 upgrade.
- **Vector store** (`src/store.py`): FAISS (`IndexFlatIP` on normalised
  vectors = cosine similarity) when installed, otherwise a plain NumPy
  index. Saved locally under `index/` — no server, no paid service.

## Day 1 — verified numbers

Run from the project root with the venv active:

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m src.ingest
```

Real results (26 Oct 2026):

- **744 documents ingested, 744 species, 744 chunks.** Each species
  entry is short (~60–100 words), so every species fits in a single
  chunk — the chunker is exercised by the unit tests, not by the corpus.
- **Embedding backend: TF-IDF. Vector store: FAISS.**
- **Spot-check: PASS.** Query *"Kashmiri name for Greylag Goose?"*
  retrieved the Greylag Goose chunk, which contains:
  > "Greylag Goose (Anser anser) Kashmiri name: **Ans**. Family:
  > Anatidae. … Length: 74–91 cm. Wingspan: 147–180 cm. Weight:
  > 2,160–4,560 g."
- **Tests: 4 passed** (`tests/test_ingest.py`).

## Day 2 — retrieval, what it does

- **Retrieval** (`src/retrieve.py`): CLI `python -m src.retrieve "question" [--k 5]`
  loads the Day 1 index without re-ingesting, embeds the question with
  the same embedder as ingestion, and prints the top-k chunks ranked by
  cosine similarity — each with score, species, source + page, and text,
  everything the Day 3 answer step needs to cite its sources.
- **TF-IDF detail:** the index stores vectors but not the fitted
  vectorizer, so for the TF-IDF backend the vectorizer is re-fitted on
  the indexed chunk texts (same corpus, same parameters → same
  vocabulary and idf weights). A test verifies the re-fit reproduces
  the stored vectors (cosine > 0.999). If the backend ever changes,
  re-run `python -m src.ingest` — retrieval refuses a dimension
  mismatch instead of silently returning wrong results.

## Day 2 — verified numbers

Run from the project root with the venv active:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m src.retrieve "Kashmiri name for Greylag Goose?" --k 5
```

Real results (27 Oct 2026):

- **Embedding backend: TF-IDF. Vector store: FAISS.** (The
  sentence-transformers install was retried with `TMPDIR` pointed at
  the workspace — the earlier failure was pip using the 512 MB `/tmp`
  tmpfs — and was still downloading torch when Day 2 was verified.
  Dense embeddings land only when that install finishes *and* a
  re-ingest + the paraphrase test pass with it.)
- **Retrieval test set: 4/5 at rank 1.** Greylag Goose → "Ans" (0.458),
  Barn Owl → "Rata Mogul" (0.443), Black-headed Gull → "Krind"
  (0.451), and the reverse lookup "Ababeel" → Asian House Martin
  (0.109, narrowly ahead of #2 at 0.105).
- **Paraphrase test: FAIL on TF-IDF, as expected.** "What do locals
  call the owl that hunts at night in barns?" returns Black-crowned
  Night Heron at rank 1 (0.114); Barn Owl is not in the top 5. TF-IDF
  matches words, not meaning — this is the measured gap dense
  embeddings are meant to close.
- **Tests: 11 passed, 1 xfailed** (`tests/test_ingest.py` +
  `tests/test_retrieve.py`; the xfail is the paraphrase test,
  recorded honestly until the dense backend lands).

## Day 3 — answers with citations + evaluation

- **Answering** (`src/answer.py`): CLI `python -m src.answer "question" [--k 5]`
  reuses the Day 2 `Retriever` (no new index, no re-ingest). The answer
  is EXTRACTIVE — it quotes the best sentence from the retrieved chunk
  and names the species it came from, with the source file + page and
  the retrieval score as its citation. No LLM, no API key, and it never
  adds a fact that is not in a retrieved chunk. Reverse lookups
  ("Which bird is called 'Bor' in Kashmir?") scan the retrieved chunks
  for the Kashmiri name instead of trusting rank 1 alone.
- **Abstention:** when the question names a species that is not in the
  corpus (checked against all 744 common names), or the top chunk
  shares no words with the question (top score 0), it prints
  "I don't know — not in the documents" instead of guessing.
- **Evaluation** (`evaluate.py`): runs the 10-question golden set
  (ground truth read from the indexed chunks) and reports fact
  accuracy, citation accuracy and abstention correctness.

## Day 3 — verified numbers

Run from the project root with the venv active:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python evaluate.py
.venv/bin/python -m src.answer "Kashmiri name for Greylag Goose?"
```

Real results (28 Oct 2026):

- **Embedding backend: TF-IDF. Vector store: FAISS.**
  sentence-transformers is still not installed in this environment
  (checked 28 Oct 2026: `import sentence_transformers` fails), and
  the CPU-torch install attempt failed with an HTTP 403 from the
  PyTorch index — so Day 3 ships on TF-IDF, as Day 2 did. The dense
  upgrade still needs another route (his laptop, or a session where
  Hugging Face downloads work).
- **Evaluation: 8/9 facts correct, 8/9 citations correct, 1/1
  abstention correct.** Greylag Goose → "Ans" (0.458), Barn Owl →
  "Rata Mogul" (0.443), Black-headed Gull → "Krind" (0.451),
  "Ababeel" → Asian House Martin, Alpine Accentor → "Zanda tsar",
  Greylag Goose length 74–91 cm, Barn Swallow → "Katij", "Bor" →
  Black-crowned Night Heron. The African Penguin question correctly
  abstains — no penguin is in the corpus.
- **The one miss is the known TF-IDF paraphrase gap.** "What do locals
  call the owl that hunts at night in barns?" answers Black-crowned
  Night Heron ("Bor", 0.114) instead of Barn Owl ("Rata Mogul") —
  TF-IDF matches words, not meaning.
- **Tests: 17 passed, 1 xfailed** (`tests/test_ingest.py` +
  `tests/test_retrieve.py` + `tests/test_answer.py`; the xfail is
  still the Day 2 paraphrase retrieval test).

## Day 4 — API + finished repo

- **API** (`src/api.py`): FastAPI service wrapping Days 1–3. The index
  loads lazily on first use, so `/` stays reachable before ingestion
  and `/health` reports honestly whether an index is loaded; `/ask`
  and `/search` return HTTP 503 with a plain-language fix
  (`python -m src.ingest`) when no usable index exists, instead of a
  traceback. Interactive docs at `/docs` once the server is running.
- **Endpoints:** `GET /`, `GET /health`, `GET /index-info`,
  `POST /ask` (`{"question": "...", "k": 5}` → answer + citation or
  abstention), `POST /search` (top-k chunks, ranked, with scores).
- **Docker:** the `Dockerfile` now ingests at build time and serves
  the API (`docker compose up` → http://localhost:8000); the old
  ingest-only command is gone.
- **Tests:** 8 new API tests (`tests/test_api.py`) against the real
  local index — cited answers, abstention, ranked search, and
  validation (empty question, oversized `k`).

## Day 4 — verified numbers

Run from the project root with the venv active:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m uvicorn src.api:app --port 8000
curl http://localhost:8000/health
curl -X POST http://localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "Kashmiri name for Greylag Goose?"}'
```

Real results (29 Oct 2026):

- **Tests: 25 passed, 1 xfailed** (the xfail is still the Day 2
  paraphrase retrieval test — TF-IDF matches words, not meaning).
- **Evaluation re-run: 8/9 facts, 8/9 citations, 1/1 abstention** —
  unchanged from Day 3, so the API layer added no regression.
- **Live API smoke test:** `/health` → `{"status": "ok",
  "index_loaded": true, "n_chunks": 744, "backend": "tfidf"}`;
  `/ask` "Kashmiri name for Greylag Goose?" → "Greylag Goose:
  Kashmiri name: Ans." with citation
  `bird_database/greylag_goose/info.json` (page 1, score 0.458);
  `/search` "Kashmiri name for Barn Owl?" ranks Barn Owl first;
  `/docs` returns 200.

## Usage

```bash
# Ingest the built-in corpus + anything in data/uploads/
python -m src.ingest

# Serve the API (interactive docs at http://localhost:8000/docs)
python -m uvicorn src.api:app --reload

# Or with Docker (ingests at build time, serves on port 8000)
docker compose up

# Ask a question — top 5 chunks, ranked, with sources
python -m src.retrieve "Kashmiri name for Greylag Goose?"

# Answer a question — extractive answer + citation, or an abstention
python -m src.answer "Kashmiri name for Greylag Goose?"
python -m src.answer "What is the Kashmiri name of the African Penguin?"

# Run the 10-question evaluation
python evaluate.py

# Fewer results
python -m src.retrieve "Which bird is called Ababeel in Kashmir?" --k 3

# Ingest only your own documents
python -m src.ingest --bird-db /nonexistent --docs data/uploads --out index

# Or point at a different bird database checkout
python -m src.ingest --bird-db /path/to/bird_database --out index
```

Drop your own PDFs / text files into `data/uploads/` and re-run
`python -m src.ingest` — they are chunked and indexed alongside the
birds, with their filename and page in the metadata.

## Project layout

```
src/
  loaders.py      bird corpus + PDF/text/Markdown loaders
  chunking.py     ~512-token chunks, ~50-token overlap
  embeddings.py   sentence-transformers, TF-IDF fallback
  store.py        FAISS / NumPy local vector store
  ingest.py       Day 1 CLI: ingest + spot-check
  retrieve.py     Day 2 CLI: retrieve top-k chunks for a question
  answer.py       Day 3 CLI: extractive answer + citation, or abstention
  api.py          Day 4: FastAPI service (/ask, /search, /health, /index-info)
tests/
  test_ingest.py  chunking, loaders, Kashmiri-name rendering
  test_retrieve.py  retrieval test set (5 questions), top-k, metadata
  test_answer.py  Day 3: facts, citations, reverse lookup, abstention
  test_api.py     Day 4: API answers, abstention, ranked search, validation
evaluate.py     Day 3: 10-question golden-set evaluation
data/
  bird_database/  744 species info.json files (from kashmir-birds-ai)
  uploads/        your own PDFs / text files go here
make_workflow_image.py  generates the pipeline diagram for the LinkedIn
  posts (needs matplotlib; run `python make_workflow_image.py --day 4`).
  The generated PNGs are post assets and are not committed (binary).
index/            built vector index (generated, gitignored)
```

## Honest limits (Day 3)

- Chunk sizes are approximate (word-based token estimate).
- TF-IDF embeddings match words, not meaning — measured on 28 Oct
  2026: 8/9 eval facts right, but the paraphrase question ("the owl
  that hunts at night in barns") still answers the wrong bird.
  Dense embeddings (sentence-transformers) are the planned upgrade
  for retrieval quality, but they cannot be installed in this
  environment yet (see Day 3 numbers above).
- Answers are extractive: the system quotes the chunk, it does not
  compose new sentences, and it can only answer what is in the corpus.

## Licence

MIT — see `LICENSE`.
