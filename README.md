# Llama Prototype

A lightweight research prototype that explores a retrieval-augmented assistant with reflective drafting and a simple policy layer. Week 2 focuses on Memory Store & Retrieval: building an ANN index (HNSW), integrating a minimal RAG runtime, and wiring a LongMemEval-lite baseline.

## What’s implemented so far

- Core pipeline (`src/pipeline.py`)
  - Policy checks (input/output) from `configs/policy.yaml`
  - Generation via Ollama (`src/llm.py`)
  - Internal memory: append-only JSONL or ChromaDB backend + cosine search (`src/memory.py`)
  - Reflection pass (critique → rewrite) (`src/reflect.py`)
  - Optional policy router support (`src/router.py`, train in `scripts/train_router.py`)
  - External memory (RAG) via HNSW (`src/rag.py`) with safe fallback
  - Per-turn logging (latency, rough tokens, recalls) to `runs/events.jsonl`

- RAG / Indexing
  - Build corpus index with embeddings from Ollama: `scripts/build_corpus_index.py`
  - HNSW runtime in `src/rag.py` (default paths `data/index/corpus.*`)

- Evaluation (Week 2 LongMemEval-lite)
  - Minimal dataset loader and evaluator: `scripts/eval_longmem_lite.py`
  - Metrics: EM, token-level F1, retrieval Hit@5
  - Dialogue replay option (default ON) to populate internal memory prior to the question
  - Sample record in `data/longmemeval_lite/sample.jsonl`

- Utilities
  - Token counter with safe fallback (`src/utils/tokenizer.py`)
  - Minimal logging helper (`src/utils/logging.py`)

- Tooling
  - `Makefile` targets: setup, smoke, index, eval-longmem-lite, router
  - `requirements.txt` for Python deps

## How it works (high level)

1) Input processing
- Policy input check (rejects disallowed content) and tagging (`src/policy.py`, `src/tags.py`).
- Optional policy router predicts an action hint (`src/router.py`).

2) Memory & Retrieval
- Internal memory: previous turns are stored in `memory.jsonl` (JSONL backend) or a ChromaDB collection (default) with normalized embeddings.
- External RAG: HNSW over local corpus (`data/index/corpus.hnsw`), with metadata in JSONL.
- The query is embedded; we retrieve top-K from both internal memory and RAG and merge their texts into the system prompt context.

3) Draft → Reflect → Final
- Draft is produced by the LLM with relevant memory context.
- A critique is generated and used to rewrite the response into a final answer.
- Output policy check + optional disclaimer.
- New turns (user, assistant, reflection) are appended to the active memory backend.

4) Logging
- Each turn logs latency, rough token count, and the recall texts to `runs/events.jsonl`.

## Quick start

- Install deps:
  - `make setup`
- Build index (optional for RAG):
  - `make index`
- Sanity test:
  - `make smoke`
- LongMemEval-lite (with dialogue replay):
  - `make eval-longmem-lite`

Note: Requires an Ollama server and the specified models in `configs/policy.yaml` (e.g., `llama3.2:1b` for chat and `all-minilm` for embeddings). If the tokenizer (transformers) is unavailable, token counting falls back to a rough `len(text)//4` estimate.

Telemetry note: If you see messages like "Failed to send telemetry event ... capture() takes 1 positional argument but 3 were given" from ChromaDB, it's a PostHog API mismatch. We've pinned `posthog<4` in `requirements.txt`. You can also disable Chroma telemetry entirely by setting:

```bash
export CHROMA_TELEMETRY_DISABLED=TRUE
export ANONYMIZED_TELEMETRY=False
```

## Config overview (`configs/policy.yaml`)

- `gen`: model, temperature, num_predict
- `memory`: embed_model, top_k, backend (`chroma`|`jsonl`), `chroma_path`, `collection`, `jsonl_path`
- `external_memory`: enable + index paths (optional; defaults available)
- `router`: optional MemGPT router bundle
- `budget`: future token budgeting defaults (context_capacity, buffer, max_retrieved_chunks)
- `logging.dir`: base output folder for events

## Known limitations / next steps

- Token budget enforcement is not active yet (Week 3 plan).
- The evaluator currently replays dialogue into internal memory; optional ingestion into RAG for larger-scale retrieval tests is a good follow-up.
- Router is optional and not required for Week 2 metrics.
- Push to GitHub is paused until repo history is cleaned of large files.

## Repository hygiene

- `.gitignore` excludes large/generated artifacts: `artifacts/`, `data/index/`, `models/`, `memory.jsonl`, `__pycache__/`.
- If you need to publish, remove large blobs from history (BFG or git filter-repo) and force-push.
