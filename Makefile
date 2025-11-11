.PHONY: setup smoke index eval-longmem-lite router health rag-build

setup:
	python3 -m pip install --upgrade pip
	python3 -m pip install -r requirements.txt

smoke:
	python3 scripts/smoke.py

index:
	python3 scripts/build_corpus_index.py

.PHONY: knowledge
knowledge:
	python3 scripts/build_chroma_knowledge.py --src data/longmemeval/longmemeval_oracle.json --chroma data/chroma --collection knowledge --embed_model "llama-2-7b-chat.Q4_K_M"

.PHONY: ingest-wiki
ingest-wiki:
	python3 scripts/ingest_hf_wiki.py --subset text-corpus --outdir data/rag_mini_wikipedia --embed_model "llama-2-7b-chat.Q4_K_M" --chroma_path data/chroma --collection wiki

eval-longmem-lite:
	python3 scripts/eval_longmem_lite.py --limit 10

router:
	python3 scripts/train_router.py --sample 5000

health:
	@curl -s http://localhost:11434/api/version && echo

rag-build:
	python3 scripts/build_chroma_hf_wiki.py --limit 20000
	python3 scripts/build_chroma_longmem.py