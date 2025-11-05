.PHONY: setup smoke index eval-longmem-lite router

setup:
	python3 -m pip install --upgrade pip
	python3 -m pip install -r requirements.txt

smoke:
	python3 scripts/smoke.py

index:
	python3 scripts/build_corpus_index.py

eval-longmem-lite:
	python3 scripts/eval_longmem_lite.py --limit 10

router:
	python3 scripts/train_router.py --sample 5000