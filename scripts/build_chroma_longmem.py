import argparse, json, uuid
from pathlib import Path
import chromadb
from ollama import Client
from tqdm import tqdm


def load_jsonl_dir(path):
    items = []
    p = Path(path)
    for fp in p.glob("*.jsonl"):
        for line in Path(fp).read_text().splitlines():
            if line.strip():
                try:
                    items.append(json.loads(line))
                except Exception:
                    pass
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/chroma")
    ap.add_argument("--name", default="longmem_docs")
    ap.add_argument("--lite_dir", default="data/longmemeval_lite")
    ap.add_argument("--embed_model", default="all-minilm")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--limit", type=int, default=10000)
    args = ap.parse_args()

    client = chromadb.PersistentClient(path=args.db)

    class EF:
        def __init__(self, host, model):
            self.c = Client(host=host)
            self.m = model
        def __call__(self, input):
            return self.c.embed(model=self.m, input=list(input))["embeddings"]
    ef = EF(args.host, args.embed_model)

    coll = client.get_or_create_collection(name=args.name, embedding_function=ef, metadata={"hnsw:space": "cosine"})

    items = load_jsonl_dir(args.lite_dir)
    ids, docs, metas = [], [], []
    for i, ex in tqdm(list(enumerate(items))[: args.limit]):
        dialogue = ex.get("dialogue") or []
        ev = ex.get("evidence") or []
        for j, turn in enumerate(dialogue):
            txt = (turn.get("content") or "").strip()
            if not txt:
                continue
            ids.append(str(uuid.uuid4()))
            docs.append(txt)
            metas.append({"source": "longmem", "kind": "turn"})
        for k, span in enumerate(ev):
            s = (span or "").strip()
            if not s:
                continue
            ids.append(str(uuid.uuid4()))
            docs.append(s)
            metas.append({"source": "longmem", "kind": "evidence"})
    if docs:
        coll.add(ids=ids, documents=docs, metadatas=metas)
    print(f"Added {len(docs)} docs to '{args.name}' from {args.lite_dir}")


if __name__ == "__main__":
    main()
