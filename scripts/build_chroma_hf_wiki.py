import argparse, uuid, math
from datasets import load_dataset
import chromadb
from ollama import Client
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/chroma")
    ap.add_argument("--name", default="wiki_mini")
    ap.add_argument("--embed_model", default="all-minilm")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--limit", type=int, default=20000, help="max passages to ingest")
    ap.add_argument("--batch", type=int, default=128, help="batch size for add()")
    args = ap.parse_args()

    client = chromadb.PersistentClient(path=args.db)

    class EF:
        def __init__(self, host, model):
            self.c = Client(host=host)
            self.m = model
        def __call__(self, input):
            # Chroma expects __call__(input=list[str]); Ollama client accepts input=list[str]
            return self.c.embed(model=self.m, input=list(input))["embeddings"]
    ef = EF(args.host, args.embed_model)
    coll = client.get_or_create_collection(name=args.name, embedding_function=ef, metadata={"hnsw:space": "cosine"})

    ds = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus", split="passages")

    ids, docs, metas = [], [], []
    added = 0
    for i, row in tqdm(enumerate(ds), total=min(len(ds), args.limit)):
        if i >= args.limit:
            break
        text = row.get("text") or row.get("passage") or row.get("content") or ""
        if not text:
            continue
        title = row.get("title") or row.get("article_title") or "wikipedia"
        ids.append(str(uuid.uuid4()))
        docs.append(text)
        metas.append({"source": "wiki", "title": title})
        if len(docs) >= args.batch:
            coll.add(ids=ids, documents=docs, metadatas=metas)
            added += len(docs)
            ids, docs, metas = [], [], []
    if docs:
        coll.add(ids=ids, documents=docs, metadatas=metas)
        added += len(docs)
    print(f"Added {added} docs to Chroma collection '{args.name}' at {args.db}")

if __name__ == "__main__":
    main()
