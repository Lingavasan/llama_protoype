# tools/build_wiki_index.py
import os, json, argparse
from itertools import islice
import numpy as np
import hnswlib
from datasets import load_dataset
from tqdm import tqdm

# ---- helpers to read flexible record formats ----
def get_text(rec):
    # Try common keys used by wiki datasets
    for k in ("passage", "text", "content", "chunk", "section_text"):
        if k in rec and rec[k]:
            return rec[k]
    # fallback: sometimes title + snippet
    title = rec.get("title") or ""
    return title

def get_precomputed_embedding(rec):
    # Try common keys
    for k in ("embedding", "vector", "emb", "embedding_ada"):
        if k in rec and rec[k]:
            return np.array(rec[k], dtype="float32")
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=20000, help="how many wiki chunks to index")
    ap.add_argument("--use-precomputed", action="store_true", help="use vectors from dataset if available")
    ap.add_argument("--embed-model", default="all-minilm", help="Ollama embedding model name (if computing)")
    ap.add_argument("--out-prefix", default="data/wiki", help="prefix for index + meta files")
    args = ap.parse_args()

    # 1) stream the HF dataset (train split)
    # NOTE: needs HF login if private/gated
    ds_iter = load_dataset("MemGPT/wikipedia-embeddings", split="train", streaming=True)  # :contentReference[oaicite:2]{index=2}

    texts, vecs, metas = [], [], []
    if not args.use_precomputed:
        # lazy import only when needed
        from ollama import Client
        client = Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))

    for rec in tqdm(islice(ds_iter, args.num), total=args.num, desc="Indexing"):
        if args.use_precomputed:
            v = get_precomputed_embedding(rec)
            if v is None:
                # skip if this row has no vector
                continue
            text = get_text(rec)
        else:
            text = get_text(rec)
            if not text:
                continue
            out = client.embed(model=args.embed_model, input=text)  # local + small
            v = np.array(out["embeddings"][0], dtype="float32")

        # normalize for cosine
        v = v / (np.linalg.norm(v) + 1e-8)
        vecs.append(v)
        metas.append({
            "title": rec.get("title"),
            "id": rec.get("id") or rec.get("page_id"),
            "text": text
        })

    if not vecs:
        raise SystemExit("No vectors collected. Try removing --use-precomputed or increase --num.")

    dim = vecs[0].shape[0]
    X = np.vstack(vecs).astype("float32")

    # 2) build HNSW index
    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=X.shape[0], ef_construction=200, M=16)
    index.add_items(X, list(range(X.shape[0])))
    index.set_ef(64)
    index.save_index(args.out_prefix + ".hnsw")

    # 3) save meta + info (to map ids back to texts)
    with open(args.out_prefix + "_meta.jsonl", "w") as f:
        for m in metas:
            f.write(json.dumps(m) + "\n")
    with open(args.out_prefix + "_info.json", "w") as f:
        json.dump({"dim": int(dim), "count": int(X.shape[0])}, f)

    print(f"Done. Indexed {X.shape[0]} items with dim={dim}")
    print(f"Index: {args.out_prefix}.hnsw")
    print(f"Meta : {args.out_prefix}_meta.jsonl")
    print(f"Info : {args.out_prefix}_info.json")

if __name__ == "__main__":
    main()
