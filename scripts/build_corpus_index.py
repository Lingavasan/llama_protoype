import os, json, argparse
from pathlib import Path
from glob import glob
import numpy as np, hnswlib
from tqdm import tqdm
from ollama import Client

def read_docs(corpus_dir):
    paths = []
    for ext in ("*.txt","*.md"):
        paths += glob(str(Path(corpus_dir)/ext))
    docs = []
    for p in paths:
        txt = Path(p).read_text(errors="ignore")
        docs.append((p, txt))
    return docs

def chunk(text, max_chars=800, overlap=120):
    text = " ".join(text.split())  # normalize whitespace
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+max_chars]
        chunks.append(chunk)
        i += max_chars - overlap
    return chunks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--out_prefix", default="data/index/corpus")
    ap.add_argument("--embed_model", default="all-minilm")
    ap.add_argument("--host", default="http://localhost:11434")
    args = ap.parse_args()

    client = Client(host=args.host)
    docs = read_docs(args.corpus)
    if not docs:
        raise SystemExit(f"No .txt/.md files found in {args.corpus}")

    texts, metas = [], []
    for path, txt in docs:
        name = Path(path).name
        for j, ch in enumerate(chunk(txt)):
            texts.append(ch)
            metas.append({"source": name, "chunk_id": j})

    # batch embed
    vecs = []
    B = 64
    for i in tqdm(range(0, len(texts), B), desc="Embedding"):
        batch = texts[i:i+B]
        out = client.embed(model=args.embed_model, input=batch)
        for v in out["embeddings"]:
            v = np.array(v, dtype="float32")
            v = v / (np.linalg.norm(v) + 1e-8)
            vecs.append(v)
    X = np.vstack(vecs).astype("float32")
    dim = X.shape[1]

    # build HNSW index
    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=X.shape[0], ef_construction=200, M=16)
    index.add_items(X, list(range(X.shape[0])))
    index.set_ef(64)
    index.save_index(args.out_prefix + ".hnsw")

    # save meta + info
    with open(args.out_prefix + "_meta.jsonl","w") as f:
        for m, t in zip(metas, texts):
            m = {**m, "text": t}
            f.write(json.dumps(m) + "\n")
    with open(args.out_prefix + "_info.json","w") as f:
        json.dump({"dim": int(dim), "count": int(X.shape[0])}, f)

    print("OK:", args.out_prefix + ".hnsw")

if __name__ == "__main__":
    main()
