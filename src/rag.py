import json, numpy as np, hnswlib
from pathlib import Path
try:
    import chromadb  # type: ignore
    from chromadb import PersistentClient  # type: ignore
    _HAS_CHROMA = True
except Exception:
    chromadb = None
    PersistentClient = None  # type: ignore
    _HAS_CHROMA = False


class HNSWRAG:
    def __init__(
        self,
        idx: str | None = None,
        meta: str | None = None,
        info: str | None = None,
        # Backward-compat kwargs
        index_path: str | None = None,
        meta_path: str | None = None,
        info_path: str | None = None,
    ):
        # Resolve paths from either the new (idx/meta/info) or old (index_path/meta_path/info_path) args
        if index_path or meta_path or info_path:
            ip = index_path or "data/index/corpus.hnsw"
            mp = meta_path or "data/index/corpus_meta.jsonl"
            inf = info_path or "data/index/corpus_info.json"
        else:
            ip = (idx or "data/index/corpus.hnsw")
            mp = (meta or "data/index/corpus_meta.jsonl")
            inf = (info or "data/index/corpus_info.json")

        info_obj = json.loads(Path(inf).read_text())
        dim = int(info_obj["dim"])
        self.index = hnswlib.Index(space="cosine", dim=dim)
        self.index.load_index(ip)
        # Load metadata lines
        self.meta = [
            json.loads(l)
            for l in Path(mp).read_text().splitlines()
            if l.strip()
        ]

    def search(self, qvec, k=3):
        q = np.array([qvec], dtype="float32")
        labels, dists = self.index.knn_query(q, k=k)
        out = []
        for idx, dist in zip(labels[0], dists[0]):
            score = 1.0 - float(dist)   # convert cosine distance → similarity-ish
            m = self.meta[int(idx)]
            out.append((score, {"kind":"rag", "title": m["source"], "text": m["text"]}))
        return out


class ChromaRAG:
    """Query a Chroma collection as an external memory source."""
    def __init__(self, path: str = "data/chroma", collection: str = "knowledge"):
        if not _HAS_CHROMA:
            raise ImportError("chromadb is not installed")
        Path(path).mkdir(parents=True, exist_ok=True)
        self.client = PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(name=collection, metadata={"hnsw:space": "cosine"})

    def search(self, qvec, k=3):
        res = self.collection.query(query_embeddings=[qvec], n_results=k, include=["documents", "distances", "metadatas"])
        docs = res.get("documents", [[]])[0]
        dists = res.get("distances", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        out = []
        for doc, dist, meta in zip(docs, dists, metas):
            score = 1.0 - float(dist if dist is not None else 1.0)
            title = (meta or {}).get("title") or (meta or {}).get("source") or "chroma"
            out.append((score, {"kind": "rag", "title": title, "text": doc}))
        return out


class JSONRAG:
    """Embed and search over a JSON/JSONL set of texts using a provided embed_fn."""
    def __init__(self, items, embed_fn):
        # items: iterable of {title?, text}
        self.items = []
        self.vecs = None
        texts = []
        for it in items:
            text = (it.get("text") or "").strip()
            if not text:
                continue
            title = it.get("title") or "json"
            self.items.append({"title": title, "text": text})
            texts.append(text)
        # embed and normalize
        vecs = []
        for t in texts:
            v = np.array(embed_fn(t), dtype="float32")
            n = np.linalg.norm(v) + 1e-8
            vecs.append(v / n)
        self.vecs = np.vstack(vecs) if vecs else np.zeros((0,1), dtype="float32")

    def search(self, qvec, k=3):
        if self.vecs.shape[0] == 0:
            return []
        q = np.array(qvec, dtype="float32")
        q /= (np.linalg.norm(q) + 1e-8)
        sims = self.vecs @ q
        idxs = np.argsort(-sims)[:k]
        out = []
        for i in idxs:
            score = float(sims[i])
            it = self.items[int(i)]
            out.append((score, {"kind": "rag", "title": it["title"], "text": it["text"]}))
        return out
