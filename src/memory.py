import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import numpy as np

try:
    import chromadb  # type: ignore
    from chromadb import PersistentClient  # type: ignore
    try:
        from chromadb.errors import InvalidDimensionException  # type: ignore
    except Exception:  # older versions
        InvalidDimensionException = Exception  # type: ignore
    _HAS_CHROMA = True
except Exception:
    chromadb = None
    PersistentClient = None  # type: ignore
    _HAS_CHROMA = False

def _normalize(v):
    v = np.array(v, dtype="float32")
    n = np.linalg.norm(v) + 1e-8
    return v / n

class JsonlMemory:
    """
    Simple memory bank:
      - appends entries to memory.jsonl (text + vector)
      - cosine similarity search with NumPy
    """
    def __init__(self, path="memory.jsonl"):
        self.path = Path(path)
        self.path.touch(exist_ok=True)

    def add(self, kind: str, text: str, vec: List[float], meta: Dict):
        rec = {
            "ts": datetime.utcnow().isoformat(),
            "kind": kind,
            "text": text,
            "vec": vec,              # already normalized
            "meta": meta or {}
        }
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    def _load(self):
        items = []
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                pass
        return items

    def search(self, query_vec: List[float], top_k: int = 3) -> List[Tuple[float, Dict]]:
        items = self._load()
        if not items:
            return []
        q = _normalize(query_vec)
        scored = []
        for it in items:
            v = np.array(it["vec"], dtype="float32")
            score = float(np.dot(q, v))
            scored.append((score, it))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def clear(self):
        try:
            self.path.write_text("")
        except Exception:
            pass


class ChromaMemory:
    """
    Chroma-backed memory with the same interface as JsonlMemory.
    Stores (kind, text, vec, meta, ts) in collection metadatas/documents.
    """

    def __init__(self, path: str = "data/chroma", collection: str = "memory"):
        if not _HAS_CHROMA:
            raise ImportError("chromadb is not installed. Please install per requirements.txt")
        # Ensure directory exists
        Path(path).mkdir(parents=True, exist_ok=True)
        self._path = path
        self._name = collection
        self.client = PersistentClient(path=path)
        # Use cosine space; our vectors are normalized as well
        self.collection = self.client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )
        self._id_counter = 0

    def _recreate_collection(self):
        try:
            # drop and recreate (resets dimensionality on first add)
            self.client.delete_collection(self._name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self._name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, kind: str, text: str, vec: List[float], meta: Optional[Dict]):
        self._id_counter += 1
        _id = f"m-{int(datetime.utcnow().timestamp()*1e3)}-{self._id_counter}"
        # Chroma metadata supports only scalar values; coerce others to JSON strings
        sanitized = {}
        for k, v in (meta or {}).items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                sanitized[k] = v
            else:
                try:
                    sanitized[k] = json.dumps(v, ensure_ascii=False)
                except Exception:
                    sanitized[k] = str(v)
        try:
            self.collection.add(
                ids=[_id],
                embeddings=[vec],
                documents=[text],
                metadatas=[{"ts": datetime.utcnow().isoformat(), "kind": kind, **sanitized}],
            )
        except InvalidDimensionException:
            # Dimension changed; reset collection and retry once
            self._recreate_collection()
            self.collection.add(
                ids=[_id],
                embeddings=[vec],
                documents=[text],
                metadatas=[{"ts": datetime.utcnow().isoformat(), "kind": kind, **sanitized}],
            )

    def search(self, query_vec: List[float], top_k: int = 3) -> List[Tuple[float, Dict]]:
        try:
            res = self.collection.query(
                query_embeddings=[query_vec],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except InvalidDimensionException:
            # Dimension mismatch; reset and return empty until repopulated
            self._recreate_collection()
            return []
        out: List[Tuple[float, Dict]] = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        # Convert cosine distance to a similarity-ish score
        for doc, meta, dist in zip(docs, metas, dists):
            score = float(1.0 - (dist if dist is not None else 1.0))
            out.append((score, {"kind": meta.get("kind", "conv"), "text": doc, "meta": meta}))
        return out

    def clear(self):
        # Delete all items in the collection
        try:
            self.collection.delete(where={})
        except Exception:
            pass

