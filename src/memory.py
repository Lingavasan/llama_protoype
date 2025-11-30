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
            "id": f"m-{int(datetime.utcnow().timestamp()*1e6)}",
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
        
        # Calculate similarity scores
        scored = []
        for it in items:
            v = np.array(it["vec"], dtype="float32")
            score = float(np.dot(q, v))
            scored.append((score, it))
        
        # Sort by similarity to get a baseline
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Re-rank with recency bias
        now = datetime.utcnow()
        reranked = []
        for sim_score, item in scored[:top_k * 3]: # Fetch more to re-rank
            ts_str = item.get("ts", "")
            if not ts_str:
                recency_score = 0.0
            else:
                ts = datetime.fromisoformat(ts_str.replace("Z", ""))
                age_seconds = (now - ts).total_seconds()
                # Recency score decays over ~1 day, falls to ~0.1 after 3 days
                recency_score = np.exp(-age_seconds / (3600 * 24))

            # Combined score with weighting
            # Boost user messages to prioritize original facts over assistant summaries
            role_boost = 1.2 if item.get("kind") == "user" else 1.0
            
            # Usefulness boost (frequency)
            use_count = item.get("meta", {}).get("use_count", 0)
            use_boost = 1.0 + 0.1 * np.log1p(use_count)
            
            combined_score = (0.9 * sim_score + 0.1 * recency_score) * role_boost * use_boost
            reranked.append((combined_score, item))
            
        reranked.sort(key=lambda x: x[0], reverse=True)
        return reranked[:top_k]

    def touch(self, ids: List[str]):
        """Update timestamp and increment use_count for accessed memories."""
        if not ids:
            return
        
        items = self._load()
        updated = False
        target_ids = set(ids)
        new_items = []
        
        for it in items:
            if it.get("id") in target_ids:
                it["ts"] = datetime.utcnow().isoformat()
                meta = it.get("meta", {})
                meta["use_count"] = meta.get("use_count", 0) + 1
                it["meta"] = meta
                updated = True
            new_items.append(it)
            
        if updated:
            with self.path.open("w") as f:
                for it in new_items:
                    f.write(json.dumps(it) + "\n")


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
            # Fetch more results to re-rank
            res = self.collection.query(
                query_embeddings=[query_vec],
                n_results=top_k * 3,
                include=["documents", "metadatas", "distances"],
            )
        except InvalidDimensionException:
            # Dimension mismatch; reset and return empty until repopulated
            self._recreate_collection()
            return []

        # Re-rank with recency
        now = datetime.utcnow()
        reranked = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        for _id, doc, meta, dist in zip(ids, docs, metas, dists):
            sim_score = float(1.0 - (dist if dist is not None else 1.0))
            
            # Hard threshold: if similarity is too low, ignore it regardless of recency
            if sim_score < 0.15:
                continue

            ts_str = meta.get("ts", "")
            if not ts_str:
                recency_score = 0.0
            else:
                ts = datetime.fromisoformat(ts_str.replace("Z", ""))
                age_seconds = (now - ts).total_seconds()
                # Recency score decays over ~1 day, falls to ~0.1 after 3 days
                recency_score = np.exp(-age_seconds / (3600 * 24))

            # Combined score with weighting (heavily favor similarity)
            combined_score = 0.9 * sim_score + 0.1 * recency_score
            
            # Use combined_score for sorting, but return original sim_score for visibility
            reranked.append((combined_score, (sim_score, {"id": _id, "kind": meta.get("kind", "conv"), "text": doc, "meta": meta})))

        reranked.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in reranked[:top_k]]

    def touch(self, ids: List[str]):
        """Update the timestamp of accessed memories to prevent TTL expiration."""
        if not ids:
            return
        now_ts = datetime.utcnow().isoformat()
        # We need to fetch existing metadatas to preserve other fields
        try:
            existing = self.collection.get(ids=ids, include=["metadatas"])
            metas = existing.get("metadatas", [])
            if not metas:
                return
            
            new_metas = []
            for m in metas:
                m["ts"] = now_ts
                new_metas.append(m)
            
            self.collection.update(ids=ids, metadatas=new_metas)
        except Exception:
            pass

    def clear(self):
        # Delete all items in the collection
        try:
            self.collection.delete(where={})
        except Exception:
            pass

