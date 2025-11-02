import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np

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

