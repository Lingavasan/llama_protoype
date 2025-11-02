import json, numpy as np, hnswlib
from pathlib import Path

class HNSWRAG:
    def __init__(self, index_path, meta_path, info_path):
        info = json.loads(Path(info_path).read_text())
        dim = int(info["dim"])
        self.index = hnswlib.Index(space="cosine", dim=dim)
        self.index.load_index(index_path)
        self.meta = []
        with open(meta_path) as f:
            for line in f:
                if line.strip():
                    self.meta.append(json.loads(line))

    def search(self, qvec, k=3):
        q = np.array([qvec], dtype="float32")
        labels, dists = self.index.knn_query(q, k=k)
        out = []
        for idx, dist in zip(labels[0], dists[0]):
            score = 1.0 - float(dist)   # convert cosine distance → similarity-ish
            m = self.meta[int(idx)]
            out.append((score, {"kind":"rag", "title": m["source"], "text": m["text"]}))
        return out
