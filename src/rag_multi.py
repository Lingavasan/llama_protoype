import chromadb
from typing import List, Tuple, Dict
from ollama import Client

class MultiChromaRAG:
    def __init__(self, sources: List[Dict], host: str = "http://localhost:11434"):
        """
        sources: list of dicts each containing:
          - db_path: path to chroma persistent directory
          - collection: collection name
          - top_k: per-source result count (default 3)
          - embed_model: embedding model name (default all-minilm)
          - name: logical display name (default collection)
        """
        self.client = Client(host=host)
        self.sources = []
        for s in sources:
            c = chromadb.PersistentClient(path=s["db_path"])
            coll = c.get_collection(name=s["collection"])
            self.sources.append({
                "name": s.get("name", s["collection"]),
                "top_k": int(s.get("top_k", 3)),
                "embed_model": s.get("embed_model", "all-minilm"),
                "coll": coll,
            })

    def _embed(self, model: str, text: str):
        return self.client.embed(model=model, input=[text])["embeddings"][0]

    def search(self, query: str) -> List[Tuple[float, Dict]]:
        results: List[Tuple[float, Dict]] = []
        for s in self.sources:
            vec = self._embed(s["embed_model"], query)
            res = s["coll"].query(
                query_embeddings=[vec],
                n_results=s["top_k"],
                include=["documents", "metadatas", "distances"],
            )
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            dists = res.get("distances", [[]])[0]
            for doc, meta, dist in zip(docs, metas, dists):
                score = 1.0 - float(dist if dist is not None else 1.0)
                meta = meta or {}
                results.append((score, {
                    "kind": "rag",
                    "source_name": s["name"],
                    "title": meta.get("title") or meta.get("source") or s["name"],
                    "text": doc,
                }))
        results.sort(key=lambda x: x[0], reverse=True)
        return results
