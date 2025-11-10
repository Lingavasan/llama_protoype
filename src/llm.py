from typing import List, Dict
from ollama import Client


class OllamaLLM:
    def __init__(self, host: str = "http://localhost:11434", model: str = "llama3.2:1b",
                 temperature: float = 0.7, num_predict: int = 256):
        self.client = Client(host=host)
        self.host = host
        self.model = model
        self.temperature = float(temperature)
        self.num_predict = int(num_predict)

    def chat(self, messages: List[Dict[str, str]], temperature=None, num_predict=None) -> str:
        opts = {
            "temperature": self.temperature if temperature is None else float(temperature),
            "num_predict": self.num_predict if num_predict is None else int(num_predict),
        }
        r = self.client.chat(model=self.model, messages=messages, stream=False, options=opts)
        return r.message.content.strip()

    def embed(self, text: str, embed_model: str):
        # Primary path: python client with `input` (maps to /api/embeddings)
        out = self.client.embed(model=embed_model, input=text)
        vec = []
        if isinstance(out, dict):
            embs = out.get("embeddings")
            if embs and isinstance(embs, list) and embs and isinstance(embs[0], list):
                vec = embs[0]
        if vec:
            return vec
        # Fallback: raw HTTP with "prompt" key (some versions/examples)
        try:
            import requests
            url = self.host.rstrip("/") + "/api/embeddings"
            resp = requests.post(url, json={"model": embed_model, "prompt": text}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                if "embedding" in data and isinstance(data["embedding"], list):
                    return data["embedding"]
                embs = data.get("embeddings")
                if embs and isinstance(embs, list) and embs and isinstance(embs[0], list):
                    return embs[0]
        except Exception:
            pass
        return vec

