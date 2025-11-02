import numpy as np, joblib
from ollama import Client

class MemGPTRouter:
    def __init__(self, bundle, host="http://localhost:11434"):
        self.le = bundle["label_encoder"]
        self.clf = bundle["classifier"]
        self.embed_model = bundle["embed_model"]
        self.client = Client(host=host)

    @classmethod
    def load(cls, path, host="http://localhost:11434"):
        return cls(joblib.load(path), host=host)

    def _embed(self, text):
        out = self.client.embed(model=self.embed_model, input=[text])
        v = np.array(out["embeddings"][0], dtype="float32")
        v = v / (np.linalg.norm(v) + 1e-8)
        return v.reshape(1, -1)

    def predict(self, text: str) -> str:
        X = self._embed(text)
        yhat = self.clf.predict(X)[0]
        return self.le.inverse_transform([yhat])[0]

