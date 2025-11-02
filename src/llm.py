from typing import List, Dict
from ollama import Client

class OllamaLLM:
    def __init__(self, host="http://localhost:11434", model="llama3.2:1b",
                 temperature=0.7, num_predict=256):
        self.client = Client(host=host)
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
        out = self.client.embed(model=embed_model, input=text)
        # out = {'embeddings': [[...]]}
        vec = out["embeddings"][0]
        return vec

