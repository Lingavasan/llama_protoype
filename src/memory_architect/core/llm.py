
import requests
import json
from typing import Optional, Dict, Any

class OllamaClient:
    """
    A simple client for the local Ollama API.
    """
    
    def __init__(self, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        
    def generate(
        self, 
        prompt: str, 
        system: Optional[str] = None,
        context_window: int = 8192,
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate text from the LLM.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": context_window,
                "temperature": 0.0  # Deterministic for benchmarking
            }
        }
        
        if system:
            payload["system"] = system
            
        if options:
            payload["options"].update(options)
            
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
            
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama: {e}")
            if response is not None:
                print(f"Response: {response.text}")
            return f"Error: {str(e)}"

if __name__ == "__main__":
    # Smoke test
    client = OllamaClient()
    print("Testing Ollama connection...")
    resp = client.generate("Say hello!", context_window=2048)
    print(f"Response: {resp}")
