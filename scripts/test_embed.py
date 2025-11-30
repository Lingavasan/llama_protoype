import sys
from pathlib import Path
import time

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.llm import OllamaLLM

def test():
    print("Initializing LLM wrapper...")
    llm = OllamaLLM(model="llama3.2:1b")
    embed_model = "nomic-embed-text"
    
    print(f"Testing embedding with {embed_model}...")
    text = "This is a test sentence to verify the embedding model."
    
    try:
        vec = llm.embed(text, embed_model=embed_model)
        print(f"Success! Vector length: {len(vec)}")
    except Exception as e:
        print(f"Failed: {e}")
        return

    print("Testing stability (10 requests)...")
    for i in range(10):
        try:
            vec = llm.embed(f"Test sentence {i}", embed_model=embed_model)
            print(f"Request {i+1}: OK")
            time.sleep(0.1)
        except Exception as e:
            print(f"Request {i+1}: Failed - {e}")

if __name__ == "__main__":
    test()
