import json
import argparse
from pathlib import Path
from datetime import datetime
import chromadb
from chromadb.config import Settings
import sys

# Ensure project root is on sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.llm import OllamaLLM

def ingest(json_path, chroma_path="data/chroma", collection_name="longmem_docs", limit=None):
    print(f"Loading {json_path}...")
    data = json.loads(Path(json_path).read_text())
    
    print(f"Connecting to Chroma at {chroma_path}...")
    client = chromadb.PersistentClient(path=chroma_path)
    # Use cosine space
    col = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    
    # Initialize embedding model (using Ollama wrapper for consistency)
    llm = OllamaLLM(model="llama3.2:1b") # Model for embedding
    embed_model = "nomic-embed-text"

    count = 0
    print("Starting ingestion...")
    
    for item in data:
        if limit and count >= limit:
            break
            
        qid = item.get("question_id")
        # haystack_sessions is a list of sessions (list of turns)
        sessions = item.get("haystack_sessions", [])
        
        for sess_idx, session in enumerate(sessions):
            for turn_idx, turn in enumerate(session):
                role = turn.get("role")
                content = turn.get("content")
                
                if not content:
                    continue
                
                # Create a unique ID
                doc_id = f"{qid}_s{sess_idx}_t{turn_idx}"
                
                # Embed content
                try:
                    vec = llm.embed(content, embed_model=embed_model)
                except Exception as e:
                    print(f"\nFailed to embed doc {doc_id}: {e}")
                    continue
                
                # Metadata
                meta = {
                    "source": "longmemeval",
                    "question_id": qid,
                    "session_idx": sess_idx,
                    "turn_idx": turn_idx,
                    "role": role,
                    "ts": datetime.utcnow().isoformat() 
                }
                
                # Add to Chroma
                try:
                    col.add(
                        ids=[doc_id],
                        embeddings=[vec],
                        documents=[content],
                        metadatas=[meta]
                    )
                except Exception as e:
                    print(f"\nFailed to add doc {doc_id} to Chroma: {e}")
                
                import time
                time.sleep(0.05) # Rate limit

                count += 1
                if count % 10 == 0:
                    print(f"Ingested {count} documents...", end="\r")
                    
    print(f"\nDone! Ingested {count} documents into '{collection_name}'.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/longmemeval/longmemeval_oracle.json")
    ap.add_argument("--chroma", default="data/chroma")
    ap.add_argument("--collection", default="longmem_docs")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    
    ingest(args.src, args.chroma, args.collection, args.limit)
