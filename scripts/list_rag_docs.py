import sys
from pathlib import Path
import chromadb

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

def list_docs():
    path = "data/chroma"
    client = chromadb.PersistentClient(path=path)
    
    collections = client.list_collections()
    print(f"Found {len(collections)} collections in {path}:")
    
    for col in collections:
        print(f"\n--- Collection: {col.name} ---")
        c = client.get_collection(col.name)
        count = c.count()
        print(f"Total Documents: {count}")
        
        if count > 0:
            # Peek at first 5
            peek = c.peek(limit=5)
            ids = peek['ids']
            metas = peek['metadatas']
            docs = peek['documents']
            
            print("Sample Documents:")
            for i, (doc_id, meta, doc) in enumerate(zip(ids, metas, docs)):
                title = meta.get('title', 'No Title')
                source = meta.get('source', 'Unknown')
                preview = doc[:100].replace('\n', ' ')
                print(f"  {i+1}. [{title}] ({source}): {preview}...")

if __name__ == "__main__":
    list_docs()
