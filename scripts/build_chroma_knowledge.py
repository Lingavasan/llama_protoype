import argparse, json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.llm import OllamaLLM
from src.memory import ChromaMemory

def load_items(path: Path):
    items = []
    if path.suffix.lower() in {".jsonl", ".jsonl.gz"}:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                pass
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            data = data.get("items") or data.get("data") or []
        for it in data:
            items.append(it)
    else:
        raise ValueError(f"Unsupported file {path}")
    return items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path to JSON/JSONL containing {title?, text}")
    ap.add_argument("--chroma", default="data/chroma", help="Chroma DB path")
    ap.add_argument("--collection", default="knowledge", help="Collection name")
    ap.add_argument("--embed_model", default="llama-2-7b-chat.Q4_K_M")
    args = ap.parse_args()

    llm = OllamaLLM()
    mem = ChromaMemory(path=args.chroma, collection=args.collection)
    items = load_items(Path(args.src))
    added = 0
    for it in items:
        text = (it.get("text") or "").strip()
        if not text:
            continue
        v = llm.embed(text, args.embed_model)
        mem.add(kind="knowledge", text=text, vec=v, meta={"title": it.get("title") or it.get("source") or "json"})
        added += 1
        if added % 100 == 0:
            print("added", added)
    print("total added", added)

if __name__ == "__main__":
    main()
