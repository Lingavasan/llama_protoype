import argparse, json
from pathlib import Path
from datasets import load_dataset
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.llm import OllamaLLM
from src.memory import ChromaMemory


def export_jsonl(ds, out_path: Path, limit: int | None = None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open('w') as f:
        for ex in ds:
            title = ex.get('title') or ex.get('wikipedia_title') or ex.get('id') or ''
            text = ex.get('text') or ex.get('passage') or ex.get('content') or ''
            if not text:
                continue
            f.write(json.dumps({"title": title, "text": text}) + "\n")
            count += 1
            if limit and count >= limit:
                break


def ingest_chroma(jsonl_path: Path, embed_model: str, chroma_path: str, collection: str, limit: int | None = None):
    llm = OllamaLLM()
    mem = ChromaMemory(path=chroma_path, collection=collection)
    added = 0
    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        ex = json.loads(line)
        text = ex.get('text') or ''
        if not text:
            continue
        v = llm.embed(text, embed_model)
        mem.add('knowledge', text, v, {"title": ex.get('title', 'wiki')})
        added += 1
        if added % 200 == 0:
            print('ingested', added)
        if limit and added >= limit:
            break
    print('total ingested', added)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', default='text-corpus')
    ap.add_argument('--outdir', default='data/rag_mini_wikipedia')
    ap.add_argument('--embed_model', default='llama-2-7b-chat.Q4_K_M')
    ap.add_argument('--chroma_path', default='data/chroma')
    ap.add_argument('--collection', default='wiki')
    ap.add_argument('--export_only', action='store_true')
    ap.add_argument('--limit', type=int, default=0, help='Optional cap for export and ingest')
    args = ap.parse_args()

    # For 'text-corpus', the split is named 'passages'
    split = 'passages' if args.subset == 'text-corpus' else 'train'
    ds = load_dataset('rag-datasets/rag-mini-wikipedia', args.subset, split=split)
    outdir = Path(args.outdir)
    out_jsonl = outdir / 'corpus.jsonl'
    export_jsonl(ds, out_jsonl, limit=(args.limit or None))
    print('exported to', out_jsonl)
    if not args.export_only:
        ingest_chroma(out_jsonl, args.embed_model, args.chroma_path, args.collection, limit=(args.limit or None))


if __name__ == '__main__':
    main()
