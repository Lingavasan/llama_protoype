import argparse, json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.llm import OllamaLLM
from src.memory import ChromaMemory


def iter_jsonl(path: Path):
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except Exception:
            continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--path', default='data/longmemeval_lite')
    ap.add_argument('--pattern', default='*.jsonl')
    ap.add_argument('--chroma_path', default='data/chroma')
    ap.add_argument('--collection', default='memory')
    ap.add_argument('--embed_model', default='llama-2-7b-chat.Q4_K_M')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    llm = OllamaLLM()
    mem = ChromaMemory(path=args.chroma_path, collection=args.collection)
    added = 0
    files = list(Path(args.path).glob(args.pattern))
    for fp in files:
        for ex in iter_jsonl(fp):
            for turn in ex.get('dialogue', []):
                content = (turn.get('content') or '').strip()
                if not content:
                    continue
                vec = llm.embed(content, args.embed_model)
                mem.add(kind=turn.get('role','user'), text=content, vec=vec,
                        meta={"seed": True, "source": fp.name})
                added += 1
                if args.limit and added >= args.limit:
                    print('seeded', added)
                    return
    print('seeded', added)


if __name__ == '__main__':
    main()
