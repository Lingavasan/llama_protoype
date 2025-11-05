import argparse
from pathlib import Path
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_prefix", default="data/index/corpus")
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()

    prefix = Path(args.index_prefix)
    idx = prefix.with_suffix(".hnsw")
    meta = Path(str(prefix) + "_meta.jsonl")
    info = Path(str(prefix) + "_info.json")

    if not (idx.exists() and meta.exists() and info.exists()):
        print("Index not found. Run: make index")
        print("Expected:")
        print(" -", idx)
        print(" -", meta)
        print(" -", info)
        return 0

    # Print info and sample a few entries
    with info.open() as f:
        j = json.load(f)
    print(f"Index OK: dim={j.get('dim')} count={j.get('count')}")

    print("Sample meta entries:")
    shown = 0
    with meta.open() as f:
        for line in f:
            if not line.strip():
                continue
            m = json.loads(line)
            print(f"- {m.get('source')}: chunk {m.get('chunk_id')}")
            shown += 1
            if shown >= args.limit:
                break
    print("Lite eval complete ✅")


if __name__ == "__main__":
    main()