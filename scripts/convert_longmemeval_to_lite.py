import json, argparse
from pathlib import Path


def to_dialogue_jsonl(in_path: str, out_path: str, limit: int = 10):
    src = json.loads(Path(in_path).read_text())
    # Expect a list of instances
    items = src if isinstance(src, list) else src.get("data", [])
    out_lines = []
    for i, ex in enumerate(items):
        if i >= limit:
            break
        # Oracle format includes only evidence sessions, so create a compact dialogue
        sessions = ex.get("haystack_sessions") or ex.get("sessions") or []
        dialogue = []
        for sess in sessions:
            for turn in sess:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if content:
                    dialogue.append({"role": role, "content": content})
        rec = {
            "dialogue": dialogue,
            "question": ex.get("question", ""),
            "answer": ex.get("answer", ""),
            "evidence": [" ".join(t.get("content","") for t in s) for s in sessions],
        }
        out_lines.append(json.dumps(rec))
    Path(out_path).write_text("\n".join(out_lines) + ("\n" if out_lines else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input longmemeval_oracle.json path")
    ap.add_argument("--out", dest="out", required=True, help="Output JSONL path")
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()
    to_dialogue_jsonl(args.inp, args.out, args.limit)


if __name__ == "__main__":
    main()