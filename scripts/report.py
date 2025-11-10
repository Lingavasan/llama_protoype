import json, argparse, statistics
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="runs")
    args = ap.parse_args()

    events = []
    for fp in Path(args.run_dir).glob("**/events.jsonl"):
        for line in Path(fp).read_text().splitlines():
            if line.strip():
                try:
                    events.append(json.loads(line))
                except Exception:
                    continue
    if not events:
        print("No events found.")
        return
    lats = [e["latency_s"] for e in events if "latency_s" in e]
    toks = [e["tokens"] for e in events if "tokens" in e]
    print("Turns:", len(events))
    if lats:
        l_sorted = sorted(lats)
        p50 = statistics.median(lats)
        p95 = l_sorted[int(0.95 * len(l_sorted)) - 1]
        print("Latency p50 / p95:", round(p50, 3), "/", round(p95, 3))
    if toks:
        print("Tokens/Turn avg:", round(sum(toks) / len(toks), 2))


if __name__ == "__main__":
    main()
