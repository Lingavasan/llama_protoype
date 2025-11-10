import json, argparse, re, statistics, sys
from pathlib import Path
# Ensure project root on sys.path for `import src.*`
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.pipeline import Engine
from typing import List, Dict
from src.verify import ungrounded_claim_rate


def normalize(s):
    return re.sub(r"\s+"," ", s.strip().lower())


def f1(a, b):
    # token-level F1 for short answers
    ta, tb = normalize(a).split(), normalize(b).split()
    common = set(ta) & set(tb)
    if not ta or not tb:
        return 0.0
    prec = len(common)/max(1,len(tb))
    rec  = len(common)/max(1,len(ta))
    if prec+rec==0:
        return 0.0
    return 2*prec*rec/(prec+rec)


def hit_at_k(evidence, recalls, k=5):
    # naive "hit" if any evidence substring appears in any retrieved text
    joined = " ".join(recalls[:k]).lower()
    return any(ev.lower() in joined for ev in evidence)


def span_overlap_metrics(evidence: list[str], rag_texts: list[str], k: int = 5):
    # Compute strict phrase coverage and bigram recall for RAG recalls
    rag_join = " \n ".join(rag_texts[:k]).lower()
    # phrase coverage: fraction of evidence strings that appear verbatim in RAG
    if evidence:
        phrase_cov = sum(1 for ev in evidence if ev.lower() in rag_join) / max(1, len(evidence))
    else:
        phrase_cov = 0.0
    # bigram recall: fraction of answer bigrams present in RAG (approx grounding of answer against RAG)
    return phrase_cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/longmemeval_lite")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--no-replay", dest="replay", action="store_false", help="Do not replay dialogue turns into memory before asking the question.")
    ap.set_defaults(replay=True)
    args = ap.parse_args()

    eng = Engine()
    files = list(Path(args.path).glob("*.jsonl"))
    items = []
    for fp in files:
        for i, line in enumerate(Path(fp).read_text().splitlines()):
            if not line.strip():
                continue
            items.append(json.loads(line))
            if len(items) >= args.limit:
                break
        if len(items) >= args.limit:
            break

    f1s, ems, hits, ucrs = [], [], [], []
    rag_phrase_covs = []
    for ex in items:
        # Reset memory for isolation between examples
        if hasattr(eng.memory, "clear"):
            eng.memory.clear()

        # Optionally replay dialogue into memory without generating model outputs
        if args.replay:
            for turn in ex.get("dialogue", []):
                content = turn.get("content", "")
                if not content:
                    continue
                try:
                    vec = eng._embed(content)
                    eng.memory.add(turn.get("role", "user"), content, vec, {"replay": True})
                except Exception:
                    # If embedding fails, skip this turn
                    continue

    out = eng.run(ex["question"], debug=False)
    final = out["output"]
    # strip disclaimer if present
    final = re.sub(r"_For research.*$", "", final).strip()
    f1s.append(f1(final, ex["answer"]))
    ems.append(1.0 if normalize(final) == normalize(ex["answer"]) else 0.0)
    recalls = out.get("recalls", [])
    hits.append(1.0 if hit_at_k(ex.get("evidence",[]), recalls, k=5) else 0.0)
    # Grounding metric: aggregate internal + RAG evidence and blend with semantic similarity
    all_evidence_texts = recalls + out.get("rag_recalls", [])
    evidence_concat = " \n ".join(all_evidence_texts)

    def _sim(a: str, b: str) -> float:
        try:
            import numpy as _np
            va = _np.array(eng._embed(a), dtype="float32")
            vb = _np.array(eng._embed(b), dtype="float32")
            na = _np.linalg.norm(va) + 1e-8
            nb = _np.linalg.norm(vb) + 1e-8
            return float((va @ vb) / (na * nb))
        except Exception:
            return 0.0

    ucr = ungrounded_claim_rate(final, evidence_concat, embed_similarity=_sim, alpha=0.7)
    ucrs.append(ucr)
    print("Ungrounded Claim Rate (approx):", round(ucr,3))
    # Perfect token retrieval (phrase coverage of evidence in RAG-only recalls)
    rag_texts = out.get("rag_recalls", [])
    pc = span_overlap_metrics(ex.get("evidence", []), rag_texts, k=5)
    rag_phrase_covs.append(pc)
    print("RAG Phrase Coverage (evidence@5):", round(pc,3))

    if not items:
        print("N = 0")
        return
    print("N =", len(items))
    print("EM =", round(sum(ems)/len(ems), 3))
    print("F1 =", round(sum(f1s)/len(f1s), 3))
    print("Hit@5 =", round(sum(hits)/len(hits), 3))
    if ucrs:
        print("UCR avg =", round(sum(ucrs)/len(ucrs), 3))
    if rag_phrase_covs:
        print("RAG Phrase Coverage avg =", round(sum(rag_phrase_covs)/len(rag_phrase_covs), 3))


if __name__ == "__main__":
    main()