import json
from pathlib import Path
from src.pipeline import Engine


def load_longmem_sample():
    # Use the bundled lite sample if available
    p = Path("data/longmemeval_lite/sample.jsonl")
    if p.exists():
        line = p.read_text().splitlines()[0]
        return json.loads(line)
    # Fallback to a static example aligned to our ingestion
    return {
        "question": "When are my office hours?",
        "answer": "Friday 2-4pm",
        "evidence": ["office hours are Friday 2-4pm."]
    }


def main():
    eng = Engine()
    sample = load_longmem_sample()

    print("[TEST] Known in-corpus question:")
    res1 = eng.run(sample["question"], debug=False)
    print("  status:", res1.get("status"), "note:", res1.get("note"))
    print("  output:", res1.get("output", "")[:160])
    print("  recalls:", len(res1.get("recalls", [])), "rag:", len(res1.get("rag_recalls", [])))

    print("\n[TEST] Unknown/gibberish question (expect fallback):")
    q2 = "blorptastic zyxw qqq about marsupial qubits?"
    res2 = eng.run(q2, debug=False)
    print("  status:", res2.get("status"), "note:", res2.get("note"))
    print("  output:", res2.get("output"))


if __name__ == "__main__":
    main()
