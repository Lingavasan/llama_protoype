import os, json, time, sys
from pathlib import Path
# Ensure project root is on sys.path for `import src.*`
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.pipeline import Engine
from src.utils.tokenizer import count_tokens
from src.utils.logging import log_event, TurnTimer

# Minimal LongMemEval-lite sample (you can replace with a real file in Week 2)
sample = {
    "dialogue": [
        {"role":"user","content":"Remember this: My advisor is Dr. Lee."},
        {"role":"assistant","content":"Got it. I'll remember your advisor is Dr. Lee."},
        {"role":"user","content":"Who is my advisor?"}
    ],
    "question": "Who is my advisor?",
    "answer": "Dr. Lee",
    "evidence": ["My advisor is Dr. Lee."]
}

def main():
        eng = Engine()
        run_dir = "runs/smoke"
        # replay dialogue; ask the final question
        with TurnTimer() as t:
                out = eng.run(sample["question"], debug=True)
        final = out["output"]
        # rough token accounting
        tok = count_tokens(sample["question"] + final)
        log_event(run_dir, "turn", {
                "latency_s": t.elapsed,
                "tokens": tok,
                "final": final
        })
        print("\n=== FINAL ===\n", final)
        print("\nLatency(s):", round(t.elapsed, 3), "Tokens:", tok)

if __name__ == "__main__":
        main()