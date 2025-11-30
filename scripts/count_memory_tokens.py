import sys
import json
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.tokenizer import count_tokens

def main():
    path = Path("memory.jsonl")
    if not path.exists():
        print("memory.jsonl not found.")
        return

    total_tokens = 0
    count = 0
    
    with path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                text = rec.get("text", "")
                t = count_tokens(text)
                total_tokens += t
                count += 1
            except Exception:
                pass
                
    print(f"Total Entries: {count}")
    print(f"Total Tokens: {total_tokens}")

if __name__ == "__main__":
    main()
