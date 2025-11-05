import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    print("Smoke check starting…")

    # Check config exists
    cfg_path = root / "configs" / "policy.yaml"
    assert cfg_path.exists(), f"Missing config: {cfg_path}"
    print("- Found config:", cfg_path)

    # Check corpus dir
    corpus_dir = root / "data" / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    n_files = len(list(corpus_dir.glob("*.txt"))) + len(list(corpus_dir.glob("*.md")))
    print(f"- Corpus directory: {corpus_dir} ({n_files} files)")

    # Check scripts presence
    for p in [
        root / "scripts" / "build_corpus_index.py",
        root / "scripts" / "train_router.py",
        root / "scripts" / "eval_router.py",
    ]:
        assert p.exists(), f"Missing script: {p}"
    print("- Core scripts present")

    # Check writable memory file path (do not modify contents)
    mem_file = root / "memory.jsonl"
    print("- Memory file path:", mem_file)

    print("Smoke check passed ✅")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Smoke check failed:", e)
        sys.exit(1)