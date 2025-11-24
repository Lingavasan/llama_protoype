import argparse
from src.pipeline import Engine

def main():
    parser = argparse.ArgumentParser(description="Run the Llama Prototype pipeline with a user query.")
    parser.add_argument("query", type=str, help="The user query to process.")
    parser.add_argument("--debug", action="store_true", help="Enable debug printing.")
    args = parser.parse_args()

    print(f"Initializing pipeline engine...")
    engine = Engine()

    print(f"Running pipeline with query: '{args.query}'")
    result = engine.run(args.query, debug=args.debug)

    print("\n--- PIPELINE OUTPUT ---")
    print(f"Status: {result.get('status')}")
    if result.get('status') == 'ok':
        print(f"Output: {result.get('output')}")
        print(f"Note: {result.get('note')}")
        print(f"Tags: {result.get('tags')}")
        if args.debug:
            print("\n--- DEBUG INFO ---")
            print(f"Recalls: {result.get('recalls')}")
            print(f"RAG Recalls: {result.get('rag_recalls')}")
            print(f"Critique: {result.get('critique')}")
    else:
        print(f"Reason: {result.get('reason')}")
    print("-----------------------\n")


if __name__ == "__main__":
    main()
