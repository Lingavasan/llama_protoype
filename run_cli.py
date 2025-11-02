import argparse
from rich import print
from src.pipeline import Engine

def main():
    ap = argparse.ArgumentParser(description="Ollama thesis skeleton")
    ap.add_argument("--config", default="configs/policy.yaml")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--ask", help="One-shot question")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    engine = Engine(config_path=args.config, host=args.host)

    if args.ask:
        res = engine.run(args.ask, debug=args.debug)
        if res["status"] == "ok":
            print("\n[bold green]FINAL[/bold green]\n", res["output"])
        else:
            print("[bold red]BLOCKED[/bold red]:", res["reason"])
        return

    print("Type 'exit' or 'quit' to leave.")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in {"exit","quit"}:
            break
        res = engine.run(q, debug=args.debug)
        if res["status"] == "ok":
            print("\n[bold green]FINAL[/bold green]\n", res["output"])
        else:
            print("[bold red]BLOCKED[/bold red]:", res["reason"])

if __name__ == "__main__":
    main()
