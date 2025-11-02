import argparse
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
from src.router import MemGPTRouter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="artifacts/memgpt_router.joblib")
    ap.add_argument("--num", type=int, default=1000)
    args = ap.parse_args()

    ds = load_dataset("MemGPT/MemGPT-DPO-Dataset", split="train").shuffle(seed=1).select(range(args.num))
    router = MemGPTRouter.load(args.bundle)
    y_true = ds["chosen"]
    y_pred = [router.predict(t) for t in ds["prompt"]]
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    main()
