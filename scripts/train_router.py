import os, argparse, numpy as np, joblib
from pathlib import Path
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
from ollama import Client

def embed_batch(client, model, texts, B=64):
    vecs = []
    for i in range(0, len(texts), B):
        out = client.embed(model=model, input=texts[i:i+B])
        for v in out["embeddings"]:
            v = np.array(v, dtype="float32")
            v = v / (np.linalg.norm(v) + 1e-8)
            vecs.append(v)
    return np.vstack(vecs).astype("float32")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=5000)
    ap.add_argument("--embed_model", default="all-minilm")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--out_dir", default="artifacts")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ds = load_dataset("MemGPT/MemGPT-DPO-Dataset", split="train")
    if args.sample < len(ds):
        ds = ds.shuffle(seed=42).select(range(args.sample))

    prompts = ds["prompt"]
    labels  = ds["chosen"]  # action string

    # Filter out classes with only one sample
    from collections import Counter
    label_counts = Counter(labels)
    valid_labels = {label for label, count in label_counts.items() if count > 1}
    filtered = [(p, l) for p, l in zip(prompts, labels) if l in valid_labels]
    if not filtered:
        raise ValueError("No classes with more than one sample. Cannot stratify.")
    prompts, labels = zip(*filtered)

    le = LabelEncoder()
    y = le.fit_transform(labels)
    Xtr_txt, Xte_txt, ytr, yte = train_test_split(prompts, y, test_size=0.2, random_state=42, stratify=y)
    client = Client(host=args.host)

    print("Embedding train…")
    Xtr = embed_batch(client, args.embed_model, Xtr_txt)
    print("Embedding test…")
    Xte = embed_batch(client, args.embed_model, Xte_txt)

    clf = LogisticRegression(max_iter=2000, n_jobs=-1, multi_class="multinomial", solver="saga")
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    print("Accuracy:", accuracy_score(yte, pred))
    print(classification_report(yte, pred, target_names=le.classes_))

    artifact = {"label_encoder": le, "classifier": clf, "embed_model": args.embed_model}
    joblib.dump(artifact, Path(args.out_dir) / "memgpt_router.joblib")
    print("Saved:", Path(args.out_dir) / "memgpt_router.joblib")

if __name__ == "__main__":
    main()
