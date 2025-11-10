import os, json, time
from datetime import datetime


def log_event(run_dir, name, payload):
    os.makedirs(run_dir, exist_ok=True)
    rec = {
        "ts": datetime.utcnow().isoformat(),
        "event": name,
        **payload
    }
    path = os.path.join(run_dir, "events.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


class TurnTimer:
    def __enter__(self):
        self.t0 = time.time(); return self
    def __exit__(self, *exc):
        self.elapsed = time.time() - self.t0
