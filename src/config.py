from pathlib import Path
import yaml

REQUIRED = {
    "system": str,
    "gen": dict,
    "memory": dict,
    "safety": dict,
}

def load_config(path="configs/policy.yaml"):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p.resolve()}")
    cfg = yaml.safe_load(p.read_text())

    # basic validation
    for key, typ in REQUIRED.items():
        if key not in cfg:
            raise ValueError(f"Missing '{key}' in {path}")
        if not isinstance(cfg[key], typ):
            raise TypeError(f"'{key}' must be {typ.__name__}")

    # defaults
    cfg.setdefault("disclaimer", "")
    cfg["gen"].setdefault("temperature", 0.7)
    cfg["gen"].setdefault("num_predict", 256)
    cfg["memory"].setdefault("top_k", 3)
    cfg["safety"].setdefault("banned_keywords", [])
    cfg["safety"].setdefault("warn_on_long_answer", 0)
    return cfg

