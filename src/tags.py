import re
from datetime import datetime

RULES = {
    "code": r"\b(python|error|function|class|bug|stack trace)\b",
    "policy": r"\b(policy|safety|guardrail|license)\b",
    "math": r"\b(matrix|equation|integral|theorem)\b",
}

def tag_text(text: str):
    tags = []
    low = text.lower()
    for name, pat in RULES.items():
        if re.search(pat, low):
            tags.append(name)
    if len(text) > 500:
        tags.append("long")
    return tags

def base_meta(version: str):
    return {"policy_version": version, "ts": datetime.utcnow().isoformat()}
