import re
from typing import Tuple, Dict

class Policy:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.banned = [b.lower() for b in cfg["safety"].get("banned_keywords", [])]
        self.warn_len = int(cfg["safety"].get("warn_on_long_answer", 0))
        self.disclaimer = cfg.get("disclaimer", "")

    def check_input(self, text: str) -> Tuple[bool, str]:
        low = text.lower()
        for bad in self.banned:
            if bad and bad in low:
                return False, f"Blocked by policy (keyword: {bad})"
        if re.search(r"\b\d{13,19}\b", text):
            return False, "Blocked: possible sensitive number detected"
        return True, "ok"

    def check_output(self, text: str) -> Tuple[bool, str]:
        if self.warn_len and len(text) > self.warn_len:
            return True, "warn: long answer"
        return True, "ok"
