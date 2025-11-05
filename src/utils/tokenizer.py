"""
Token counter utility with safe fallback.
 - Tries to use Hugging Face AutoTokenizer (Llama 3.2 1B)
 - Falls back to a rough estimate: len(text)//4 tokens
"""
from functools import lru_cache


@lru_cache(maxsize=1)
def _get_tokenizer():
    try:
        from transformers import AutoTokenizer  # type: ignore
        # Use a reasonable default tokenizer; will fail gracefully offline
        return AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-1B", use_fast=True
        )
    except Exception:
        return None


def count_tokens(text: str) -> int:
    tok = _get_tokenizer()
    if tok is None:
        # Rough fallback: average ~4 chars per token
        return max(1, len(text) // 4)
    return len(tok(text, add_special_tokens=False)["input_ids"])
