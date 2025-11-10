import re
from typing import Callable, Optional


def _normalize(text: str) -> list[str]:
    t = re.sub(r"[^a-z0-9 ]", " ", (text or "").lower())
    return [tok for tok in t.split() if tok]


def ungrounded_claim_rate(
    answer: str,
    evidence_texts: str,
    *,
    embed_similarity: Optional[Callable[[str, str], float]] = None,
    alpha: float = 0.7,
) -> float:
    """
    Estimate how much of `answer` is ungrounded by the provided `evidence_texts`.

    Method:
    - Token recall baseline: fraction of answer tokens unseen in evidence.
      UCR_tokens = 1 - (|tokens(answer) ∩ tokens(evidence)| / |tokens(answer)|)
    - Optional embedding similarity provides a semantic check in [0,1] where higher is better.
      If provided, we blend: UCR = alpha * UCR_tokens + (1-alpha) * (1 - sim)

    Args:
        answer: model output text.
        evidence_texts: concatenated evidence/memory/RAG context.
        embed_similarity: function that returns cosine-like similarity in [0,1] for (answer, evidence).
        alpha: weight for the token-based term when embedding similarity is used.

    Returns:
        Float in [0,1], higher means more ungrounded.
    """
    a_toks = _normalize(answer)
    if not a_toks:
        return 0.0
    ev_toks = set(_normalize(evidence_texts))
    covered = sum(1 for t in a_toks if t in ev_toks)
    token_ucr = 1.0 - (covered / max(1, len(a_toks)))

    if embed_similarity is None:
        return token_ucr

    try:
        sim = float(embed_similarity(answer, evidence_texts))
        sim = max(0.0, min(1.0, sim))  # clamp
        return alpha * token_ucr + (1.0 - alpha) * (1.0 - sim)
    except Exception:
        # Fallback if similarity fn fails
        return token_ucr
