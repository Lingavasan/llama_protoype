from dataclasses import dataclass
from typing import List, Tuple
from src.utils.tokenizer import count_tokens


@dataclass
class BudgetConfig:
    C: int          # context capacity
    eps: int        # buffer
    max_retrieved_chunks: int = 5


@dataclass
class Segment:
    name: str
    text: str
    tokens: int


def pack_with_budget(system: str, user: str,
                     memory_chunks: List[str], retrieved_chunks: List[str],
                     gen_tokens: int, cfg: BudgetConfig) -> Tuple[str, List[str], List[str], int]:
    """
    Returns (system_out, M_used, R_used, G_final) such that:
    S + U + sum(M) + sum(R) + G + eps <= C
    Strategy:
      1) Compute S,U,G,eps
      2) Add memory chunks until budget tight (oldest/micro first)
      3) Add retrieved chunks in order of value (caller sorts by relevance)
      4) If overflow, trim R then M; as last resort, reduce G.
    """
    S = count_tokens(system)
    U = count_tokens(user)
    eps = cfg.eps
    M_used: List[str] = []
    R_used: List[str] = []
    budget_now = S + U + gen_tokens + eps

    # Add memory chunks
    for m in memory_chunks:
        t = count_tokens(m)
        if budget_now + t <= cfg.C:
            M_used.append(m)
            budget_now += t
        else:
            break

    # Add retrieved chunks
    for r in retrieved_chunks[: cfg.max_retrieved_chunks]:
        t = count_tokens(r)
        if budget_now + t <= cfg.C:
            R_used.append(r)
            budget_now += t
        else:
            break

    # If still over (shouldn't happen), reduce R then M then G
    total = budget_now
    if total > cfg.C:
        # trim R first
        while R_used and total > cfg.C:
            rm = R_used.pop()
            total -= count_tokens(rm)
        # trim M
        while M_used and total > cfg.C:
            rm = M_used.pop()
            total -= count_tokens(rm)
        # reduce G
        while gen_tokens > 64 and total > cfg.C:
            gen_tokens -= 32
            total -= 32

    return system, M_used, R_used, gen_tokens
