"""
Week 4 Day 1: Ebbinghaus Forgetting Curve
==========================================
Implements mathematical decay of memory strength over time.
Models human forgetting curves: unused memories fade, strong memories persist.
"""

import math
import time
from typing import Optional


def apply_forgetting_curve(
    current_score: float,
    last_access_ts: float,
    stability: float = 2.0,
    current_time: Optional[float] = None
) -> float:
    """
    Apply exponential decay to memory score based on Ebbinghaus forgetting curve.
    
    Formula: R = current_score * exp(-t/S)
    
    Where:
    - R = Retention (new score after decay)
    - t = Time elapsed since last access (hours)
    - S = Stability (memory strength factor)
    
    Theoretical Background:
    Hermann Ebbinghaus (1885) discovered that memory retention follows an
    exponential decay pattern. Without reinforcement, memories fade over time.
    The stability factor S determines how quickly a memory decays - higher
    stability means slower decay.
    
    Dynamic Stability:
    We model stability as adaptive based on the memory's current score:
    S_effective = base_stability * (current_score / 50.0)
    
    This means:
    - Stronger memories (higher scores) have higher stability → decay slower
    - Weaker memories decay faster
    - Mimics human memory: important facts are retained longer
    
    Args:
        current_score: Current reflection score (0-100)
        last_access_ts: Unix timestamp of last access
        stability: Base stability factor in hours (default: 2.0)
        current_time: Current time (for testing; uses time.time() if None)
    
    Returns:
        New score after applying decay (>= 0.0)
    
    Example:
        >>> # Memory accessed 24 hours ago with score 80
        >>> import time
        >>> ts_24h_ago = time.time() - 86400
        >>> new_score = apply_forgetting_curve(80.0, ts_24h_ago, stability=2.0)
        >>> print(f"Score after 24h: {new_score:.1f}")
        >>> # Score will be significantly lower due to decay
    
    Reference: Ebbinghaus, H. (1885). Memory: A Contribution to Experimental Psychology
    """
    # Get current time
    if current_time is None:
        current_time = time.time()
    
    # Calculate elapsed time in hours
    elapsed_seconds = current_time - last_access_ts
    elapsed_hours = elapsed_seconds / 3600.0
    
    # Ensure non-negative elapsed time
    if elapsed_hours < 0:
        # Future timestamp (shouldn't happen in practice)
        return current_score
    
    # Dynamic stability: stronger memories decay slower
    # Normalize by 50.0 (mid-point of 0-100 scale)
    score_factor = current_score / 50.0
    dynamic_stability = stability * score_factor
    
    # Ensure minimum stability to avoid division by zero or extreme decay
    effective_stability = max(0.1, dynamic_stability)
    
    # Apply exponential decay: R = current_score * e^(-t/S)
    retention_factor = math.exp(-elapsed_hours / effective_stability)
    new_score = current_score * retention_factor
    
    # Floor at 0.0 (scores cannot go negative)
    return max(0.0, new_score)


def calculate_decay_rate(
    score: float,
    stability: float = 2.0
) -> float:
    """
    Calculate the hourly decay rate for a memory at given score.
    
    Useful for predicting when a memory will fall below a threshold.
    
    Args:
        score: Current reflection score
        stability: Base stability factor
    
    Returns:
        Percentage of score lost per hour
    
    Example:
        >>> rate = calculate_decay_rate(80.0, stability=2.0)
        >>> print(f"Memory loses {rate:.2%} per hour")
    """
    dynamic_stability = stability * (score / 50.0)
    effective_stability = max(0.1, dynamic_stability)
    
    # After 1 hour: retention = e^(-1/S)
    # Decay rate = 1 - retention
    one_hour_retention = math.exp(-1.0 / effective_stability)
    decay_rate = 1.0 - one_hour_retention
    
    return decay_rate


def estimate_time_to_threshold(
    current_score: float,
    threshold_score: float,
    stability: float = 2.0
) -> float:
    """
    Estimate hours until memory score falls below threshold (without access).
    
    Useful for:
    - Determining when to consolidate/summarize memories
    - Predicting memory lifespan
    - Scheduling maintenance tasks
    
    Args:
        current_score: Current reflection score
        threshold_score: Target threshold score
        stability: Base stability factor
    
    Returns:
        Estimated hours until score drops below threshold
    
    Example:
        >>> hours = estimate_time_to_threshold(80.0, 20.0, stability=2.0)
        >>> print(f"Memory will fade below threshold in {hours:.1f} hours")
    """
    if current_score <= threshold_score:
        return 0.0
    
    # Dynamic stability
    score_factor = current_score / 50.0
    dynamic_stability = stability * score_factor
    effective_stability = max(0.1, dynamic_stability)
    
    # Solve: threshold = current * e^(-t/S) for t
    # t = -S * ln(threshold / current)
    ratio = threshold_score / current_score
    hours_to_threshold = -effective_stability * math.log(ratio)
    
    return hours_to_threshold


def apply_reinforcement(
    current_score: float,
    boost: float = 10.0,
    max_score: float = 100.0
) -> float:
    """
    Apply reinforcement boost to memory score when accessed/used.
    
    Counterpart to decay - memories that are used get stronger.
    
    Args:
        current_score: Current reflection score
        boost: Score increase on access (default: 10.0)
        max_score: Maximum allowed score (default: 100.0)
    
    Returns:
        New score after reinforcement (capped at max_score)
    
    Example:
        >>> # Memory was used in LLM response
        >>> new_score = apply_reinforcement(50.0, boost=10.0)
        >>> print(f"Reinforced score: {new_score}")  # 60.0
    """
    new_score = current_score + boost
    return min(max_score, new_score)
