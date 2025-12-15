"""
Week 3 Day 1: Token Budget Management
======================================
Implements greedy knapsack algorithm for context window optimization.
Treats LLM context window as a constrained resource (0/1 Knapsack Problem).
"""

from typing import List, Dict, Callable, Any


def simple_tokenizer(text: str) -> List[str]:
    """
    Simple tokenizer approximation for token counting.
    
    Uses character-based estimation: tokens ≈ len(text) / 4
    This is a rough approximation. For production use, integrate actual
    tokenizer (e.g., tiktoken for GPT models, transformers for Llama).
    
    Args:
        text: Input text to tokenize
    
    Returns:
        List of pseudo-tokens (split by spaces for compatibility)
    
    Note: Returns list for compatibility with tokenizer interface,
    but primary purpose is length estimation via len().
    """
    # Simple word-based split
    # Token count approximation: ~4 characters per token on average
    tokens = text.split()
    return tokens


def greedy_knapsack_budgeting(
    candidates: List[Dict[str, Any]],
    max_tokens: int,
    tokenizer_fn: Callable[[str], List[str]] = simple_tokenizer
) -> List[str]:
    """
    Select memories to maximize score within token budget using greedy algorithm.
    
    Implements greedy approximation of 0/1 Knapsack Problem:
    - Capacity: LLM context window (max_tokens)
    - Items: Retrieved memory chunks
    - Weight: Token count of chunk
    - Value: Relevance score from ranking
    
    Algorithm:
    1. Calculate token cost for each candidate
    2. Compute information density: density = score / tokens
    3. Sort by density (descending) - items with highest value per token first
    4. Greedily select items until budget exhausted
    
    Complexity:
    - Time: O(n log n) for sorting + O(n) for selection = O(n log n)
    - Space: O(n) for items list
    
    Theoretical Note:
    While Dynamic Programming gives optimal solution in O(nW), it's too slow
    for runtime RAG. Greedy is near-optimal when items have similar costs
    and provides 2-approximation for fractional knapsack.
    
    Args:
        candidates: List of dicts with 'content' (str) and 'score' (float) keys
        max_tokens: Maximum token budget (context window capacity)
        tokenizer_fn: Function to tokenize text and return token list
    
    Returns:
        List of selected content strings that fit within budget
    
    Example:
        >>> candidates = [
        ...     {'content': 'Important fact', 'score': 0.9},
        ...     {'content': 'Less relevant but short', 'score': 0.3},
        ...     {'content': 'Very long but highly relevant...', 'score': 0.95}
        ... ]
        >>> selected = greedy_knapsack_budgeting(candidates, max_tokens=100)
    
    Reference: [16] Greedy approximation for runtime efficiency
    """
    selected_content = []
    current_tokens = 0
    
    # Handle edge case: empty candidates
    if not candidates:
        return selected_content
    
    # 1. Pre-calculate costs and density for all candidates
    items = []
    for candidate in candidates:
        content = candidate.get('content', '')
        score = candidate.get('score', 0.0)
        
        # Calculate token cost using provided tokenizer
        tokens = tokenizer_fn(content)
        cost = len(tokens)
        
        # Information density = Score per Token
        # Avoid division by zero
        density = score / cost if cost > 0 else 0.0
        
        items.append({
            'content': content,
            'score': score,
            'cost': cost,
            'density': density
        })
    
    # 2. Sort by information density (descending)
    # Highest value-per-token items selected first
    items.sort(key=lambda x: x['density'], reverse=True)
    
    # 3. Greedy selection: fill the knapsack
    for item in items:
        # Check if adding this item would exceed budget
        if current_tokens + item['cost'] <= max_tokens:
            selected_content.append(item['content'])
            current_tokens += item['cost']
        # else: skip items that don't fit (standard greedy behavior)
    
    return selected_content


def estimate_token_budget(
    model_context_window: int,
    system_prompt_tokens: int = 200,
    response_tokens: int = 500,
    safety_margin: float = 0.1
) -> int:
    """
    Estimate available token budget for memory context.
    
    Context window allocation:
    - System prompt (instructions)
    - User query
    - Memory context (computed here)
    - Response generation space
    - Safety margin
    
    Args:
        model_context_window: Total context window size (e.g., 4096 for Llama)
        system_prompt_tokens: Estimated tokens for system prompt
        response_tokens: Reserved tokens for model response
        safety_margin: Percentage to reserve as safety buffer (default: 10%)
    
    Returns:
        Available token budget for memory context
    
    Example:
        >>> budget = estimate_token_budget(4096)
        >>> print(budget)  # ~2977 tokens available for memories
    """
    # Calculate overhead
    overhead = system_prompt_tokens + response_tokens
    
    # Apply safety margin
    usable_window = int(model_context_window * (1 - safety_margin))
    
    # Memory budget = usable window - overhead
    memory_budget = usable_window - overhead
    
    # Ensure non-negative
    return max(0, memory_budget)


def analyze_selection_quality(
    candidates: List[Dict[str, Any]],
    selected: List[str],
    tokenizer_fn: Callable[[str], List[str]] = simple_tokenizer
) -> Dict[str, Any]:
    """
    Analyze the quality of greedy selection for debugging/tuning.
    
    Args:
        candidates: Original candidate list
        selected: Selected content strings
        tokenizer_fn: Tokenizer function used
    
    Returns:
        Dictionary with quality metrics:
        - total_score: Sum of scores of selected items
        - total_tokens: Total tokens used
        - avg_density: Average information density
        - selection_ratio: Fraction of candidates selected
        - utilization: Fraction of budget used
    """
    selected_set = set(selected)
    
    total_score = 0.0
    total_tokens = 0
    densities = []
    
    for candidate in candidates:
        content = candidate.get('content', '')
        if content in selected_set:
            score = candidate.get('score', 0.0)
            tokens = len(tokenizer_fn(content))
            
            total_score += score
            total_tokens += tokens
            if tokens > 0:
                densities.append(score / tokens)
    
    return {
        'total_score': total_score,
        'total_tokens': total_tokens,
        'avg_density': sum(densities) / len(densities) if densities else 0.0,
        'selection_ratio': len(selected) / len(candidates) if candidates else 0.0,
        'num_selected': len(selected)
    }
