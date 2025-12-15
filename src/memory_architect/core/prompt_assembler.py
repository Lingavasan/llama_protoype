"""
Week 3 Day 4: Context Assembly and Prompt Injection
===================================================
Formats selected memories into prompt structure for LLM grounding.
"""

from typing import List, Optional


# Standard system template for memory-grounded responses
SYSTEM_TEMPLATE = """You are an AI assistant with access to long-term memory.
Below is a list of retrieved memories relevant to the user's current query.
Use these facts to answer the question. If the memories contradict, trust the most recent one.

{memory_context}

User Query: {user_query}"""


def assemble_context(
    selected_memories: List[str],
    user_query: str,
    template: Optional[str] = None,
    memory_separator: str = "\n\n"
) -> str:
    """
    Assemble selected memories and user query into a formatted prompt.
    
    This function creates the final prompt that will be sent to the LLM,
    injecting retrieved memory context before the user's query to ground
    the model's responses in factual information.
    
    Args:
        selected_memories: List of memory content strings (already budgeted)
        user_query: The user's current question/request
        template: Custom template (uses SYSTEM_TEMPLATE if None)
        memory_separator: String to join memories (default: double newline)
    
    Returns:
        Formatted prompt string ready for LLM
    
    Example:
        >>> memories = [
        ...     "User prefers Python programming.",
        ...     "Alice works on machine learning projects."
        ... ]
        >>> query = "What programming language should I use?"
        >>> prompt = assemble_context(memories, query)
        >>> # Returns formatted prompt with memories injected
    
    Design Notes:
    - Memories injected BEFORE query to establish context first
    - Template instructs model on memory usage and contradiction handling
    - Separator allows clear distinction between different memories
    """
    # Use default template if none provided
    if template is None:
        template = SYSTEM_TEMPLATE
    
    # Join memories with separator
    memory_context = memory_separator.join(selected_memories) if selected_memories else "(No relevant memories found)"
    
    # Format template with memory context and query
    prompt = template.format(
        memory_context=memory_context,
        user_query=user_query
    )
    
    return prompt


def create_custom_template(
    system_instructions: str,
    memory_instructions: str = "Use the following memories to inform your response:",
    contradiction_handling: str = "If memories contradict, trust the most recent one."
) -> str:
    """
    Create a custom prompt template with specified instructions.
    
    Args:
        system_instructions: Opening system instructions
        memory_instructions: Instructions for using memories
        contradiction_handling: How to handle contradictions
    
    Returns:
        Custom template string with placeholders
    
    Example:
        >>> template = create_custom_template(
        ...     system_instructions="You are a helpful coding assistant.",
        ...     memory_instructions="Reference these code examples:"
        ... )
    """
    return f"""{system_instructions}

{memory_instructions}

{contradiction_handling}

{{memory_context}}

User Query: {{user_query}}"""


def format_memory_with_metadata(
    memories: List[tuple],
    include_timestamp: bool = True,
    include_source: bool = False
) -> List[str]:
    """
    Format memories with metadata for richer context.
    
    Args:
        memories: List of (content, metadata) tuples
        include_timestamp: Include creation timestamp
        include_source: Include source session ID
    
    Returns:
        List of formatted memory strings
    
    Example:
        >>> memories = [
        ...     ("User likes Python", {"created_at": 1234567890, "source_session_id": "s1"}),
        ...     ("Alice is a developer", {"created_at": 1234567900, "source_session_id": "s2"})
        ... ]
        >>> formatted = format_memory_with_metadata(memories)
    """
    from datetime import datetime
    
    formatted = []
    for content, metadata in memories:
        parts = [content]
        
        if include_timestamp and 'created_at' in metadata:
            timestamp = metadata['created_at']
            date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
            parts.append(f"[Recorded: {date_str}]")
        
        if include_source and 'source_session_id' in metadata:
            parts.append(f"[Session: {metadata['source_session_id']}]")
        
        formatted.append(" ".join(parts))
    
    return formatted


def estimate_prompt_tokens(
    selected_memories: List[str],
    user_query: str,
    template: Optional[str] = None,
    tokenizer_fn=None
) -> int:
    """
    Estimate total tokens in assembled prompt.
    
    Useful for validation that final prompt fits in context window.
    
    Args:
        selected_memories: Memory content list
        user_query: User's query
        template: Custom template (optional)
        tokenizer_fn: Tokenizer function (uses simple estimate if None)
    
    Returns:
        Estimated token count
    """
    prompt = assemble_context(selected_memories, user_query, template)
    
    if tokenizer_fn is None:
        # Simple word-based estimate
        return len(prompt.split())
    else:
        return len(tokenizer_fn(prompt))
