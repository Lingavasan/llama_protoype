"""
Week 4 Day 3: Summarization Pipeline
=====================================
Compresses episodic memories into semantic knowledge.
Converts conversation logs into consolidated facts.
"""

from typing import List, Optional
from src.memory_architect.core.schema import MemoryChunk, MemoryType, PolicyClass
import time


# Summarization prompt template
SUMMARIZATION_PROMPT = """Condense the following conversation into a concise list of key facts about the user.

Guidelines:
- Ignore pleasantries and small talk
- Preserve important dates, names, and specific details
- Focus on facts that would be useful for future conversations
- Use bullet points for clarity
- Be specific and factual

Conversation:
{episodic_content}

Key facts:"""


def generate_summarization_prompt(episodic_chunks: List[MemoryChunk]) -> str:
    """
    Generate LLM prompt for summarizing episodic memories.
    
    Args:
        episodic_chunks: List of episodic memory chunks to summarize
    
    Returns:
        Formatted prompt string for LLM
    
    Example:
        >>> chunks = [
        ...     MemoryChunk(content="User: I like Python", ...),
        ...     MemoryChunk(content="User: I work on ML", ...)
        ... ]
        >>> prompt = generate_summarization_prompt(chunks)
    """
    # Concatenate episodic content with separators
    conversation_parts = []
    for i, chunk in enumerate(episodic_chunks, 1):
        conversation_parts.append(f"[{i}] {chunk.content}")
    
    episodic_content = "\n".join(conversation_parts)
    
    # Format prompt
    prompt = SUMMARIZATION_PROMPT.format(episodic_content=episodic_content)
    
    return prompt


def summarize_episodic_memories(
    episodic_chunks: List[MemoryChunk],
    llm_generate_fn,
    user_id: str,
    source_session_id: Optional[str] = None
) -> MemoryChunk:
    """
    Compress episodic memories into a single semantic summary.
    
    Process:
    1. Generate summarization prompt
    2. Call LLM to extract key facts
    3. Create new semantic memory chunk
    4. Inherit metadata from source chunks
    
    Args:
        episodic_chunks: List of episodic memories to summarize
        llm_generate_fn: Function that takes prompt and returns response
        user_id: User ID for the summary
        source_session_id: Optional session ID for tracking
    
    Returns:
        New semantic MemoryChunk with consolidated facts
    
    Example:
        >>> def my_llm(prompt):
        ...     # Call your LLM here
        ...     return "- User prefers Python\\n- Works on ML projects"
        >>> 
        >>> episodic = [chunk1, chunk2, chunk3]
        >>> summary = summarize_episodic_memories(episodic, my_llm, "user_alice")
        >>> print(summary.type)  # MemoryType.SEMANTIC
    """
    # Generate prompt
    prompt = generate_summarization_prompt(episodic_chunks)
    
    # Call LLM
    summary_text = llm_generate_fn(prompt)
    
    # Calculate aggregated metadata
    total_access_count = sum(
        getattr(chunk, 'access_count', 0) for chunk in episodic_chunks
    )
    avg_reflection_score = sum(
        chunk.reflection_score for chunk in episodic_chunks
    ) / len(episodic_chunks) if episodic_chunks else 50.0
    
    # Boost score for semantic memories (they're more valuable)
    semantic_score = min(100.0, avg_reflection_score * 1.2)
    
    # Extract tags from all episodic chunks
    all_tags = set()
    for chunk in episodic_chunks:
        all_tags.update(chunk.tags)
    
    # Create semantic memory chunk
    summary_chunk = MemoryChunk(
        content=summary_text,
        type=MemoryType.SEMANTIC,
        policy=PolicyClass.CANONICAL,  # Semantic memories are canonical
        source_session_id=source_session_id or "summarization_pipeline",
        user_id=user_id,
        tags=list(all_tags) + ["summarized"],
        reflection_score=semantic_score,
        created_at=time.time(),
        last_accessed=time.time(),
        ttl_seconds=None  # Canonical memories don't expire
    )
    
    # Preserve access count
    summary_chunk.access_count = total_access_count
    
    return summary_chunk


def should_trigger_summarization(
    user_id: str,
    vector_store,
    episodic_threshold: int = 20,
    time_threshold_hours: float = 24.0
) -> bool:
    """
    Determine if summarization should be triggered for a user.
    
    Triggers when:
    - User has >= episodic_threshold episodic memories
    - Oldest episodic memory is >= time_threshold_hours old
    
    Args:
        user_id: User to check
        vector_store: ChromaManager instance
        episodic_threshold: Minimum number of episodic memories to trigger
        time_threshold_hours: Minimum age of oldest memory (hours)
    
    Returns:
        True if summarization should run
    
    Example:
        >>> if should_trigger_summarization("user_alice", db):
        ...     run_summarization_job("user_alice")
    """
    # Get episodic memories for user
    episodic_memories = vector_store.get_memories_by_type(
        user_id=user_id,
        memory_type=MemoryType.EPISODIC
    )
    
    # Check count threshold
    if len(episodic_memories) < episodic_threshold:
        return False
    
    # Check time threshold
    current_time = time.time()
    oldest_timestamp = min(chunk.created_at for chunk in episodic_memories)
    age_hours = (current_time - oldest_timestamp) / 3600.0
    
    return age_hours >= time_threshold_hours


def run_summarization_job(
    user_id: str,
    vector_store,
    llm_generate_fn,
    batch_size: int = 20,
    delete_episodic: bool = False
) -> Optional[MemoryChunk]:
    """
    Run summarization job for a user (background task).
    
    Workflow:
    1. Check if summarization should trigger
    2. Collect oldest episodic memories
    3. Summarize into semantic memory
    4. Store semantic memory
    5. Optionally delete episodic memories
    
    Args:
        user_id: User to process
        vector_store: ChromaManager instance
        llm_generate_fn: LLM generation function
        batch_size: Number of episodic memories to summarize
        delete_episodic: Whether to delete episodic chunks after summarization
    
    Returns:
        Created semantic memory chunk, or None if skipped
    
    Example:
        >>> # Background job (run periodically)
        >>> def llm_fn(prompt):
        ...     # Your LLM client
        ...     return llm.generate(prompt)
        >>> 
        >>> summary = run_summarization_job(
        ...     "user_alice",
        ...     db,
        ...     llm_fn,
        ...     batch_size=20,
        ...     delete_episodic=True
        ... )
    """
    # Check if we should run
    if not should_trigger_summarization(user_id, vector_store):
        return None
    
    # Get oldest episodic memories
    episodic_memories = vector_store.get_memories_by_type(
        user_id=user_id,
        memory_type=MemoryType.EPISODIC
    )
    
    # Sort by creation time, take oldest batch
    episodic_memories.sort(key=lambda x: x.created_at)
    to_summarize = episodic_memories[:batch_size]
    
    # Summarize
    summary_chunk = summarize_episodic_memories(
        to_summarize,
        llm_generate_fn,
        user_id
    )
    
    # Store semantic memory
    vector_store.add_memory(summary_chunk)
    
    # Optionally delete episodic chunks
    if delete_episodic:
        for chunk in to_summarize:
            vector_store.delete_memory(chunk.id)
    
    return summary_chunk
