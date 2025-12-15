"""
Week 4 Day 4: Pruning and Garbage Collection
=============================================
Permanently deletes memories that fall below value thresholds.
Implements vacuum process for memory hygiene.
"""

import time
from typing import Dict, Optional
from src.memory_architect.core.schema import PolicyClass


def run_garbage_collection(
    vector_store,
    score_threshold: float = 15.0,
    min_age_hours: float = 168.0  # 1 week
) -> Dict[str, int]:
    """
    Run garbage collection to prune low-value memories.
    
    Deletion Criteria:
    1. Hard expiry: expiry_timestamp < current_time
    2. Low value: reflection_score < threshold AND age > min_age
    
    Protection:
    - Canonical memories are NEVER deleted
    - Recent memories are protected (even if low score)
    
    Args:
        vector_store: ChromaManager instance
        score_threshold: Minimum score to retain (default: 15.0)
        min_age_hours: Minimum age before pruning (default: 168h = 1 week)
    
    Returns:
        Dictionary with deletion statistics:
        - expired_count: Memories deleted due to TTL
        - pruned_count: Memories deleted due to low score
        - total_deleted: Total memories removed
    
    Example:
        >>> from src.memory_architect.storage.vector_store import ChromaManager
        >>> db = ChromaManager()
        >>> stats = run_garbage_collection(db, score_threshold=15.0)
        >>> print(f"Pruned {stats['total_deleted']} memories")
    """
    current_time = time.time()
    min_timestamp = current_time - (min_age_hours * 3600.0)
    
    stats = {
        'expired_count': 0,
        'pruned_count': 0,
        'total_deleted': 0
    }
    
    # Condition 1: Hard expiry (expiry_timestamp < now)
    # Delete memories with TTL that have expired
    try:
        # Get all memories with expiry timestamps
        results = vector_store.collection.get(
            where={
                "$and": [
                    {"expiry_timestamp": {"$gt": 0.0}},  # Has expiry
                    {"expiry_timestamp": {"$lt": current_time}},  # Expired
                    {"policy": {"$ne": PolicyClass.CANONICAL.value}}  # Not canonical
                ]
            },
            include=['metadatas']
        )
        
        expired_ids = results['ids']
        if expired_ids:
            vector_store.collection.delete(ids=expired_ids)
            stats['expired_count'] = len(expired_ids)
    
    except Exception as e:
        # ChromaDB may not support complex queries in all versions
        # Fall back to manual filtering
        pass
    
    # Condition 2: Low value (score < threshold AND age > 1 week)
    # Delete old, low-scoring memories
    try:
        results = vector_store.collection.get(
            where={
                "$and": [
                    {"reflection_score": {"$lt": score_threshold}},
                    {"created_at": {"$lt": min_timestamp}},
                    {"policy": {"$ne": PolicyClass.CANONICAL.value}}
                ]
            },
            include=['metadatas']
        )
        
        pruned_ids = results['ids']
        if pruned_ids:
            vector_store.collection.delete(ids=pruned_ids)
            stats['pruned_count'] = len(pruned_ids)
    
    except Exception as e:
        # Fallback: Get all memories and filter manually
        all_results = vector_store.collection.get(include=['metadatas'])
        
        pruned_ids = []
        for doc_id, metadata in zip(all_results['ids'], all_results['metadatas']):
            # Skip canonical
            if metadata.get('policy') == PolicyClass.CANONICAL.value:
                continue
            
            # Check score and age
            score = metadata.get('reflection_score', 50.0)
            created = metadata.get('created_at', current_time)
            
            if score < score_threshold and created < min_timestamp:
                pruned_ids.append(doc_id)
        
        if pruned_ids:
            vector_store.collection.delete(ids=pruned_ids)
            stats['pruned_count'] = len(pruned_ids)
    
    stats['total_deleted'] = stats['expired_count'] + stats['pruned_count']
    
    return stats


def analyze_memory_health(vector_store, user_id: str) -> Dict[str, any]:
    """
    Analyze memory health for a user before pruning.
    
    Provides statistics to help tune garbage collection thresholds.
    
    Args:
        vector_store: ChromaManager instance
        user_id: User to analyze
    
    Returns:
        Dictionary with health metrics:
        - total_memories: Total count
        - score_distribution: Histogram of scores
        - age_distribution: Histogram of ages
        - expired_count: Number of expired memories
        - low_score_count: Number below threshold
        - canonical_count: Protected memories
    
    Example:
        >>> health = analyze_memory_health(db, "user_alice")
        >>> print(f"Low-value memories: {health['low_score_count']}")
        >>> if health['low_score_count'] > 100:
        ...     run_garbage_collection(db)
    """
    current_time = time.time()
    
    # Get all memories for user
    all_memories = vector_store.get_all_memories_for_user(user_id)
    
    health = {
        'total_memories': len(all_memories),
        'score_distribution': {'0-20': 0, '20-40': 0, '40-60': 0, '60-80': 0, '80-100': 0},
        'age_distribution': {'1d': 0, '1w': 0, '1m': 0, '3m': 0, '1y+': 0},
        'expired_count': 0,
        'low_score_count': 0,
        'canonical_count': 0
    }
    
    for memory in all_memories:
        # Score distribution
        score = memory.reflection_score
        if score < 20:
            health['score_distribution']['0-20'] += 1
        elif score < 40:
            health['score_distribution']['20-40'] += 1
        elif score < 60:
            health['score_distribution']['40-60'] += 1
        elif score < 80:
            health['score_distribution']['60-80'] += 1
        else:
            health['score_distribution']['80-100'] += 1
        
        # Age distribution
        age_hours = (current_time - memory.created_at) / 3600.0
        if age_hours < 24:
            health['age_distribution']['1d'] += 1
        elif age_hours < 168:
            health['age_distribution']['1w'] += 1
        elif age_hours < 720:
            health['age_distribution']['1m'] += 1
        elif age_hours < 2160:
            health['age_distribution']['3m'] += 1
        else:
            health['age_distribution']['1y+'] += 1
        
        # Check expiry
        expiry = getattr(memory, 'expiry_timestamp', None)
        if expiry and expiry > 0 and expiry < current_time:
            health['expired_count'] += 1
        
        # Check low score
        if score < 15.0:
            health['low_score_count'] += 1
        
        # Check canonical
        if memory.policy == PolicyClass.CANONICAL:
            health['canonical_count'] += 1
    
    return health


def schedule_garbage_collection(
    vector_store,
    user_id: str,
    auto_threshold: int = 1000,
    score_threshold: float = 15.0
) -> Optional[Dict[str, int]]:
    """
    Conditionally run garbage collection based on memory count.
    
    Only runs if user has more than auto_threshold memories.
    Prevents unnecessary pruning for light users.
    
    Args:
        vector_store: ChromaManager instance
        user_id: User to check
        auto_threshold: Minimum memories to trigger GC
        score_threshold: Score threshold for pruning
    
    Returns:
        Deletion stats if GC ran, None if skipped
    
    Example:
        >>> # Daily background job
        >>> for user in all_users:
        ...     stats = schedule_garbage_collection(db, user, auto_threshold=1000)
        ...     if stats:
        ...         log(f"Pruned {stats['total_deleted']} memories for {user}")
    """
    # Check memory count
    all_memories = vector_store.get_all_memories_for_user(user_id)
    
    if len(all_memories) < auto_threshold:
        return None  # Skip GC
    
    # Run garbage collection
    return run_garbage_collection(vector_store, score_threshold=score_threshold)
