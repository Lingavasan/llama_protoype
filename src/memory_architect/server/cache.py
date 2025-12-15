"""
Week 5 Day 3: Caching Strategy
===============================
LRU cache for embedding optimization.
"""

from functools import lru_cache
import hashlib
from typing import List, Optional


# Global cache statistics
cache_stats = {
    'hits': 0,
    'misses': 0,
    'total_requests': 0
}


@lru_cache(maxsize=1000)
def get_cached_embedding(text: str) -> str:
    """
    Cache embeddings to avoid recomputation.
    
    Uses LRU (Least Recently Used) eviction policy.
    Returns cache key for the embedding (actual embedding
    would be retrieved from embedding model).
    
    Args:
        text: Text to generate embedding for
    
    Returns:
        Cache key (in production, return actual embedding)
    
    Note:
        In production, this would call the embedding model:
        return embedding_model.encode(text).tolist()
    """
    global cache_stats
    cache_stats['total_requests'] += 1
    
    # Generate cache key
    key = cache_key(text)
    
    # In production, generate actual embedding here
    # For now, return the key to demonstrate caching
    return key


def cache_key(text: str) -> str:
    """
    Generate cache key from text.
    
    Uses MD5 hash for deterministic, fixed-length keys.
    
    Args:
        text: Input text
    
    Returns:
        MD5 hash of text
    """
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def get_cache_stats() -> dict:
    """
    Get cache statistics.
    
    Returns:
        Dictionary with:
        - hits: Number of cache hits
        - misses: Number of cache misses
        - total_requests: Total requests
        - hit_rate: Percentage of hits
    """
    global cache_stats
    total = cache_stats['total_requests']
    hits = cache_stats['hits']
    
    hit_rate = (hits / total * 100) if total > 0 else 0.0
    
    return {
        'hits': hits,
        'misses': cache_stats['misses'],
        'total_requests': total,
        'hit_rate': round(hit_rate, 2)
    }


def clear_cache():
    """Clear the embedding cache."""
    get_cached_embedding.cache_clear()
    global cache_stats
    cache_stats = {
        'hits': 0,
        'misses': 0,
        'total_requests': 0
    }


def cache_info():
    """Get LRU cache info."""
    return get_cached_embedding.cache_info()
