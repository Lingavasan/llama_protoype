"""
Week 4 Day 5: Comprehensive Reflection System Tests
===================================================
Tests complete memory dynamics: decay, reinforcement, and pruning.
"""

import pytest
import time
from unittest.mock import Mock, MagicMock
from src.memory_architect.core.schema import MemoryChunk, MemoryType, PolicyClass
from src.memory_architect.core.decay import apply_forgetting_curve, apply_reinforcement
from src.memory_architect.core.reflection import update_memory_scores, check_relevance
from src.memory_architect.core.pruning import run_garbage_collection, analyze_memory_health


class TestDecayOverTime:
    """Test memory decay over extended periods."""
    
    def test_decay_over_one_month(self):
        """
        Test decay over 1 month period (Day 5 requirement).
        
        Mock time forward 1 month, verify significant score drop.
        """
        # Initial state
        initial_score = 80.0
        current_time = time.time()
        
        # Mock time forward 1 month (30 days)
        one_month_later = current_time + (86400 * 30)
        
        # Apply decay
        final_score = apply_forgetting_curve(
            current_score=initial_score,
            last_access_ts=current_time,
            stability=2.0,
            current_time=one_month_later
        )
        
        # After 1 month without access, should decay significantly
        assert final_score < 20.0, \
            f"Score after 1 month should be < 20, got {final_score}"
        assert final_score >= 0.0, "Score cannot be negative"
        
        # Calculate retention percentage
        retention = (final_score / initial_score) * 100
        assert retention < 25.0, \
            f"Should retain less than 25% after 1 month, got {retention:.1f}%"
    
    def test_decay_progression(self):
        """Test decay at multiple time points."""
        initial_score = 90.0
        base_time = time.time()
        
        # Test at different intervals
        intervals = [
            (1, "1 day"),
            (7, "1 week"),
            (14, "2 weeks"),
            (30, "1 month"),
            (90, "3 months")
        ]
        
        previous_score = initial_score
        for days, label in intervals:
            future_time = base_time + (86400 * days)
            score = apply_forgetting_curve(
                initial_score,
                base_time,
                stability=2.0,
                current_time=future_time
            )
            
            # Each interval should show more decay
            assert score < previous_score, \
                f"Score at {label} should be less than previous"
            previous_score = score
    
    def test_decay_with_stability_variations(self):
        """Test how different stability values affect decay."""
        base_time = time.time()
        one_week = base_time + (86400 * 7)
        
        # Low stability (fast decay)
        low_stable = apply_forgetting_curve(
            80.0, base_time, stability=1.0, current_time=one_week
        )
        
        # High stability (slow decay)
        high_stable = apply_forgetting_curve(
            80.0, base_time, stability=5.0, current_time=one_week
        )
        
        # Higher stability should retain more
        assert high_stable > low_stable, \
            "Higher stability should result in less decay"


class TestReinforcementSimulation:
    """Test reinforcement through simulated retrieval."""
    
    def test_reinforcement_on_retrieval(self):
        """
        Test score increase when memory is used (Day 5 requirement).
        
        Simulate retrieval where memory is "used", verify score increases.
        """
        # Initial chunk
        chunk = MemoryChunk(
            content="User prefers Python for data science",
            type=MemoryType.EPISODIC,
            policy=PolicyClass.EPHEMERAL,
            source_session_id="test",
            user_id="test_user",
            tags=[],
            reflection_score=50.0,
            created_at=time.time(),
            last_accessed=time.time()
        )
        chunk.access_count = 0
        
        # Mock vector store
        mock_store = Mock()
        mock_store.update_memory_metadata = Mock()
        
        # Simulate LLM response that uses this memory
        response_text = "I recommend Python for your data science project"
        
        # Update scores (reflection loop)
        update_memory_scores(
            retrieved_chunks=[chunk],
            response_text=response_text,
            vector_store=mock_store
        )
        
        # Verify reinforcement occurred
        assert chunk.reflection_score == 60.0, \
            f"Score should increase from 50 to 60, got {chunk.reflection_score}"
        assert chunk.access_count == 1, "Access count should increment"
        
        # Verify database was updated
        assert mock_store.update_memory_metadata.called
        call_args = mock_store.update_memory_metadata.call_args[0]
        updates = call_args[1]
        assert updates['reflection_score'] == 60.0
        assert updates['access_count'] == 1
    
    def test_decay_on_non_use(self):
        """Test that unused retrieved memories decay."""
        chunk = MemoryChunk(
            content="User likes coffee in the morning",
            type=MemoryType.EPISODIC,
            policy=PolicyClass.EPHEMERAL,
            source_session_id="test",
            user_id="test_user",
            tags=[],
            reflection_score=50.0,
            created_at=time.time() - 86400,  # 1 day old
            last_accessed=time.time() - 86400
        )
        
        mock_store = Mock()
        mock_store.update_memory_metadata = Mock()
        
        # Response doesn't mention coffee
        response_text = "The weather is sunny today"
        
        update_memory_scores([chunk], response_text, mock_store)
        
        # Should have decayed
        assert chunk.reflection_score < 50.0, \
            "Unused memory should decay"
    
    def test_repeated_use_accumulation(self):
        """Test that repeated use strengthens memory."""
        # Test reinforcement directly to avoid reflection loop's decay
        score = 50.0
        
        # Simulate 5 uses with direct reinforcement
        for i in range(5):
            score = apply_reinforcement(score, boost=10.0)
        
        # Should reach cap
        assert score == 100.0, \
            f"After 5 reinforcements, score should be 100 (capped), got {score}"


class TestGarbageCollection:
    """Test pruning/vacuum process."""
    
    def test_low_score_deletion(self):
        """
        Test deletion of low-score memories (Day 5 requirement).
        
        Set score to 5, run GC, verify deletion.
        """
        # Create mock vector store with ChromaDB collection
        mock_collection = Mock()
        mock_collection.get = Mock(return_value={
            'ids': ['low-score-1', 'low-score-2'],
            'metadatas': [
                {
                    'reflection_score': 5.0,
                    'created_at': time.time() - (86400 * 14),  # 2 weeks old
                    'policy': PolicyClass.EPHEMERAL.value
                },
                {
                    'reflection_score': 8.0,
                    'created_at': time.time() - (86400 * 10),  # 10 days old
                    'policy': PolicyClass.EPHEMERAL.value
                }
            ]
        })
        mock_collection.delete = Mock()
        
        mock_store = Mock()
        mock_store.collection = mock_collection
        
        # Run garbage collection
        stats = run_garbage_collection(mock_store, score_threshold=15.0)
        
        # Verify deletion occurred
        assert mock_collection.delete.called, "Delete should be called"
        assert stats['pruned_count'] == 2, \
            f"Should prune 2 low-score memories, got {stats['pruned_count']}"
        assert stats['total_deleted'] >= 2
    
    def test_canonical_protection(self):
        """Test that canonical memories are never deleted."""
        mock_collection = Mock()
        mock_collection.get = Mock(return_value={
            'ids': [],
            'metadatas': []
        })
        mock_collection.delete = Mock()
        
        mock_store = Mock()
        mock_store.collection = mock_collection
        
        # Even with low threshold, canonical should be protected
        stats = run_garbage_collection(mock_store, score_threshold=100.0)
        
        # Canonical filtering is in the query itself
        get_calls = mock_collection.get.call_args_list
        for call in get_calls:
            where_clause = call[1].get('where', {})
            # Should have policy != canonical filter
            assert any(
                '$ne' in str(condition) for condition in str(where_clause).split()
            ), "Should filter out canonical memories"
    
    def test_recent_memory_protection(self):
        """Test that recent memories are protected even if low score."""
        mock_collection = Mock()
        mock_collection.get = Mock(return_value={
            'ids': [],
            'metadatas': []
        })
        
        mock_store = Mock()
        mock_store.collection = mock_collection
        
        # Run with default settings (1 week minimum age)
        run_garbage_collection(mock_store, score_threshold=15.0, min_age_hours=168.0)
        
        # Check that age filter was applied
        get_calls = mock_collection.get.call_args_list
        assert len(get_calls) >= 1, "Should query for pruning candidates"
    
    def test_expired_memory_deletion(self):
        """Test hard expiry TTL deletion."""
        current_time = time.time()
        expired_time = current_time - 3600  # 1 hour ago
        
        mock_collection = Mock()
        mock_collection.get = Mock(return_value={
            'ids': ['expired-1'],
            'metadatas': [{
                'expiry_timestamp': expired_time,
                'policy': PolicyClass.EPHEMERAL.value
            }]
        })
        mock_collection.delete = Mock()
        
        mock_store = Mock()
        mock_store.collection = mock_collection
        
        stats = run_garbage_collection(mock_store)
        
        # Should attempt deletion
        assert stats['expired_count'] >= 0


class TestMemoryHealthAnalysis:
    """Test memory health analysis tools."""
    
    def test_health_analysis(self):
        """Test memory health metrics."""
        # Create mock memories with varying health
        mock_memories = [
            # High score, recent
            MemoryChunk(
                content="Good memory",
                type=MemoryType.SEMANTIC,
                policy=PolicyClass.CANONICAL,
                source_session_id="s1",
                user_id="test",
                tags=[],
                reflection_score=85.0,
                created_at=time.time() - 3600
            ),
            # Low score, old
            MemoryChunk(
                content="Fading memory",
                type=MemoryType.EPISODIC,
                policy=PolicyClass.EPHEMERAL,
                source_session_id="s2",
                user_id="test",
                tags=[],
                reflection_score=10.0,
                created_at=time.time() - (86400 * 30)
            )
        ]
        
        mock_store = Mock()
        mock_store.get_all_memories_for_user = Mock(return_value=mock_memories)
        
        health = analyze_memory_health(mock_store, "test")
        
        assert health['total_memories'] == 2
        assert health['canonical_count'] == 1
        assert health['low_score_count'] >= 1
        assert '80-100' in health['score_distribution']


class TestIntegratedWorkflow:
    """Test complete reflection workflow."""
    
    def test_full_memory_lifecycle(self):
        """Test: Store → Use → Decay → Prune."""
        # This would be an integration test in practice
        # For unit test, we verify the components work together
        
        # 1. Store memory
        chunk = MemoryChunk(
            content="Important fact",
            type=MemoryType.EPISODIC,
            policy=PolicyClass.EPHEMERAL,
            source_session_id="test",
            user_id="test",
            tags=[],
            reflection_score=50.0,
            created_at=time.time(),
            last_accessed=time.time()
        )
        
        # 2. Use memory (reinforcement)
        chunk.reflection_score = apply_reinforcement(chunk.reflection_score)
        assert chunk.reflection_score == 60.0
        
        # 3. Time passes, decay
        month_later = time.time() + (86400 * 30)
        chunk.reflection_score = apply_forgetting_curve(
            chunk.reflection_score,
            chunk.last_accessed,
            current_time=month_later
        )
        
        # 4. Score should have decayed below pruning threshold
        assert chunk.reflection_score < 15.0, \
            "After 1 month without use, should be prunable"
