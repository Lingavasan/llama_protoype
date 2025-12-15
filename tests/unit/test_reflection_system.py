"""
Unit tests for Week 4: Reflection, Summarization, and Adaptation
================================================================
Tests forgetting curves, reflection loop, and summarization pipeline.
"""

import pytest
import time
from src.memory_architect.core.decay import (
    apply_forgetting_curve,
    apply_reinforcement,
    calculate_decay_rate,
    estimate_time_to_threshold
)
from src.memory_architect.core.reflection import check_relevance
from src.memory_architect.core.schema import MemoryChunk, MemoryType, PolicyClass


class TestForgettingCurve:
    """Test Ebbinghaus forgetting curve implementation."""
    
    def test_recent_memory_minimal_decay(self):
        """Test that recently accessed memories decay minimally."""
        current_time = time.time()
        recent_access = current_time - 3600  # 1 hour ago
        
        score = apply_forgetting_curve(
            current_score=80.0,
            last_access_ts=recent_access,
            stability=2.0,
            current_time=current_time
        )
        
        # After 1 hour with dynamic stability, expect moderate decay
        # Dynamic stability for score 80: S = 2.0 * (80/50) = 3.2
        # Retention = e^(-1/3.2) ≈ 0.73
        assert score > 55.0, f"Score {score} decayed too much for recent access"
        assert score <= 80.0, "Score should not increase"
    
    def test_old_memory_significant_decay(self):
        """Test that old memories decay significantly."""
        current_time = time.time()
        old_access = current_time - (86400 * 7)  # 7 days ago
        
        score = apply_forgetting_curve(
            current_score=80.0,
            last_access_ts=old_access,
            stability=2.0,
            current_time=current_time
        )
        
        # After 7 days, should have decayed significantly
        assert score < 50.0, f"Old memory score {score} didn't decay enough"
    
    def test_dynamic_stability(self):
        """Test that higher scores lead to slower decay."""
        current_time = time.time()
        access_time = current_time - 86400  # 1 day ago
        
        # High score memory
        high_score = apply_forgetting_curve(
            current_score=90.0,
            last_access_ts=access_time,
            stability=2.0,
            current_time=current_time
        )
        
        # Low score memory
        low_score = apply_forgetting_curve(
            current_score=30.0,
            last_access_ts=access_time,
            stability=2.0,
            current_time=current_time
        )
        
        # High score should retain more (higher % of original)
        high_retention = high_score / 90.0
        low_retention = low_score / 30.0
        
        assert high_retention > low_retention, \
            "Higher scored memories should decay slower"
    
    def test_score_floor(self):
        """Test that scores cannot go negative."""
        current_time = time.time()
        very_old = current_time - (86400 * 365)  # 1 year ago
        
        score = apply_forgetting_curve(
            current_score=10.0,
            last_access_ts=very_old,
            stability=2.0,
            current_time=current_time
        )
        
        assert score >= 0.0, "Score should not go negative"
    
    def test_zero_elapsed_time(self):
        """Test handling of zero elapsed time."""
        current_time = time.time()
        
        score = apply_forgetting_curve(
            current_score=75.0,
            last_access_ts=current_time,
            stability=2.0,
            current_time=current_time
        )
        
        # No time elapsed = no decay
        assert score == pytest.approx(75.0, rel=0.01)


class TestReinforcement:
    """Test memory reinforcement (score increases on use)."""
    
    def test_basic_reinforcement(self):
        """Test basic score reinforcement."""
        new_score = apply_reinforcement(50.0, boost=10.0)
        assert new_score == 60.0
    
    def test_reinforcement_cap(self):
        """Test that scores cap at max_score."""
        new_score = apply_reinforcement(95.0, boost=10.0, max_score=100.0)
        assert new_score == 100.0
    
    def test_custom_boost(self):
        """Test custom boost values."""
        new_score = apply_reinforcement(40.0, boost=5.0)
        assert new_score == 45.0


class TestDecayUtilities:
    """Test utility functions for decay analysis."""
    
    def test_calculate_decay_rate(self):
        """Test decay rate calculation."""
        rate = calculate_decay_rate(score=80.0, stability=2.0)
        
        # Should be a percentage
        assert 0.0 <= rate <= 1.0
        #  decay rate should be reasonable
        assert rate < 0.5, "Hourly decay should be < 50%"
    
    def test_estimate_time_to_threshold(self):
        """Test threshold estimation."""
        hours = estimate_time_to_threshold(
            current_score=80.0,
            threshold_score=20.0,
            stability=2.0
        )
        
        # Should return positive hours
        assert hours > 0
        
        # Verify by applying decay
        final_score = apply_forgetting_curve(
            current_score=80.0,
            last_access_ts=time.time() - (hours * 3600),
            stability=2.0
        )
        
        assert final_score <= 20.0, "Estimated time should bring score to threshold"


class TestRelevanceChecking:
    """Test memory relevance checking for reflection loop."""
    
    def test_high_relevance(self):
        """Test detecting high relevance."""
        memory = "User prefers Python programming for data science"
        response = "I recommend Python for your data science project"
        
        is_relevant = check_relevance(memory, response, threshold=0.2)
        assert is_relevant, "Should detect shared keywords"
    
    def test_low_relevance(self):
        """Test detecting low relevance."""
        memory = "User likes coffee in the morning"
        response = "The weather forecast predicts rain tomorrow"
        
        is_relevant = check_relevance(memory, response, threshold=0.3)
        assert not is_relevant, "Should not detect relevance"
    
    def test_exact_match(self):
        """Test exact content match."""
        content = "Python is great for machine learning"
        
        is_relevant = check_relevance(content, content, threshold=0.3)
        assert is_relevant, "Identical content should be relevant"
    
    def test_empty_content(self):
        """Test handling of empty content."""
        is_relevant = check_relevance("", "some text", threshold=0.3)
        assert not is_relevant
        
        is_relevant = check_relevance("some text", "", threshold=0.3)
        assert not is_relevant


class TestMemoryLifecycle:
    """Integration tests for complete memory lifecycle."""
    
    def test_use_then_decay_cycle(self):
        """Test memory that's used then forgotten."""
        # Initial state
        score = 50.0
        last_use = time.time()
        
        # Used in response - reinforce
        score = apply_reinforcement(score, boost=10.0)
        assert score == 60.0
        
        # Not used for a week - decay
        one_week_later = last_use + (86400 * 7)
        score = apply_forgetting_curve(
            score,
            last_use,
            stability=2.0,
            current_time=one_week_later
        )
        
        # Should have decayed below original
        assert score < 50.0, "Unused memory should eventually decay"
    
    def test_repeated_reinforcement(self):
        """Test memory with repeated use."""
        score = 50.0
        
        # Used 5 times
        for _ in range(5):
            score = apply_reinforcement(score, boost=10.0)
        
        # Should be significantly higher
        assert score == 100.0, "Repeated use should strengthen memory"
    
    def test_decay_stabilization(self):
        """Test that high-score memories are more stable."""
        current_time = time.time()
        day_ago = current_time - 86400
        
        # Strong memory (score 90)
        strong_final = apply_forgetting_curve(90.0, day_ago, 2.0, current_time)
        
        # Weak memory (score 30)
        weak_final = apply_forgetting_curve(30.0, day_ago, 2.0, current_time)
        
        # Strong should retain more of its score
        strong_retention = strong_final / 90.0
        weak_retention = weak_final / 30.0
        
        assert strong_retention > weak_retention


class TestSummarizationPrompt:
    """Test summarization prompt generation."""
    
    def test_prompt_generation(self):
        """Test that prompts are generated correctly."""
        from src.memory_architect.core.summarization import generate_summarization_prompt
        
        chunks = [
            MemoryChunk(
                content="User: I like Python",
                type=MemoryType.EPISODIC,
                policy=PolicyClass.EPHEMERAL,
                source_session_id="s1",
                user_id="test",
                tags=[]
            ),
            MemoryChunk(
                content="User: I work on ML",
                type=MemoryType.EPISODIC,
                policy=PolicyClass.EPHEMERAL,
                source_session_id="s1",
                user_id="test",
                tags=[]
            )
        ]
        
        prompt = generate_summarization_prompt(chunks)
        
        assert "Python" in prompt
        assert "ML" in prompt
        assert "Key facts" in prompt


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_future_timestamp(self):
        """Test handling of future timestamp."""
        future = time.time() + 3600
        score = apply_forgetting_curve(80.0, future, 2.0)
        
        # Should return original score (no decay for future)
        assert score == 80.0
    
    def test_very_low_stability(self):
        """Test minimum stability floor."""
        score = apply_forgetting_curve(
            current_score=50.0,
            last_access_ts=time.time() - 3600,
            stability=0.001,  # Very low
            current_time=time.time()
        )
        
        # Should still work (floor prevents extreme decay)
        assert score >= 0.0
    
    def test_zero_score_decay(self):
        """Test decay on zero score."""
        score = apply_forgetting_curve(0.0, time.time() - 86400, 2.0)
        assert score == 0.0
