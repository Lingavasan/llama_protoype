"""
Unit tests for Token Budget Management (Week 3 Day 1)
=====================================================
Tests greedy knapsack algorithm for context window optimization.
"""

import pytest
from src.memory_architect.core.budgeter import (
    greedy_knapsack_budgeting,
    simple_tokenizer,
    estimate_token_budget,
    analyze_selection_quality
)


class TestSimpleTokenizer:
    """Test the simple tokenizer approximation."""
    
    def test_basic_tokenization(self):
        """Test basic word-based tokenization."""
        text = "Hello world this is a test"
        tokens = simple_tokenizer(text)
        assert len(tokens) == 6
        assert tokens == ["Hello", "world", "this", "is", "a", "test"]
    
    def test_empty_string(self):
        """Test tokenization of empty string."""
        tokens = simple_tokenizer("")
        assert len(tokens) == 0
    
    def test_single_word(self):
        """Test single word tokenization."""
        tokens = simple_tokenizer("hello")
        assert len(tokens) == 1


class TestGreedyKnapsackBudgeting:
    """Test the greedy knapsack selection algorithm."""
    
    def test_budget_enforcement(self):
        """Test that selected items never exceed budget."""
        candidates = [
            {'content': 'Short text', 'score': 0.5},
            {'content': 'Another short piece', 'score': 0.7},
            {'content': 'This is a much longer piece of text', 'score': 0.9},
            {'content': 'Medium length text here', 'score': 0.6}
        ]
        
        max_tokens = 10
        selected = greedy_knapsack_budgeting(candidates, max_tokens)
        
        # Count actual tokens used
        total_tokens = sum(len(simple_tokenizer(content)) for content in selected)
        
        assert total_tokens <= max_tokens
    
    def test_density_based_selection(self):
        """Test that high density items are selected first."""
        # Create candidates with known densities
        # High density: short + high score
        # Low density: long + low score
        candidates = [
            {'content': 'A B C', 'score': 0.9},  # 3 tokens, density = 0.3
            {'content': 'X Y Z W V U T S R Q', 'score': 0.5},  # 10 tokens, density = 0.05
            {'content': 'Short high', 'score': 1.0}  # 2 tokens, density = 0.5
        ]
        
        max_tokens = 5  # Can fit "Short high" + "A B C"
        selected = greedy_knapsack_budgeting(candidates, max_tokens)
        
        # Should select highest density items
        assert 'Short high' in selected  # Highest density
        assert 'A B C' in selected  # Second highest density
        assert 'X Y Z W V U T S R Q' not in selected  # Lowest density, doesn't fit
    
    def test_empty_candidates(self):
        """Test with empty candidate list."""
        selected = greedy_knapsack_budgeting([], max_tokens=100)
        assert selected == []
    
    def test_all_items_under_budget(self):
        """Test when all items fit within budget."""
        candidates = [
            {'content': 'A', 'score': 0.5},
            {'content': 'B', 'score': 0.7},
            {'content': 'C', 'score': 0.9}
        ]
        
        max_tokens = 100  # Very large budget
        selected = greedy_knapsack_budgeting(candidates, max_tokens)
        
        # All items should be selected
        assert len(selected) == 3
    
    def test_all_items_over_budget(self):
        """Test when no items fit within budget."""
        candidates = [
            {'content': 'This is a very long piece of text', 'score': 0.9},
            {'content': 'Another extremely long text here', 'score': 0.8}
        ]
        
        max_tokens = 2  # Very small budget
        selected = greedy_knapsack_budgeting(candidates, max_tokens)
        
        # No items should be selected (or only very short ones)
        total_tokens = sum(len(simple_tokenizer(c)) for c in selected)
        assert total_tokens <= 2
    
    def test_zero_budget(self):
        """Test with zero token budget."""
        candidates = [
            {'content': 'Text', 'score': 0.5}
        ]
        
        selected = greedy_knapsack_budgeting(candidates, max_tokens=0)
        assert selected == []
    
    def test_custom_tokenizer(self):
        """Test with custom tokenizer function."""
        def char_tokenizer(text):
            """Tokenize by characters."""
            return list(text)
        
        candidates = [
            {'content': 'ABC', 'score': 0.9},  # 3 chars
            {'content': 'XY', 'score': 0.5}  # 2 chars
        ]
        
        max_tokens = 4
        selected = greedy_knapsack_budgeting(
            candidates, 
            max_tokens, 
            tokenizer_fn=char_tokenizer
        )
        
        # Should select ABC (higher score, 3 tokens) but not XY (would exceed)
        # Or select XY if it has higher density
        total_chars = sum(len(c) for c in selected)
        assert total_chars <= 4
    
    def test_score_zero_handling(self):
        """Test handling of zero scores."""
        candidates = [
            {'content': 'Text with zero score', 'score': 0.0},
            {'content': 'Text with some score', 'score': 0.5}
        ]
        
        max_tokens = 20
        selected = greedy_knapsack_budgeting(candidates, max_tokens)
        
        # Should prefer non-zero scores
        assert 'Text with some score' in selected
    
    def test_missing_score_field(self):
        """Test graceful handling of missing score field."""
        candidates = [
            {'content': 'Text without score'}
        ]
        
        selected = greedy_knapsack_budgeting(candidates, max_tokens=10)
        
        # Should handle gracefully (treat as score=0)
        assert isinstance(selected, list)
    
    def test_score_maximization(self):
        """Test that greedy selection approximates score maximization."""
        # Create scenario where greedy should perform well
        candidates = [
            {'content': 'A B', 'score': 0.8},  # 2 tokens, density 0.4
            {'content': 'C', 'score': 0.5},  # 1 token, density 0.5
            {'content': 'D E F', 'score': 0.9},  # 3 tokens, density 0.3
        ]
        
        max_tokens = 3
        selected = greedy_knapsack_budgeting(candidates, max_tokens)
        
        # Best selection should be C (density 0.5) + A B (density 0.4) = total 3 tokens
        # Total score = 0.5 + 0.8 = 1.3
        total_score = sum(
            c['score'] for c in candidates 
            if c['content'] in selected
        )
        
        # Should achieve reasonably high score
        assert total_score >= 1.0


class TestEstimateTokenBudget:
    """Test token budget estimation function."""
    
    def test_default_estimation(self):
        """Test default budget estimation."""
        budget = estimate_token_budget(4096)
        
        # Should reserve space for system, response, and safety margin
        assert budget > 0
        assert budget < 4096
        
        # Rough validation: should be around 2900-3000 tokens
        assert 2500 <= budget <= 3500
    
    def test_custom_parameters(self):
        """Test with custom system and response token counts."""
        budget = estimate_token_budget(
            model_context_window=2048,
            system_prompt_tokens=100,
            response_tokens=300,
            safety_margin=0.2  # 20% safety margin
        )
        
        # 2048 * 0.8 = 1638.4 (after safety margin)
        # 1638 - 100 - 300 = 1238
        expected = int(2048 * 0.8) - 100 - 300
        assert budget == expected
    
    def test_small_context_window(self):
        """Test with very small context window."""
        budget = estimate_token_budget(1024)
        
        # Should still return non-negative value
        assert budget >= 0
    
    def test_zero_safety_margin(self):
        """Test with no safety margin."""
        budget = estimate_token_budget(
            model_context_window=4096,
            system_prompt_tokens=200,
            response_tokens=500,
            safety_margin=0.0
        )
        
        # 4096 - 200 - 500 = 3396
        assert budget == 3396


class TestAnalyzeSelectionQuality:
    """Test selection quality analysis function."""
    
    def test_quality_metrics(self):
        """Test that quality metrics are computed correctly."""
        candidates = [
            {'content': 'A B', 'score': 0.8},
            {'content': 'C D E', 'score': 0.6},
            {'content': 'F', 'score': 0.9}
        ]
        
        selected = ['A B', 'F']
        
        metrics = analyze_selection_quality(candidates, selected)
        
        assert 'total_score' in metrics
        assert 'total_tokens' in metrics
        assert 'avg_density' in metrics
        assert 'selection_ratio' in metrics
        assert 'num_selected' in metrics
        
        # Verify values
        assert metrics['total_score'] == 0.8 + 0.9  # 1.7
        assert metrics['total_tokens'] == 2 + 1  # 3 tokens
        assert metrics['num_selected'] == 2
        assert metrics['selection_ratio'] == pytest.approx(2/3)
    
    def test_empty_selection(self):
        """Test with no items selected."""
        candidates = [
            {'content': 'Text', 'score': 0.5}
        ]
        
        selected = []
        metrics = analyze_selection_quality(candidates, selected)
        
        assert metrics['total_score'] == 0.0
        assert metrics['total_tokens'] == 0
        assert metrics['num_selected'] == 0


class TestIntegrationScenarios:
    """Integration tests combining multiple components."""
    
    def test_realistic_scenario(self):
        """Test realistic memory selection scenario."""
        # Simulate retrieved memories with varying lengths and scores
        candidates = [
            {'content': 'User prefers Python programming.', 'score': 0.95},
            {'content': 'Alice mentioned she likes coffee.', 'score': 0.60},
            {'content': 'The project deadline is next week.', 'score': 0.85},
            {'content': 'Very long context about historical events that happened many years ago with extensive details.', 'score': 0.40},
            {'content': 'Bob is a data scientist.', 'score': 0.70},
            {'content': 'Meeting scheduled for Monday.', 'score': 0.75}
        ]
        
        # Estimate budget for Llama 4K context
        max_tokens = estimate_token_budget(4096)
        
        # Select memories
        selected = greedy_knapsack_budgeting(candidates, max_tokens)
        
        # Analyze quality
        metrics = analyze_selection_quality(candidates, selected, simple_tokenizer)
        
        # Validations
        assert len(selected) > 0
        assert metrics['total_tokens'] <= max_tokens
        assert metrics['total_score'] > 0
        
        # High-score items should be prioritized
        assert 'User prefers Python programming.' in selected or metrics['total_tokens'] < 10
