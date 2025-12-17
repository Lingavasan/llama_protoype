"""
Reflection Engine (Heuristic)
=============================
This engine decides what's truly important.
It looks at a memory and asks: "Is this a golden nugget of info, or just chitchat?"
"""

import re

class ReflectionEngine:
    """
    Assigns an 'Importance Score' to every thought.
    
    In a real brain (or big LLM), this would be complex reasoning.
    Here, we use a simple rule for our "Sudoku Demo":
    - Numbers & Settings ("9x9 grid") = GOLD (High Score).
    - "Hi", "Cool" = DUST (Low Score).
    """
    
    def __init__(self):
        # Words that signal "Hey! This is a rule!"
        self.constraint_keywords = [
            "set", "config", "mode", "level", "difficulty", 
            "grid", "size", "9x9", 
            "must", "always", "never", "ensure", "remember", 
            "important", "alert", "critical",
            "secret", "code", "token", "key", "password", # High value targets
            "live", "reside", "location", "city", # Personal constraints
            "seattle", "john", "sushi", "rover", "python", # Demo Entities
            "hard", "easy", "enable", "disable"
        ]
        
    def evaluate(self, text: str) -> float:
        """Rate how important this text is from 0.0 (Useless) to 100.0 (Vital)."""
        text_lower = text.lower()
        score = 50.0 # Start neutral
        
        # 1. Hunt for the "Needle" (Constraints)
        is_constraint = any(k in text_lower for k in self.constraint_keywords)
        has_numbers = bool(re.search(r'\d', text))
        
        if is_constraint:
            score = 85.0
            if has_numbers:
                # Specific numbers like "9x9" are extremely valuable instructions.
                score = 95.0
        
        # 2. Hunt for the "Haystack" (Chitchat)
        # If it's short and not a constraint, it's likely just noise.
        if len(text.split()) < 5 and not is_constraint:
            score = 15.0 
            
        if text_lower in ["hi", "hello", "ok", "cool", "thanks", "sounds good"]:
            score = 10.0
            
        return score

    def categorize(self, score: float) -> str:
        if score >= 80: return "CRITICAL (Keep Verbatim)"
        if score <= 20: return "NOISE (Purge)"
        return "GENERAL (Summarize?)"
