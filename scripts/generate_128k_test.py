"""
128k Token Dataset Generator
============================
This script creates a MASSIVE file to stress-test the AI.
It fills a conversation with 130,000+ tokens of "noise" (random sentences)
and hides one "Needle" (the secret code) inside it.

Goal: See if the AI crashes or if it can find the needle.
"""

import json
import random
import os

def generate_128k_dataset(filename="data/locomo_128k.json"):
    print(f"Generating 128k+ Token Dataset: {filename}...")
    
    # Target: ~130,000 tokens (approx 520,000 chars) to exceed 128k limit
    target_chars = 550000 
    
    # 1. The Needle (Critical Info to Retrieve)
    # We place it effectively "long ago" in conversation terms
    needle = "The secret operation code is 'OMEGA-BLUE-77'."
    
    # 2. The Haystack (Chitchat/Noise)
    noise_phrases = [
        "I like iterating on code.",
        "The weather is nice.",
        "System check nominal.",
        "Waiting for input.",
        "Just thinking about algorithms.",
        "Optimization is key.",
        "Memory management is hard.",
        "Python is great.",
        "Keep the context clean.",
        "Redundancy check."
    ]
    
    # Build huge history
    turns = []
    current_chars = 0
    
    # Add Needle Early (at approx 10% mark)
    turns.append({"speaker": "user", "text": "Note this important secret."})
    turns.append({"speaker": "assistant", "text": "Ready."})
    turns.append({"speaker": "user", "text": needle}) # CRITICAL
    current_chars += len(needle)
    
    # Fill remaining space with noise
    while current_chars < target_chars:
        text = " ".join(random.choices(noise_phrases, k=50)) # ~300 chars per turn
        turns.append({"speaker": "user", "text": text})
        turns.append({"speaker": "assistant", "text": "Acknowledged."})
        current_chars += len(text)
        
    print(f"Generated {len(turns)} turns. Total Chars: {current_chars} (~{current_chars//4} tokens).")
    
    # 3. The Question (Requires the Needle)
    qa = [
        {
            "question": "What is the secret operation code?",
            "answer": "OMEGA-BLUE-77",
            "evidence": [] 
        }
    ]
    
    data = [{
        "sample_id": "stress_test_128k",
        "conversation": [{"session_id": 1, "turns": turns}],
        "qa": qa
    }]
    
    os.makedirs("data", exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(data, f)
        
    print("✅ Dataset Created.")

if __name__ == "__main__":
    generate_128k_dataset()
