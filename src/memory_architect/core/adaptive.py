"""
Adaptive Context Policy Manager
==============================
Think of this as the "Customs Officer" for your AI's brain.
It checks if the context suitcase is too heavy, and if so, decides what to throw out.
"""

import yaml
from typing import List, Dict, Any, Tuple
from pathlib import Path

# We try to use a real tokenizer, detailed below.
try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

class AdaptiveContextManager:
    """
    This class guards the context window.
    
    It reads `policy.yaml` to know the limit (e.g., 8192 tokens).
    If the AI is talking too much or remembering too much, this manager steps in
    and removes the least important stuff ("Noise") so the AI doesn't crash.
    """
    
    def __init__(self, config_path: str = "configs/policy.yaml"):
        self.config = self._load_config(config_path)
        self.budget_cfg = self.config.get('budget', {})
        self.adaptive_cfg = self.budget_cfg.get('adaptive', {})
        
        # How much can we carry?
        self.capacity = self.budget_cfg.get('context_capacity', 8192)
        # Safety margin
        self.buffer = self.budget_cfg.get('buffer', 500)
        self.limit = self.capacity - self.buffer
        
        # What do we throw out first? (e.g., Boring memories, then old history)
        self.strategies = self.adaptive_cfg.get('strategy_priority', ["low_score_memories", "old_history"])
        # Aim to get back down to this % utilization
        self.target_util = self.adaptive_cfg.get('target_utilization', 0.8)
        
        self.purge_history = []
        
    def _load_config(self, path: str) -> Dict:
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Oops: Could not load policy.yaml: {e}. Using defaults.")
            return {}

    def count_tokens(self, text: str) -> int:
        """A rough guess at token count (Char / 4) to be fast."""
        return len(text) // 4

    def monitor_and_optimize(
        self, 
        history: List[Dict], 
        memories: List[Dict], 
        system_prompt: str,
        user_input: str
    ) -> Dict[str, Any]:
        """
        The main check point.
        Calculates how full we are and triggers a purge if we are over the limit.
        """
        if not self.adaptive_cfg.get('enabled', False):
            return {"history": history, "memories": memories, "action": "DISABLED"}
            
        # 1. Weigh the luggage
        sys_tokens = self.count_tokens(system_prompt)
        hist_tokens = sum(self.count_tokens(t.get('content', t.get('text', ''))) for t in history)
        mem_tokens = sum(self.count_tokens(m.get('content', '')) for m in memories)
        input_tokens = self.count_tokens(user_input)
        
        total_load = sys_tokens + hist_tokens + mem_tokens + input_tokens
        utilization = total_load / self.capacity
        
        # DEBUG Breakdown
        print(f"[DEBUG] Load: Sys={sys_tokens} Hist={hist_tokens} Mem={mem_tokens} Input={input_tokens} Total={total_load}")
        
        # 2. Are we too heavy?
        critical = self.adaptive_cfg.get('critical_threshold', 0.95)
        
        status = {
            "load": total_load,
            "max": self.limit,
            "utilization": utilization,
            "action": "NONE"
        }
        
        if utilization > critical:
            print(f"\n[AdaptiveManager] 🚨 OVERLOAD ALERT: {utilization:.1%}full > {critical:.0%} limit")
            return self._execute_purge(history, memories, total_load)
            
        return {"history": history, "memories": memories, "status": status}

    def _execute_purge(self, history, memories, current_load):
        """Throw things out until we are back to a safe weight."""
        target_load = int(self.capacity * self.target_util)
        tokens_to_free = current_load - target_load
        
        print(f"[AdaptiveManager] Action: Tossing out {tokens_to_free} tokens to get back to {self.target_util:.0%} capacity")
        
        optimized_memories = list(memories)
        optimized_history = list(history)
        freed = 0
        
        # Try expected strategies in order
        for strategy in self.strategies:
            if freed >= tokens_to_free:
                break
                
            if strategy == "low_score_memories":
                freed += self._prune_memories(optimized_memories, tokens_to_free - freed)
            elif strategy == "old_history":
                freed += self._prune_history(optimized_history, tokens_to_free - freed)
                
        status = {
            "load": current_load - freed,
            "freed": freed,
            "action": "PURGED"
        }
        
        # Log the event for analysis
        from datetime import datetime
        self.purge_history.append({
            "timestamp": datetime.now().isoformat(),
            "load_before": current_load,
            "load_after": current_load - freed,
            "purged_amount": freed, 
            "target": target_load
        })
        
        return {"history": optimized_history, "memories": optimized_memories, "status": status}

    def _prune_memories(self, memories, target) -> int:
        """Strategy: Get rid of memories with low 'Reflection Scores' (the boring stuff)."""
        # Sort so weakest are first
        memories.sort(key=lambda x: x.get('reflection_score', 0.0))
        
        freed = 0
        kept = []
        for m in memories:
            cost = self.count_tokens(m.get('content', ''))
            score = m.get('reflection_score', 0.0)
            
            # PROTECTED CLASS: Never delete Critical memories (Score >= 80)
            # regardless of how much debt we have. We must find savings in History instead.
            if score >= 80.0:
                kept.append(m)
                continue
                
            if freed < target:
                print(f"   [Policy] Dropping Memory (Score {score:.2f}): {m.get('content')[:30]}...")
                freed += cost
            else:
                kept.append(m)
        
        memories[:] = kept # Update the original list
        return freed

    def _prune_history(self, history, target) -> int:
        """Strategy: Forget what we said a long time ago (oldest chat turns)."""
        freed = 0
        while history and freed < target:
            removed = history.pop(0) # Bye bye oldest turn
            text = removed.get('content', removed.get('text', ''))
            cost = self.count_tokens(text)
            # print(f"   [Policy] Truncating History: {text[:30]}...")
            freed += cost
        return freed

    def get_purge_history(self) -> List[Dict]:
        return self.purge_history
