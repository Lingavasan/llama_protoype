"""
LoCoMo Dataset Ingestion
========================
This tool feeds our memory system with test data.
It pretends to be a user having a conversation so we can test if the AI remembers it later.
"""

import json
import time
from typing import List, Dict, Optional
from pathlib import Path

from src.memory_architect.storage.vector_store import ChromaManager
from src.memory_architect.policy.privacy import PrivacyGuard
from src.memory_architect.core.schema import MemoryChunk, MemoryType, PolicyClass
from src.memory_architect.core.reflection import ReflectionEngine


def load_locomo(file_path: str) -> List[Dict]:
    """
    Read the benchmark file.
    
    The file looks like a list of conversations, plus some questions to ask at the end.
    Structure:
      - Conversation: "Hi, I'm Alice."
      - QA: "What is my name?" -> "Alice"
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Oops, can't find the dataset here: {file_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Just in case it's a single object, Wrap it in a list so we can loop over it.
    if isinstance(data, dict):
        data = [data]
    
    # Quick sanity check
    for sample in data:
        assert 'sample_id' in sample, "Broken Data: Missing 'sample_id'"
        assert 'conversation' in sample, "Broken Data: Missing 'conversation'"
        assert 'qa' in sample, "Broken Data: Missing 'qa'"
    
    return data


def replay_conversation(
    sample: Dict,
    memory_system: ChromaManager,
    privacy_guard: Optional[PrivacyGuard] = None,
    sanitize: bool = True
) -> Dict[str, any]:
    """
    The Time Machine.
    
    It plays back a recorded conversation as if it's happening right now.
    Every time the User speaks, we save that as a memory.
    """
    sample_id = sample['sample_id']
    conversation = sample['conversation']
    
    # Initialize the Judge
    reflector = ReflectionEngine()
    
    memory_ids = []
    turn_count = 0
    
    for session in conversation:
        session_id = session.get('session_id', 0)
        
        for turn in session['turns']:
            # We only memorize what the USER said. (The Assistant already knows what *it* said).
            if turn.get('speaker') == 'user':
                text = turn['text']
                
                # Should we censor secrets?
                if sanitize and privacy_guard:
                    text = privacy_guard.sanitize(text, role="user")
                
                # Judge the memory importance
                importance_score = reflector.evaluate(text)
                
                # Create the memory object
                chunk = MemoryChunk(
                    content=text,
                    type=MemoryType.EPISODIC,
                    policy=PolicyClass.EPHEMERAL,
                    source_session_id=f"{sample_id}_session_{session_id}",
                    user_id=sample_id,
                    tags=["locomo", f"session_{session_id}"],
                    reflection_score=importance_score,
                    created_at=time.time(),
                    last_accessed=time.time(),
                    ttl_seconds=86400 * 7  # 1 week memory
                )
                
                # Save it
                memory_system.add_memory(chunk)
                memory_ids.append(chunk.id)
                turn_count += 1
    
    return {
        'memory_ids': memory_ids,
        'turn_count': turn_count,
        'session_count': len(conversation),
        'sample_id': sample_id
    }


def batch_ingest_locomo(
    dataset_path: str,
    memory_system: ChromaManager,
    privacy_guard: Optional[PrivacyGuard] = None,
    limit: Optional[int] = None
) -> List[Dict]:
    """
    Process a whole bunch of test conversations at once.
    """
    samples = load_locomo(dataset_path)
    
    if limit:
        samples = samples[:limit]
    
    results = []
    
    for i, sample in enumerate(samples):
        print(f"Ingesting story {i+1} of {len(samples)}: {sample['sample_id']}")
        
        result = replay_conversation(
            sample,
            memory_system,
            privacy_guard,
            sanitize=False # Keep it RAW so we can test the Read-Path Privacy later.
        )
        
        results.append(result)
    
    return results


def create_sample_locomo_data(output_path: str):
    """
    Generates a tiny dummy dataset so you can test if things are working.
    """
    sample_data = [
        {
            "sample_id": "test_001",
            "conversation": [
                {
                    "session_id": 1,
                    "turns": [
                        {"speaker": "user", "text": "My name is Alice"},
                        {"speaker": "assistant", "text": "Nice to meet you, Alice!"}
                    ]
                },
                {
                    "session_id": 2,
                    "turns": [
                        {"speaker": "user", "text": "I prefer Python for ML"},
                        {"speaker": "assistant", "text": "Python is great for ML!"}
                    ]
                }
            ],
            "qa": [
                {
                    "question": "What is my name?",
                    "answer": "Alice",
                    "evidence": [] 
                },
                {
                    "question": "What programming language do I prefer?",
                    "answer": "Python",
                    "evidence": []
                }
            ]
        }
    ]
    
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"Made a dummy file at: {output_path}")


if __name__ == "__main__":
    # If you run this file directly, it runs a quick test.
    create_sample_locomo_data("data/locomo_test.json")
    
    db = ChromaManager()
    guard = PrivacyGuard()
    
    results = batch_ingest_locomo("data/locomo_test.json", db, guard)
    
    print("\nDone:")
    for result in results:
        print(f"  Sample {result['sample_id']}: Remembered {len(result['memory_ids'])} things.")
