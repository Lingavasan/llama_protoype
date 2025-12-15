"""
Evaluation Harness (The Exam Proctor)
=====================================
This script runs the "Exams" for our Memory Architect.
It checks:
1. Retrieval Recall: Did we find the right facts?
2. Factuality: Did we answer the question correctly?
"""

import time
from typing import List, Dict, Optional, Callable
import numpy as np

from src.memory_architect.storage.vector_store import ChromaManager
from src.memory_architect.core.schema import MemoryChunk, MemoryType, PolicyClass
from src.memory_architect.core.adaptive import AdaptiveContextManager
from src.memory_architect.eval.ingest import load_locomo, replay_conversation


def calculate_retrieval_recall(
    retrieved_ids: List[str],
    evidence_ids: List[str]
) -> float:
    """
    Score: How many of the required facts did we find?
    
    Formula: (Found & Needed) / Needed
    """
    if not evidence_ids:
        return 1.0  # Perfect score if we didn't need anything specific.
    
    retrieved_set = set(retrieved_ids)
    evidence_set = set(evidence_ids)
    
    intersection = retrieved_set & evidence_set
    recall = len(intersection) / len(evidence_set)
    
    return recall


def calculate_factuality(
    predicted_answer: str,
    ground_truth: str
) -> float:
    """
    Score: Did the AI say the right words?
    
    It compares the AI's answer to the "Correct Answer" using a simple word overlap.
    (In the future, we could use a smarter judge, but this works for now).
    """
    # Simple word cleaner
    import string
    def clean_words(text):
        # Remove punctuation and split
        text = text.translate(str.maketrans('', '', string.punctuation))
        return set(text.lower().split())

    pred_words = clean_words(predicted_answer)
    truth_words = clean_words(ground_truth)
    
    # Ignore boring words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'based', 'on', 'memory'}
    pred_words -= stop_words
    truth_words -= stop_words
    
    if not truth_words:
        return 1.0
    
    overlap = pred_words & truth_words
    factuality = len(overlap) / len(truth_words)
    
    return factuality


class MetricsCalculator:
    """
    The Scorekeeper.
    It tracks all the grades and prints a report card at the end.
    """
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.recalls = []
        self.factualities = []
        self.qa_details = []
    
    def add_qa_result(
        self,
        question: str,
        predicted: str,
        truth: str,
        retrieved_ids: List[str],
        evidence_ids: List[str]
    ):
        """Record the score for one question."""
        recall = calculate_retrieval_recall(retrieved_ids, evidence_ids)
        factuality = calculate_factuality(predicted, truth)
        
        self.recalls.append(recall)
        self.factualities.append(factuality)
        
        self.qa_details.append({
            'question': question,
            'predicted': predicted,
            'truth': truth,
            'recall': recall,
            'factuality': factuality
        })
    
    def get_summary(self) -> Dict[str, any]:
        """Calculate averages."""
        if not self.recalls:
            return {
                'retrieval_recall': {'mean': 0.0, 'std': 0.0, 'count': 0},
                'factuality': {'mean': 0.0, 'std': 0.0, 'count': 0}
            }
        
        return {
            'retrieval_recall': {
                'mean': float(np.mean(self.recalls)),
                'std': float(np.std(self.recalls)),
                'count': len(self.recalls)
            },
            'factuality': {
                'mean': float(np.mean(self.factualities)),
                'std': float(np.std(self.factualities)),
                'count': len(self.factualities)
            },
            'qa_count': len(self.qa_details)
        }
    
    def print_summary(self):
        """Print the Report Card."""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("EVALUATION REPORT CARD")
        print("="*60)
        
        recall = summary['retrieval_recall']
        print(f"Memory Accuracy: {recall['mean']:.1%} (±{recall['std']:.1%})")
        
        fact = summary['factuality']
        print(f"Answer Accuracy: {fact['mean']:.1%} (±{fact['std']:.1%})")
        
        print(f"Questions Asked: {summary['qa_count']}")
        print("="*60)


class EvaluationHarness:
    """
    The Exam Proctor.
    It takes a Test (Locomo Dataset) and administers it to the AI.
    """
    
    def __init__(
        self,
        memory_system: ChromaManager,
        answer_generator: Optional[Callable] = None
    ):
        """
        Initialize evaluation harness.
        
        Args:
            memory_system: ChromaManager instance
            answer_generator: Function that takes (question, memories) and returns answer
                            If None, uses simple mock
        """
        self.db = memory_system
        # If we don't have a real brain (LLM), we use a dummy one.
        self.answer_generator = answer_generator or self._mock_answer_generator
        self.metrics = MetricsCalculator()
        self.adaptive_mgr = AdaptiveContextManager()
    
    def _mock_answer_generator(
        self,
        question: str,
        memories: List[MemoryChunk]
    ) -> str:
        """
        A smarter dummy brain that respects context window limits.
        """
        # 1. Simulate Context Window (Std Recency Bias if not managed)
        # Assume a standard model has ~8k token limit (~32k chars)
        CONTEXT_LIMIT_CHARS = 32000 
        
        full_context = "\n".join([m.content for m in memories])
        
        
        # If context is too long, standard LLMs truncate the beginning (keep recent)
        if len(full_context) > CONTEXT_LIMIT_CHARS:
             # Keep the LAST N chars (Simulating "Sliding Window")
             effective_context = full_context[-CONTEXT_LIMIT_CHARS:]
        else:
             effective_context = full_context

        # DEBUG: See what we are looking for and what we have
        needle = "OMEGA-BLUE-77"
        if needle not in effective_context:
             print(f"   [MockBrain] ❌ Needle NOT FOUND in context (Len: {len(effective_context)})")
             # print(f"   [Context Snippet]: {effective_context[:100]}...")
        else:
             print(f"   [MockBrain] ✅ Needle FOUND!")
        
        if needle in effective_context:
            return f"Based on memory: The secret code is {needle}."
        
        if needle in effective_context:
            return f"Based on memory: The secret code is {needle}."
        else:
            return "Based on memory: I don't know the answer. The context didn't contain it."
    
    def evaluate_sample(
        self,
        sample: Dict,
        verbose: bool = False
    ) -> Dict:
        """
        Test the AI on a SINGLE conversation.
        """
        sample_id = sample['sample_id']
        
        if verbose:
            print(f"\nEvaluating Story: {sample_id}")
        
        # Run QA tests
        qa_results = []
        
        for qa in sample['qa']:
            question = qa['question']
            truth = qa['answer']
            evidence_ids = qa.get('evidence', [])
            
            # 1. RETRIEVE: Ask the memory system for facts
            raw_results = self.db.retrieve_candidates(
                query_text=question,
                user_id=sample_id,
                k=5
            )
            
            # 2. RANK: Sort them by importance
            ranked = self.db.rank_results(raw_results)
            
            # Extract memory IDs and chunks
            retrieved_ids = [memory_id for memory_id, _, _ in ranked]
            
            # 3. PREPARE: Get the text of the memories
            memories = []
            candidates_dicts = [] # Prepare for AdaptiveContextManager
            for memory_id, score, metadata in ranked:
                chunk = MemoryChunk(
                    id=memory_id,
                    content=metadata.get('content', ''),
                    # Add required fields with defaults
                    type=MemoryType(metadata.get('type', 'episodic')),
                    policy=PolicyClass(metadata.get('policy', 'ephemeral')),
                    source_session_id=metadata.get('source_session_id', 'unknown_session'),
                    user_id=sample_id,
                    reflection_score=score
                )
                memories.append(chunk)
                
                # We need dicts for the Adaptive Manager
                candidates_dicts.append(chunk.model_dump()) # Convert to dict
            
            # --- SYSTEM PROMPT TEMPLATE ---
            SYSTEM_TEMPLATE = """
You are an AI assistant with access to long-term memory.
Below is a list of retrieved memories relevant to the user's current query.
Use these facts to answer the question. If the memories contradict, trust the most recent one.


{memory_context}


User Query: {user_query}
"""
            # Format Context
            context_str = "\n".join([f"- {m.content}" for m in memories])
            full_prompt = SYSTEM_TEMPLATE.format(memory_context=context_str, user_query=question)
            
            # (In a real system, full_prompt is passed to LLM. Here we just document we used it)

            # 4. OPTIMIZE: Use the Adaptive Manager to prune the context
            # (See if we can fit it in the budget)
            
            # Simulate Session History
            session_history = sample['conversation'][0]['turns']
            
            policy_result = self.adaptive_mgr.monitor_and_optimize(
                history=session_history,
                memories=candidates_dicts,
                system_prompt="You are a helpful assistant.", # Fixed overhead
                user_input=question
            )
            
            # 5. SELECT: Who survived the cut?
            final_memories = []
            if 'memories' in policy_result:
                # Reconstruct MemoryChunk objects from the Optimized List
                for m_dict in policy_result['memories']:
                     chunk = MemoryChunk(
                        id=m_dict['id'],
                        content=m_dict['content'],
                        type=MemoryType(m_dict.get('type', 'episodic')),
                        policy=PolicyClass.EPHEMERAL,
                        source_session_id="eval",
                        user_id=sample_id,
                        reflection_score=m_dict['reflection_score']
                     )
                     final_memories.append(chunk)
            
            # If Adaptive Manager is off or returned weird data, fallback to raw recall
            if not final_memories and memories:
                final_memories = memories

            # 6. ANSWER: Ask the AI using the optimized context
            predicted = self.answer_generator(question, final_memories)
            
            # 7. GRADE: Calculate scores
            self.metrics.add_qa_result(
                question=question,
                predicted=predicted,
                truth=truth,
                retrieved_ids=retrieved_ids,
                evidence_ids=evidence_ids
            )
            
            qa_results.append({
                'question': question,
                'predicted': predicted,
                'truth': truth,
                'retrieved_count': len(retrieved_ids)
            })
            
            if verbose:
                print(f"  Q: {question}")
                print(f"  A (AI):   {predicted[:80]}...")
                print(f"  A (True): {truth}")
        
        return {
            'sample_id': sample_id,
            'qa_results': qa_results
        }
    
    def run_evaluation(
        self,
        dataset_path: str,
        limit: Optional[int] = None,
        verbose: bool = False
    ) -> Dict:
        """
        Run the full battery of tests.
        """
        # Load dataset
        samples = load_locomo(dataset_path)
        
        if limit:
            samples = samples[:limit]
        
        print(f"Starting Exam on {len(samples)} stories...")
        
        # Evaluate each sample
        for i, sample in enumerate(samples):
            if verbose or (i + 1) % 10 == 0:
                print(f"Progress: {i+1}/{len(samples)}")
            
            self.evaluate_sample(sample, verbose=verbose)
        
        # Get summary
        summary = self.metrics.get_summary()
        self.metrics.print_summary()
        
        return summary


if __name__ == "__main__":
    # Self-test: Run a quick exam
    from src.memory_architect.eval.ingest import batch_ingest_locomo
    from src.memory_architect.policy.privacy import PrivacyGuard
    
    # Setup
    db = ChromaManager()
    guard = PrivacyGuard()
    
    print("Loading practice questions...")
    batch_ingest_locomo("data/locomo_test.json", db, guard)
    
    print("\nStarting Practice Exam...")
    harness = EvaluationHarness(db)
    results = harness.run_evaluation("data/locomo_test.json", verbose=True)
