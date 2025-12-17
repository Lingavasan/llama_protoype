"""
Custom Evaluation Runner
========================
A simple script to run the "Exam" locally on your machine.
It creates a temporary memory space, loads the data, and checks the score.
"""

import sys
import os

# Ensure src is in path so we can import our modules
sys.path.append(os.getcwd())

from src.memory_architect.storage.vector_store import ChromaManager
from src.memory_architect.policy.privacy import PrivacyGuard
from src.memory_architect.eval.ingest import batch_ingest_locomo
from src.memory_architect.eval.harness import EvaluationHarness

def run_reflection_demo_metrics():
    """Print a little reminder about what just happened."""
    print("\n" + "="*60)
    print("ADAPTIVE POLICY EXECUTION COMPLETE")
    print("="*60)
    print("Check the [Policy Sentinel] logs above to verify:")
    print("1. Did we notice the memory was too full?")
    print("2. Did we delete the junk?")
    print("3. Is the context size safe now?")
    print("="*60)

def main():
    # Setup - Using a temporary collection for evaluation to keep it clean
    print("Initializing components...")
    
    # 0. CLEANUP: Ensure we start fresh (Student-Proofing)
    db_path = "./data/.chroma_eval_user"
    if os.path.exists(db_path):
        import shutil
        shutil.rmtree(db_path)
        print(f"   [System] Cleared temporary database at {db_path}")

    # We use a separate folder so we don't mess up your real memories.
    db = ChromaManager(persist_path=db_path) 
    # Disable Privacy for the Demo so we can see the "Seattle" answer clearly
    guard = PrivacyGuard(enabled=False)
    
    dataset_path = "data/locomo_test.json"
    
    # 1. Memorize the Study Material
    print("\n1. Ingesting LoCoMo Dataset...")
    batch_ingest_locomo(dataset_path, db, guard)
    
    # 2. Take the Test
    print("\n2. Running Performance Evaluation...")
    harness = EvaluationHarness(db)
    results = harness.run_evaluation(dataset_path, verbose=True)
    
    # Show detailed metrics
    run_reflection_demo_metrics()

if __name__ == "__main__":
    main()
