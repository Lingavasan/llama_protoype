
import yaml
import time
import sys
import os

# Make sure Python can find our code in 'src'
sys.path.append(os.getcwd())

from src.memory_architect.storage.vector_store import ChromaManager
from src.memory_architect.policy.privacy import PrivacyGuard
from src.memory_architect.eval.ingest import batch_ingest_locomo
from src.memory_architect.eval.harness import EvaluationHarness
from src.memory_architect.core.adaptive import AdaptiveContextManager

CONFIG_PATH = "configs/policy.yaml"
DATASET_PATH = "data/locomo_128k.json"

def update_config(adaptive_enabled: bool, privacy_enabled: bool):
    """
    Hack: We modify the 'policy.yaml' file on the fly
    to turn features ON or OFF for the test.
    """
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
        
    config['budget']['adaptive']['enabled'] = adaptive_enabled
    config['privacy']['enabled'] = privacy_enabled
    
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(config, f)

def run_test_case(name: str, adaptive: bool):
    print(f"\n🚀 Running Case: {name} (Adaptive={adaptive})...")
    # Make sure we see PII redaction by keeping privacy ON
    update_config(adaptive_enabled=adaptive, privacy_enabled=True)
    
    # 1. Setup
    # Use a separate DB path for this big test so we don't mess up main memory
    db_path = "./data/.chroma_128k"
    
    db = ChromaManager(persist_path=db_path)
    guard = PrivacyGuard()
    
    # Lazy Load: Only ingest 128k tokens if we haven't already.
    # It takes a while, so this saves time on re-runs.
    if db.collection.count() < 100:
        print("   Ingesting 128k Dataset (First Run)...")
        batch_ingest_locomo(DATASET_PATH, db, guard)
    else:
        print("   Using existing 128k Vector Data.")

    # 2. Run the Test
    harness = EvaluationHarness(db)
    
    start_time = time.time()
    
    # What are we testing?
    # - If Adaptive is OFF: The model tries to read ALL 137k tokens. (Danger Zone)
    # - If Adaptive is ON: The Manager sees 137k > 128k, and PURGES 9k tokens of noise. (Safe Zone)
    
    
    try:
        summary = harness.run_evaluation(DATASET_PATH, verbose=False) # Keep logs clean
        duration = time.time() - start_time
        
        return {
            "Configuration": name,
            "Recall": f"{summary['retrieval_recall']['mean']:.1%}",
            "Factuality": f"{summary['factuality']['mean']:.1%}",
            "Success": "✅ Yes",
            "Duration": f"{duration:.2f}s",
            "Notes": "Adaptive Purge Triggered" if adaptive else "Processed Full Context (Risk of Overflow)"
        }
    except Exception as e:
        return {
            "Configuration": name,
            "Recall": "0%",
            "Factuality": "0%",
            "Success": "❌ Crash",
            "Duration": f"{time.time() - start_time:.2f}s",
            "Notes": str(e)[:50]
        }

def main():
    print("="*60)
    print("      MEMORY ARCHITECT: 128K HALLUCINATION BENCHMARK      ")
    print("="*60)
    
    results = []
    
    # Case 1: Without Architecture (Baseline)
    # "Just let the model figure it out."
    res_base = run_test_case("Baseline (Raw 128k)", adaptive=False)
    results.append(res_base)
    
    # Case 2: With Memory Architect
    # "Manage the context intelligently."
    res_arch = run_test_case("Memory Architect (Adaptive)", adaptive=True)
    results.append(res_arch)
    
    print("\n" + "="*80)
    print("FINAL RESULTS TABLE")
    print("="*80)
    
    # Print a nice table
    print("| Configuration               | Recall | Factuality | Success | Duration | Notes                                      |")
    print("|-----------------------------|--------|------------|---------|----------|--------------------------------------------|")
    for r in results:
        print(f"| {r['Configuration']:<27} | {r['Recall']:<6} | {r['Factuality']:<10} | {r['Success']:<7} | {r['Duration']:<8} | {r['Notes']:<42} |")
        
    print("\nAnalysis:")
    print("- Baseline: Processed all tokens. If Factuality is low, it means it 'hallucinated' or missed the needle.")
    print("- Architect: If Factuality is high, it means the Adaptive Policy successfully prioritized the needle.")

if __name__ == "__main__":
    main()
