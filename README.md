# Memory Architect (Llama 3.1 8B) 🧠

> **A Strategic Memory Layer for LLMs**  
> *Solve the 128k Context Wall with Adaptive Token Budgeting, Policy-Driven Reflection, and Read-Path Privacy.*

---

## 🚀 The Mission
Modern LLMs have huge context windows (128k+), but sending everything to the model is:
1.  **Expensive**: Pay for tokens you don't need.
2.  **Unstable**: Causes OOM (Out-Of-Memory/Crash) failures on local hardware.
3.  **Inaccurate**: The "Lost-in-the-Middle" problem.

**Memory Architect** provides a "Hippocampus Layer" that monitors context load and intelligently purges "Noise" (Chitchat) while protecting "Needles" (Secret codes, rules, constraints).

---

## 🛠️ Step-by-Step Setup

### 1. Prerequisites
Before you begin, ensure you have:
*   **Python 3.11** (Highly Recommended for stability; 3.13 currently has build issues with AI dependencies).
*   **Ollama**: Installed and running ([Download here](https://ollama.com/))
*   **Llama 3.1 Model**: 
    ```bash
    ollama pull llama3.1:8b
    ```

### 2. Installation
Clone the repository and set up the environment using **Poetry**:

```bash
# Clone the repo
git clone <repo_url>
cd llama_protoype

# Force Poetry to use Python 3.11 (if multiple versions installed)
poetry env use python3.11

# Install dependencies
poetry install
```

### 3. Verification: Proof of Life
Run the verification script to ensure all layers (Privacy, Adaptive, Vector DB) are communicating:

```bash
poetry run python scripts/run_eval_custom.py
```
**Expected Outcome**: A "100.0% Accuracy" report card.

---

### Step 1: Data Setup
The benchmarking datasets are pre-included in the `data/` folder. However, you can regenerate the 128k stress test at any time:
```bash
# Optional: Regenerate the 128k dataset
poetry run python scripts/generate_128k_test.py
```

### Step 2: Run the A/B Test (Baseline vs. Architect)
Compare how a "Raw LLM" handles 128k tokens vs. the "Memory Architect":
```bash
poetry run python scripts/benchmark_128k.py
```

**What happens during this test?**
1.  **Baseline**: Sends the full 128k context (~550,000 chars) to Ollama. 
    - *Expected Result*: **Server Crash (500 Error)**. Resource exhaustion.
2.  **Memory Architect**: Intercepts the load, detects the overload, and **purges 94% of the noise**.
    - *Expected Result*: **Success**. The model survives and finds the secret code.
*   **Logs**: Check `purge_log.json` for the exact timestamps and amount of context purged.

---

## ⚙️ Configuration (`configs/policy.yaml`)
Tune your "Brain" from a single YAML file:

```yaml
budget:
  adaptive:
    enabled: true            # Toggles Token Budgeting
    target_utilization: 0.8  # Aim for 80% usage
    critical_threshold: 0.95 # Purge if > 95% full

privacy:
  enabled: true              # Redact PII (Names, Emails) on the fly
```

---

## 📂 Project Navigation
*   `src/memory_architect/core`: 
    - `adaptive.py`: The context manager (Purge Logic).
    - `reflection.py`: The scoring engine (Importance Logic).
*   `src/memory_architect/policy`: 
    - `privacy.py`: The redaction gate (Security Logic).
*   `src/memory_architect/storage`: 
    - `vector_store.py`: ChromaDB integration.
*   `scripts/`: 
    - `benchmark_128k.py`: The 128k Stress Test.
    - `run_eval_custom.py`: Final verification script.

---

## 📜 Detailed Analysis
For deep-dive metrics on token reduction and latency, see [ANALYSIS_REPORT.md](ANALYSIS_REPORT.md).

---

## 🔧 Troubleshooting
*   **Poetry Build Errors**: If `srsly` or `llama-cpp-python` fails to build, ensure you are using **Python 3.11**. Run `poetry env use python3.11` before installing.
*   **Ollama Connection**: Ensure `ollama serve` is running in the background.
*   **ChromaDB Busy**: If the DB is locked, delete the `./data/.chroma_128k` folder to reset.

---
*Created by the Memory Architect Team.*
