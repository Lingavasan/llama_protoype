# Memory Architect (Llama 3.1 8B) 🧠

> **A Strategic Memory Layer for LLMs**  
> *Solve the 128k Context Wall with Adaptive Token Budgeting, Policy-Driven Reflection, and Read-Path Privacy.*

---

## 🚀 The Mission
Modern LLMs have huge context windows (128k+), but sending everything to the model is:
1.  **Expensive**: Pay for tokens you don't need.
2.  **Unstable**: Causes OOM (Out-Of-Memory) crashes on local hardware.
3.  **Inaccurate**: The "Lost-in-the-Middle" problem.

**Memory Architect** provides a "Hippocampus Layer" that monitors context load and intelligently purges "Noise" (Chitchat) while protecting "Needles" (Secret codes, rules, constraints).

---

## 🛠️ Step-by-Step Setup

### 1. Prerequisites
Before you begin, ensure you have:
*   **Python 3.10+**
*   **Ollama**: Installed and running ([Download here](https://ollama.com/))
*   **Llama 3.1 Model**: Run this command to pull the model:
    ```bash
    ollama pull llama3.1:8b
    ```

### 2. Installation
Clone the repository and set up the environment using **Poetry**:

```bash
# Clone the repo
git clone <repo_url>
cd llama_protoype

# Install dependencies with Poetry
poetry install

# Alternatively, if you don't use poetry:
# pip install -r requirements.txt
```

### 3. Verification: Proof of Life
Once installed, run the system verification script to ensure all layers (Privacy, Adaptive, Vector DB) are communicating:

```bash
poetry run python scripts/run_eval_custom.py
```
**Expected Outcome**: You should see a "100% Accuracy" report for a small control set.

---

## 📊 Benchmarking the 128k Stress Test
This is the core of the project. We simulate a 128k-token "Haystack" and bury a "Needle" (Secret Code) inside it.

### Step 1: Generate the Massive Dataset
If the dataset isn't present, generate the 128k stress test:
```bash
poetry run python scripts/generate_128k_test.py
```

### Step 2: Run the A/B Test (Baseline vs. Architect)
Compare how a "Raw LLM" handles 128k tokens vs. the "Memory Architect".
```bash
poetry run python scripts/benchmark_128k.py
```

**What happens during this test?**
1.  **Baseline**: Sends the full 128k context (~550,000 chars) to Ollama. 
    - *Expected Result*: **Server Crash / 500 Error** (Local hardware cannot handle it).
2.  **Memory Architect**: Intercepts the load, detects the overload, and **purges 94% of the noise**.
    - *Expected Result*: **Success**. The model survives the load and finds the secret code.

---

## ⚙️ Configuration (`configs/policy.yaml`)
You can tune the "Brain" from a single YAML file:

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
*   `scripts/`: Automation for benchmarks and evaluations.

---

## 📜 Detailed Analysis
For deep-dive metrics on token reduction, latency, and factuality scores, see the [ANALYSIS_REPORT.md](ANALYSIS_REPORT.md).

---
*Created by the Memory Architect Team.*
