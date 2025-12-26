# Active Policy Catalog
**Date**: 2025-12-21
**System**: Memory Architect

This document details the mathematical models, logic, and code implementations for the active policies in the system.

---

## Domain A: Lifecycle & Hygiene (The "Forgetting" Engine)

### 1. DSR Decay Policy (Spacing Effect)
**Goal:** Mimic human forgetting curves; retain what is repeated, delete what is ignored.
*   **Formula:**
    *   **Retrievability ($R$)**: Probability of recall at time $t$.
        $$R = C \cdot e^{\frac{-t}{S}}$$
        *Where $S$ is Stability (days until R drops to 90%) and $C$ is a scaling factor.*
    *   **Stability Update ($S_{new}$)**: How much stronger a memory gets after review.
        $$S_{new} = S_{old} \cdot (1 + \text{Factor} \cdot \text{Difficulty})$$
*   **Codebase**: `src/memory_architect/core/decay.py`

### 2. Scene Memory Policy (Working Memory)
**Goal:** Protect active thoughts from decay until the topic changes.
*   **Formula (Topic Shift)**:
    *   **Cosine Similarity**:
        $$Sim(v_{new}, \mu_{scene}) = \frac{v_{new} \cdot \mu_{scene}}{\|v_{new}\| \|\mu_{scene}\|}$$
    *   **Trigger**:
        $$\text{If } Sim < 0.6 \implies \text{Flush to LTM}$$
*   **Codebase**: `src/memory_architect/core/working_memory.py`
*   **Role**: Acts as a buffer. Memories in the Scene Buffer have effective $R=1.0$ (Perfect Recall) regardless of time.

### 3. Consolidation Policy (Episodic $\to$ Semantic)
**Goal:** Compress raw chat logs into concise facts.
*   **Logic**: Trigger when `count(episodic) > 20` OR `oldest > 24h`.
*   **Codebase**: `src/memory_architect/core/summarization.py`

---

## Domain B: Consistency & Truth (The "Anti-Hallucination" Engine)

### 4. Provenance & Trust Policy
**Goal:** Distinguish between System Axioms and User Statements.
*   **Hierarchy**: 1.0 (Axiom) > 0.8 (User) > 0.6 (Model) > 0.4 (Web).
*   **Codebase**: `src/memory_architect/core/schema.py`

### 5. Conflict Resolution Policy
**Goal:** Arbitrate contradictory beliefs.
*   **Logic**:
    $$Winner = \max(Trust_A, Trust_B)$$
    *(Tie-breaker: Recent Timestamp wins)*
*   **Codebase**: `src/memory_architect/core/consistency.py`

---

## Domain C: Adaptive Retrieval (The "Utility" Engine)

### 6. Kalman Filter Utility Policy
**Goal:** Track intrinsic usefulness of a memory.
*   **Formula (1-D Kalman)**:
    *   Updates `reflection_score` ($\hat{x}$) based on retrieval frequency.
    *   $$K = P / (P + R_{noise})$$
    *   $$\hat{x}_{new} = \hat{x} + K \cdot (Observed - \hat{x})$$
*   **Codebase**: `src/memory_architect/core/kalman.py`

### 7. Spreading Activation Policy (Associative Retrieval)
**Goal:** Simulate "Stream of Consciousness" by retrieving contextually linked memories.
*   **Formula**:
    *   Find neighbors in time window $T \pm 10 \text{ min}$.
    *   $$Score_{neighbor} = Score_{parent} \cdot 0.5$$
*   **Codebase**: `src/memory_architect/storage/vector_store.py`

### 8. Adaptive Budgeting Policy
**Goal:** Maximize information density in the context window.
*   **Codebase**: `src/memory_architect/core/adaptive.py`

---

## Domain D: Governance & Safety (The "Compliance" Engine)

### 9. Privacy & PII Policy
**Goal:** Redact sensitive data via regex patterns.
*   **Codebase**: `src/memory_architect/policy/privacy.py`

---

## Policy Relationships (The "Ecosystem")

How these policies interact to form a cohesive mind:

1.  **DSR $\leftrightarrow$ Scene Memory (The Handoff)**
    *   **Relationship**: Handover.
    *   **Status**: ✅ **Implemented & Validated**.
    *   **Verification**: `tests/week5/test_fsrs_update.py` verified the math. LoCoMo Thesis Validation confirmed **93% token reduction**.
    *   **Logic**: Memories start in **Scene Buffer** (Protected from decay). When the scene *ends* (Topic Shift), they are handed to **DSR** (Decay begins).
    *   *Effect*: Ensures short-term focus isn't penalized by long-term forgetting rules.

2.  **DSR $\leftrightarrow$ Kalman (The Ranking Mix)**
    *   **Relationship**: Component Forces.
    *   **Logic**: Final Ranking Score = $w_1 \cdot \text{DSR}(Time) + w_2 \cdot \text{Kalman}(Utility) + w_3 \cdot \text{Similarity}$.
    *   *Effect*: A memory can be old (Low DSR) but highly useful (High Kalman) and still be retrieved.

3.  **Consolidation $\leftrightarrow$ Consistency (The Truth Filter)**
    *   **Relationship**: Gatekeeper.
    *   **Logic**: **Consolidation** generates new facts. **Consistency** checks them against existing facts before saving.
    *   *Effect*: Prevents the "Schizophrenic Mind" by ensuring new summaries don't contradict old axioms.

4.  **Spreading Activation $\leftrightarrow$ Budgeting (The Pressure)**
    *   **Relationship**: Supply vs Demand.
    *   **Logic**: **Spreading Activation** *increases* the number of candidates (Supply). **Budgeting** *limits* the context window (Demand).
    *   *Effect*: Budgeting becomes more crucial because Spreading Activation floods the pipeline with "Contextual" memories that might not match the query keywords.

---

## Hardcoded Logic & Constants

Explicit values defining system behavior:

| File | Constant / Value | Purpose |
| :--- | :--- | :--- |
| `consistency.py` | `TrustLevel.AXIOM = 1.0` | Absolute truth (System Instructions). |
| `consistency.py` | `TrustLevel.DIRECT_USER = 0.8` | High trust, but allows System override. |
| `consistency.py` | `TrustLevel.INFERRED = 0.6` | Model summaries are treated as "Heresay". |
| `working_memory.py` | `TOPIC_SHIFT_THRESHOLD = 0.6` | Cosine Sim drop < 0.6 triggers flush. |
| `working_memory.py` | `BUFFER_CAPACITY = 10` | Max turns before forced flush to LTM. |
| `vector_store.py` | `window_minutes = 10` | Spreading Activation looks +/- 10 mins. |
| `vector_store.py` | `Score Boost = 0.5` | Neighboring memories get 50% score penalty. |
| `schema.py` | `default_stability = 48.0` | Memories naturally decay to 90% recall in 48h. |
| `api.py` | `Mock LLM Response` | **Simulation Only**. Must be replaced with Real LLM. |
