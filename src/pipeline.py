from typing import Dict
import numpy as np
from src.config import load_config
from src.llm import OllamaLLM
from src.memory import JsonlMemory, ChromaMemory
from src.policy import Policy
from src.rag import HNSWRAG, ChromaRAG
from src.rag_multi import MultiChromaRAG
from src.utils.tokenizer import count_tokens
from src.utils.logging import log_event
from src.tags import tag_text, base_meta
from src.reflect import critique_messages, rewrite_messages
from src.router import MemGPTRouter
from src.budget import BudgetConfig, pack_with_budget

class Engine:
    def __init__(self, config_path="configs/policy.yaml", host="http://localhost:11434"):
        self.cfg = load_config(config_path)
        self.llm = OllamaLLM(
            host=host,
            model=self.cfg["gen"]["model"],
            temperature=self.cfg["gen"]["temperature"],
            num_predict=self.cfg["gen"]["num_predict"],
        )

        # Load RAG (external memory)
        ext = self.cfg.get("external_memory", {})
        self.rag_enabled = bool(ext.get("enabled", False))
        self.rag_sources = []
        self.rag_top_k = int(ext.get("top_k", 3))
        self.rag_embed_model = ext.get("embed_model", self.cfg["memory"]["embed_model"]) 
        # Multi-source RAG aggregator (preferred if chroma_sources provided)
        self.multi_rag = None
        if ext.get("enabled") and ext.get("chroma_sources"):
            try:
                self.multi_rag = MultiChromaRAG(ext["chroma_sources"], host=host)
            except Exception:
                self.multi_rag = None
        if self.rag_enabled:
            # Enable multiple sources: corpus (HNSW), chroma collection(s), and optional JSON index
            try:
                # HNSW corpus
                idx = ext.get("index", {})
                if idx:
                    from pathlib import Path
                    ip, mp, infp = idx["index_path"], idx["meta_path"], idx["info_path"]
                    if Path(ip).exists() and Path(mp).exists() and Path(infp).exists():
                        self.rag_sources.append(HNSWRAG(index_path=ip, meta_path=mp, info_path=infp))
                # Chroma collections
                for chroma in ext.get("chroma", []):
                    try:
                        self.rag_sources.append(ChromaRAG(path=chroma.get("path", "data/chroma"), collection=chroma.get("collection", "knowledge")))
                    except Exception:
                        pass
            except Exception:
                self.rag_sources = []

        # Load router 
        r_cfg = self.cfg.get("router", {})
        self.router = None
        if bool(r_cfg.get("enabled", False)):
            try:
                self.router = MemGPTRouter.load(r_cfg["bundle_path"])
            except Exception:
                self.router = None  

        # Initialize memory and policy (Chroma preferred if available/configured)
        mem_cfg = self.cfg.get("memory", {})
        backend = mem_cfg.get("backend", "chroma").lower()
        self.memory = None
        if backend == "chroma":
            try:
                self.memory = ChromaMemory(
                    path=mem_cfg.get("chroma_path", "data/chroma"),
                    collection=mem_cfg.get("collection", "memory"),
                )
            except Exception:
                # Fallback to JSONL if Chroma unavailable
                self.memory = JsonlMemory(mem_cfg.get("jsonl_path", "memory.jsonl"))
        else:
            self.memory = JsonlMemory(mem_cfg.get("jsonl_path", "memory.jsonl"))
        self.policy = Policy(self.cfg)


    def _embed(self, text: str):
        vec = self.llm.embed(text, self.cfg["memory"]["embed_model"])
        v = np.array(vec, dtype="float32")
        v /= (np.linalg.norm(v) + 1e-8)
        return v.tolist()

    def _embed_with_model(self, text: str, embed_model: str):
        vec = self.llm.embed(text, embed_model)
        v = np.array(vec, dtype="float32")
        v /= (np.linalg.norm(v) + 1e-8)
        return v.tolist()


    def run(self, user_text: str, debug: bool=False) -> Dict:  # the main pipeline!
        run_dir = self.cfg.get("logging", {}).get("dir", "runs")
        from time import time as _now
        _t0 = _now()

        # 1) Policy + tags
        ok, why = self.policy.check_input(user_text)
        if not ok:
            return {"status": "blocked", "reason": why}
        if not ok:
            return {"status": "blocked", "reason": why}
        tags_data = tag_text(user_text, client=self.llm)
        tags = tags_data.get("tags", [])
        category = tags_data.get("category", "other")

        pred_action = self.router.predict(user_text) if self.router else None

        # 2) Memory search (internal memory)
        qv = self._embed(user_text)
        recalls = self.memory.search(qv, top_k=int(self.cfg["memory"]["top_k"]))
        
        # Refresh retrieved memories (update timestamp to avoid TTL)
        if recalls and hasattr(self.memory, "touch"):
            recall_ids = [it.get("id") for _, it in recalls if it.get("id")]
            self.memory.touch(recall_ids)

        # Optional external memory (RAG)
        rag_recalls: list = []
        if self.multi_rag is not None:
            try:
                rag_recalls = self.multi_rag.search(user_text)
            except Exception:
                rag_recalls = []
        elif getattr(self, "rag_enabled", False) and getattr(self, "rag_sources", None):
            try:
                qv_rag = self._embed_with_model(user_text, self.rag_embed_model)
                for src in self.rag_sources:
                    rag_recalls.extend(src.search(qv_rag, k=self.rag_top_k))
            except Exception:
                rag_recalls = []

        # 3) Budgeted context packing
        router_hint = f"\nPolicy router suggests action: {pred_action}" if pred_action else ""
        system = self.cfg.get("system", "You are a helpful assistant.") + router_hint
        b = self.cfg.get("budget", {})
        bcfg = BudgetConfig(
            C=b.get("context_capacity", 4096),
            eps=b.get("buffer", 128),
            max_retrieved_chunks=b.get("max_retrieved_chunks", 5),
        )
        memory_texts = [it["text"] for _, it in recalls]
        retrieved_texts = [it["text"] for _, it in rag_recalls]

        # 3a) Evidence sufficiency check -> conservative fallback
        def _has_minimal_evidence(q: str, mem: list[str], rag: list[str]) -> bool:
            # Bypass for conversational/personal inputs
            q_lower = q.lower().strip()
            if any(q_lower.startswith(p) for p in ["i ", "i'm", "my ", "hello", "hi ", "hey "]):
                return True

            import re
            # Focus on alphanumeric tokens length >= 4
            toks = [t for t in re.sub(r"[^a-z0-9 ]", " ", q_lower).split() if len(t) >= 4]
            if not toks:
                return False
            ev = " \n ".join((mem or []) + (rag or [])).lower()
            # token-based signal: at least two distinct focus tokens present OR one rare token
            hits = {t for t in set(toks) if t in ev}
            token_ok = (len(hits) >= 2) or any(len(t) >= 6 for t in hits)
            # similarity-based signal: use top similarity from recalls if available
            max_sim = 0.0
            try:
                if recalls:
                    max_sim = max(max_sim, max(s for s, _ in recalls))
                if rag_recalls:
                    max_sim = max(max_sim, max(s for s, _ in rag_recalls))
            except Exception:
                pass
            sim_ok = max_sim >= 0.20  # conservative threshold
            return token_ok or sim_ok

        if not _has_minimal_evidence(user_text, memory_texts, retrieved_texts):
            # Log and store only the user turn, respond with fallback
            fallback = "Sorry, I don't know."
            meta = base_meta(self.cfg.get("version", "0.0.0"))
            if pred_action:
                meta["router_action"] = pred_action
            self.memory.add("user", user_text, qv, {"tags": tags, **meta})
            # no assistant/reflection memory for fallback to avoid reinforcing gaps
            try:
                log_event(run_dir, "turn", {"latency_s": _now() - _t0, "tokens": count_tokens(user_text + "\n" + fallback), "recalls": []})
            except Exception:
                pass
            return {"status": "ok", "output": fallback, "note": "insufficient_evidence", "tags": tags, "recalls": [], "rag_recalls": [], "critique": ""}
        # Build combined recalls for context display (memory + rag)
        all_recalls = recalls + rag_recalls
        recall_text = "\n".join([
            f"- (rag:{it.get('source_name','')}) {(it.get('title') + ': ') if it.get('title') else ''}{it['text']}"
            for _, it in all_recalls
        ]) or "None."

        system_out, M_used, R_used, G_final = pack_with_budget(
            system, user_text, memory_texts, retrieved_texts,
            gen_tokens=self.cfg["gen"]["num_predict"], cfg=bcfg
        )
        mem_block = "\n".join([f"- {m}" for m in M_used]) or "None."
        rag_block = recall_text  # use merged recall_text instead of only R_used
        context = f"Relevant memory:\n{mem_block}\n\nRetrieved evidence:\n{rag_block}"

    # 4) Draft
        draft = self.llm.chat([
            {"role": "system", "content": system_out + "\n" + context + "\n\nInstruction: Use the provided memory to answer the user's question. If the memory contains the answer, output it directly."},
            {"role": "user",   "content": user_text}
        ], num_predict=G_final)

    # 5) Reflection (critique -> rewrite)
        critique = self.llm.chat(critique_messages(user_text, draft, context), num_predict=256)
        if "No critique needed" in critique or len(critique) < 10:
            final = draft
        else:
            final = self.llm.chat(rewrite_messages(user_text, draft, critique), num_predict=256)
        latency = _now() - _t0
        used_tokens = count_tokens(user_text + "\n" + final)

    # 6) Post-policy + disclaimer
        _, note = self.policy.check_output(final)
        if self.policy.disclaimer:
            final = f"{final}\n\n_{self.policy.disclaimer}_"

        # 7) Persist memory
        meta = base_meta(self.cfg.get("version", "0.0.0"))
        if pred_action:
            meta["router_action"] = pred_action
        
        # Deduplication check
        should_save = True
        dedup_cfg = self.cfg.get("memory", {}).get("deduplication", {})
        if dedup_cfg.get("enabled", False):
            # Check if similar exists
            # We search for the exact text. If we find a high match, we skip.
            # We use a high threshold (e.g. 0.9) to only catch duplicates/near-duplicates.
            thresh = float(dedup_cfg.get("threshold", 0.9))
            existing = self.memory.search(qv, top_k=1)
            if existing:
                # existing is list of (score, item)
                top_score, top_item = existing[0]
                if top_score >= thresh:
                    should_save = False
                    print(f"DEBUG: Deduplication - Skipped saving (score={top_score:.2f} >= {thresh}). Matched: '{top_item['text'][:50]}...'")

        if should_save:
            self.memory.add("user", user_text, qv, {"tags": tags, "category": category, **meta})
        self.memory.add("assistant", final, self._embed(final), meta)
        self.memory.add("reflection", critique, self._embed(critique), meta)

        # Governance: TTL + micro-merge
        try:
            from src.governance import apply_ttl, select_for_merge, micro_summarize, golden_summary
            # Pull last N entries to see if we should merge
            recent = self.memory._load()[-50:]
            # Governance: Only run if we are nearing context capacity (e.g. > 80%)
            # We estimate current usage from the last turn's packing
            current_usage = count_tokens(system_out + "\n" + context + "\n" + final)
            capacity = bcfg.C
            
            if current_usage > 0.8 * capacity:
                 # TTL pass (example: 30 days)
                recent = apply_ttl(recent, days=30)
                # Merge pass every ~10 turns (simple trigger)
                if len(recent) % 10 == 0 and len(recent) >= 10:
                    to_merge = select_for_merge(recent, limit=6)
                if to_merge:
                    client = self.llm.client
                    summary = micro_summarize(client, to_merge)
                    # Replace merged items with 1 summary item (we just add a summary record)
                    self.memory.add("summary", summary, self._embed(summary), {"merged": [e.get("ts") for e in to_merge]})
            # Golden summary every 50 turns
            if len(recent) % 50 == 0 and len(recent) >= 20:
                client = self.llm.client
                history_texts = [e.get("text","") for e in recent if e.get("kind") in {"user","assistant","summary"}]
                if history_texts:
                    gsum = golden_summary(client, history_texts[-100:])  # last ~100 items
                    self.memory.add("golden", gsum, self._embed(gsum), {"window": len(history_texts[-100:])})
        except Exception:
            pass

        result = {
            "status": "ok",
            "output": final,
            "note": note,
            "tags": tags,
            "recalls": [f"[{s:.2f}] {it['text']}" for s, it in recalls],
            "rag_recalls": [f"[{s:.2f}] {it['text']}" for s, it in rag_recalls],
            "critique": critique,
        }
    # 8) lightweight run logging
        try:
            retrieval_tokens = count_tokens(context) # Context contains both memory and RAG
            log_event(run_dir, "turn", {
                "latency_s": latency,
                "total_tokens": used_tokens,
                "retrieval_tokens": retrieval_tokens,
                "recalls": result["recalls"],
            })
        except Exception:
            pass
        if debug:
            print("\n--- DEBUG: RECALLS ---")
            for r in result["recalls"]:
                print("-", r)
            print("\n--- DEBUG: RAG RECALLS ---")
            for r in result["rag_recalls"]:
                print("-", r)
            print("\n--- DEBUG: CRITIQUE ---\n", result["critique"])
        return result
