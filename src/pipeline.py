from typing import Dict
import numpy as np
from src.config import load_config
from src.llm import OllamaLLM
from src.memory import JsonlMemory
from src.policy import Policy
from src.rag import HNSWRAG
from src.tags import tag_text, base_meta
from src.reflect import critique_messages, rewrite_messages
from src.router import MemGPTRouter

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
        if self.rag_enabled:
            idx = ext["index"]
            self.rag = HNSWRAG(
                index_path=idx["index_path"],
                meta_path=idx["meta_path"],
                info_path=idx["info_path"]
            )
            self.rag_top_k = int(ext.get("top_k", 3))
            self.rag_embed_model = ext.get("embed_model", self.cfg["memory"]["embed_model"])

        # Load router 
        r_cfg = self.cfg.get("router", {})
        self.router = None
        if bool(r_cfg.get("enabled", False)):
            try:
                self.router = MemGPTRouter.load(r_cfg["bundle_path"])
            except Exception:
                self.router = None  

    # Initialize memory and policy
        self.memory = JsonlMemory("memory.jsonl")
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


    def run(self, user_text: str, debug: bool=False) -> Dict: #the main pipeline!
        # 1) Policy + tags
        ok, why = self.policy.check_input(user_text)
        if not ok:
            return {"status":"blocked", "reason":why}
        tags = tag_text(user_text)

        pred_action = self.router.predict(user_text) if self.router else None


        # 2) Memory search
        qv = self._embed(user_text)
        recalls = self.memory.search(qv, top_k=int(self.cfg["memory"]["top_k"]))
        recall_text = "\n".join([f"- {it['text']}" for _, it in recalls]) or "None."
        
        rag_recalls = []
        if self.rag_enabled:
            qv_rag = self._embed_with_model(user_text, self.rag_embed_model)
            rag_recalls = self.rag.search(qv_rag, k=self.rag_top_k)

        
        all_recalls = recalls + rag_recalls
        recall_text = "\n".join([
        f"- ({it.get('kind','conv')}) {(it.get('title')+': ') if it.get('title') else ''}{it['text']}"
            for _, it in all_recalls
        ]) or "None."
        
        router_hint = f"\nPolicy router suggests action: {pred_action}" if pred_action else ""
        router_hint = f"\nPolicy router suggests action: {pred_action}" if pred_action else ""
        system = self.cfg.get("system", "You are a helpful assistant.")
        draft = self.llm.chat([
            {"role": "system", "content": system + router_hint + f"\nRelevant memory:\n{recall_text}"},
            {"role": "user",   "content": user_text}
        ])


    # 3) Draft
    # (already handled above)

        # 4) Reflection (critique -> rewrite)
        critique = self.llm.chat(critique_messages(user_text, draft), num_predict=256)
        final = self.llm.chat(rewrite_messages(user_text, draft, critique), num_predict=256)

        # 5) Post-policy + disclaimer
        _, note = self.policy.check_output(final)
        if self.policy.disclaimer:
            final = f"{final}\n\n_{self.policy.disclaimer}_"

        # 6) Persist memory
        meta = base_meta(self.cfg.get("version","0.0.0"))
        if pred_action:
            meta["router_action"] = pred_action
        self.memory.add("user", user_text, qv, {"tags": tags, **meta})
        self.memory.add("assistant", final, self._embed(final), meta)
        self.memory.add("reflection", critique, self._embed(critique), meta)

        result = {
            "status": "ok",
            "output": final,
            "note": note,
            "tags": tags,
            "recalls": [it["text"] for _, it in recalls],
            "critique": critique
        }
        if debug:
            print("\n--- DEBUG: RECALLS ---")
            for r in result["recalls"]:
                print("-", r)
            print("\n--- DEBUG: CRITIQUE ---\n", result["critique"])
        return result
