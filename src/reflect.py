from typing import List, Dict
CRITIC_SYS = "You are a helpful reviewer. If the draft answers the question using the context, it is good. Do not be pedantic."

def critique_messages(task: str, draft: str, context: str = "") -> List[Dict[str,str]]:
    return [
        {"role": "system", "content": CRITIC_SYS},
        {"role": "user", "content": f"Context:\n{context}\n\nTask:\n{task}\n\nDraft:\n{draft}\n\nCheck if the draft answers the task. If it does, say 'No critique needed.' Only critique if it is harmful or completely wrong."}
    ]

def rewrite_messages(task: str, draft: str, critique: str) -> List[Dict[str,str]]:
    return [
        {"role": "system", "content": "You are a careful editor. Improve the draft using the critique. Be concise. Output ONLY the final answer. Do not include introductory text."},
        {"role": "user", "content": f"Task:\n{task}\n\nDraft:\n{draft}\n\nCritique:\n{critique}\n\nRewrite the answer addressing every point. Output ONLY the rewritten answer."}
    ]
