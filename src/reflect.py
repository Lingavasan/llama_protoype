from typing import List, Dict
CRITIC_SYS = "You are a strict reviewer. Point out concrete issues, missing steps, ambiguity, or policy concerns."

def critique_messages(task: str, draft: str) -> List[Dict[str,str]]:
    return [
        {"role": "system", "content": CRITIC_SYS},
        {"role": "user", "content": f"Task:\n{task}\n\nDraft:\n{draft}\n\nList 3-5 critiques as bullets."}
    ]

def rewrite_messages(task: str, draft: str, critique: str) -> List[Dict[str,str]]:
    return [
        {"role": "system", "content": "You are a careful editor. Improve the draft using the critique. Be concise."},
        {"role": "user", "content": f"Task:\n{task}\n\nDraft:\n{draft}\n\nCritique:\n{critique}\n\nRewrite the answer addressing every point."}
    ]
