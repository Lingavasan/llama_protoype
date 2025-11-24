from typing import List, Dict
from datetime import datetime, timedelta
from ollama import Client


def micro_summarize(client: Client, entries: List[Dict]) -> str:
    # Compact a set of short memory items into a micro-summary
    text = "\n".join([e.get("text", "") for e in entries])
    prompt = [
      {"role":"system","content":"Summarize the following notes into a concise micro-summary."},
      {"role":"user","content":text}
    ]
    r = client.chat(model="llama3.2:1b", messages=prompt, stream=False)
    return r.message.content.strip()


def golden_summary(client: Client, history: List[str]) -> str:
    text = "\n".join(history)
    prompt = [
      {"role":"system","content":"Create a compact 'golden summary' that preserves key facts and persona; be factual."},
      {"role":"user","content":text}
    ]
    r = client.chat(model="llama3.2:1b", messages=prompt, stream=False)
    return r.message.content.strip()


def apply_ttl(entries: List[Dict], days:int=14) -> List[Dict]:
    cutoff = datetime.utcnow() - timedelta(days=days)
    keep = []
    for e in entries:
        ts = datetime.fromisoformat(e.get("ts", "").replace("Z",""))
        if ts >= cutoff:
            keep.append(e)
    return keep


def select_for_merge(entries: List[Dict], limit:int=6) -> List[Dict]:
    """Pick older, lower-importance entries to merge.
    Importance heuristic: longer texts are more important.
    Score is inversely proportional to text length.
    We sort by (importance_score ASC, timestamp ASC)."""
    scored = []
    for e in entries:
        # Skip seed (pre-ingested) entries from merge
        if e.get('meta', {}).get('seed') or e.get('seed'):  # handle both meta nesting and flat
            continue
        
        text = e.get("text", "")
        # Score is inversely proportional to length; add 1 to avoid division by zero
        score = 1.0 / (len(text) + 1)
        
        scored.append((score, e.get("ts",""), e))

    scored.sort(key=lambda x: (x[0], x[1]))
    return [e for _,_,e in scored[:limit]]
