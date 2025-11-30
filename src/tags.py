import re
from datetime import datetime

RULES = {
    "code": r"\b(python|error|function|class|bug|stack trace)\b",
    "policy": r"\b(policy|safety|guardrail|license)\b",
    "math": r"\b(matrix|equation|integral|theorem)\b",
}

def tag_text(text: str, client=None):
    if not client:
        return []
    
    prompt = [
        {"role": "system", "content": "You are a classifier. Analyze the user message and output a JSON object with two keys: 'category' (one of: plans, to-do list, preferences, schedule, messages, other) and 'tags' (list of keywords like personal, trip, vacation, work, coursework). Output ONLY the JSON."},
        {"role": "user", "content": text}
    ]
    try:
        # Assuming client is an OllamaLLM instance or similar that has a chat method
        # We use a small model for speed if possible, or the main model
        response = client.chat(prompt, num_predict=128)
        # Extract JSON from response (simple heuristic)
        import json
        # Find first { and last }
        start = response.find("{")
        end = response.rfind("}")
        if start != -1 and end != -1:
            data = json.loads(response[start:end+1])
            return data
    except Exception:
        pass
    return {"category": "other", "tags": []}

def base_meta(version: str):
    return {"policy_version": version, "ts": datetime.utcnow().isoformat()}
