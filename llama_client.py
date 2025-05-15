import requests, json
from typing import Dict

GROQ_API_KEY = "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"

# Expanded fail‑over list of LLaMA models (in descending order of priority)
MODEL_IDS = [
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-guard-4-12b",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "llama-3.1-8b-instant"
]

def define_word(word: str) -> Dict[str, str]:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = (
        "You are an exact, authoritative Cebuano dictionary API. "
        "return a JSON object with these keys for the Cebuano word "
        f"'{word}':\n"
        "  • part_of_speech (noun, verb, etc.)\n"
        "  • pronunciation (Cebuano phonetics)\n"
        "  • definition (one concise sentence in English)\n"
        "  • example (one Cebuano sentence)\n"
        "  • translation (English equivalent(s))\n"
        "Only output the raw JSON."
    )

    for model_id in MODEL_IDS:
        try:
            data = {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": "You are a precise Cebuano dictionary API."},
                    {"role": "user",   "content": prompt}
                ],
                "temperature": 0.2
            }
            resp = requests.post(url, headers=headers, json=data, timeout=10)
            resp.raise_for_status()

            raw = resp.json()["choices"][0]["message"]["content"].strip()

            # 1) Remove code fences if present
            if raw.startswith("```"):
                parts = raw.split("```")
                if len(parts) >= 3:
                    raw = parts[1].strip()

            # 2) Remove a leading 'json' line if present
            lines = raw.splitlines()
            if lines and lines[0].strip().lower() == "json":
                raw = "\n".join(lines[1:]).strip()

            # 3) Parse JSON
            try:
                return json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"❌ Invalid JSON from {model_id}: {e}. Output was:\n{raw}\nTrying next model…")
                continue

        except requests.HTTPError as http_err:
            if resp.status_code == 429:
                print(f"⚠️ {model_id} hit rate limit, trying next model…")
                continue
            else:
                print(f"❌ HTTP error with {model_id}: {http_err}")
                break

        except Exception as e:
            print(f"❌ Error with {model_id}: {e}, trying next…")
            continue

    print("❌ All models exhausted or errored out")
    return {
        "part_of_speech": "",
        "pronunciation": "",
        "definition": "",
        "example": "",
        "translation": ""
    }
