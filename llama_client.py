import requests, json
from typing import Dict

GROQ_API_KEY = "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"

def define_word(word: str) -> Dict[str, str]:
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        prompt = (
            "Return a JSON object with keys:\n"
            "  part_of_speech, pronunciation, definition, example, translation\n"
            f"for the Cebuano word '{word}'.\n"
            "Definition: one concise sentence. Example: one Cebuano sentence.\n"
            "Only output the raw JSON."
        )
        data = {
            "model": "llama3-70b-8192",
            "messages": [
                {"role": "system", "content": "You are a precise Cebuano dictionary API."},
                {"role": "user",   "content": prompt}
            ],
            "temperature": 0.2
        }
        resp = requests.post(url, headers=headers, json=data)
        resp.raise_for_status()
        return json.loads(resp.json()["choices"][0]["message"]["content"].strip())
    except Exception as e:
        print("❌ Error from Groq:", e)
        return {
            "part_of_speech": "",
            "pronunciation": "",
            "definition": "",
            "example": "",
            "translation": ""
        }
