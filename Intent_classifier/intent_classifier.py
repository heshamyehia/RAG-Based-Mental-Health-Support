"""
intent_classifier.py
Core classification logic for Module 3.
Loads all prompts and config from prompts.yaml.
"""

import os
import yaml
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

from schemas import Intent

load_dotenv()

# ─── Load prompts.yaml ────────────────────────────────────────────────────────

_PROMPTS_PATH = Path(__file__).parent / "prompts.yaml"

with open(_PROMPTS_PATH, "r", encoding="utf-8") as f:
    _CFG = yaml.safe_load(f)["intent_classifier"]

SYSTEM_PROMPT      = _CFG["system_prompt"].strip()
FEW_SHOT_EXAMPLES  = _CFG["few_shot_examples"]       # list of {role, content} dicts
DIRECT_RESPONSES   = _CFG["direct_responses"]         # dict intent → reply string
MODEL_NAME         = _CFG["model"]["name"]
MAX_TOKENS         = _CFG["model"]["max_tokens"]
TEMPERATURE        = _CFG["model"]["temperature"]
STOP_SEQUENCES     = _CFG["model"]["stop_sequences"]

# ─── Groq client ─────────────────────────────────────────────────────────────

_client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))

# ─── Classifier ───────────────────────────────────────────────────────────────

def classify_intent(user_message: str) -> Intent:
    """
    Classify the intent of a user message using few-shot LLM prompting.

    Args:
        user_message: Raw text from the user.

    Returns:
        An Intent enum value.
        Falls back to Intent.OUT_OF_SCOPE on API errors or unexpected output.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *FEW_SHOT_EXAMPLES,
        {"role": "user", "content": user_message}
    ]

    try:
        response = _client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            stop=STOP_SEQUENCES
        )

        raw_label = response.choices[0].message.content.strip().lower()

        # Exact match first
        valid = [i.value for i in Intent]
        if raw_label in valid:
            return Intent(raw_label)

        # Fuzzy fallback — handle model adding punctuation / extra words
        for intent_value in valid:
            if intent_value in raw_label:
                return Intent(intent_value)

        return Intent.OUT_OF_SCOPE

    except Exception as e:
        print(f"[IntentClassifier] API error: {e}")
        return Intent.OUT_OF_SCOPE


def get_direct_response(intent: Intent) -> str:
    """
    Return the pre-defined direct response for non-RAG intents.
    Returns None for asking_mental_health_question (handled by RAG).
    """
    return DIRECT_RESPONSES.get(intent.value, "")
