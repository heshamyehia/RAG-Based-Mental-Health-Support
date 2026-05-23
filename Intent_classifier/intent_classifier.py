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

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))  # project root on path
from schemas import Intent, Emotion

load_dotenv()

# ─── Load prompts.yaml ────────────────────────────────────────────────────────

_PROMPTS_PATH = Path(__file__).parent / "prompts.yaml"

with open(_PROMPTS_PATH, "r", encoding="utf-8") as f:
    _CFG = yaml.safe_load(f)["intent_classifier"]

SYSTEM_PROMPT           = _CFG["system_prompt"].strip()
FEW_SHOT_EXAMPLES       = _CFG["few_shot_examples"]
DIRECT_RESPONSE_PROMPT  = _CFG["direct_response_prompt"].strip()
MODEL_NAME              = _CFG["model"]["name"]
MAX_TOKENS              = _CFG["model"]["max_tokens"]
TEMPERATURE             = _CFG["model"]["temperature"]
STOP_SEQUENCES          = _CFG["model"]["stop_sequences"]
DR_MODEL_NAME           = _CFG["direct_response_model"]["name"]
DR_MAX_TOKENS           = _CFG["direct_response_model"]["max_tokens"]
DR_TEMPERATURE          = _CFG["direct_response_model"]["temperature"]

# ─── Groq client ─────────────────────────────────────────────────────────────

_client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))

# ─── Intent Classifier ────────────────────────────────────────────────────────

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


# ─── Direct Response ──────────────────────────────────────────────────────────

def get_direct_response(intent: Intent, emotion: Emotion, language_code: str) -> str:
    """
    Generate a context-aware direct response for non-RAG intents using Groq.
    Takes emotion and language into account for a more personalised reply.

    Args:
        intent:        Classified intent (never asking_mental_health_question here).
        emotion:       Detected emotion from Module 2.
        language_code: Detected language from Module 1 (ISO 639-1, e.g. 'en', 'ar').

    Returns:
        A short, empathetic response string.
        Falls back to a generic string on API errors.
    """
    prompt = DIRECT_RESPONSE_PROMPT.format(
        intent=intent.value,
        emotion=emotion.value,
        language=language_code,
    )

    try:
        response = _client.chat.completions.create(
            model=DR_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=DR_MAX_TOKENS,
            temperature=DR_TEMPERATURE,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[IntentClassifier] Direct response error: {e}")
        return "I'm here for you. Feel free to share how you're feeling."