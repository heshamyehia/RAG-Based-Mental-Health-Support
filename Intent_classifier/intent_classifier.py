"""
intent_classifier.py
Core classification logic for Module 3.
Loads all prompts and config from prompts.yaml.
"""

import os
import yaml
from pathlib import Path
from google import genai
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
DR_MODEL_NAME           = _CFG["direct_response_model"]["name"]
DR_MAX_TOKENS           = _CFG["direct_response_model"]["max_tokens"]
DR_TEMPERATURE          = _CFG["direct_response_model"]["temperature"]

# ─── Gemini client ────────────────────────────────────────────────────────────

_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))

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
    # Build few-shot context from examples
    few_shot_text = "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}"
        for msg in FEW_SHOT_EXAMPLES
    )

    prompt = f"{few_shot_text}\nUser: {user_message}"

    try:
        response = _client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config={
                "system_instruction": SYSTEM_PROMPT,
                "max_output_tokens":  MAX_TOKENS,
                "temperature":        TEMPERATURE,
                "thinking_config":   {"thinking_budget": 0},
            },
        )

        raw_label = response.text.strip().lower()

        # Exact match first
        valid = [i.value for i in Intent]
        if raw_label in valid:
            return Intent(raw_label)

        # Fuzzy fallback
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
    Generate a context-aware direct response for non-RAG intents using Gemini.

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
        response = _client.models.generate_content(
            model=DR_MODEL_NAME,
            contents=prompt,
            config={
                "max_output_tokens": DR_MAX_TOKENS,
                "temperature":       DR_TEMPERATURE,
                "thinking_config":   {"thinking_budget": 0},
            },
        )
        return response.text.strip()

    except Exception as e:
        print(f"[IntentClassifier] Direct response error: {e}")
        return "I'm here for you. Feel free to share how you're feeling."