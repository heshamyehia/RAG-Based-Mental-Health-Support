"""
intent_classifier.py
Core classification logic for Module 3.
Loads all prompts and config from prompts.yaml.
Uses Groq (llama-3.3-70b-versatile) for high-throughput, low-latency inference.
"""

import logging
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from groq import Groq

sys.path.insert(0, str(Path(__file__).parent.parent))  # project root on path
from schemas import Emotion, Intent

load_dotenv()

logger = logging.getLogger(__name__)

# ─── Load prompts.yaml ────────────────────────────────────────────────────────

_PROMPTS_PATH = Path(__file__).parent / "prompts.yaml"

with open(_PROMPTS_PATH, "r", encoding="utf-8") as f:
    _CFG = yaml.safe_load(f)["intent_classifier"]

SYSTEM_PROMPT          = _CFG["system_prompt"].strip()
FEW_SHOT_EXAMPLES      = _CFG["few_shot_examples"]
DIRECT_RESPONSE_PROMPT = _CFG["direct_response_prompt"].strip()
MODEL_NAME             = _CFG["model"]["name"]
MAX_TOKENS             = _CFG["model"]["max_tokens"]
TEMPERATURE            = _CFG["model"]["temperature"]
STOP_SEQUENCES         = _CFG["model"]["stop_sequences"]
DR_MODEL_NAME          = _CFG["direct_response_model"]["name"]
DR_MAX_TOKENS          = _CFG["direct_response_model"]["max_tokens"]
DR_TEMPERATURE         = _CFG["direct_response_model"]["temperature"]

# ─── Groq client ─────────────────────────────────────────────────────────────

_client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))

# ─── Intent Classifier ────────────────────────────────────────────────────────


def classify_intent(user_message: str) -> Intent:
    """
    Classify the intent of a user message using few-shot LLM prompting.

    Returns:
        An Intent enum value.
        Falls back to Intent.CLASSIFICATION_ERROR on API errors.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *FEW_SHOT_EXAMPLES,
        {"role": "user", "content": user_message},
    ]

    try:
        response = _client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            stop=STOP_SEQUENCES,
        )

        raw_label = response.choices[0].message.content.strip().lower()

        valid = [i.value for i in Intent]
        if raw_label in valid:
            return Intent(raw_label)

        # Fuzzy fallback — handle model adding punctuation / extra words
        for intent_value in valid:
            if intent_value in raw_label:
                return Intent(intent_value)

        return Intent.OUT_OF_SCOPE

    except Exception:
        logger.exception("[IntentClassifier] Groq API error")
        return Intent.CLASSIFICATION_ERROR


# ─── Direct Response ──────────────────────────────────────────────────────────


def get_direct_response(intent: Intent, emotion: Emotion, language_code: str) -> str:
    """
    Generate a context-aware direct response for non-RAG intents using Groq.
    Takes emotion and language into account for a more personalised reply.

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

    except Exception:
        logger.exception("[IntentClassifier] Direct response generation failed")
        return "I'm here for you. Feel free to share how you're feeling."
