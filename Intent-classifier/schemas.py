"""
schemas.py
Pydantic models for Module 3 – Intent Classifier ONLY.
RAG response shape is Module 4's responsibility.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ─── Enums ────────────────────────────────────────────────────────────────────

class Intent(str, Enum):
    GREETING             = "greeting"
    GOODBYE              = "goodbye"
    GRATITUDE            = "gratitude"
    ASKING_MENTAL_HEALTH = "asking_mental_health_question"
    OUT_OF_SCOPE         = "out_of_scope"


# ─── Request ──────────────────────────────────────────────────────────────────

class IntentRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The raw user message to classify.",
        examples=["I've been feeling really anxious lately."]
    )
    language_code: Optional[str] = Field(
        default=None,
        description="ISO 639-1 language code from Module 1 (e.g. 'en', 'ar').",
        examples=["en"]
    )
    emotion: Optional[str] = Field(
        default=None,
        description="Emotion label from Module 2 (e.g. 'sadness', 'fear').",
        examples=["sadness"]
    )


# ─── Response ─────────────────────────────────────────────────────────────────

class IntentResponse(BaseModel):
    """
    Returned for all intents EXCEPT asking_mental_health_question.
    For that intent, main.py proxies the request to Module 4 directly.
    """
    intent:          Intent        = Field(..., description="Classified intent.")
    direct_response: str           = Field(..., description="Pre-defined reply from prompts.yaml.")
    language_code:   Optional[str] = Field(default=None)
    emotion:         Optional[str] = Field(default=None)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "intent": "greeting",
                    "direct_response": "Hello! I'm here to support you. How are you feeling today?",
                    "language_code": "en",
                    "emotion": None
                }
            ]
        }
    }


# ─── Health check ─────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:  str = "ok"
    module:  str = "intent_classifier"
    version: str = "1.0.0"