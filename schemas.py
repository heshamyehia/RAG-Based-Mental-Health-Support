"""
schemas.py  (project root)
Unified Pydantic models for the /chat endpoint.
Each module's output is carried forward through the pipeline.
"""

from enum import Enum
from typing import Optional, Literal
from pydantic import BaseModel, Field


# ─── Enums ────────────────────────────────────────────────────────────────────

class Intent(str, Enum):
    GREETING             = "greeting"
    GOODBYE              = "goodbye"
    GRATITUDE            = "gratitude"
    ASKING_MENTAL_HEALTH = "asking_mental_health_question"
    OUT_OF_SCOPE         = "out_of_scope"
    CLASSIFICATION_ERROR = "classification_error"


class Emotion(str, Enum):
    JOY      = "joy"
    SADNESS  = "sadness"
    ANGER    = "anger"
    FEAR     = "fear"
    LOVE     = "love"
    SURPRISE = "surprise"
    NEUTRAL  = "neutral"
    UNKNOWN  = "unknown"


# ─── /chat  Request ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(
        None,
        description="Optional unique ID for the chat session. If omitted, the server creates one.",
        examples=["session_12345"]
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Raw user message.",
        examples=["I've been feeling really anxious lately."]
    )


# ─── /feedback Request ───────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    session_id: Optional[str] = Field(
        None,
        description="Optional chat session ID associated with the feedback.",
        examples=["session_12345"]
    )
    vote: Literal["thumbs_up", "thumbs_down"] = Field(
        ...,
        description="Feedback vote for the assistant response.",
        examples=["thumbs_up", "thumbs_down"]
    )
    comment: Optional[str] = Field(
        None,
        max_length=1000,
        description="Optional free-text feedback comment."
    )


# ─── /chat  Response ──────────────────────────────────────────────────────────

class ChatResponse(BaseModel):
    """
    Unified response returned by the /chat endpoint.
    Carries the output of every module so the front-end (or tests) can inspect
    the full pipeline trace.
    """
    # Session
    session_id: str = Field(..., description="Chat session ID for continuity and feedback tracking.")

    # Module 1
    language_code: str = Field(..., description="ISO 639-1 code detected by Module 1.")

    # Module 2
    emotion: Emotion = Field(..., description="Emotion label detected by Module 2.")

    # Module 3
    intent: Intent = Field(..., description="Intent used for routing (post-fallback).")
    raw_intent: Optional[Intent] = Field(None, description="Original classifier output before any fallback override.")
    
    # Final answer — either a direct response (M3) or a RAG answer (M4)
    response: str = Field(..., description="The chatbot's reply to the user.")

    # Source tag so the client knows which module produced the answer
    response_source: Literal["rag", "direct"] = Field(
        ...,
        description="'direct' for non-RAG intents, 'rag' for mental-health questions.",
        examples=["direct", "rag"]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "language_code": "en",
                    "emotion": "sadness",
                    "intent": "asking_mental_health_question",
                    "response": "It sounds like you're going through a hard time …",
                    "response_source": "rag"
                }
            ]
        }
    }


# ─── /feedback Response ─────────────────────────────────────────────────────

class FeedbackResponse(BaseModel):
    status: str = "ok"
    vote: Literal["thumbs_up", "thumbs_down"]
    recorded_at: str


# ─── Health check ─────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:  str = "ok"
    module:  str = "chat_pipeline"
    version: str = "1.0.0"