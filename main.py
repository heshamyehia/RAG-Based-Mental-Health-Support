"""
main.py  (project root)
FastAPI app — unified /chat endpoint.

Pipeline per request:
  1. Module 1: Language Detection   (TF-IDF + LinearSVM, loaded from language_detector/)
  2. Module 2: Emotion Classifier   (DistilBERT, loaded from emotion_classifier/)
  3. Module 3: Intent Classifier    (Groq few-shot, loaded from intent_classifier/)
  4. Routing:
       asking_mental_health_question → Module 4 RAG  (proxy, enable when M4 is ready)
       everything else               → direct response from intent_classifier/prompts.yaml

Project layout expected:
  project_root/
  ├── main.py                         ← this file
  ├── schemas.py
  ├── .env
  ├── language_detector/
  │   ├── language_detector.joblib
  │   └── language_detector_meta.joblib
  ├── emotion_classifier/
  │   ├── __init__.py
  │   ├── predictor.py
  │   └── final_emotion_model/
  └── intent_classifier/
      ├── __init__.py
      ├── intent_classifier.py
      └── prompts.yaml
"""

import os
import warnings
import joblib
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# ─── Project-level schemas ────────────────────────────────────────────────────
from schemas import ChatRequest, ChatResponse, HealthResponse, Intent

# ─── Module 2 – Emotion ──────────────────────────────────────────────────────
from emotion_classifier.predictor import predict_emotion

# ─── Module 3 – Intent ───────────────────────────────────────────────────────
from Intent_classifier.intent_classifier import classify_intent, get_direct_response

load_dotenv()

# ─── Module 1 – Language Detection  ────────────────────
_BASE_DIR = os.path.dirname(__file__)
_model = joblib.load(os.path.join(_BASE_DIR, "language_detector", "language_detector.joblib"))
_meta  = joblib.load(os.path.join(_BASE_DIR, "language_detector", "language_detector_meta.joblib"))

RAG_ENDPOINT = os.getenv("RAG_ENDPOINT", "http://localhost:8001/answer")

app = FastAPI(
    title="Mental Health Support Chatbot",
    version="1.0.0",
    description="RAG-based mental health chatbot — NLP Final Task 2026.",
)


# ─── Module 1 ─────────────────────────────────────────────────────────────────
def detect_language(text: str) -> str:
    """
    Module 1 — TF-IDF (char n-grams) + LinearSVM language detector.
    Returns an ISO 639-1 code e.g. 'en', 'ar', 'fr'.
    Falls back to 'en' if confidence < 0.75 or text is too short.
    Supports 20 languages: ar, bg, de, el, en, es, fr, hi, it, ja,
                           nl, pl, pt, ru, sw, th, tr, ur, vi, zh.
    """
    if not text or len(text.strip()) < 3:
        return "en"
    try:
        proba      = _model.predict_proba([text])[0]
        confidence = proba.max()
        lang       = _meta["classes"][proba.argmax()]
        return lang if confidence >= 0.75 else "en"
    except Exception:
        return "en"


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    return HealthResponse()


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Full pipeline endpoint.

    1. Detect language  (Module 1)
    2. Classify emotion (Module 2)
    3. Classify intent  (Module 3)
    4. Route to direct response or RAG (Module 4)
    """

    # ── Step 1: Language detection ────────────────────────────────────────────
    language_code = detect_language(request.message)

    # ── Step 2: Emotion classification ───────────────────────────────────────
    emotion = predict_emotion(request.message)

    # ── Step 3: Intent classification ────────────────────────────────────────
    intent = classify_intent(request.message)

    # ── Step 4: Route ─────────────────────────────────────────────────────────
    if intent == Intent.ASKING_MENTAL_HEALTH:
        # ── Module 4 RAG (uncomment when Module 4 is ready) ──────────────────
        # return await _proxy_to_rag(request.message, language_code, emotion)

        # Placeholder until Module 4 is built
        return ChatResponse(
            language_code=language_code,
            emotion=emotion,
            intent=intent,
            response=(
                "I hear you. That sounds really difficult. "
                "(RAG module coming soon — this is a placeholder response.)"
            ),
            response_source="rag_placeholder",
        )

    # ── Non-RAG intents: direct response from prompts.yaml ───────────────────
    direct_response = get_direct_response(intent, emotion, language_code)

    return ChatResponse(
        language_code=language_code,
        emotion=emotion,
        intent=intent,
        response=direct_response,
        response_source="direct",
    )


# ─── RAG proxy ────────────────────────────────────────────────────────────────

async def _proxy_to_rag(message: str, language_code: str, emotion: str) -> ChatResponse:
    """
    Forward the request to Module 4 RAG and wrap its reply in a ChatResponse.
    Uncomment the call above when Module 4 is ready.
    """
    payload = {
        "question":      message,
        "language_code": language_code,
        "emotion":       emotion,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(RAG_ENDPOINT, json=payload)
            resp.raise_for_status()
            rag_data = resp.json()

        return ChatResponse(
            language_code=language_code,
            emotion=emotion,
            intent=Intent.ASKING_MENTAL_HEALTH,
            response=rag_data.get("answer", ""),
            response_source="rag",
        )

    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"Module 4 RAG is unreachable at {RAG_ENDPOINT}.",
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Module 4 returned an error: {e.response.status_code}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)