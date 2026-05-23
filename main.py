"""
main.py  (project root)
FastAPI app — unified /chat endpoint.

Pipeline per request:
  1. Module 1 – Language Detection   (stub → replace with real model later)
  2. Module 2 – Emotion Classifier   (DistilBERT, loaded from emotion_classifier/)
  3. Module 3 – Intent Classifier    (Groq few-shot, loaded from intent_classifier/)
  4. Routing:
       asking_mental_health_question → Module 4 RAG  (proxy, enable when M4 is ready)
       everything else               → direct response from intent_classifier/prompts.yaml

Project layout expected:
  project_root/
  ├── main.py                         ← this file
  ├── schemas.py
  ├── .env
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
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# ─── Project-level schemas ────────────────────────────────────────────────────
from schemas import ChatRequest, ChatResponse, HealthResponse, Intent

# ─── Module 2 – Emotion ──────────────────────────────────────────────────────
from emotion_classifier.predictor import predict_emotion

# ─── Module 3 – Intent ───────────────────────────────────────────────────────
from Intent_classifier.intent_classifier import classify_intent, get_direct_response
import history_manager

load_dotenv()

RAG_ENDPOINT = os.getenv("RAG_ENDPOINT", "http://localhost:8001/answer")

app = FastAPI(
    title="Mental Health Support Chatbot",
    version="1.0.0",
    description="RAG-based mental health chatbot — NLP Final Task 2026.",
)


# ─── Module 1 stub ────────────────────────────────────────────────────────────
# TODO: replace with your trained language-detection model (TF-IDF + ML).
def detect_language(text: str) -> str:
    """
    Stub for Module 1.
    Returns 'en' until the real model is integrated.
    Replace this function body with your trained pipeline.
    """
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

    # ── Step 0: History retrieval ─────────────────────────────────────────────
    chat_history = history_manager.get_history(request.session_id)

    # ── Step 1: Language detection ────────────────────────────────────────────
    language_code = detect_language(request.message)

    # ── Step 2: Emotion classification ───────────────────────────────────────
    emotion = predict_emotion(request.message)

    # ── Step 3: Intent classification ────────────────────────────────────────
    intent = classify_intent(request.message)

    # ── Step 4: Route ─────────────────────────────────────────────────────────
    if intent == Intent.ASKING_MENTAL_HEALTH:
        # ── Module 4 RAG (uncomment when Module 4 is ready) ──────────────────
        # response_obj = await _proxy_to_rag(request.message, language_code, emotion, chat_history)
        # history_manager.append_history(request.session_id, request.message, response_obj.response)
        # return response_obj

        # Placeholder until Module 4 is built
        placeholder_response = (
            "I hear you. That sounds really difficult. "
            "(RAG module coming soon — this is a placeholder response.)"
        )
        history_manager.append_history(request.session_id, request.message, placeholder_response)
        
        return ChatResponse(
            language_code=language_code,
            emotion=emotion,
            intent=intent,
            response=placeholder_response,
            response_source="rag_placeholder",
        )

    # ── Non-RAG intents: direct response from prompts.yaml ───────────────────
    direct_response = get_direct_response(intent, emotion, language_code)
    history_manager.append_history(request.session_id, request.message, direct_response)

    return ChatResponse(
        language_code=language_code,
        emotion=emotion,
        intent=intent,
        response=direct_response,
        response_source="direct",
    )


# ─── RAG proxy ────────────────────────────────────────────────────────────────

async def _proxy_to_rag(message: str, language_code: str, emotion: str, chat_history: list) -> ChatResponse:
    """
    Forward the request to Module 4 RAG and wrap its reply in a ChatResponse.
    Uncomment the call above when Module 4 is ready.
    """
    payload = {
        "question":      message,
        "language_code": language_code,
        "emotion":       emotion,
        "chat_history":  chat_history,
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