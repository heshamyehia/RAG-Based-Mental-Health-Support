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
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

# ─── Project-level schemas ────────────────────────────────────────────────────
from schemas import ChatRequest, ChatResponse, HealthResponse, Intent

# ─── Module 2 – Emotion ──────────────────────────────────────────────────────
from emotion_classifier.predictor import predict_emotion

# ─── Module 3 – Intent ───────────────────────────────────────────────────────
from Intent_classifier.intent_classifier import classify_intent, get_direct_response

# ─── Module 4 – RAG ───────────────────────────────────────────────────────
from module4_rag.rag_pipeline import RAGConfig, RAGPipeline

load_dotenv()

# ─── Module 4 – Pipeline instance ─────────────────────────────────────────
_pipeline: RAGPipeline | None = None

def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(RAGConfig())
    return _pipeline


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

    # ── Step 1: Language detection ────────────────────────────────────────────
    language_code = detect_language(request.message)

    # ── Step 2: Emotion classification ───────────────────────────────────────
    emotion = predict_emotion(request.message)

    # ── Step 3: Intent classification ────────────────────────────────────────
    intent = classify_intent(request.message)

    # ── Step 4: Route ─────────────────────────────────────────────────────────
    if intent == Intent.ASKING_MENTAL_HEALTH:
        try:
            pipeline = get_pipeline()
            result = pipeline.answer(request.message)
            return ChatResponse(
                language_code=language_code,
                emotion=emotion,
                intent=intent,
                response=result["answer"],
                response_source="rag",
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        
    # ── Non-RAG intents: direct response from prompts.yaml ───────────────────
    direct_response = get_direct_response(intent, emotion, language_code)

    return ChatResponse(
        language_code=language_code,
        emotion=emotion,
        intent=intent,
        response=direct_response,
        response_source="direct",
    )


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)