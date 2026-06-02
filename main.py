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
  ├── Intent_classifier/
  │   ├── __init__.py
  │   ├── intent_classifier.py
  │   └── prompts.yaml
  └── module4_rag/
      ├── __init__.py
      ├── rag_pipeline.py
      └── config.yaml
"""

import pathlib
import warnings
import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ─── Project-level schemas ────────────────────────────────────────────────────
from schemas import ChatRequest, ChatResponse, HealthResponse, Intent

# ─── Module 1 – Language Detection ───────────────────────────────────────────
from language_detector.language_detector import detect_language

# ─── Module 2 – Emotion ──────────────────────────────────────────────────────
from emotion_classifier.predictor import predict_emotion

# ─── Module 3 – Intent ───────────────────────────────────────────────────────
from Intent_classifier.intent_classifier import classify_intent, get_direct_response

# ─── Module 4 – RAG ───────────────────────────────────────────────────────
from module4_rag.rag_pipeline import RAGPipeline

BASE_DIR = pathlib.Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "module4_rag" / "config.yaml"

# ─── Module 4 – Pipeline instance ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.pipeline = RAGPipeline(CONFIG_PATH)
    logger.info("RAG pipeline initialized")

    yield

    # Shutdown (optional cleanup)
    logger.info("Shutting down application")

app = FastAPI(
    title="Mental Health Support Chatbot",
    version="1.0.0",
    description="RAG-based mental health chatbot — NLP Final Task 2026.",
    lifespan=lifespan,
)

def get_pipeline() -> RAGPipeline:
    pipeline = getattr(app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    return pipeline


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    return HealthResponse()


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(chat_request: ChatRequest):
    """
    Full pipeline endpoint.

    1. Detect language  (Module 1)
    2. Classify emotion (Module 2)
    3. Classify intent  (Module 3)
    4. Route to direct response or RAG (Module 4)
    """

    # ── Step 1: Language detection ────────────────────────────────────────────
    language_task = run_in_threadpool(detect_language, chat_request.message)

    # ── Step 2: Emotion classification ───────────────────────────────────────
    emotion_task = run_in_threadpool(predict_emotion, chat_request.message)

    # ── Step 3: Intent classification ────────────────────────────────────────
    intent_task = run_in_threadpool(classify_intent, chat_request.message)

    try:
        language_code, emotion, intent = await asyncio.gather(
            language_task, emotion_task, intent_task
        )
    except Exception as exc:
        logger.exception("Pipeline classification failed")
        raise HTTPException(status_code=500, detail={
            "error_code": "CLASSIFICATION_ERROR",
            "message": "An internal server error occurred."
        }) from exc
    
    # ── Step 4: Route ─────────────────────────────────────────────────────────
    if intent == Intent.ASKING_MENTAL_HEALTH:
        pipeline = get_pipeline()
        try:            
            result = await run_in_threadpool(
                pipeline.answer,
                chat_request.message,
                emotion=emotion,
                language_code=language_code,
            )
            return ChatResponse(
                language_code=language_code,
                emotion=emotion,
                intent=intent,
                response=result["answer"],
                response_source="rag",
            )

        except Exception as exc:
            logger.exception("RAG pipeline failed")
            raise HTTPException(
                status_code=500,
                detail={
                    "error_code": "RAG_PIPELINE_ERROR",
                    "message": "An internal server error occurred."
                }
            ) from exc
        
    # ── Non-RAG intents: direct response from prompts.yaml ───────────────────
    try:
        direct_response = await run_in_threadpool(get_direct_response, intent, emotion, language_code)
    except Exception as exc:
        logger.exception("Direct response generation failed")
        raise HTTPException(status_code=500, detail={
            "error_code": "DIRECT_RESPONSE_ERROR",
            "message": "An internal server error occurred."
        }) from exc

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