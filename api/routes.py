import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

import feedback_manager
import history_manager
from emotion_classifier.predictor import predict_emotion
from Intent_classifier.intent_classifier import classify_intent, get_direct_response
from language_detector.language_detector import detect_language
from module4_rag.rag_pipeline import RAGPipeline
from schemas import (
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    Intent,
)

import monitoring.telemetry as tel

from .dependencies import get_pipeline

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    logger.info("Health check requested")
    return HealthResponse()

@router.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
async def submit_feedback(request: FeedbackRequest):
    logger.info(
        "Feedback submitted vote=%s session_id=%s",
        request.vote,
        request.session_id,
    )
    try:
        record = await run_in_threadpool(feedback_manager.append_feedback, request.model_dump())
    except Exception as exc:
        logger.exception("Feedback recording failed")
        raise HTTPException(status_code=500, detail={
            "error_code": "FEEDBACK_ERROR",
            "message": "An internal server error occurred."
        }) from exc
    return FeedbackResponse(
        vote=record["vote"],
        recorded_at=record["recorded_at"],
    )

@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest, pipeline: RAGPipeline = Depends(get_pipeline)):
    """
    Full pipeline endpoint.

    1. Detect language  (Module 1)
    2. Classify emotion (Module 2)
    3. Classify intent  (Module 3)
    4. Route to direct response or RAG (Module 4)
    """

    session_id = request.session_id or history_manager.generate_session_id()
    logger.info("Chat request received session_id=%s", session_id)

    # Metric 3 (server): count every request
    tel.record_request()
    # Metric 2 (data): message length distribution
    tel.record_message_length(len(request.message))

    chat_history = history_manager.get_history(session_id)

    language_task = run_in_threadpool(detect_language, request.message)
    emotion_task = run_in_threadpool(predict_emotion, request.message)
    intent_task = run_in_threadpool(classify_intent, request.message)

    try:
        language_code, emotion, intent = await asyncio.gather(
            language_task, emotion_task, intent_task
        )
        logger.info(
            "Chat classification completed session_id=%s language=%s emotion=%s intent=%s",
            session_id,
            language_code,
            emotion,
            intent,
        )
    except Exception as exc:
        logger.exception("Pipeline classification failed")
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": "CLASSIFICATION_ERROR",
                "message": "An internal server error occurred.",
            },
        ) from exc

    # Metric 1 (model/NLP): intent distribution
    tel.record_intent(intent.value)

    raw_intent = intent
    if intent == Intent.CLASSIFICATION_ERROR:
        logger.warning(
            "Intent classification failed, routing session_id=%s to RAG as a safe default",
            session_id,
        )
        intent = Intent.ASKING_MENTAL_HEALTH

    if intent == Intent.ASKING_MENTAL_HEALTH:
        try:
            logger.info("Routing to RAG pipeline session_id=%s", session_id)
            result = await run_in_threadpool(
                pipeline.answer,
                request.message,
                emotion=emotion,
                language_code=language_code,
                chat_history=chat_history,
            )
            history_manager.append_history(
                session_id,
                request.message,
                result["answer"],
            )
            return ChatResponse(
                session_id=session_id,
                language_code=language_code,
                emotion=emotion,
                intent=intent,
                raw_intent=raw_intent,
                response=result["answer"],
                response_source="rag",
            )
        except Exception as exc:
            logger.exception("RAG pipeline failed")
            tel.record_error("RAG_PIPELINE_ERROR")
            raise HTTPException(
                status_code=500,
                detail={
                    "error_code": "RAG_PIPELINE_ERROR",
                    "message": "An internal server error occurred.",
                },
            ) from exc

    try:
        logger.info("Routing to direct response session_id=%s", session_id)
        direct_response = await run_in_threadpool(
            get_direct_response,
            intent,
            emotion,
            language_code,
        )
        history_manager.append_history(session_id, request.message, direct_response)
    except Exception as exc:
        logger.exception("Direct response generation failed")
        tel.record_error("DIRECT_RESPONSE_ERROR")
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": "DIRECT_RESPONSE_ERROR",
                "message": "An internal server error occurred.",
            },
        ) from exc

    return ChatResponse(
        session_id=session_id,
        language_code=language_code,
        emotion=emotion,
        intent=intent,
        raw_intent=raw_intent,
        response=direct_response,
        response_source="direct",
    )
