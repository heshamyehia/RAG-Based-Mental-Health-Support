"""
main.py
FastAPI app for Module 3 – Intent Classifier.

Routing logic:
  asking_mental_health_question → proxy request to Module 4 RAG, return its response as-is
  everything else               → return IntentResponse with direct_response from prompts.yaml
"""

import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from schemas import IntentRequest, IntentResponse, HealthResponse, Intent
from intent_classifier import classify_intent, get_direct_response

load_dotenv()

RAG_ENDPOINT = os.getenv("RAG_ENDPOINT", "http://localhost:8001/answer")

app = FastAPI(
    title="Module 3 – Intent Classifier",
    version="1.0.0",
)


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    return HealthResponse()


@app.post("/classify", tags=["Intent"])
async def classify(request: IntentRequest):
    """
    Classify user intent and route accordingly.

    - greeting / goodbye / gratitude / out_of_scope
        → returns IntentResponse (our schema, direct response from prompts.yaml)

    - asking_mental_health_question
        → proxies request to Module 4 RAG endpoint
        → returns Module 4's response as-is (their schema, not ours)
    """
    intent = classify_intent(request.message)

    # ── Mental health → forward to Module 4, return their response as-is ──────
    # if intent == Intent.ASKING_MENTAL_HEALTH:
    #     return await _proxy_to_rag(request)

    # ── All other intents → our direct response ───────────────────────────────
    return IntentResponse(
        intent=intent,
        direct_response=get_direct_response(intent),
        language_code=request.language_code,
        emotion=request.emotion,
    )


# ─── RAG proxy ────────────────────────────────────────────────────────────────

async def _proxy_to_rag(request: IntentRequest) -> JSONResponse:
    """
    Forward the user's message to Module 4 RAG endpoint.
    We pass message + language_code + emotion so RAG can use them for
    language-aware retrieval and emotion-aware response generation.
    We return Module 4's response exactly as received — their schema, not ours.
    """
    payload = {
        "question":      request.message,
        "language_code": request.language_code,
        "emotion":       request.emotion,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(RAG_ENDPOINT, json=payload)
            response.raise_for_status()
            return JSONResponse(content=response.json(), status_code=response.status_code)

    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"Module 4 RAG is unreachable at {RAG_ENDPOINT}. Make sure it is running."
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Module 4 returned an error: {e.response.status_code}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)