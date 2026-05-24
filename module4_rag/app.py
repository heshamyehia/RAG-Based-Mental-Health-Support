"""
app.py
======
FastAPI deployment for the RAG-Based Mental Health Support Chatbot – Module 4.

Endpoints
---------
  GET  /health       – liveness check
  POST /chat         – main chat endpoint (RAG)
  POST /build-index  – (re)build the Qdrant index from the dataset
  GET  /docs         – auto-generated Swagger UI

Run
---
  uvicorn app:app --reload

Environment variables (see .env):
  GEMINI_API_KEY, QDRANT_URL, QDRANT_API_KEY
"""

import os
import time
import traceback

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from rag_pipeline import RAGConfig, RAGPipeline

# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

load_dotenv()

app = FastAPI(
    title="Mental Health RAG Chatbot",
    description="RAG-based mental health support chatbot – Module 4",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        print("[app] Initialising RAG pipeline …")
        _pipeline = RAGPipeline(RAGConfig())
    return _pipeline


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    query: str
    answer: str
    sources: list
    latency_ms: int

class BuildIndexRequest(BaseModel):
    max_samples: int | None = None
    recreate: bool = False

class BuildIndexResponse(BaseModel):
    status: str
    message: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness check."""
    return {"status": "ok", "service": "Mental Health RAG Chatbot – Module 4"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Main RAG chat endpoint.
    Send a mental health question and get a grounded, empathetic response.
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="'query' must not be empty.")

    try:
        pipeline = get_pipeline()
        t0 = time.perf_counter()
        result = pipeline.answer(query)
        result["latency_ms"] = int((time.perf_counter() - t0) * 1000)
        return result

    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/build-index", response_model=BuildIndexResponse)
def build_index(request: BuildIndexRequest):
    """
    Build or rebuild the Qdrant knowledge-base index from the dataset.
    Set recreate=true to wipe and re-index from scratch.
    """
    try:
        pipeline = get_pipeline()
        pipeline.build_index(max_samples=request.max_samples, recreate=request.recreate)
        count = pipeline.vectorstore.count()
        return {"status": "success", "message": f"Index built. Total documents in Qdrant: {count}"}

    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"[app] Starting FastAPI server on http://0.0.0.0:{port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)