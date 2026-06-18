import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from module4_rag.rag_pipeline import RAGPipeline

from .routes import router

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "module4_rag" / "config.yaml"


def _get_frontend_origins() -> list[str]:
    raw_origins = os.getenv("FRONTEND_ORIGINS") or os.getenv("FRONTEND_ORIGIN")
    if raw_origins:
        return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]

    return ["http://localhost:3000", "http://localhost:5173"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = RAGPipeline(CONFIG_PATH)
    logger.info("RAG pipeline initialized")

    yield

    logger.info("Shutting down application")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Mental Health Support Chatbot",
        version="1.0.0",
        description="RAG-based mental health chatbot — NLP Final Task 2026.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_get_frontend_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    return app


app = create_app()
