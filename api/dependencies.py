from fastapi import HTTPException, Request

from module4_rag.rag_pipeline import RAGPipeline


def get_pipeline(request: Request) -> RAGPipeline:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    return pipeline