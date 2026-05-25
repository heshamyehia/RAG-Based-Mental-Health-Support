"""
rag_pipeline.py
===============
Reusable RAG pipeline for the Mental Health Support Chatbot.

Components
----------
1. DataLoader          – loads & pre-processes the Hugging Face dataset
2. EmbeddingEngine     – wraps sentence-transformers for encoding
3. QdrantVectorStore   – manages the cloud Qdrant collection
4. GeminiLLM           – thin wrapper around the Google Gemini API
5. RAGPipeline         – orchestrates retrieve → augment → generate
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Config Loader
# ---------------------------------------------------------------------------

def load_yaml_config(path: str | Path = "config.yaml") -> Dict[str, Any]:
    """Load YAML config and return the inner 'rag' block."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    return raw.get("rag", raw)


# ---------------------------------------------------------------------------
# RAG Config
# ---------------------------------------------------------------------------

@dataclass
class RAGConfig:
    # Vector Store
    collection_name: str = "mental_health_kb"

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Retrieval
    top_k: int = 5

    # Dataset
    dataset_name: str = "Amod/mental_health_counseling_conversations"
    dataset_file: str = "combined_dataset.json"
    max_response_words: int = 600

    # LLM
    gemini_model: str = "gemini-3.5-flash"

    # Prompt
    system_prompt: str = ""

    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> "RAGConfig":
        cfg = load_yaml_config(path)

        return cls(
            collection_name=os.getenv(
                "COLLECTION_NAME",
                cfg.get("collection_name", cls.collection_name),
            ),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL",
                cfg.get("embedding_model", cls.embedding_model),
            ),
            top_k=int(
                os.getenv(
                    "TOP_K",
                    cfg.get("top_k", cls.top_k),
                )
            ),
            dataset_name=cfg.get(
                "dataset_name",
                cls.dataset_name,
            ),
            dataset_file=cfg.get(
                "dataset_file",
                cls.dataset_file,
            ),
            max_response_words=int(
                cfg.get(
                    "max_response_words",
                    cls.max_response_words,
                )
            ),
            gemini_model=os.getenv(
                "GEMINI_MODEL",
                cfg.get("gemini_model", cls.gemini_model),
            ),
            system_prompt=cfg.get(
                "system_prompt",
                cls.system_prompt,
            ),
        )


# ---------------------------------------------------------------------------
# Document Schema
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """A single unit stored in the vector database."""

    doc_id: str
    context: str
    response: str
    source: str = "mental_health_counseling_conversations"

    @property
    def text_to_embed(self) -> str:
        """Embed the user question/context."""
        return self.context

    @property
    def full_text(self) -> str:
        """Text used inside retrieved context."""
        return f"Question: {self.context}\nAnswer: {self.response}"


# ---------------------------------------------------------------------------
# Data Loader
# ---------------------------------------------------------------------------

class DataLoader:
    """
    Loads and preprocesses the mental health counseling dataset.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_file: str,
        max_response_words: int,
        split: str = "train",
        max_samples: Optional[int] = None,
    ):
        self.dataset_name = dataset_name
        self.dataset_file = dataset_file
        self.max_response_words = max_response_words
        self.split = split
        self.max_samples = max_samples

    # ------------------------------------------------------------------

    def load(self) -> List[Document]:
        """Return cleaned Document objects."""

        raw = self._load_raw()

        docs = [
            self._to_document(i, row)
            for i, row in enumerate(raw)
        ]

        docs = [d for d in docs if self._is_valid(d)]

        print(f"[DataLoader] Loaded {len(docs)} valid documents.")

        return docs

    # ------------------------------------------------------------------

    def _load_raw(self):
        from datasets import load_dataset

        ds = load_dataset(
            self.dataset_name,
            data_files=self.dataset_file,
            split=self.split,
        )

        if self.max_samples:
            ds = ds.select(
                range(min(self.max_samples, len(ds)))
            )

        return ds

    # ------------------------------------------------------------------

    def _to_document(self, idx: int, row: dict) -> Document:
        context = self._clean(row.get("Context", ""))
        response = self._clean(row.get("Response", ""))

        response = self._truncate(
            response,
            self.max_response_words,
        )

        return Document(
            doc_id=f"doc_{idx:06d}",
            context=context,
            response=response,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _clean(text: str) -> str:
        """Basic text normalization."""

        text = text.strip()

        text = re.sub(r"\n{3,}", "\n\n", text)

        text = "\n".join(
            line.strip()
            for line in text.splitlines()
        )

        return text

    # ------------------------------------------------------------------

    @staticmethod
    def _truncate(text: str, max_words: int) -> str:
        words = text.split()

        if len(words) > max_words:
            return " ".join(words[:max_words]) + " ..."

        return text

    # ------------------------------------------------------------------

    @staticmethod
    def _is_valid(doc: Document) -> bool:
        return (
            len(doc.context) > 10
            and len(doc.response) > 10
        )


# ---------------------------------------------------------------------------
# Embedding Engine
# ---------------------------------------------------------------------------

class EmbeddingEngine:
    """
    Wraps SentenceTransformer for generating embeddings.
    """

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name

        print(f"[EmbeddingEngine] Loading model: {model_name} ...")

        self.model = SentenceTransformer(model_name)

        self.dimension = (
            self.model.get_sentence_embedding_dimension()
        )

        print(
            f"[EmbeddingEngine] Ready. "
            f"Vector dimension = {self.dimension}"
        )

    # ------------------------------------------------------------------

    def encode(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode multiple texts."""

        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    # ------------------------------------------------------------------

    def encode_single(self, text: str) -> np.ndarray:
        """Encode one query."""

        return self.encode(
            [text],
            show_progress=False,
        )[0]


# ---------------------------------------------------------------------------
# Qdrant Vector Store
# ---------------------------------------------------------------------------

class QdrantVectorStore:
    """
    Qdrant cloud vector database wrapper.
    """

    def __init__(
        self,
        collection_name: str,
        vector_size: int,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
    ):
        from qdrant_client import QdrantClient

        self.collection_name = collection_name
        self.vector_size = vector_size

        url = qdrant_url or os.getenv("QDRANT_URL")
        key = qdrant_api_key or os.getenv("QDRANT_API_KEY")

        if not url or not key:
            raise ValueError(
                "QDRANT_URL or QDRANT_API_KEY missing."
            )

        self.client = QdrantClient(
            url=url,
            api_key=key,
        )

        print(f"[QdrantVectorStore] Connected to: {url}")

    # ------------------------------------------------------------------

    def create_collection(self, recreate: bool = False) -> None:
        from qdrant_client.models import (
            Distance,
            VectorParams,
        )

        existing = [
            c.name
            for c in self.client.get_collections().collections
        ]

        if self.collection_name in existing:
            if recreate:
                print(
                    f"[QdrantVectorStore] "
                    f"Deleting existing collection "
                    f"'{self.collection_name}' ..."
                )

                self.client.delete_collection(
                    self.collection_name
                )

            else:
                print(
                    f"[QdrantVectorStore] Collection "
                    f"'{self.collection_name}' already exists."
                )
                return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
            ),
        )

        print(
            f"[QdrantVectorStore] Collection "
            f"'{self.collection_name}' created."
        )

    # ------------------------------------------------------------------

    def collection_exists(self) -> bool:
        existing = [
            c.name
            for c in self.client.get_collections().collections
        ]

        return self.collection_name in existing

    # ------------------------------------------------------------------

    def count(self) -> int:
        return self.client.count(
            self.collection_name
        ).count

    # ------------------------------------------------------------------

    def upsert_documents(
        self,
        documents: List[Document],
        embeddings: np.ndarray,
        batch_size: int = 128,
    ) -> None:
        from qdrant_client.models import PointStruct

        total = len(documents)

        print(
            f"[QdrantVectorStore] "
            f"Upserting {total} points ..."
        )

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)

            batch_docs = documents[start:end]
            batch_vecs = embeddings[start:end]

            points = [
                PointStruct(
                    id=idx + start,
                    vector=vec.tolist(),
                    payload={
                        "doc_id": doc.doc_id,
                        "context": doc.context,
                        "response": doc.response,
                        "source": doc.source,
                    },
                )
                for idx, (doc, vec) in enumerate(
                    zip(batch_docs, batch_vecs)
                )
            ]

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

            print(f"  → Upserted {end}/{total}", end="\r")

        print(
            f"\n[QdrantVectorStore] Done. "
            f"Total indexed: {self.count()} points."
        )

    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> List[dict]:
        """Retrieve top-k documents."""

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k,
            with_payload=True,
        )

        return [
            {
                "score": hit.score,
                "doc_id": hit.payload["doc_id"],
                "context": hit.payload["context"],
                "response": hit.payload["response"],
            }
            for hit in results
        ]


# ---------------------------------------------------------------------------
# Gemini LLM
# ---------------------------------------------------------------------------

class GeminiLLM:
    """
    Thin wrapper around Google Gemini.
    """

    def __init__(
        self,
        model: str,
        system_prompt: str,
        api_key: Optional[str] = None,
    ):
        from google import genai

        key = api_key or os.getenv("GEMINI_API_KEY")

        if not key:
            raise ValueError(
                "GEMINI_API_KEY is not set."
            )

        self.client = genai.Client(api_key=key)

        self.model_name = model
        self.system_prompt = system_prompt

        print(f"[GeminiLLM] Using model: {self.model_name}")

    # ------------------------------------------------------------------

    def generate(
        self,
        user_message: str,
        context_block: str,
    ) -> str:
        """
        Generate grounded response.
        """

        augmented_user_msg = (
            "Here are relevant excerpts from professional "
            "mental health counselling sessions that may "
            "help answer the user's question:\n\n"
            f"--- Retrieved Context ---\n"
            f"{context_block}\n"
            f"--- End of Context ---\n\n"
            f"User Question: {user_message}\n\n"
            "Please provide a supportive, grounded response "
            "based on the context above."
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=augmented_user_msg,
            config={
                "system_instruction": self.system_prompt
            },
        )

        return response.text.strip()


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    Main RAG Orchestrator.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config = RAGConfig.from_yaml(config_path)

        # Embeddings
        self.embedder = EmbeddingEngine(
            self.config.embedding_model
        )

        # Vector DB
        self.vectorstore = QdrantVectorStore(
            collection_name=self.config.collection_name,
            vector_size=self.embedder.dimension,
        )

        # LLM
        self.llm = GeminiLLM(
            model=self.config.gemini_model,
            system_prompt=self.config.system_prompt,
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def build_index(
        self,
        max_samples: Optional[int] = None,
        recreate: bool = False,
    ) -> None:
        """
        Load dataset → embed → upload to Qdrant.
        """

        if (
            self.vectorstore.collection_exists()
            and not recreate
        ):
            count = self.vectorstore.count()

            print(
                f"[RAGPipeline] Collection already has "
                f"{count} documents."
            )

            print(
                "Pass recreate=True to force re-indexing."
            )

            return

        # Load dataset
        loader = DataLoader(
            dataset_name=self.config.dataset_name,
            dataset_file=self.config.dataset_file,
            max_response_words=self.config.max_response_words,
            max_samples=max_samples,
        )

        documents = loader.load()

        # Generate embeddings
        print("[RAGPipeline] Encoding documents ...")

        texts = [
            doc.text_to_embed
            for doc in documents
        ]

        embeddings = self.embedder.encode(texts)

        # Create collection
        self.vectorstore.create_collection(
            recreate=recreate
        )

        # Upload vectors
        self.vectorstore.upsert_documents(
            documents,
            embeddings,
        )

        print("[RAGPipeline] Index built successfully.")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> List[dict]:
        """
        Retrieve relevant documents.
        """

        query_vec = self.embedder.encode_single(query)

        return self.vectorstore.search(
            query_vec,
            top_k=self.config.top_k,
        )

    # ------------------------------------------------------------------
    # Full RAG
    # ------------------------------------------------------------------

    def answer(self, query: str) -> dict:
        """
        Full Retrieve → Augment → Generate pipeline.
        """

        # Retrieve
        retrieved = self.retrieve(query)

        # Build context
        context_block = self._format_context(
            retrieved
        )

        # Generate answer
        answer_text = self.llm.generate(
            query,
            context_block,
        )

        return {
            "query": query,
            "answer": answer_text,
            "sources": retrieved,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_context(
        retrieved_docs: List[dict],
    ) -> str:
        """
        Format retrieved docs into readable context.
        """

        parts = []

        for i, doc in enumerate(
            retrieved_docs,
            start=1,
        ):
            parts.append(
                f"[Excerpt {i}] "
                f"(relevance score: {doc['score']:.3f})\n"
                f"Q: {doc['context']}\n"
                f"A: {doc['response']}"
            )

        return "\n\n".join(parts)