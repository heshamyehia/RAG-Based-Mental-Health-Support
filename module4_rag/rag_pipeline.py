"""
rag_pipeline.py
===============
Reusable RAG pipeline for the Mental Health Support Chatbot - Module 4.
This module is imported both by the notebook and the Flask deployment app.

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
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from dotenv import load_dotenv

load_dotenv()  # read .env file if present

# ---------------------------------------------------------------------------
# 1. DataLoader
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """A single unit stored in the vector database."""
    doc_id: str
    context: str          # original user question from the dataset
    response: str         # therapist / counsellor response
    source: str = "mental_health_counseling_conversations"

    @property
    def text_to_embed(self) -> str:
        """We embed CONTEXT so that a user query matches similar past questions."""
        return self.context

    @property
    def full_text(self) -> str:
        """Text shown to the LLM as retrieved context."""
        return f"Question: {self.context}\nAnswer: {self.response}"


class DataLoader:
    """
    Loads and pre-processes the Mental Health Counseling Conversations dataset
    from Hugging Face: `Amod/mental_health_counseling_conversations`
    """

    DATASET_NAME = "Amod/mental_health_counseling_conversations"
    MAX_RESPONSE_TOKENS = 600  # truncate very long counsellor responses

    def __init__(self, split: str = "train", max_samples: Optional[int] = None):
        self.split = split
        self.max_samples = max_samples

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> List[Document]:
        """Return a cleaned list of Document objects."""
        raw = self._load_raw()
        docs = [self._to_document(i, row) for i, row in enumerate(raw)]
        docs = [d for d in docs if self._is_valid(d)]
        print(f"[DataLoader] Loaded {len(docs)} valid documents.")
        return docs

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_raw(self):
        from datasets import load_dataset
        ds = load_dataset(self.DATASET_NAME, data_files="combined_dataset.json", split=self.split)
        if self.max_samples:
            ds = ds.select(range(min(self.max_samples, len(ds))))
        return ds

    def _to_document(self, idx: int, row: dict) -> Document:
        context  = self._clean(row.get("Context",  ""))
        response = self._clean(row.get("Response", ""))
        # Truncate extremely long responses to keep prompts manageable
        response = self._truncate(response, self.MAX_RESPONSE_TOKENS)
        return Document(
            doc_id=f"doc_{idx:06d}",
            context=context,
            response=response,
        )

    @staticmethod
    def _clean(text: str) -> str:
        """Basic text normalisation."""
        text = text.strip()
        # Collapse multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Remove leading/trailing whitespace per line
        text = "\n".join(line.strip() for line in text.splitlines())
        return text

    @staticmethod
    def _truncate(text: str, max_words: int) -> str:
        words = text.split()
        if len(words) > max_words:
            return " ".join(words[:max_words]) + " ..."
        return text

    @staticmethod
    def _is_valid(doc: Document) -> bool:
        return len(doc.context) > 10 and len(doc.response) > 10


# ---------------------------------------------------------------------------
# 2. EmbeddingEngine
# ---------------------------------------------------------------------------

class EmbeddingEngine:
    """
    Wraps a SentenceTransformer model for encoding text into dense vectors.

    Default model : sentence-transformers/all-MiniLM-L6-v2
      - Dimension  : 384
      - Fast, lightweight, good semantic quality
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        print(f"[EmbeddingEngine] Loading model: {model_name} …")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"[EmbeddingEngine] Ready. Vector dimension = {self.dimension}")

    def encode(self, texts: List[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
        """Return a 2-D numpy array of shape (len(texts), dimension)."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,   # cosine similarity ≡ dot product
            convert_to_numpy=True,
        )

    def encode_single(self, text: str) -> np.ndarray:
        """Convenience wrapper for a single query string."""
        return self.encode([text], show_progress=False)[0]


# ---------------------------------------------------------------------------
# 3. QdrantVectorStore
# ---------------------------------------------------------------------------

class QdrantVectorStore:
    """
    Manages a Qdrant cloud collection for the mental-health knowledge base.

    Setup
    -----
    Free cloud Qdrant: https://cloud.qdrant.io
    Set QDRANT_URL and QDRANT_API_KEY in your .env file.
    """

    def __init__(
        self,
        collection_name: str = "mental_health_kb",
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        vector_size: int = 384,
    ):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.collection_name = collection_name
        self.vector_size = vector_size
        self.Distance = Distance
        self.VectorParams = VectorParams

        url = qdrant_url or os.getenv("QDRANT_URL")
        key = qdrant_api_key or os.getenv("QDRANT_API_KEY")

        if not url or not key:
            raise ValueError(
                "Qdrant credentials missing. "
                "Set QDRANT_URL and QDRANT_API_KEY in your .env file."
            )

        self.client = QdrantClient(url=url, api_key=key)
        print(f"[QdrantVectorStore] Connected to: {url}")

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def create_collection(self, recreate: bool = False) -> None:
        """Create (or re-create) the Qdrant collection."""
        from qdrant_client.models import Distance, VectorParams

        existing = [c.name for c in self.client.get_collections().collections]

        if self.collection_name in existing:
            if recreate:
                print(f"[QdrantVectorStore] Deleting existing collection '{self.collection_name}' …")
                self.client.delete_collection(self.collection_name)
            else:
                print(f"[QdrantVectorStore] Collection '{self.collection_name}' already exists. Skipping creation.")
                return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
        )
        print(f"[QdrantVectorStore] Collection '{self.collection_name}' created (dim={self.vector_size}).")

    def collection_exists(self) -> bool:
        existing = [c.name for c in self.client.get_collections().collections]
        return self.collection_name in existing

    def count(self) -> int:
        return self.client.count(self.collection_name).count

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def upsert_documents(
        self,
        documents: List[Document],
        embeddings: np.ndarray,
        batch_size: int = 128,
    ) -> None:
        """Upload document vectors + payloads to Qdrant in batches."""
        from qdrant_client.models import PointStruct

        total = len(documents)
        print(f"[QdrantVectorStore] Upserting {total} points …")

        for start in range(0, total, batch_size):
            end   = min(start + batch_size, total)
            batch_docs = documents[start:end]
            batch_vecs = embeddings[start:end]

            points = [
                PointStruct(
                    id=idx + start,
                    vector=vec.tolist(),
                    payload={
                        "doc_id":   doc.doc_id,
                        "context":  doc.context,
                        "response": doc.response,
                        "source":   doc.source,
                    },
                )
                for idx, (doc, vec) in enumerate(zip(batch_docs, batch_vecs))
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)
            print(f"  → Upserted {end}/{total}", end="\r")

        print(f"\n[QdrantVectorStore] Done. Total indexed: {self.count()} points.")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[dict]:
        """Return top-k similar documents with their payloads and scores."""
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k,
            with_payload=True,
        )
        return [
            {
                "score":    hit.score,
                "doc_id":   hit.payload["doc_id"],
                "context":  hit.payload["context"],
                "response": hit.payload["response"],
            }
            for hit in results
        ]


# ---------------------------------------------------------------------------
# 4. GeminiLLM
# ---------------------------------------------------------------------------
 
class GeminiLLM:
    """
    Thin wrapper around the Google Gemini API.
 
    Default model: gemini-3.5-flash (free tier)
    Get your API key at: https://aistudio.google.com/apikey
    """
 
    SYSTEM_PROMPT = """You are a compassionate and professional mental health support assistant.
Your role is to provide empathetic, evidence-based guidance to individuals
experiencing anxiety, depression, stress, or emotional distress.
 
IMPORTANT GUIDELINES:
- Always respond with empathy, warmth, and respect.
- Use the provided context from counselling conversations to ground your answer.
- If the user appears to be in crisis, always encourage them to contact a
  professional or a crisis helpline immediately.
- Do NOT diagnose or prescribe medication.
- Keep responses clear, supportive, and actionable.
- Match the language of the user (Arabic if they write in Arabic, etc.)."""
 
    def __init__(
        self,
        model: str = "gemini-3.5-flash",
        api_key: Optional[str] = None,
    ):
        from google import genai

        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY is not set. Check your .env file.")
        self.client = genai.Client(api_key=key)
        self.model_name = model
        print(f"[GeminiLLM] Using model: {self.model_name}")
 
    def generate(self, user_message: str, context_block: str) -> str:
        """
        Generate a response given a user query and retrieved context.
 
        Parameters
        ----------
        user_message : str
            The user's raw input question.
        context_block : str
            Concatenated retrieved documents (already formatted).
 
        Returns
        -------
        str
            The model's response text.
        """
        augmented_user_msg = (
            f"Here are relevant excerpts from professional mental health counselling sessions "
            f"that may help answer the user's question:\n\n"
            f"--- Retrieved Context ---\n{context_block}\n"
            f"--- End of Context ---\n\n"
            f"User Question: {user_message}\n\n"
            f"Please provide a supportive, grounded response based on the context above."
        )
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=augmented_user_msg,
            config={"system_instruction": self.SYSTEM_PROMPT},
        )
        return response.text.strip()
 
 
# ---------------------------------------------------------------------------
# 5. RAGPipeline  (main orchestrator)
# ---------------------------------------------------------------------------

@dataclass
class RAGConfig:
    """Centralised configuration object."""
    collection_name: str = field(default_factory=lambda: os.getenv("COLLECTION_NAME", "mental_health_kb"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    gemini_model: str    = field(default_factory=lambda: os.getenv("GEMINI_MODEL",    "gemini-3.5-flash"))
    top_k: int           = field(default_factory=lambda: int(os.getenv("TOP_K", "5")))
 
 
class RAGPipeline:
    """
    Orchestrates the full Retrieve → Augment → Generate cycle.
 
    Usage
    -----
    pipeline = RAGPipeline(config)
    pipeline.build_index(max_samples=500)   # run once to populate Qdrant
    answer   = pipeline.answer("I feel overwhelmed all the time.")
    """
 
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.embedder    = EmbeddingEngine(self.config.embedding_model)
        self.vectorstore = QdrantVectorStore(
            collection_name=self.config.collection_name,
            vector_size=self.embedder.dimension,
        )
        self.llm = GeminiLLM(model=self.config.gemini_model)

    # ------------------------------------------------------------------
    # Index building  (run once)
    # ------------------------------------------------------------------

    def build_index(self, max_samples: Optional[int] = None, recreate: bool = False) -> None:
        """
        Load dataset → embed → upsert into Qdrant.
        Set recreate=True to wipe and re-index from scratch.
        """
        if self.vectorstore.collection_exists() and not recreate:
            count = self.vectorstore.count()
            print(f"[RAGPipeline] Collection already has {count} documents. Skipping indexing.")
            print("              Pass recreate=True to force re-indexing.")
            return

        # 1. Load & preprocess
        loader = DataLoader(max_samples=max_samples)
        documents = loader.load()

        # 2. Embed
        print("[RAGPipeline] Encoding documents …")
        texts = [doc.text_to_embed for doc in documents]
        embeddings = self.embedder.encode(texts)

        # 3. Upload
        self.vectorstore.create_collection(recreate=recreate)
        self.vectorstore.upsert_documents(documents, embeddings)
        print("[RAGPipeline] Index built successfully.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> List[dict]:
        """Retrieve the top-K most relevant documents for a query."""
        query_vec = self.embedder.encode_single(query)
        return self.vectorstore.search(query_vec, top_k=self.config.top_k)

    def answer(self, query: str) -> dict:
        """
        Full RAG cycle for a user query.

        Returns
        -------
        dict with keys:
            query      – original user question
            answer     – generated response
            sources    – list of retrieved documents (with scores)
        """
        # Step 1 – Retrieve
        retrieved = self.retrieve(query)

        # Step 2 – Build context block
        context_block = self._format_context(retrieved)

        # Step 3 – Generate
        answer_text = self.llm.generate(query, context_block)

        return {
            "query":   query,
            "answer":  answer_text,
            "sources": retrieved,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_context(retrieved_docs: List[dict]) -> str:
        """Convert a list of retrieved dicts into a readable context block."""
        parts = []
        for i, doc in enumerate(retrieved_docs, start=1):
            score = doc["score"]
            parts.append(
                f"[Excerpt {i}] (relevance score: {score:.3f})\n"
                f"Q: {doc['context']}\n"
                f"A: {doc['response']}"
            )
        return "\n\n".join(parts)
