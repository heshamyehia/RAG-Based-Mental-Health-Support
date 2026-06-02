# Module 4 – Q&A RAG Pipeline
## NLP Final Task 2026 · Mental Health Support Chatbot

---

## Architecture

```
User Query
    │
    ▼
[Embedding Engine]       sentence-transformers/all-MiniLM-L6-v2
    │  384-dim vector
    ▼
[Qdrant Cloud]           Free cloud vector database (cosine similarity)
    │  top-K relevant Q&A pairs
    ▼
[Gemini LLM]             gemini-3.5-flash (free tier)
    │  context-grounded, empathetic response
    ▼
Final Answer
```

---

## File Structure

```
module4_rag/
├── module4_rag.ipynb       ← Main Jupyter notebook (step-by-step)
├── rag_pipeline.py         ← Reusable pipeline module
├── app.py                  ← FastAPI deployment server
├── requirements.txt        ← Python dependencies
├── .env.example            ← Environment variable template
└── README.md               ← This file
```

---

## Quick Start

### Step 1 – Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 – Set up credentials

Copy `.env.example` to `.env` and fill in:

```env
GEMINI_API_KEY  = ...   # from https://aistudio.google.com/apikey
QDRANT_URL      = ...   # from https://cloud.qdrant.io
QDRANT_API_KEY  = ...   # from https://cloud.qdrant.io
```

### Step 3 – Build the knowledge base index (run once)

Open `module4_rag.ipynb` and run all cells **in order**. This will:
1. Load the mental health counselling dataset from Hugging Face
2. Generate sentence-transformer embeddings
3. Upload all vectors to your Qdrant Cloud collection

### Step 4 – Start the FastAPI server

```bash
uvicorn app:app --reload
```

The server starts on `http://localhost:8000`.  
Swagger UI available at `http://localhost:8000/docs`.

---

## API Reference

### `GET /health`
Liveness check.

```bash
curl http://localhost:8000/health
```

Response:
```json
{"status": "ok", "service": "Mental Health RAG Chatbot – Module 4"}
```

---

### `POST /chat`
Main chat endpoint.

```bash
curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"query": "I feel anxious all the time. What can I do?"}'
```

Response:
```json
{
  "query":   "I feel anxious all the time. What can I do?",
  "answer":  "It sounds like you're going through a really difficult time...",
  "sources": [
    {
      "score":    0.87,
      "doc_id":   "doc_000123",
      "context":  "I have been dealing with anxiety for years...",
      "response": "Anxiety can be incredibly challenging..."
    }
  ],
  "latency_ms": 1340
}
```

---

### `POST /build-index`
(Re)build the Qdrant index from the dataset.

```bash
curl -X POST http://localhost:8000/build-index \
     -H "Content-Type: application/json" \
     -d '{"max_samples": 500, "recreate": false}'
```

---

## Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Embedding model | `all-MiniLM-L6-v2` | Fast, 384-dim, strong semantic quality |
| What we embed | `Context` column only | User queries match counsellor questions better |
| Similarity metric | Cosine | Normalised embeddings → stable similarity scores |
| Qdrant batch size | 128 | Balance between speed and memory |
| LLM | `gemini-3.5-flash` | Generous free tier, no token limit issues |
| Max response length | 600 words (truncated) | Avoid exceeding LLM context window |

## Integration with Other Modules

```
User Query
    │
    ├── Module 1 → Language Detection  → ensure correct KB / response language
    ├── Module 2 → Emotion Classifier  → tune system prompt for empathy level  
    ├── Module 3 → Intent Classifier:
    │       greeting / goodbye / gratitude → direct reply (no RAG)
    │       asking_mental_health_question  → MODULE 4 (this module)
    │       out_of_scope                  → polite redirection
    └── Module 4 → RAG → final answer
```