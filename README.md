---
title: Mental Health Support API
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# RAG-Based Mental Health Support Chatbot

[![CI/CD Pipeline](https://github.com/heshamyehia/RAG-Based-Mental-Health-Support/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/heshamyehia/RAG-Based-Mental-Health-Support/actions/workflows/ci-cd.yml)

An empathetic, multi-lingual, and context-aware mental health support assistant built as a unified NLP pipeline. The chatbot combines advanced natural language processing (NLP) classification models, semantic search with retrieval-augmented generation (RAG), and a session-based history manager to provide secure, empathetic, and tailored guidance.

---

## 🏗️ System Architecture

The pipeline processes user queries sequentially, classifying language, emotion, and intent in parallel before routing the request to either the Q&A RAG pipeline or a direct prompt-based fallback:

```mermaid
graph TD
    User([User Query]) --> History[Retrieve Chat History]
    History --> ParallelClf[Parallel Classification]
    
    subgraph Parallel Classification
        M1[Module 1: Language Detection<br/>TF-IDF + LinearSVM]
        M2[Module 2: Emotion Classifier<br/>Fine-Tuned DistilBERT]
        M3[Module 3: Intent Classifier<br/>Few-Shot LLM / prompts.yaml]
    end
    
    User --> M1
    User --> M2
    User --> M3
    
    M1 & M2 & M3 --> Router{Intent Router}
    
    Router -- asking_mental_health_question --> M4[Module 4: RAG Pipeline<br/>Qdrant + HF Dataset + Gemini]
    Router -- greeting / goodbye / gratitude / out_of_scope --> Direct[Direct Response<br/>Gemini + prompts.yaml]
    
    M4 --> Response[Construct ChatResponse]
    Direct --> Response
    Response --> SaveHistory[Save Exchange to History]
    SaveHistory --> Output([JSON Output])
```

---

## 📂 Project Structure

```
RAG-Based-Mental-Health-Support/
├── main.py                     # FastAPI entrypoint exposing the unified pipeline
├── schemas.py                  # Shared Pydantic data schemas & Enums (Intent, Emotion)
├── history_manager.py          # Session-based local JSON chat history (sliding window)
├── pyproject.toml              # Project metadata and dependencies
├── uv.lock                     # Locked dependencies for reproducible environments
├── .env.example                # Template for environment configuration keys
│
├── language_detector/          # Module 1: Language Identification
│   ├── language_detector.py    # Detector script (public API & standalone CLI test)
│   ├── language_detector.joblib     # Pre-trained TF-IDF + SVM classifier pipeline
│   ├── language_detector_meta.joblib# Class labels metadata mapping
│   └── train_language_detection.ipynb  # Notebook for model training & evaluation
│
├── emotion_classifier/         # Module 2: Emotion Classifier
│   ├── predictor.py            # DistilBERT predictor class interface
│   ├── final_emotion_model/    # Hugging Face fine-tuned DistilBERT weights & config
│   └── DistilBertTuned_EmotionClassifier.ipynb # Notebook detailing the fine-tuning process
│
├── Intent_classifier/          # Module 3: Intent Identification
│   ├── intent_classifier.py    # Intent classifier & context-aware direct responder
│   ├── prompts.yaml            # YAML configuration for LLM instructions and examples
│   └── module3_intent_classifier.ipynb # Notebook exploring Groq/Gemini intent classify
│
└── module4_rag/                # Module 4: Q&A RAG Pipeline
    ├── rag_pipeline.py         # Knowledge-base search and context-grounded response pipeline
    ├── config.yaml             # Vector DB, Embedding Model, and LLM configuration
    ├── pipeline_config.json    # Pipeline metadata cache
    ├── README.md               # Dedicated RAG module documentation
    └── module4_rag.ipynb       # Step-by-step vector indexing & testing notebook
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (fast Python package manager)

### Step 1: Install uv
If you don't have `uv` installed:
```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Step 2: Set Up the Environment and Install Dependencies
Clone the repository, then let `uv` create a virtual environment and install all dependencies in one step:
```bash
uv sync
```

To also install development dependencies (linters, formatters):
```bash
uv sync --group dev
- [uv](https://github.com/astral-sh/uv) (Extremely fast Python package installer and resolver)

### Step 1: Install Dependencies
Clone the repository and install the project dependencies using `uv`:
```bash
uv sync
```

### Step 3: Set Up Environment Variables
Copy `.env.example` to `.env` and populate your API credentials:
```bash
cp .env.example .env
```
Update `.env` with the following variables:
- `GEMINI_API_KEY`: API key from Google AI Studio.
- `QDRANT_URL`: Endpoint for your Qdrant vector database.
- `QDRANT_API_KEY`: Authentication key for Qdrant.
- `GROQ_API_KEY`: API key for Groq (optional/fallback).
- `TOKENIZERS_PARALLELISM`: Set to `false` to avoid warning logs in multi-threaded workflows.
- `FRONTEND_ORIGIN`: Exact frontend origin for CORS. For GitHub Pages, use your site origin such as `https://your-username.github.io`.

If you need to allow more than one origin, set `FRONTEND_ORIGINS` to a comma-separated list.

### Step 4: Populate the Q&A Vector Database (Qdrant)
Run the indexing pipeline inside `module4_rag/module4_rag.ipynb` to download the Hugging Face dataset (`Amod/mental_health_counseling_conversations`), generate embeddings using `sentence-transformers/all-MiniLM-L6-v2`, and upload them to your Qdrant instance.

### Step 5: Run the FastAPI Server
Start the API server locally:
```bash
uv run python main.py
```
*Note: The server runs at `http://localhost:8000`. You can access the interactive API docs (Swagger UI) at `http://localhost:8000/docs`.*

---

## 🔌 API Endpoints

### 1. Liveness & Health Check
* **Endpoint:** `GET /health`
* **Response:**
  ```json
  {
    "status": "ok",
    "module": "chat_pipeline",
    "version": "1.0.0"
  }
  ```

### 2. Feedback
* **Endpoint:** `POST /feedback`
* **Request Body:**
  ```json
  {
    "session_id": "user_session_999",
    "vote": "thumbs_up"
  }
  ```
* **Accepted votes:** `thumbs_up`, `thumbs_down`

### 3. Unified Chat Pipeline
* **Endpoint:** `POST /chat`
* **Request Body:**
  ```json
  {
    "session_id": "user_session_999",
    "message": "I have been feeling extremely overwhelmed and anxious this week."
  }
  ```
* **Response Body:**
  ```json
  {
    "language_code": "en",
    "emotion": "sadness",
    "intent": "asking_mental_health_question",
    "response": "Feeling overwhelmed and anxious can be an incredibly heavy burden to carry... Here are some grounding exercises...",
    "response_source": "rag"
  }
  ```

---

## 🧠 Core NLP Modules

### 🌐 Module 1: Language Detection
* **Model Type:** character-level TF-IDF n-grams + Linear Support Vector Machine (LinearSVM).
* **Trained On:** `papluca/language-identification` (dataset covering 20 languages).
* **Behavior:** Extracts ISO 639-1 language codes. Fallback defaults to `"en"` when input text is too short (<3 chars) or confidence is less than 50%.

### 🎭 Module 2: Emotion Classifier
* **Model Type:** Fine-tuned `DistilBERT` sequence classifier.
* **Location:** Stored locally in `emotion_classifier/final_emotion_model/`.
* **Output Labels:** `joy`, `sadness`, `anger`, `fear`, `love`, `surprise`. Falls back to `unknown` on exception.

### 🧭 Module 3: Intent Classifier
* **Model Type:** LLM-based few-shot classifier leveraging `gemini-3.5-flash` or Groq.
* **Configured Via:** `Intent_classifier/prompts.yaml`.
* **Classification categories:** `greeting`, `goodbye`, `gratitude`, `asking_mental_health_question`, and `out_of_scope`.
* **Routing logic:** Non-RAG intents bypass the search database and receive an immediate contextually tailored and empathetic reply mapping the detected emotion and language.

### 📚 Module 4: Q&A RAG Pipeline
* **Knowledge Source:** `Amod/mental_health_counseling_conversations` dataset.
* **Vector Store:** Qdrant Cloud.
* **Embeddings:** `all-MiniLM-L6-v2` (384-dimensional cosine similarity).
* **LLM Engine:** `gemini-3.5-flash`.
* **Behavior:** RAG fetches the top 5 counselling QA pairs, matches the user's input language, and drafts an empathetic, safe, and actionable response incorporating the chat history.

### 💾 History Manager
* **Implementation:** `history_manager.py` maintains conversation threads locally under `chat_sessions/` as JSON documents.
* **Sliding Window:** Limits context to the last `10` turns (user/assistant pairs) to preserve token constraints and model focus.

---

## 📊 Monitoring

The chatbot is instrumented with **OpenTelemetry** and ships telemetry to **Axiom** via the OpenTelemetry Collector.

### Architecture

```
FastAPI app → OTLP/HTTP (port 4318) → OTel Collector (Docker) → Axiom
```

### Metrics

| # | Metric | Type | Rationale |
|---|--------|------|-----------|
| 1 | `chatbot.intent.count` | Model/NLP | Tracks intent distribution over time. A spike in `out_of_scope` signals model degradation or abuse; a rise in `asking_mental_health_question` shows real demand hitting the RAG path. |
| 2 | `chatbot.message.length` | Data | Monitors character length of incoming messages. Very short messages indicate bot/test traffic; unusually long messages may signal prompt injection attempts. |
| 3 | `chatbot.requests.total` / `chatbot.errors.total` | Server | Standard SRE signal. Error rate = errors ÷ requests. Catches Gemini quota exhaustion (429s), Qdrant failures, and pipeline crashes. |

Metrics are emitted as structured OTel log events (Axiom ingests OTLP logs and traces natively).

### Running the OTel Collector

Requires Docker. Add the following to your `.env`:

```env
AXIOM_API_TOKEN=your_axiom_api_token_here
AXIOM_DATASET_NAME=mental-health-chatbot
OTEL_COLLECTOR_ENDPOINT=http://localhost:4318
```

Then start the collector:

```bash
docker compose up -d
```

### Axiom Dashboard

Build a dashboard in Axiom with 3 panels using APL queries:

**Panel 1 — Intent Distribution:**
```kusto
['mental-health-chatbot']
| where ['attributes.event'] == "chatbot.intent.count"
| summarize count() by ['attributes.intent'], bin_auto(_time)
```

**Panel 2 — Message Length:**
```kusto
['mental-health-chatbot']
| where ['attributes.event'] == "chatbot.message.length"
| summarize avg(todouble(['attributes.length'])) by bin_auto(_time)
```

**Panel 3 — Request & Error Rate:**
```kusto
['mental-health-chatbot']
| where ['attributes.event'] in ("chatbot.requests.total", "chatbot.errors.total")
| summarize count() by ['attributes.event'], bin_auto(_time)
```

**Dashboard screenshot:**

![Axiom Dashboard](https://github.com/user-attachments/assets/bfb018b8-e97a-431d-b6e1-bc8dbe95dce9)

---

## ⚠️ Medical Disclaimer

This project is a final academic/research NLP task and is **not** a substitute for professional medical advice, diagnosis, or treatment. If you or someone you know is in crisis or distress, please reach out to a professional mental health provider or contact your local emergency response hotline immediately.

---

## 🌐 Deployment

### Deployed API (Hugging Face Spaces)

| Item | Details |
|------|---------|
| **Live API URL** | [`https://emam2231-mental-health-api.hf.space`](https://emam2231-mental-health-api.hf.space) |
| **Swagger Docs** | [`https://emam2231-mental-health-api.hf.space/docs`](https://emam2231-mental-health-api.hf.space/docs) |
| **Health Check** | `GET /health` → `{"status": "ok"}` |
| **HTTPS** | ✅ Enabled by default (provided by Hugging Face Spaces) |

### CORS

CORS is configured via the `FRONTEND_ORIGIN` / `FRONTEND_ORIGINS` environment variable. On the deployed Space, set this to your frontend origin (e.g., `https://heshamyehia.github.io`). Locally it defaults to `http://localhost:3000` and `http://localhost:5173`.

### CI/CD Pipeline

Every push to `main` triggers the [CI/CD workflow](.github/workflows/ci-cd.yml):

1. **Lint** — `ruff check .`
2. **Test** — `pytest tests/ -v`
3. **Build & Push** — Docker image → Docker Hub
4. **Deploy** — Push to Hugging Face Spaces (only if lint + tests pass)
