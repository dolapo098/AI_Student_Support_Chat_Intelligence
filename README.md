# Kent AI Student Support Service

## Overview

The **Kent AI Student Support Service** is the backend AI engine for the University of Kent
student support chatbot. It provides intelligent conversational assistance for student queries
using a Retrieval-Augmented Generation (RAG) pipeline with FAISS + a configurable LLM (Gemini, OpenAI, or Groq).

Key capabilities include:

- **RAG Pipeline**: FAISS semantic search grounds responses in Kent University page content
- **Conversational Memory**: Session-based history maintains context across a conversation
- **Wellbeing Detection**: Detects student distress and immediately signposts crisis resources
- **Escalation Handling**: Gracefully directs to human support when the bot cannot answer
- **Clean Architecture**: Interfaces, factories, and dependency injection throughout

---

## Requirements

| Component          | Specification                        |
| ------------------ | ------------------------------------ |
| **Runtime**        | Python 3.11+                         |
| **Environment**    | Virtual Environment (`venv`)         |
| **Vector Store**   | FAISS (local, no external server)    |
| **LLM Provider**   | Configurable (Gemini / OpenAI / Groq)  |
| **Embeddings**     | Local `sentence-transformers` (MiniLM) |
| **API Framework**  | FastAPI                              |
| **Testing**        | Pytest                               |

---

## Setup

### 1. Python Environment

1. Create and activate virtual environment:

   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   **Optional - name / entity redaction:** the scrubber runs **SpaCy NER on the original message**, then **regex** (so NER does not see ``[REDACTED_*]`` tokens, which can be misparsed as ORG). Download a small English model once:

   ```bash
   python -m spacy download en_core_web_sm
   ```

   Then set `ENABLE_SPACY_NER=true` in `.env`. If the model or package is missing, the API still runs using **regex-only** redaction (with a startup-time warning when NER is enabled).

3. Configure environment variables in `.env` (pick **one** LLM provider). If you omit `LLM_PROVIDER` / `LLM_MODEL`, the service defaults to **OpenAI** + **gpt-4o-mini`.

   **OpenAI** (default)
   ```env
   LLM_PROVIDER=openai
   OPENAI_API_KEY=your_key
   LLM_MODEL=gpt-4o-mini
   ENVIRONMENT=development
   ```

   **Gemini**
   ```env
   LLM_PROVIDER=gemini
   GEMINI_API_KEY=your_key
   LLM_MODEL=gemini-2.5-flash
   ENVIRONMENT=development
   ```

   **Groq**
   ```env
   LLM_PROVIDER=groq
   GROQ_API_KEY=your_key
   LLM_MODEL=llama-3.3-70b-versatile
   ENVIRONMENT=development
   ```

   Optional: `FAISS_INDEX_PATH` (default `faiss_index`), `ALLOWED_ORIGINS` (comma-separated, no spaces - must match the **exact** browser origin, e.g. both `http://localhost:5173` **and** `http://127.0.0.1:5173` if you use either). The server defaults include both for Vite.

   **Privacy / transport (production):**
   - `SANITIZE_LOGS` - default `true`; strips common PII patterns from log lines and **conversation audit** fields.
   - **Conversation audit (optional):** set `ENABLE_CONVERSATION_AUDIT_LOG=true` to append **scrubbed** JSON Lines to `CONVERSATION_AUDIT_LOG_PATH` (default `logs/conversation_audit.jsonl`). Each line includes `session_id`, scrubbed user/assistant text (assistant capped by `CONVERSATION_AUDIT_MAX_RESPONSE_CHARS`, default 8000), `retrieved_chunk_indices`, `context_empty`, `wellbeing_flag`, and stage latencies (`latency_ms_retrieval`, `latency_ms_llm`, etc.) for RAG debugging and review. The `logs/` folder is gitignored. **Production** would typically sink the same schema to Postgres/object storage with IAM and retention policies - not raw prints.
   - `ENABLE_SECURITY_HEADERS` - set `true` behind HTTPS to add `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, and optional HSTS.
   - `HSTS_MAX_AGE` - e.g. `31536000` (only when the app is served over HTTPS end-to-end to the browser; unsafe on plain HTTP dev URLs).
   - **`ENABLE_SPACY_NER`** - default off; when `true`, runs **SpaCy NER** on the raw text, then regex, for configured entity types (see below).
   - **`NER_REDACT_LABELS`** - comma-separated spaCy labels to replace (default `PERSON`). Example: `PERSON` or `PERSON,GPE,ORG`.
   - **`SPACY_MODEL`** - default `en_core_web_sm` (larger models improve NER quality at the cost of RAM/CPU).

   **Hybrid scrubbing:** **SpaCy** (optional) runs on the **original** text for contextual entities (e.g. **person names**). **Regex** then redacts deterministic PII (emails, UK-style phones and postcodes, long numeric IDs, labelled student references). NER runs before regex so bracketed placeholders never confuse the entity model. **Wellbeing** checks and **FAISS** search still use the **original** message.

4. Build the knowledge base (FAISS index) before starting the API:

   ```bash
   python scripts/build_knowledge_base.py
   ```

   This scrapes configured Kent pages, embeds chunks locally, and writes `faiss_index/index.faiss` and `chunks.pkl`.

### 2. Start the Service

```bash
# Set Python path
$env:PYTHONPATH="."  # Windows PowerShell
export PYTHONPATH="."  # Linux/Mac

# Run the service
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**API Documentation**: http://localhost:8000/docs (use your chosen port, e.g. `8001`).

### TLS (HTTPS) in production

Uvicorn serves **HTTP** by default. For encryption **in transit**, terminate **TLS** in front of the app (Render, Azure, nginx, Caddy, etc.) or run uvicorn with certificates, for example:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --ssl-keyfile /path/to/key.pem --ssl-certfile /path/to/cert.pem
```

Ensure the browser and the API both use `https://` URLs in production; point `ALLOWED_ORIGINS` at your real frontend origin (`https://your-app.vercel.app`, etc.).

### Encryption at rest (operational)

Conversation history is held **in memory** only in this service. The **FAISS index** and `chunks.pkl` live on disk under `FAISS_INDEX_PATH`. Protect those paths with **OS or cloud volume encryption**, restricted file ACLs, and **secrets** (API keys) via the host environment or a secrets manager-not committed to git.

---

## API Endpoints

| Endpoint | Method | Description |
| --- | --- | --- |
| `/api/v1/chat/health` | GET | Health check - service and FAISS load status |
| `/api/v1/chat` | POST | Send a student message, receive Kay's response |
| `/api/v1/chat/session/{session_id}` | DELETE | Clear conversation memory for a session |

### POST /api/v1/chat - Example

```json
{
  "message": "What are the entry requirements for Computer Science?",
  "session_id": "uuid-string"
}
```

Prior messages for that `session_id` are kept on the server. **Response:**
```json
{
  "answer": "For BSc Computer Science at Kent, you will need AAB at A-Level...",
  "session_id": "uuid-string",
  "is_wellbeing": false,
  "suggested_topics": ["Application process", "Fees and funding"]
}
```

---

## Architecture

The service follows **Clean Architecture** principles with strict layer separation:

- **Routes**: HTTP endpoint definitions only - thin, delegates immediately
- **Controllers**: Translates HTTP requests to service calls
- **Contracts**: Abstract interfaces (ABC) and Pydantic DTOs
- **Application**: Concrete service implementations and business logic
- **Infrastructure**: LLM and FAISS providers, factories, middleware, DI

### Key Design Patterns

- **Strategy Pattern**: `ILLMProvider` and `IKnowledgeProvider` interfaces
- **Factory Pattern**: `LLMProviderFactory` with dictionary-based registry
- **Dependency Injection**: FastAPI's `Depends()` wired in `di.py`
- **Clean Architecture**: Strict layer boundaries, no cross-layer imports

---

## Knowledge Base

The FAISS index is **not** built at API startup. Run `python scripts/build_knowledge_base.py`
to scrape configured pages, embed with the local sentence-transformers model, and write
`faiss_index/` (or `FAISS_INDEX_PATH`). The API loads that index on startup; if files are
missing, startup logs a warning - run the build script first.

---

## Troubleshooting

| Issue | Cause | Resolution |
| --- | --- | --- |
| **No knowledge / weak answers** | Index missing or stale | Run `python scripts/build_knowledge_base.py` |
| **API Error 429** | Provider quota exceeded | Wait or switch provider / model in `.env` |
| **Connection Refused** | Service not running | Start uvicorn on the port you chose |
| **Empty / error from LLM** | Wrong API key or `LLM_PROVIDER` | Match keys and model to the selected provider |

---

## License

University of Kent - AI Software Engineer Technical Assessment, March 2026.
