# Advanced RAG PDF Pipeline

A full-stack RAG (Retrieval-Augmented Generation) pipeline for PDF documents featuring:
- **FastAPI** for API endpoints.
- **RabbitMQ** for task queuing and dead-letter handling.
- **Redis** for state management, rate-limiting, and SSE-based progress tracking.
- **PostgreSQL (pgvector)** for vector search and metadata management.
- **Ollama** for local LLM generation.
- **Hybrid Search** (Vector + BM25) with reranking.
- **RBAC** for secure multi-department access.

## Architecture

1. **Upload**: User sends PDFs via FastAPI.
2. **Queueing**: PDF jobs are published to RabbitMQ (Priority, Normal, or Large queues).
3. **Processing**: Worker threads consume jobs, extract text (via OCR if needed), chunk, embed, and store in PostgreSQL.
4. **Query**: Users query the indexed documents with hybrid search.

## Setup

1. **Prerequisites**:
   - Docker (for PostgreSQL, Redis, RabbitMQ)
   - Python 3.9+
   - Ollama (running locally)

2. **Run Infrastructure**:
   ```bash
   docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=secret -e POSTGRES_DB=ragchat pgvector/pgvector:pg16
   docker run -d -p 6379:6379 redis:7-alpine
   docker run -d -p 5672:5672 -p 15672:15672 rabbitmq:3-management
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Application**:
   ```bash
   python main.py
   ```

5. **Interact**:
   Access the UI at `http://localhost:8000/ui` (if implemented) or use Swagger docs at `http://localhost:8000/docs`.
