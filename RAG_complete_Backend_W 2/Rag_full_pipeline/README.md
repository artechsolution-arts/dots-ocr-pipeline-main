# RAG Full Pipeline

Production ingestion service for the dots.ocr RAG pipeline. Exposes a FastAPI server that accepts PDF uploads, runs them through a 7-stage parallel processing pipeline, and stores chunks + embeddings in PostgreSQL/pgvector.

See the [root README](../../README.md) for full architecture, setup, and API documentation.

## Quick Start

```bash
# 1. Configure environment
cp .env.example .env   # edit PG_HOST, REDIS_HOST, RABBIT_HOST, DOTS_OCR_WEIGHTS

# 2. Start infrastructure
docker-compose up -d

# 3. Launch
python main.py
```

API: `http://localhost:8000`
Docs: `http://localhost:8000/docs`

## Directory Structure

```
src/
  api/          FastAPI routes (ingest, progress SSE, storage, health)
  config.py     All settings — override via .env
  database/     postgres_db.py, redis_db.py, rabbitmq_broker.py
  ingestion/
    pipeline/   stage_pipeline.py — 7-stage threaded pipeline
    chunking/   DocumentChunker — Markdown-aware, token-limited, overlapping
    embedding/  MxbaiEmbedder — mxbai-embed-large-v1 (1024-dim)
    parsing/    TextCleaner
    preprocessing/ DocumentPreprocessor — PDF → PIL pages at 400 DPI
  models/       Pydantic schemas
  services/     RBAC
  storage/      SeaweedFS client
  worker/       pool.py — RabbitMQ consumer, decoupled ack
main.py         Lifespan startup, worker mode flag
```
