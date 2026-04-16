# dots.ocr RAG Ingestion Pipeline

A production-grade, on-premises RAG (Retrieval-Augmented Generation) ingestion pipeline built on top of [dots.ocr](https://github.com/rednote-hilab/dots.ocr) — a state-of-the-art multilingual document layout VLM. Designed to ingest thousands of PDF documents without breaking, with real-time progress tracking, content deduplication, and scalable multi-stage parallel processing.

---

## What This Is

This project wraps the dots.ocr Vision-Language Model in a full production pipeline:

- Upload N PDFs (no count or size limit) via REST API
- Each PDF is OCR-processed page-by-page using dots.ocr (Qwen2.5-VL 1.7B)
- Text is chunked with Markdown-aware, token-limited, overlap-preserving logic
- Chunks are embedded using `mxbai-embed-large-v1` (1024-dim) and stored in PostgreSQL/pgvector
- Raw PDFs and extracted Markdown are archived in SeaweedFS object storage
- Duplicate documents are detected and skipped before any OCR work begins
- Progress is streamed in real time to clients via SSE

Optimized for Apple Silicon (M3 Ultra, MPS) but runs on CPU as a fallback.

---

## Architecture

```
┌─ Stage 1 ──────────────────────────────────────────────────┐
│  Preprocessing  · 6 CPU threads                            │
│  Reads PDF from disk → SHA-256 → dedup check → pages      │
│  Skips immediately if doc already in Postgres or Redis     │
└────────────────────────────┬───────────────────────────────┘
                             │ page_q  (maxsize=60)
┌─ Stage 2+3 ────────────────▼───────────────────────────────┐
│  OCR Workers  · 3 MPS threads  (each owns DotsOCRParser)   │
│  PIL page → VLM generate() → JSON → Markdown              │
└────────────────────────────┬───────────────────────────────┘
                             │ markdown_q  (maxsize=300)
┌─ Stage 4 ──────────────────▼───────────────────────────────┐
│  Document Assembler  · 1 thread                            │
│  Collects all pages, sorts, concatenates, cleans           │
│  SeaweedFS upload runs in a daemon thread (non-blocking)   │
└────────────────────────────┬───────────────────────────────┘
                             │ assembled_q  (maxsize=50)
┌─ Stage 5 ──────────────────▼───────────────────────────────┐
│  Chunking  · 4 CPU threads                                 │
│  Markdown-aware, token-limited (500 tok), 150-tok overlap  │
└────────────────────────────┬───────────────────────────────┘
                             │ chunk_q  (maxsize=2000)
┌─ Stage 6 ──────────────────▼───────────────────────────────┐
│  Embedding Batcher  · 1 MPS thread                         │
│  mxbai-embed-large-v1 · 128 chunks per forward pass       │
└────────────────────────────┬───────────────────────────────┘
                             │ store_q  (maxsize=2000)
┌─ Stage 7 ──────────────────▼───────────────────────────────┐
│  Storage Writers  · 4 IO threads                           │
│  PostgreSQL: chunks + pgvector (HNSW, m=24, ef=128)       │
│  Sets 1-year Redis dedup key after successful storage      │
│  Deletes local upload file after successful storage        │
└────────────────────────────────────────────────────────────┘
```

**Infrastructure:**

| Service | Role |
|---|---|
| FastAPI | REST API + SSE progress streaming |
| RabbitMQ | Job queue (priority / normal / large queues) |
| Redis | Session state, SSE pub/sub, 1-year dedup cache |
| PostgreSQL + pgvector | Chunks, embeddings (HNSW index), RBAC metadata |
| SeaweedFS | Raw PDF + Markdown archival object storage |

---

## Key Features

**Unlimited scale** — No PDF count or size limits. DocJob carries only a file path (not bytes), so `_doc_q` can hold thousands of entries without filling RAM. At most 6 files are in memory simultaneously regardless of batch size.

**Content deduplication** — SHA-256 hash checked against Redis (microseconds) then Postgres (authoritative) before any OCR work begins. Re-uploading a document returns "already exists" instantly.

**MPS meta tensor fix** — DotsOCRParser is patched at runtime to use `device_map={"": device}` so accelerate places all layers directly on MPS via `set_module_tensor_to_device()`. No bare `.to(device)` calls on meta tensors. Stable across all concurrent OCR workers regardless of batch size.

**Chunk overlap** — Exact tiktoken-based 150-token tail of each chunk is prepended to the next, improving RAG retrieval recall near section boundaries.

**Real-time SSE progress** — Per-file stage updates (`ocr → chunking → embedding → storing → done`) with percentage, chunk count, and document ID. Redis pubsub connection is released immediately when the client disconnects.

**RBAC** — Multi-department scoping. Documents are stored under `dept_id`; vector search filters by department.

---

## Requirements

- Python 3.12
- PostgreSQL with pgvector extension
- Redis 7+
- RabbitMQ 3+
- SeaweedFS (optional — archival only)
- Apple Silicon M1/M2/M3/M4 (MPS) or any CPU

---

## Installation

### 1. Clone and install dots.ocr

```bash
git clone https://github.com/rednote-hilab/dots.ocr.git
cd dots.ocr
pip install -e .
```

### 2. Download model weights

```bash
python3 tools/download_model.py
# with modelscope:
python3 tools/download_model.py --type modelscope
```

> Use a directory name without periods for the model save path (e.g. `DotsOCR` not `dots.ocr`).

### 3. Install pipeline dependencies

```bash
cd "RAG_complete_Backend_W 2/Rag_full_pipeline"
pip install -r requirements.txt
```

### 4. Start infrastructure

```bash
# PostgreSQL with pgvector
docker run -d -p 5433:5432 \
  -e POSTGRES_PASSWORD=yourpassword \
  -e POSTGRES_DB=ragdb \
  pgvector/pgvector:pg16

# Redis
docker run -d -p 6379:6379 redis:7-alpine

# RabbitMQ
docker run -d -p 5672:5672 -p 15672:15672 rabbitmq:3-management
```

Or use the provided `docker-compose.yml`:

```bash
docker-compose up -d
```

### 5. Configure environment

Create a `.env` file inside `RAG_complete_Backend_W 2/Rag_full_pipeline/`:

```env
PG_HOST=localhost
PG_PORT=5433
PG_DATABASE=ragdb
PG_USER=postgres
PG_PASSWORD=yourpassword

REDIS_HOST=localhost
REDIS_PORT=6379

RABBIT_HOST=localhost
RABBIT_PORT=5672
RABBIT_USER=guest
RABBIT_PASS=guest

DOTS_OCR_WEIGHTS=./weights/DotsOCR
EMBEDDING_DEVICE=mps        # or cpu
EMBEDDING_BATCH_SIZE=128

SEAWEEDFS_FILER_URL=http://localhost:8889   # optional
```

### 6. Launch

```bash
cd "RAG_complete_Backend_W 2/Rag_full_pipeline"
python main.py
```

API available at `http://localhost:8000` — Swagger docs at `http://localhost:8000/docs`.

---

## API Reference

### POST `/ingest`

Upload one or more PDFs for ingestion.

```bash
curl -X POST http://localhost:8000/ingest \
  -F "files=@invoice.pdf" \
  -F "files=@report.pdf" \
  -F "dept_id=<department-uuid>"
```

**Response:**
```json
{
  "session_id": "uuid",
  "dept_id": "uuid",
  "files": [
    {"file_id": "uuid", "filename": "invoice.pdf", "size_kb": 423.1}
  ]
}
```

No count limit. No size limit. Any number of PDFs accepted per request.

### GET `/ingest/progress/{session_id}`

Server-Sent Events stream of per-file progress. Each event:

```json
{
  "type": "file_progress",
  "data": {
    "file_id": "uuid",
    "filename": "invoice.pdf",
    "stage": "ocr | chunking | embedding | storing | done | error",
    "pct": 0-100,
    "chunks": 42,
    "doc_id": "uuid"
  }
}
```

Stream closes when all files in the session reach a terminal stage.

### GET `/health`

Returns live status of all infrastructure components:

```json
{
  "status": "ok",
  "postgres": true,
  "redis": true,
  "rabbitmq": true,
  "seaweedfs": true
}
```

### GET `/storage/jobs/{job_id}/files`

List all SeaweedFS objects (raw PDF, extracted Markdown) for a job.

### DELETE `/storage/jobs/{job_id}/files`

Remove SeaweedFS artefacts for a completed or failed job.

---

## Pipeline Configuration

All settings in `src/config.py` / `.env`:

| Setting | Default | Description |
|---|---|---|
| `DOTS_OCR_WEIGHTS` | `./weights/DotsOCR` | Path to downloaded model weights |
| `EMBEDDING_DEVICE` | `cpu` | `mps` for Apple Silicon, `cuda` for NVIDIA |
| `EMBEDDING_BATCH_SIZE` | `128` | Chunks per embedding forward pass |
| `chunk_size` | `1500` | Max tokens per chunk |
| `chunk_overlap` | `150` | Overlap tokens between adjacent chunks |
| `ocr_dpi` | `400` | PDF rasterization DPI |
| `top_k_retrieval` | `50` | Candidates before reranking |
| `top_k_rerank` | `5` | Final chunks returned to LLM |
| `similarity_threshold` | `0.45` | Minimum cosine similarity for retrieval |
| `DEDUP_TTL` | 31,536,000 s | 1-year Redis dedup cache TTL |

---

## Deduplication Behaviour

When a PDF is uploaded:

1. SHA-256 hash computed from raw bytes
2. Redis checked (O(1), 1-year TTL) — skip if found
3. Postgres `documents` table checked (authoritative) — skip if found
4. If new: full OCR → chunk → embed → store pipeline runs
5. After successful store: Redis dedup key set for all future uploads

Re-uploading the same file (even with a different filename) skips OCR entirely and returns the existing `doc_id`.

---

## dots.ocr Model

This pipeline uses [dots.ocr](https://github.com/rednote-hilab/dots.ocr) as its OCR engine — a 1.7B-parameter vision-language model that achieves SOTA performance on OmniDocBench for text, tables, and reading order across English, Chinese, and 100+ languages.

Key pipeline adaptations:
- `attn_implementation="eager"` — avoids SDPA/flash_attn probing noise
- `device_map={"": device}` — loads all layers directly to MPS; prevents meta tensor crash on Apple Silicon with transformers ≥4.38
- Each OCR worker thread owns its own parser instance for lock-free parallelism
- Portrait and landscape pages handled automatically via PIL normalization

---

## Known Limitations

- **Formula extraction**: Complex multi-line formulas may have imperfect LaTeX output (inherent to dots.ocr 1.7B).
- **Picture content**: Images embedded in PDFs are detected and bounded but their content is not described or indexed.
- **Continuous special characters**: Long runs of ellipses (`...`) or underscores (`___`) can cause the VLM to loop. Affected pages are caught by `max_new_tokens` and logged.
- **Very high character density**: Pages with extremely small fonts at low DPI may produce degraded output. The default 400 DPI handles most cases; increase `ocr_dpi` for dense text.
- **Cold start**: Each OCR worker loads the 1.7B model on startup (~20-40 s per worker on MPS). Subsequent documents process without reloading.
- **SeaweedFS optional**: If SeaweedFS is not configured, raw files and Markdown are not archived — only chunks and embeddings are stored in Postgres.

---

## License

The pipeline code in `RAG_complete_Backend_W 2/` is proprietary.

The dots.ocr model and library (`dots_ocr/`) are licensed under the [dots.ocr LICENSE AGREEMENT](./dots.ocr%20LICENSE%20AGREEMENT). See also [NOTICE](./NOTICE).
