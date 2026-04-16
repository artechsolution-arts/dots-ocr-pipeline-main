import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(Path(__file__).parent.parent / ".env")

# ── PostgreSQL ────────────────────────────────────────────────────────────────
PG_HOST     = os.getenv("PG_HOST",     "192.168.10.10")
PG_PORT     = int(os.getenv("PG_PORT", "5433"))
PG_DATABASE = os.getenv("PG_DATABASE", "virchow_dev")
PG_USER     = os.getenv("PG_USER",     "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "Eppl$456!")

# ── Redis ─────────────────────────────────────────────────────────────────────
REDIS_HOST     = os.getenv("REDIS_HOST",     "localhost")
REDIS_PORT     = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB       = int(os.getenv("REDIS_DB",   "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

# ── RabbitMQ ──────────────────────────────────────────────────────────────────
RABBIT_HOST  = os.getenv("RABBIT_HOST",  "localhost")
RABBIT_PORT  = int(os.getenv("RABBIT_PORT",  "5672"))
RABBIT_USER  = os.getenv("RABBIT_USER",  "guest")
RABBIT_PASS  = os.getenv("RABBIT_PASS",  "guest")
RABBIT_VHOST = os.getenv("RABBIT_VHOST", "/")

# ── Embedding ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
EMBEDDING_DIM   = 1024

# ── Upload dir ────────────────────────────────────────────────────────────────
UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── RabbitMQ topology names ───────────────────────────────────────────────────
MQ_EXCHANGE_JOBS = "rag.jobs"
MQ_EXCHANGE_DLX  = "rag.dlx"
MQ_QUEUE_PRIORITY = "rag.q.priority"
MQ_QUEUE_NORMAL   = "rag.q.normal"
MQ_QUEUE_LARGE    = "rag.q.large"
MQ_QUEUE_DEAD     = "rag.q.dead"

RK_PRIORITY = "job.priority"
RK_NORMAL   = "job.normal"
RK_LARGE    = "job.large"

# ── Routing thresholds ────────────────────────────────────────────────────────
PRIORITY_MAX_KB = 1_024
LARGE_MIN_KB    = 10_240

# ── Redis TTLs ────────────────────────────────────────────────────────────────
SESSION_TTL   = 86_400
FILE_TTL      = 86_400
WORKER_HB_TTL = 10
DEDUP_TTL     = 31_536_000   # 1 year (Permanent Deduplication)

MAX_RETRIES   = 3

class RAGConfig:
    def __init__(self):
        # Embedding
        self.embedding_model:      str   = EMBEDDING_MODEL
        self.embedding_dim:        int   = EMBEDDING_DIM
        self.embedding_batch:      int   = 32
        self.embedding_device:     str   = os.getenv("EMBEDDING_DEVICE", "cpu")
        self.upload_dir:           Path  = UPLOAD_DIR
        # Chunking
        self.chunk_size:           int   = 1500
        self.chunk_overlap:        int   = 150
        # Retrieval
        self.top_k_retrieval:      int   = 50
        self.top_k_rerank:         int   = 5
        self.similarity_threshold: float = 0.45
        # LLM
        self.llm_model:            str   = "qwen2.5:latest"
        self.max_tokens:           int   = 2048
        self.temperature:          float = 0.0
        self.alpha:                float = 0.6
        self.beta:                 float = 0.4
        # PDF
        self.max_pdf_size_mb:      float = 200.0
        self.max_pdf_pages:        int   = 2000
        self.max_batch_files:      int   = 100
        self.upload_workers:       int   = int(os.getenv("UPLOAD_WORKERS", "1"))
        # OCR
        self.ocr_dpi:              int   = 400
        self.ocr_fallback:         bool  = True
        # DotsOCR (VLM Engine) — read from env so Docker service names resolve
        self.dots_ocr_ip:          str   = os.getenv("DOTS_OCR_IP",   "localhost")
        self.dots_ocr_port:        int   = int(os.getenv("DOTS_OCR_PORT", "8001"))
        # HuggingFace model ID for DotsOCR (NOT the Ollama chat model)
        self.dots_ocr_model:       str   = os.getenv("DOTS_OCR_MODEL", "rednote-hilab/dots.ocr")
        self.dots_ocr_use_hf:      bool  = os.getenv("DOTS_OCR_USE_HF", "true").lower() in ("true", "1", "yes")
        self.dots_ocr_weights_path:str   = os.getenv("DOTS_OCR_WEIGHTS", "./weights/DotsOCR")
        self.dots_ocr_prompt_mode: str   = os.getenv("DOTS_OCR_PROMPT_MODE", "prompt_layout_all_en")
        self.dots_ocr_fitz_preprocess: bool = os.getenv("DOTS_OCR_FITZ_PREPROCESS", "true").lower() in ("true", "1", "yes")
        # DE-DUPLICATION (PREVENTS RE-PROCESSING)
        self.skip_duplicates:      bool  = True    # ALWAYS SKIP RE-UPLOADS
        # Rate limiting
        self.rate_limit_per_hour:  int   = 200
        # SeaweedFS Object Storage — read from env so Docker service names resolve
        self.SEAWEEDFS_FILER_URL:  str   = os.getenv("SEAWEEDFS_FILER_URL",  "http://192.168.10.10:8889")
        self.SEAWEEDFS_MASTER_URL: str   = os.getenv("SEAWEEDFS_MASTER_URL", "http://192.168.10.10:9333")
        self.SEAWEEDFS_S3_ENDPOINT:str   = os.getenv("SEAWEEDFS_S3_ENDPOINT","http://192.168.10.10:8333")
        self.SEAWEEDFS_BUCKET:     str   = os.getenv("SEAWEEDFS_BUCKET",     "rag-docs")
        self.SEAWEEDFS_ACCESS_KEY: str   = os.getenv("SEAWEEDFS_ACCESS_KEY", "any")
        self.SEAWEEDFS_SECRET_KEY: str   = os.getenv("SEAWEEDFS_SECRET_KEY", "any")
        self.SEAWEEDFS_UPLOAD_TO_STORAGE: bool = True

cfg = RAGConfig()
CFG = cfg
