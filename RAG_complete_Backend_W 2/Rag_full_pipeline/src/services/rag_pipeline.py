import logging, time, threading
from src.config import cfg
from src.database.postgres_db import RBACManager
from src.ingestion.orchestrator import IngestionOrchestrator

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    RAG Ingestion Pipeline.

    Scope  : PDF → DotsOCR → Text Cleaning → Chunking → Embedding
              → store chunks + vectors in PostgreSQL (pgvector).
    Chat / retrieval are out of scope for this container.
    """

    def __init__(self, conn, rsm, storage=None):
        self.conn    = conn
        self.rsm     = rsm
        self.storage = storage
        self.rbac    = RBACManager(conn)
        self._orchestrator = None # Lazy loaded
        self._lock = threading.Lock()
        logger.info("RAGPipeline (ingestion-only) initialised.")

    @property
    def orchestrator(self):
        with self._lock:
            if self._orchestrator is None:
                from src.ingestion.orchestrator import IngestionOrchestrator
                self._orchestrator = IngestionOrchestrator(
                    rsm=self.rsm, rbac=self.rbac, storage=self.storage
                )
        return self._orchestrator

    # ─────────────────────────────────────────────────────────────────────────
    # INGESTION  — called by the WorkerPool (PDFWorker._on_message)
    # ─────────────────────────────────────────────────────────────────────────

    def process_pdf(
        self,
        raw_bytes:   bytes,
        filename:    str,
        user_id:     str,
        dept_id:     str,
        file_id:     str,
        session_id:  str,
        upload_type: str = "user",
        upload_id:   str = None,
        **kwargs,          # absorbs chat_id / retry passed by worker
    ):
        """
        Execute the modular ingestion flow:
          1. Validation + dedup check
          2. DotsOCR extraction
          3. Text cleaning
          4. Chunking (Markdown-aware, token-aware)
          5. Embedding (sentence-transformers / mxbai-embed-large)
          6. Persist chunks + embeddings → PostgreSQL / pgvector
          7. Persist raw PDF → SeaweedFS  (if storage service available)
        """
        logger.info(f"[Pipeline] Ingesting: {filename} ({len(raw_bytes)//1024} KB)")
        t0 = time.time()

        result = self.orchestrator.run_ingestion(
            raw_bytes=raw_bytes,
            filename=filename,
            user_id=user_id,
            dept_id=dept_id,
            file_id=file_id,
            session_id=session_id,
            upload_type=upload_type,
            upload_id=upload_id,
        )

        latency = round(time.time() - t0, 2)
        logger.info(
            f"[Pipeline] {filename} → {result.stage} "
            f"| chunks={result.chunks} | {latency}s"
        )
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # STUB — keeps routes.py from crashing if /query is hit
    # ─────────────────────────────────────────────────────────────────────────

    def query(self, question, user_id, dept_id, chat_id, search="hybrid"):
        return {
            "answer": (
                "Query/chat endpoint is disabled in this deployment. "
                "This container handles document ingestion only."
            ),
            "citations": [],
        }
