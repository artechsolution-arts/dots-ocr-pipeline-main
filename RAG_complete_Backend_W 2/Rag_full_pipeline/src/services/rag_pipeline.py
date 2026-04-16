"""
RAG Pipeline  —  ingestion facade
==================================
Wraps StagePipeline.  All processing happens inside the stage workers;
this class is only responsible for initialisation and the public submit API.

Architecture
------------
  routes.py / PDFWorker
       │
       ▼
  RAGPipeline.submit(DocJob)
       │
       ▼
  StagePipeline._doc_q
       │
       ▼  (stage workers: preprocess → ocr → assemble → chunk → embed → store)
  PostgreSQL + pgvector
"""

import logging

from src.database.postgres_db import RBACManager
from src.ingestion.pipeline.datatypes import DocJob
from src.ingestion.pipeline.stage_pipeline import StagePipeline

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Public facade for the ingestion pipeline.

    Holds shared infrastructure references (DB pool, Redis, SeaweedFS) and
    owns a StagePipeline that contains all processing threads.
    """

    def __init__(self, conn, rsm, storage=None):
        self.conn    = conn
        self.rsm     = rsm
        self.storage = storage
        self.rbac    = RBACManager(conn)

        # StagePipeline is created here but started externally via start()
        self._stage_pipeline = StagePipeline(
            rsm=rsm,
            rbac=self.rbac,
            storage=storage,
        )
        logger.info("RAGPipeline (stage-based ingestion) initialised.")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        """Start all stage worker threads."""
        self._stage_pipeline.start()
        logger.info("StagePipeline started")

    def stop(self, timeout: float = 30.0):
        """Graceful shutdown — drain queues then stop threads."""
        self._stage_pipeline.stop(timeout=timeout)
        logger.info("StagePipeline stopped")

    # ── Public API ────────────────────────────────────────────────────────────

    def submit(self, doc_job: DocJob):
        """
        Submit a document into the stage pipeline.
        Returns immediately — processing is async through the stage queues.
        """
        self._stage_pipeline.submit(doc_job)

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
        **kwargs,
    ):
        """
        Convenience wrapper used by PDFWorker.
        Creates a DocJob and submits it to the stage pipeline.
        """
        doc_job = DocJob(
            file_id=file_id,
            session_id=session_id,
            filename=filename,
            raw_bytes=raw_bytes,
            user_id=user_id,
            dept_id=dept_id,
            upload_id=upload_id,
            upload_type=upload_type,
        )
        self.submit(doc_job)
        logger.info("[Pipeline] Submitted '%s' (%d KB) to stage pipeline",
                    filename, len(raw_bytes) // 1024)
