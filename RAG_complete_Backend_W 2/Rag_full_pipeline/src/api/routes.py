"""
Ingestion API Routes
====================
Single-purpose API for the on-prem RAG ingestion pipeline.

Endpoints
---------
POST /ingest                          Upload PDFs → OCR → Chunk → Embed → pgvector + SeaweedFS
GET  /ingest/progress/{session_id}    SSE stream of per-file ingestion progress
GET  /health                          Service health (postgres, redis, rabbitmq, seaweedfs)
GET  /storage/jobs/{job_id}/files     List SeaweedFS objects for a job
DELETE /storage/jobs/{job_id}/files   Remove SeaweedFS artefacts for a job
"""

import json
import time
import uuid
import asyncio
import logging
import threading
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from src.config import UPLOAD_DIR, cfg
from src.models.schemas import BatchSession, FileProgress, JobPayload

logger = logging.getLogger(__name__)


def create_router(rsm, ids, pipeline, mq_conn):
    router = APIRouter()

    # ── Health Check ──────────────────────────────────────────────────────────

    @router.get("/health")
    async def health():
        """
        Returns the live status of all infrastructure components.
        Your query system can poll this before sending ingestion requests.
        """
        seaweedfs_ok = False
        if pipeline.storage:
            try:
                seaweedfs_ok = await pipeline.storage.health() is not None
            except Exception:
                seaweedfs_ok = False

        status = {
            "status": "ok",
            "postgres": True,   # If we reached this point postgres is up
            "redis":    bool(rsm and rsm.ping()),
            "rabbitmq": bool(mq_conn and mq_conn.is_open),
            "seaweedfs": seaweedfs_ok,
        }
        # Overall ok only if core services are up
        if not status["redis"] or not status["rabbitmq"]:
            status["status"] = "degraded"
        return JSONResponse(status)

    # ── Ingestion ─────────────────────────────────────────────────────────────

    @router.post("/ingest")
    async def ingest(
        files: List[UploadFile] = File(...),
        dept_id: Optional[str] = Form(None),
        user_id: Optional[str] = Form(None),
    ):
        """
        Upload one or more PDF files for ingestion.

        The pipeline runs asynchronously:
          1. DotsOCR  — VLM layout detection + text extraction
          2. Chunking — Markdown-aware, token-limited chunks
          3. Embedding — mxbai-embed-large-v1 (1024-dim)
          4. Storage  — chunks + vectors → PostgreSQL/pgvector
                        raw PDF + markdown → SeaweedFS

        Parameters
        ----------
        files    : PDF file(s) to ingest
        dept_id  : Department UUID to scope vectors under (optional).
                   Defaults to the system default department.
                   Your query system should filter by the same dept_id.
        user_id  : Uploader UUID (optional). Defaults to system user.

        Returns
        -------
        {
          "session_id": "uuid",          -- poll /ingest/progress/{session_id}
          "dept_id": "uuid",             -- use this in your query system for vector filtering
          "files": [
            {"file_id": "uuid", "filename": "doc.pdf", "size_kb": 123.4}
          ]
        }
        """
        if not rsm or not rsm.ping():
            raise HTTPException(503, "Redis is offline — cannot accept ingestion jobs")

        # Resolve dept/user — fall back to seeded system defaults
        resolved_dept = dept_id or ids.get("dept_default")
        resolved_user = user_id or ids.get("user_default")
        session_id = str(uuid.uuid4())
        session = BatchSession(
            session_id=session_id,
            total=len(files),
            user_id=str(resolved_user),
            dept_id=str(resolved_dept),
            upload_type="user",
        )
        rsm.create_session(session)

        ingested_files = []

        for f in files:
            if not f.filename.lower().endswith(".pdf"):
                raise HTTPException(400, f"'{f.filename}' is not a PDF")

            contents = await f.read()

            # Magic byte check — reject non-PDF content regardless of filename
            if not contents.startswith(b"%PDF-"):
                raise HTTPException(
                    400,
                    f"'{f.filename}' does not appear to be a valid PDF file",
                )

            file_id = str(uuid.uuid4())
            fpath = UPLOAD_DIR / f"{file_id}_{f.filename}"
            fpath.write_bytes(contents)

            # Register upload in PostgreSQL (FK integrity before publishing job)
            upload_id = None
            try:
                upload_id = pipeline.rbac.register_user_upload(
                    user_id=str(resolved_user),
                    dept_id=str(resolved_dept),
                    file_name=f.filename,
                    file_path=str(fpath),
                    file_size_bytes=len(contents),
                    chat_id=None,
                    upload_scope="dept",
                )
            except Exception as e:
                logger.warning(f"Upload registration failed for '{f.filename}': {e}")

            fp = FileProgress(
                file_id=file_id,
                session_id=session_id,
                filename=f.filename,
                size_kb=len(contents) / 1024,
                started_at=time.time(),
            )
            rsm.register_file(session_id, fp)

            job = JobPayload(
                session_id=session_id,
                file_id=file_id,
                filename=f.filename,
                file_path=str(fpath),
                file_size_kb=len(contents) / 1024,
                user_id=str(resolved_user),
                dept_id=str(resolved_dept),
                upload_type="user",
                upload_id=upload_id,
            )

            from src.database.rabbitmq_broker import publish_job
            publish_job(job)

            ingested_files.append({
                "file_id":  file_id,
                "filename": f.filename,
                "size_kb":  round(len(contents) / 1024, 1),
            })

            logger.info(
                "Job queued — file=%s file_id=%s session=%s dept=%s",
                f.filename, file_id, session_id, resolved_dept,
            )

        return JSONResponse({
            "session_id": session_id,
            "dept_id":    str(resolved_dept),
            "files":      ingested_files,
        })

    # ── Progress (SSE) ────────────────────────────────────────────────────────

    @router.get("/ingest/progress/{session_id}")
    async def ingest_progress(session_id: str):
        """
        Server-Sent Events stream for real-time ingestion progress.

        Each event has the shape:
          data: {"type": "file_progress", "data": {
            "file_id": "uuid",
            "filename": "doc.pdf",
            "stage": "ocr" | "chunking" | "embedding" | "storing" | "done" | "error",
            "pct": 0-100,
            "chunks": 42,
            "doc_id": "uuid",    -- available once stored; use for PG queries
            "error": null
          }}

        The stream closes when all files in the session reach a terminal stage
        (done / error / skipped).
        """
        if not rsm or not rsm.ping():
            raise HTTPException(503, "Redis is offline")

        async def _event_stream():
            # Immediately emit current state for any already-progressed files
            summary = rsm.session_summary(session_id)
            if summary:
                for f in summary.get("files", []):
                    yield f"data: {json.dumps({'type': 'file_progress', 'data': f})}\n\n"

            # stop_event is set when the client disconnects, signalling the
            # Redis subscription thread to exit and release its connection.
            stop_event = threading.Event()
            q    = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def _subscribe():
                try:
                    for event in rsm.subscribe_session(session_id,
                                                       stop_event=stop_event):
                        loop.call_soon_threadsafe(q.put_nowait, event)
                except Exception as e:
                    logger.warning("SSE subscribe error: %s", e)
                finally:
                    loop.call_soon_threadsafe(q.put_nowait, None)

            thread = threading.Thread(target=_subscribe, daemon=True)
            thread.start()

            try:
                while True:
                    event = await q.get()
                    if event is None:
                        break
                    yield f"data: {json.dumps(event)}\n\n"
            finally:
                # Client disconnected or session complete — release Redis connection
                stop_event.set()

        return StreamingResponse(_event_stream(), media_type="text/event-stream")

    # ── SeaweedFS Storage Routes ───────────────────────────────────────────────

    @router.get("/storage/health")
    async def storage_health():
        if not pipeline.storage:
            return JSONResponse({"seaweedfs": "not configured"})
        return await pipeline.storage.health()

    @router.get("/storage/jobs/{job_id}/files")
    async def list_job_files(job_id: str):
        """List all SeaweedFS objects for a given job (raw PDF, extracted markdown)."""
        if not pipeline.storage:
            raise HTTPException(501, "Object storage not configured")
        return await pipeline.storage.list_job_files(job_id)

    @router.delete("/storage/jobs/{job_id}/files")
    async def delete_job_files(job_id: str):
        """Remove all SeaweedFS artefacts for a completed or failed job."""
        if not pipeline.storage:
            raise HTTPException(501, "Object storage not configured")
        deleted = await pipeline.storage.delete_job_artefacts(job_id)
        return {"deleted_count": deleted}

    @router.get("/storage/jobs/{job_id}/pdf-url")
    async def get_pdf_url(job_id: str, filename: str):
        """Return the SeaweedFS filer URL for the raw PDF of a job."""
        if not pipeline.storage:
            raise HTTPException(501, "Object storage not configured")
        return {"url": pipeline.storage.pdf_url(job_id, filename)}

    return router
