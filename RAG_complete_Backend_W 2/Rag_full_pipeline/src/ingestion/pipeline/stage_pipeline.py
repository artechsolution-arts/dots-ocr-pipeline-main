"""
Stage Pipeline  —  M3 Ultra  (24 perf + 8 eff CPU cores, 80-core MPS GPU)
==========================================================================

┌─ Stage 1 ──────────────────────────────────────────────────┐
│  Preprocessing  · 6 CPU threads                            │
│  Reads PDF from disk → SHA-256 → dedup check → pages      │
│  Skips immediately if doc already in Postgres or Redis     │
└────────────────────────────┬───────────────────────────────┘
                             │ page_q  (maxsize=60)
┌─ Stage 2+3 ────────────────▼───────────────────────────────┐
│  OCR Workers  · 3 MPS threads  (each owns DotsOCRParser)   │
│  PIL page → model.generate() → raw JSON                    │
│  ↓ immediate CPU post-processing on same thread:           │
│    post_process_output()  →  cells dict                    │
│    layoutjson2md()        →  page markdown string          │
└────────────────────────────┬───────────────────────────────┘
                             │ markdown_q  (maxsize=300)
┌─ Stage 4 ──────────────────▼───────────────────────────────┐
│  Document Assembler  · 1 thread                            │
│  Collects all pages per doc, sorts, concatenates           │
│  Text cleaning (whitespace, markers)                       │
│  SeaweedFS upload runs in a daemon thread (non-blocking)   │
└────────────────────────────┬───────────────────────────────┘
                             │ assembled_q  (maxsize=50)
┌─ Stage 5 ──────────────────▼───────────────────────────────┐
│  Chunking  · 4 CPU threads                                 │
│  Markdown-aware, token-limited, overlap-preserving         │
│  Emits accurate tiktoken token count per chunk             │
└────────────────────────────┬───────────────────────────────┘
                             │ chunk_q  (individual ChunkItems, maxsize=2000)
┌─ Stage 6 ──────────────────▼───────────────────────────────┐
│  Embedding Batcher  · 1 MPS thread                         │
│  Accumulates chunks across documents                       │
│  One model.encode(batch) call per batch                    │
└────────────────────────────┬───────────────────────────────┘
                             │ store_q  (EmbeddedItems, maxsize=2000)
┌─ Stage 7 ──────────────────▼───────────────────────────────┐
│  Storage Writers  · 4 IO threads                           │
│  PostgreSQL: chunks + pgvector embeddings                  │
│  Sets Redis 1-year dedup key after successful storage      │
│  Deletes the local upload file after successful storage    │
└────────────────────────────────────────────────────────────┘

Memory model for 500+ documents
--------------------------------
DocJob carries only a file_path (not raw bytes).  _doc_q can hold thousands
of entries without filling RAM.  Each preprocess worker reads its file from
disk only when it dequeues a job — at most N_PREPROCESS files are in memory
simultaneously.  PIL images are bounded by _page_q.maxsize=60.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Optional

from src.config import cfg, UPLOAD_DIR
from src.ingestion.chunking.chunker import DocumentChunker
from src.ingestion.embedding.embedder import MxbaiEmbedder
from src.ingestion.parsing.text_cleaner import TextCleaner
from src.ingestion.pipeline.datatypes import (
    AssembledDoc, ChunkItem, DocJob, EmbeddedItem,
    PageJob, PageMarkdown, PageOCRResult,
)
from src.ingestion.preprocessing.preprocessor import DocumentPreprocessor

logger = logging.getLogger(__name__)

# ── Sentinel ──────────────────────────────────────────────────────────────────
_STOP = object()

# ── Stage thread counts (tuned for M3 Ultra) ─────────────────────────────────
N_PREPROCESS = 6
N_OCR        = 3
N_CHUNK      = 4
N_STORE      = 4
EMBED_BATCH  = int(cfg.embedding_batch)   # default 128
EMBED_TIMEOUT = 3.0   # seconds — wider window improves GPU batch fill under OCR backpressure

# Max time a document may spend in the assembler waiting for all its pages.
# Prevents in-flight table from leaking on GPU hang / worker crash.
DOC_ASSEMBLY_TIMEOUT = 600   # 10 minutes


class StagePipeline:
    """
    Shared pipeline instance.  Workers submit DocJobs; stages do the rest.
    Thread-safe: multiple RabbitMQ feeder threads can call submit() concurrently.
    """

    def __init__(self, rsm, rbac, storage=None):
        self.rsm     = rsm
        self.rbac    = rbac
        self.storage = storage

        # ── Inter-stage queues ────────────────────────────────────────────────
        # _doc_q holds only file_path metadata (tiny), so maxsize=5000 is safe
        # even for 500+ concurrent uploads — no raw bytes are stored here.
        self._doc_q       = Queue(maxsize=5000)   # DocJobs (file path only — no bytes)
        self._page_q      = Queue(maxsize=60)     # PageJobs (PIL images — capped!)
        self._markdown_q  = Queue(maxsize=300)    # PageMarkdown (text only)
        self._assembled_q = Queue(maxsize=50)     # AssembledDoc
        self._chunk_q     = Queue(maxsize=2000)   # ChunkItem
        self._store_q     = Queue(maxsize=2000)   # EmbeddedItem

        # ── Shared state ──────────────────────────────────────────────────────
        self._shutdown    = threading.Event()
        # Per-file chunk completion tracking: {file_id: {received, total, doc_id, ...}}
        self._chunk_counter: dict[str, dict] = {}
        self._chunk_lock   = threading.Lock()

        # ── Worker threads ────────────────────────────────────────────────────
        self._threads: list[threading.Thread] = []

        # ── Shared singletons ─────────────────────────────────────────────────
        self._chunker  = DocumentChunker(
            chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap
        )
        self._cleaner  = TextCleaner()
        self._preprocessor = DocumentPreprocessor()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        spawns = [
            *[self._make("preprocess", self._preprocess_worker) for _ in range(N_PREPROCESS)],
            *[self._make(f"ocr-{i}",  self._ocr_worker)        for i in range(N_OCR)],
            self._make("assembler",   self._assembler_worker),
            *[self._make("chunk",     self._chunk_worker)       for _ in range(N_CHUNK)],
            self._make("embed-batcher", self._embed_batcher_worker),
            *[self._make("store",     self._store_worker)       for _ in range(N_STORE)],
        ]
        for t in spawns:
            t.start()
            self._threads.append(t)
        logger.info(
            "StagePipeline started — %d threads "
            "(preprocess=%d ocr=%d assembler=1 chunk=%d embed=1 store=%d)",
            len(self._threads), N_PREPROCESS, N_OCR, N_CHUNK, N_STORE,
        )

    def stop(self, timeout: float = 30.0):
        logger.info("StagePipeline shutting down…")
        self._shutdown.set()
        # Inject sentinels so threads blocked on _get() wake immediately.
        for _ in range(len(self._threads) * 2):
            for q in (self._doc_q, self._page_q, self._markdown_q,
                      self._assembled_q, self._chunk_q, self._store_q):
                try:
                    q.put_nowait(_STOP)
                except Full:
                    pass
        for t in self._threads:
            t.join(timeout=timeout)
        logger.info("StagePipeline stopped")

    def submit(self, doc_job: DocJob):
        """Submit a document into the pipeline. Returns as soon as the job is queued."""
        self._put(self._doc_q, doc_job)

    # ── Stage 1: Preprocessing ────────────────────────────────────────────────

    def _preprocess_worker(self):
        """
        1. Reads PDF bytes from disk (DocJob carries only file_path).
        2. Computes SHA-256 → dedup check (Redis fast-path, then Postgres).
           Skips the entire pipeline if the document already exists — no OCR.
        3. Streams PDF pages as enhanced PIL images into _page_q.

        At most N_PREPROCESS files are held in memory simultaneously regardless
        of how many jobs are queued in _doc_q.
        """
        while not self._shutdown.is_set():
            item = self._get(self._doc_q)
            if item is None:
                break
            doc: DocJob = item
            try:
                # ── Read bytes from disk ──────────────────────────────────────
                try:
                    raw_bytes = Path(doc.file_path).read_bytes()
                except OSError as e:
                    logger.error("[Preprocess] Cannot read '%s': %s", doc.filename, e)
                    self._fail(doc.file_id, doc.session_id, f"File read error: {e}")
                    continue

                # ── SHA-256 ───────────────────────────────────────────────────
                doc.content_hash = hashlib.sha256(raw_bytes).hexdigest()

                # ── Dedup: Redis fast-path (1-year cache set after completion) ─
                if self.rsm:
                    existing_id = self.rsm.check_dedup(doc.content_hash)
                    if existing_id:
                        logger.info("[Preprocess] '%s' already processed (cached) — skipped",
                                    doc.filename)
                        self._skip(doc.file_id, doc.session_id, doc_id=existing_id)
                        del raw_bytes
                        continue

                # ── Dedup: Postgres authoritative check ───────────────────────
                if self.rbac:
                    existing_doc_id = self.rbac.find_doc_by_hash(
                        doc.content_hash, doc.dept_id)
                    if existing_doc_id:
                        logger.info(
                            "[Preprocess] '%s' already exists in Postgres (doc=%s) — skipped",
                            doc.filename, existing_doc_id)
                        # Prime Redis cache for future fast-path hits
                        if self.rsm:
                            self.rsm.set_dedup(doc.content_hash, existing_doc_id)
                        self._skip(doc.file_id, doc.session_id, doc_id=existing_doc_id)
                        del raw_bytes
                        continue

                # ── New document — start pipeline ─────────────────────────────
                self._update_stage(doc.file_id, doc.session_id, "preprocessing", 5,
                                   started_at=time.time())

                pages_yielded = 0
                for page_idx, total_pages, enhanced, origin in \
                        self._preprocessor.stream_pages(raw_bytes):
                    page_job = PageJob(
                        file_id=doc.file_id, session_id=doc.session_id,
                        page_idx=page_idx, total_pages=total_pages,
                        image=enhanced, origin_image=origin, doc_job=doc,
                    )
                    if not self._put(self._page_q, page_job):
                        break   # Shutdown signalled during put
                    pages_yielded += 1

                # Free bytes — pages have been converted to PIL images in _page_q
                del raw_bytes
                logger.debug("[Preprocess] '%s' → %d pages queued", doc.filename, pages_yielded)

            except Exception as e:
                logger.error("[Preprocess] Failed '%s': %s", doc.filename, e, exc_info=True)
                self._fail(doc.file_id, doc.session_id, str(e))

    # ── Stage 2+3: OCR + Layout + Markdown (combined on GPU thread) ──────────

    def _ocr_worker(self):
        """
        Each OCR worker owns its own DotsOCRParser loaded onto MPS.
        If the parser fails to load, the worker retries every 30 s — preserving
        full OCR capacity once the GPU recovers.
        """
        parser = None
        while not self._shutdown.is_set():
            if parser is None:
                parser = self._load_ocr_parser()
                if parser is None:
                    logger.error("[OCR] Parser load failed — retrying in 30 s")
                    for _ in range(60):
                        if self._shutdown.is_set():
                            return
                        time.sleep(0.5)
                    continue

            item = self._get(self._page_q)
            if item is None:
                break
            pjob: PageJob = item
            try:
                self._update_stage(pjob.file_id, pjob.session_id, "ocr", 30)
                markdown = self._run_ocr_page(parser, pjob)
            except Exception as e:
                logger.error("[OCR] page %d of '%s' failed: %s",
                             pjob.page_idx, pjob.doc_job.filename, e)
                markdown = ""

            pm = PageMarkdown(
                file_id=pjob.file_id, session_id=pjob.session_id,
                page_idx=pjob.page_idx, total_pages=pjob.total_pages,
                markdown=markdown, doc_job=pjob.doc_job,
                error=None if markdown else "ocr_failed",
            )
            del pjob.image, pjob.origin_image   # Free PIL image memory
            self._put(self._markdown_q, pm)

    def _load_ocr_parser(self):
        try:
            from dots_ocr import DotsOCRParser
            parser = DotsOCRParser(
                ip=cfg.dots_ocr_ip, port=cfg.dots_ocr_port,
                model_name=cfg.dots_ocr_weights_path,
                use_hf=True, num_thread=1,
            )
            logger.info("[OCR] Parser loaded on %s",
                        "MPS" if self._mps_available() else "CPU")
            return parser
        except Exception as e:
            logger.error("[OCR] Parser load failed: %s", e, exc_info=True)
            return None

    def _run_ocr_page(self, parser, pjob: PageJob) -> str:
        from dots_ocr.utils.prompts import dict_promptmode_to_prompt
        from dots_ocr.utils.layout_utils import post_process_output
        from dots_ocr.utils.format_transformer import layoutjson2md

        prompt_mode = cfg.dots_ocr_prompt_mode
        prompt = dict_promptmode_to_prompt[prompt_mode]

        raw_response = parser._inference_with_hf(pjob.image, prompt)
        if not raw_response or not raw_response.strip():
            return ""

        try:
            cells, filtered = post_process_output(
                raw_response, prompt_mode,
                pjob.origin_image, pjob.image,
            )
        except Exception as e:
            logger.warning("[Layout] page %d parse failed: %s", pjob.page_idx, e)
            return raw_response

        try:
            return layoutjson2md(pjob.origin_image, cells, text_key="text",
                                 no_page_hf=True) or ""
        except Exception as e:
            logger.warning("[Markdown] page %d generation failed: %s", pjob.page_idx, e)
            return ""

    # ── Stage 4: Document Assembler ───────────────────────────────────────────

    def _assembler_worker(self):
        """
        Accumulates PageMarkdowns per document until all pages arrive, then
        assembles the full markdown and pushes to chunking.

        SeaweedFS upload is fired in a background daemon thread after reading
        the PDF bytes from disk synchronously — the assembler never blocks on
        network I/O.

        A per-document 10-minute timeout fails stalled documents so the
        in-flight table never leaks on GPU hang or worker crash.
        """
        in_flight: dict[str, dict] = {}
        last_timeout_check = time.monotonic()

        while not self._shutdown.is_set():
            item = self._get(self._markdown_q, timeout=0.5)

            # Periodic timeout sweep every 30 s
            now = time.monotonic()
            if now - last_timeout_check > 30:
                last_timeout_check = now
                for fid in list(in_flight):
                    age = now - in_flight[fid]["started_at"]
                    if age > DOC_ASSEMBLY_TIMEOUT:
                        slot = in_flight.pop(fid)
                        logger.error(
                            "[Assembler] '%s' timed out after %.0fs (%d/%d pages received)",
                            slot["doc_job"].filename, age,
                            len(slot["pages"]), slot["total"],
                        )
                        self._fail(fid, slot["session"],
                                   "Assembly timed out: incomplete OCR results")

            if item is None:
                continue

            pm: PageMarkdown = item
            fid = pm.file_id

            if fid not in in_flight:
                in_flight[fid] = {
                    "pages":      {},
                    "total":      pm.total_pages,
                    "doc_job":    pm.doc_job,
                    "session":    pm.session_id,
                    "started_at": time.monotonic(),
                }
                self._update_stage(fid, pm.session_id, "assembling", 50)

            in_flight[fid]["pages"][pm.page_idx] = pm.markdown or ""

            if len(in_flight[fid]["pages"]) < in_flight[fid]["total"]:
                continue

            # ── All pages received — assemble document ────────────────────
            slot    = in_flight.pop(fid)
            doc_job = slot["doc_job"]
            session = slot["session"]
            ordered = [slot["pages"].get(i, "") for i in range(slot["total"])]
            raw_md  = "\n\n".join(p for p in ordered if p)

            clean_md = self._cleaner.clean(raw_md)
            if not clean_md.strip():
                logger.warning("[Assembler] '%s' produced empty OCR — using pypdf fallback",
                               doc_job.filename)
                clean_md = self._pypdf_fallback(doc_job.file_path)

            if not clean_md.strip():
                logger.error("[Assembler] '%s' — no text extracted", doc_job.filename)
                self._fail(fid, session, "No text extracted from document")
                continue

            # SeaweedFS upload: read bytes from disk (fast local read), then
            # upload asynchronously in a daemon thread so the assembler is never
            # blocked on network I/O.
            if self.storage:
                self._store_seaweed_async(doc_job, raw_md, clean_md)

            self._put(self._assembled_q, AssembledDoc(
                file_id=fid, session_id=session,
                markdown=clean_md, page_count=slot["total"],
                content_hash=doc_job.content_hash,
                doc_job=doc_job,
            ))
            logger.info("[Assembler] '%s' assembled (%d pages, %d chars)",
                        doc_job.filename, slot["total"], len(clean_md))

    def _pypdf_fallback(self, file_path: str) -> str:
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            return "\n\n".join(p.extract_text() or "" for p in reader.pages)
        except Exception:
            return ""

    def _store_seaweed_async(self, doc_job: DocJob, raw_md: str, clean_md: str):
        """
        Read the PDF bytes from disk immediately (synchronous, fast local I/O),
        then upload to SeaweedFS in a daemon thread.  Reading synchronously here
        ensures the bytes are captured before the store_worker deletes the file.
        """
        import asyncio
        try:
            raw_bytes = Path(doc_job.file_path).read_bytes()
        except Exception as e:
            logger.warning("[SeaweedFS] Cannot read '%s' for upload: %s",
                           doc_job.filename, e)
            return

        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(
                    self.storage.store_uploaded_pdf(
                        doc_job.file_id, doc_job.filename, raw_bytes))
                loop.run_until_complete(
                    self.storage.store_extracted_text(
                        doc_job.file_id, doc_job.filename,
                        {"raw": raw_md, "clean": clean_md}))
            except Exception as e:
                logger.warning("[SeaweedFS] Async store failed for '%s': %s",
                               doc_job.filename, e)
            finally:
                loop.close()

        threading.Thread(target=_run, daemon=True,
                         name=f"seaweed-{doc_job.file_id[:8]}").start()

    # ── Stage 5: Chunking ─────────────────────────────────────────────────────

    def _chunk_worker(self):
        """
        Creates a Postgres document record then emits ChunkItems.
        Dedup is handled upstream in the preprocess worker — by the time a
        document reaches here it is guaranteed to be new.
        """
        while not self._shutdown.is_set():
            item = self._get(self._assembled_q)
            if item is None:
                break
            adoc: AssembledDoc = item
            try:
                self._update_stage(adoc.file_id, adoc.session_id, "chunking", 65)

                if self.rbac:
                    doc_id = self.rbac.create_document(
                        file_name=adoc.doc_job.filename,
                        file_path=adoc.doc_job.file_path,
                        dept_id=adoc.doc_job.dept_id,
                        uploaded_by=adoc.doc_job.user_id,
                        content_hash=adoc.content_hash,
                        page_count=adoc.page_count,
                        ocr_used=True,
                        source_user_upload_id=adoc.doc_job.upload_id
                            if adoc.doc_job.upload_type == "user" else None,
                        source_admin_upload_id=adoc.doc_job.upload_id
                            if adoc.doc_job.upload_type == "admin" else None,
                    )
                else:
                    doc_id = adoc.file_id

                raw_chunks = self._chunker.chunk_document(adoc.markdown)
                total = len(raw_chunks)

                with self._chunk_lock:
                    self._chunk_counter[adoc.file_id] = {
                        "received":   0,
                        "total":      total,
                        "doc_id":     doc_id,
                        "doc_job":    adoc.doc_job,
                        "session_id": adoc.session_id,
                        "content_hash": adoc.content_hash,
                    }

                for idx, chunk in enumerate(raw_chunks):
                    token_count = self._chunker.count_tokens(chunk["content"])
                    ci = ChunkItem(
                        file_id=adoc.file_id, session_id=adoc.session_id,
                        chunk_idx=idx, total_chunks=total,
                        content=chunk["content"], metadata=chunk["metadata"],
                        content_hash=adoc.content_hash, doc_job=adoc.doc_job,
                        token_count=token_count,
                    )
                    if not self._put(self._chunk_q, ci):
                        break

                logger.debug("[Chunk] '%s' → %d chunks", adoc.doc_job.filename, total)
            except Exception as e:
                logger.error("[Chunk] '%s' failed: %s",
                             adoc.doc_job.filename, e, exc_info=True)
                self._fail(adoc.file_id, adoc.session_id, str(e))

    # ── Stage 6: Cross-Document Embedding Batcher ─────────────────────────────

    def _embed_batcher_worker(self):
        embedder = MxbaiEmbedder()
        batch: list[ChunkItem] = []
        deadline = time.monotonic() + EMBED_TIMEOUT

        while not self._shutdown.is_set():
            remaining = max(0.05, deadline - time.monotonic())
            try:
                item = self._chunk_q.get(timeout=remaining)
                if item is _STOP:
                    break
                batch.append(item)
            except Empty:
                pass

            flush = len(batch) >= EMBED_BATCH or time.monotonic() >= deadline
            if flush and batch:
                self._flush_embed_batch(batch, embedder)
                batch = []
                deadline = time.monotonic() + EMBED_TIMEOUT

        if batch:
            self._flush_embed_batch(batch, embedder)

    def _flush_embed_batch(self, batch: list[ChunkItem], embedder: MxbaiEmbedder):
        texts = [c.content for c in batch]
        try:
            vectors = embedder.embed_batch(texts)
        except Exception as e:
            logger.error("[Embed] Batch failed: %s", e)
            vectors = [[0.0] * cfg.embedding_dim] * len(batch)

        seen_files: set = set()
        for chunk_item, vector in zip(batch, vectors):
            if chunk_item.file_id not in seen_files:
                self._update_stage(chunk_item.file_id, chunk_item.session_id, "embedding", 75)
                seen_files.add(chunk_item.file_id)
            self._put(self._store_q, EmbeddedItem(chunk_item=chunk_item, embedding=vector))

        logger.debug("[Embed] Flushed batch of %d chunks", len(batch))

    # ── Stage 7: Storage Writers ──────────────────────────────────────────────

    def _store_worker(self):
        """
        Writes each chunk + embedding to PostgreSQL.
        Single lock acquisition atomically reads doc_id, detects first-chunk,
        increments received, and detects completion — no TOCTOU race.

        After a document is fully stored:
        - Sets a 1-year Redis dedup key (fast-path for future re-uploads)
        - Deletes the local upload file from disk
        """
        while not self._shutdown.is_set():
            item = self._get(self._store_q)
            if item is None:
                break
            ei: EmbeddedItem = item
            ci: ChunkItem    = ei.chunk_item

            # ── Atomically claim this chunk and detect state transitions ──────
            doc_id        = None
            first_chunk   = False
            done          = False
            content_hash  = ""
            doc_job_ref   = None
            total_chunks  = 0

            with self._chunk_lock:
                counter = self._chunk_counter.get(ci.file_id)
                if counter is None:
                    continue   # Counter already removed — doc completed by another thread
                doc_id       = counter["doc_id"]
                first_chunk  = counter["received"] == 0
                content_hash = counter.get("content_hash", "")
                doc_job_ref  = counter["doc_job"]
                counter["received"] += 1
                rec   = counter["received"]
                total_chunks = counter["total"]
                done  = rec >= total_chunks
                if done:
                    self._chunk_counter.pop(ci.file_id)

            if first_chunk:
                self._update_stage(ci.file_id, ci.session_id, "storing", 85)

            u_id = ci.doc_job.upload_id if ci.doc_job.upload_type == "user"  else None
            a_id = ci.doc_job.upload_id if ci.doc_job.upload_type == "admin" else None

            try:
                if self.rbac:
                    chunk_id = self.rbac.add_chunk(
                        doc_id=doc_id, chunk_index=ci.chunk_idx,
                        chunk_text=ci.content,
                        chunk_token_count=ci.token_count,
                        page_num=ci.metadata.get("page", 0),
                        source_user_upload_id=u_id,
                        source_admin_upload_id=a_id,
                    )
                    self.rbac.store_embedding(
                        chunk_id=chunk_id, dept_id=ci.doc_job.dept_id,
                        embedding=ei.embedding,
                        source_user_upload_id=u_id,
                        source_admin_upload_id=a_id,
                    )
            except Exception as e:
                logger.error("[Store] chunk %d of '%s' failed: %s",
                             ci.chunk_idx, ci.doc_job.filename, e, exc_info=True)

            if done:
                if self.rbac:
                    try:
                        self.rbac.update_document_status(doc_id, "completed")
                        if ci.doc_job.upload_id:
                            self.rbac.update_upload_status(
                                ci.doc_job.upload_id, ci.doc_job.upload_type, "completed")
                    except Exception as e:
                        logger.warning("[Store] Status update failed for '%s': %s",
                                       ci.doc_job.filename, e)

                # Set 1-year Redis dedup key — future re-uploads hit the Redis
                # fast-path in the preprocess worker without touching Postgres.
                if self.rsm and content_hash:
                    try:
                        self.rsm.set_dedup(content_hash, doc_id)
                    except Exception:
                        pass

                self._update_stage(ci.file_id, ci.session_id, "done", 100)
                logger.info("[Store] '%s' fully stored (%d chunks)",
                            ci.doc_job.filename, total_chunks)

                # Delete the local upload file — processed and optionally in SeaweedFS.
                file_path = doc_job_ref.file_path if doc_job_ref else None
                if file_path:
                    try:
                        fpath = Path(file_path)
                        if fpath.exists():
                            fpath.unlink()
                            logger.debug("[Store] Deleted upload file: %s", fpath)
                    except Exception as e:
                        logger.warning("[Store] Could not delete '%s': %s", file_path, e)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get(self, q: Queue, timeout: float = 2.0):
        """Get from queue; return None on shutdown sentinel or shutdown event."""
        while not self._shutdown.is_set():
            try:
                item = q.get(timeout=timeout)
                if item is _STOP:
                    return None
                return item
            except Empty:
                continue
        return None

    def _put(self, q: Queue, item, timeout: float = 1.0) -> bool:
        """
        Shutdown-aware put.  Loops until the item is enqueued or shutdown fires.
        Returns True if enqueued, False if shutdown was signalled.
        Prevents any stage thread from blocking indefinitely on a full downstream
        queue, which would deadlock shutdown and mask backpressure.
        """
        while not self._shutdown.is_set():
            try:
                q.put(item, timeout=timeout)
                return True
            except Full:
                continue
        return False

    def _make(self, name: str, target) -> threading.Thread:
        return threading.Thread(target=target, name=name, daemon=True)

    def _update_stage(self, file_id: str, session_id: str,
                      stage: str, pct: int, **extra):
        if self.rsm:
            try:
                self.rsm.update_stage(file_id, session_id, stage, pct,
                                      extra=extra or None)
            except Exception:
                pass

    def _fail(self, file_id: str, session_id: str, error: str):
        self._update_stage(file_id, session_id, "error", 0, error=error)
        if self.rsm:
            try:
                self.rsm.incr_stat("total_failed")
            except Exception:
                pass

    def _skip(self, file_id: str, session_id: str, doc_id: str = ""):
        """Mark a document as skipped (duplicate) without processing it."""
        self._update_stage(file_id, session_id, "skipped", 100, doc_id=doc_id)
        if self.rsm:
            try:
                self.rsm.incr_stat("total_skipped")
            except Exception:
                pass

    @staticmethod
    def _mps_available() -> bool:
        try:
            import torch
            return torch.backends.mps.is_available()
        except Exception:
            return False
