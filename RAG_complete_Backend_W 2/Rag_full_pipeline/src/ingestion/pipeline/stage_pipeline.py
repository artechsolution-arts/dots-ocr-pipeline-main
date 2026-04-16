"""
Stage Pipeline  —  M3 Ultra  (24 perf + 8 eff CPU cores, 80-core MPS GPU)
==========================================================================

┌─ Stage 1 ──────────────────────────────────────────────────┐
│  Preprocessing  · 6 CPU threads                            │
│  PDF bytes → OpenCV enhanced PIL pages (streamed)          │
└────────────────────────────┬───────────────────────────────┘
                             │ page_q  (maxsize=30)
┌─ Stage 2+3 ────────────────▼───────────────────────────────┐
│  OCR Workers  · 3 MPS threads  (each owns DotsOCRParser)   │
│  PIL page → model.generate() → raw JSON                    │
│  ↓ immediate CPU post-processing on same thread:           │
│    post_process_output()  →  cells dict                    │
│    layoutjson2md()        →  page markdown string          │
│  Why combined: avoids passing 26 MB images between queues  │
└────────────────────────────┬───────────────────────────────┘
                             │ markdown_q  (maxsize=200)
┌─ Stage 4 ──────────────────▼───────────────────────────────┐
│  Document Assembler  · 1 thread                            │
│  Collects all pages per doc, sorts, concatenates           │
│  Text cleaning (whitespace, markers)                       │
└────────────────────────────┬───────────────────────────────┘
                             │ assembled_q  (maxsize=50)
┌─ Stage 5 ──────────────────▼───────────────────────────────┐
│  Chunking  · 4 CPU threads                                 │
│  Markdown-aware, token-limited, overlap-preserving         │
└────────────────────────────┬───────────────────────────────┘
                             │ chunk_q  (individual ChunkItems, maxsize=2000)
┌─ Stage 6 ──────────────────▼───────────────────────────────┐
│  Embedding Batcher  · 1 MPS thread                         │
│  Accumulates chunks across documents                       │
│  One model.encode(batch=256) call per batch                │
│  → 256 GPU forward passes → 1 GPU forward pass             │
└────────────────────────────┬───────────────────────────────┘
                             │ store_q  (EmbeddedItems, maxsize=2000)
┌─ Stage 7 ──────────────────▼───────────────────────────────┐
│  Storage Writers  · 4 IO threads                           │
│  PostgreSQL: chunks + pgvector embeddings                  │
│  SeaweedFS: raw PDF + extracted markdown                   │
└────────────────────────────────────────────────────────────┘

Thread budget (M3 Ultra, 32 cores):
  Preprocessing   6
  OCR (MPS)       3   → GPU command queue always fed
  Assembler       1
  Chunking        4
  Embedding       1   → single batcher, 256-chunk batches on MPS
  Storage         4
  RabbitMQ feeder 3
  ─────────────── ──
  Total          22   → leaves 10 cores for macOS + Redis + RabbitMQ
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from queue import Empty, Queue
from typing import Optional

from src.config import cfg
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
EMBED_BATCH  = int(cfg.embedding_batch)   # 256 from env
EMBED_TIMEOUT = 1.5   # seconds — max wait before flushing a partial embed batch


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
        # maxsize caps in-flight work to bound peak memory usage
        self._doc_q       = Queue(maxsize=50)     # DocJobs from feeders
        self._page_q      = Queue(maxsize=30)     # PageJobs (carry PIL images — capped!)
        self._markdown_q  = Queue(maxsize=300)    # PageMarkdown (text only, cheap)
        self._assembled_q = Queue(maxsize=50)     # AssembledDoc
        self._chunk_q     = Queue(maxsize=2000)   # ChunkItem
        self._store_q     = Queue(maxsize=2000)   # EmbeddedItem

        # ── Shared state ──────────────────────────────────────────────────────
        self._shutdown    = threading.Event()
        # Per-file chunk completion tracking: {file_id: [received, total, doc_job]}
        self._chunk_counter: dict[str, list] = {}
        self._chunk_lock   = threading.Lock()

        # ── Worker threads ────────────────────────────────────────────────────
        self._threads: list[threading.Thread] = []

        # ── Lazy-loaded per-OCR-worker parsers and per-stage singletons ───────
        self._chunker  = DocumentChunker(
            chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap
        )
        self._cleaner  = TextCleaner()
        self._preprocessor = DocumentPreprocessor()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        spawns = [
            # Stage 1: Preprocessing (CPU)
            *[self._make("preprocess", self._preprocess_worker)
              for _ in range(N_PREPROCESS)],
            # Stage 2+3: OCR + Layout + Markdown (MPS)
            *[self._make(f"ocr-{i}", self._ocr_worker)
              for i in range(N_OCR)],
            # Stage 4: Document Assembler (1 thread)
            self._make("assembler", self._assembler_worker),
            # Stage 5: Chunking (CPU)
            *[self._make("chunk", self._chunk_worker) for _ in range(N_CHUNK)],
            # Stage 6: Embedding Batcher (MPS)
            self._make("embed-batcher", self._embed_batcher_worker),
            # Stage 7: Storage Writers (IO)
            *[self._make("store", self._store_worker) for _ in range(N_STORE)],
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
        # Unblock all queue.get() calls
        for _ in range(len(self._threads) * 2):
            for q in (self._doc_q, self._page_q, self._markdown_q,
                      self._assembled_q, self._chunk_q, self._store_q):
                try:
                    q.put_nowait(_STOP)
                except Exception:
                    pass
        for t in self._threads:
            t.join(timeout=timeout)
        logger.info("StagePipeline stopped")

    def submit(self, doc_job: DocJob):
        """Public: submit a document into the pipeline."""
        self._doc_q.put(doc_job)

    # ── Stage 1: Preprocessing ────────────────────────────────────────────────

    def _preprocess_worker(self):
        """
        Pulls DocJobs, converts each PDF page to an enhanced PIL image,
        and pushes PageJobs to the OCR queue.
        Streams pages so OCR workers can start before all pages are ready.
        """
        while not self._shutdown.is_set():
            item = self._get(self._doc_q)
            if item is None:
                break
            doc: DocJob = item
            try:
                self._update_stage(doc.file_id, doc.session_id, "preprocessing", 5)
                pages_yielded = 0
                for page_idx, total_pages, enhanced, origin in \
                        self._preprocessor.stream_pages(doc.raw_bytes):
                    self._page_q.put(PageJob(
                        file_id=doc.file_id, session_id=doc.session_id,
                        page_idx=page_idx, total_pages=total_pages,
                        image=enhanced, origin_image=origin, doc_job=doc,
                    ))
                    pages_yielded += 1
                logger.debug("[Preprocess] %s → %d pages queued", doc.filename, pages_yielded)
            except Exception as e:
                logger.error("[Preprocess] Failed '%s': %s", doc.filename, e, exc_info=True)
                self._fail(doc.file_id, doc.session_id, str(e))

    # ── Stage 2+3: OCR + Layout + Markdown (combined on GPU thread) ──────────

    def _ocr_worker(self):
        """
        Each OCR worker owns its own DotsOCRParser loaded onto MPS.
        GPU inference (model.generate) is followed immediately by fast CPU
        post-processing (post_process_output + layoutjson2md) on the same thread
        so we avoid passing large PIL images through queues.
        """
        parser = self._load_ocr_parser()
        if parser is None:
            logger.error("[OCR] Parser failed to load — worker exiting")
            return

        while not self._shutdown.is_set():
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
                markdown = ""   # Empty markdown for failed pages (assembler handles it)

            self._markdown_q.put(PageMarkdown(
                file_id=pjob.file_id, session_id=pjob.session_id,
                page_idx=pjob.page_idx, total_pages=pjob.total_pages,
                markdown=markdown, doc_job=pjob.doc_job,
                error=None if markdown else "ocr_failed",
            ))
            # Release image memory immediately
            del pjob.image, pjob.origin_image

    def _load_ocr_parser(self):
        """Load a DotsOCRParser onto MPS (one per OCR thread)."""
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
        """
        GPU: model.generate() → raw JSON string
        CPU: post_process_output() → cells → layoutjson2md() → markdown
        Returns markdown string for this page.
        """
        from dots_ocr.utils.prompts import dict_promptmode_to_prompt
        from dots_ocr.utils.layout_utils import post_process_output
        from dots_ocr.utils.format_transformer import layoutjson2md

        prompt_mode = cfg.dots_ocr_prompt_mode
        prompt = dict_promptmode_to_prompt[prompt_mode]

        # ── GPU: inference ──────────────────────────────────────────────────
        raw_response = parser._inference_with_hf(pjob.image, prompt)

        if not raw_response or not raw_response.strip():
            # Fallback: try pypdf text for this page (handled upstream via HybridOCR)
            return ""

        # ── CPU: layout parsing ─────────────────────────────────────────────
        try:
            cells, filtered = post_process_output(
                raw_response, prompt_mode,
                pjob.origin_image, pjob.image,
            )
        except Exception as e:
            logger.warning("[Layout] page %d parse failed: %s", pjob.page_idx, e)
            return raw_response   # Return raw text as fallback

        # ── CPU: markdown generation ────────────────────────────────────────
        try:
            markdown = layoutjson2md(pjob.origin_image, cells, text_key="text",
                                     no_page_hf=True)
            return markdown or ""
        except Exception as e:
            logger.warning("[Markdown] page %d generation failed: %s", pjob.page_idx, e)
            return ""

    # ── Stage 4: Document Assembler ───────────────────────────────────────────

    def _assembler_worker(self):
        """
        Accumulates PageMarkdowns per document until all pages arrive,
        then assembles the full markdown and pushes to chunking.
        """
        # {file_id: {"pages": {idx: md}, "total": N, "doc_job": DocJob, "session": str}}
        in_flight: dict[str, dict] = {}

        while not self._shutdown.is_set():
            item = self._get(self._markdown_q, timeout=0.5)
            if item is None:
                continue

            pm: PageMarkdown = item
            fid = pm.file_id

            # Initialise slot for new document
            if fid not in in_flight:
                in_flight[fid] = {
                    "pages":   {},
                    "total":   pm.total_pages,
                    "doc_job": pm.doc_job,
                    "session": pm.session_id,
                }
                self._update_stage(fid, pm.session_id, "assembling", 50)

            in_flight[fid]["pages"][pm.page_idx] = pm.markdown or ""

            # Check completion
            if len(in_flight[fid]["pages"]) < in_flight[fid]["total"]:
                continue

            # ── Complete: assemble document ───────────────────────────────
            slot     = in_flight.pop(fid)
            doc_job  = slot["doc_job"]
            session  = slot["session"]
            ordered  = [slot["pages"].get(i, "") for i in range(slot["total"])]
            raw_md   = "\n\n".join(p for p in ordered if p)

            # ── Text cleaning ─────────────────────────────────────────────
            clean_md = self._cleaner.clean(raw_md)
            if not clean_md.strip():
                # Pure fallback: pypdf extraction for entire document
                logger.warning("[Assembler] '%s' produced empty OCR text — using pypdf",
                               doc_job.filename)
                clean_md = self._pypdf_fallback(doc_job.raw_bytes)

            if not clean_md.strip():
                logger.error("[Assembler] '%s' — no text extracted", doc_job.filename)
                self._fail(fid, session, "No text extracted from document")
                continue

            content_hash = hashlib.sha256(doc_job.raw_bytes).hexdigest()

            # ── SeaweedFS: persist raw PDF + extracted markdown ───────────
            if self.storage:
                self._store_seaweed(doc_job, raw_md, clean_md)

            self._assembled_q.put(AssembledDoc(
                file_id=fid, session_id=session,
                markdown=clean_md, page_count=slot["total"],
                content_hash=content_hash, doc_job=doc_job,
            ))
            logger.info("[Assembler] '%s' assembled (%d pages, %d chars)",
                        doc_job.filename, slot["total"], len(clean_md))

    def _pypdf_fallback(self, pdf_bytes: bytes) -> str:
        try:
            from pypdf import PdfReader
            import io
            reader = PdfReader(io.BytesIO(pdf_bytes))
            return "\n\n".join(
                p.extract_text() or "" for p in reader.pages
            )
        except Exception:
            return ""

    def _store_seaweed(self, doc_job: DocJob, raw_md: str, clean_md: str):
        import asyncio, concurrent.futures
        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(
                    self.storage.store_uploaded_pdf(
                        doc_job.file_id, doc_job.filename, doc_job.raw_bytes))
                loop.run_until_complete(
                    self.storage.store_extracted_text(
                        doc_job.file_id, doc_job.filename,
                        {"raw": raw_md, "clean": clean_md}))
            finally:
                loop.close()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                ex.submit(_run).result(timeout=30)
        except Exception as e:
            logger.warning("[SeaweedFS] Store failed: %s", e)

    # ── Stage 5: Chunking ─────────────────────────────────────────────────────

    def _chunk_worker(self):
        """
        Splits assembled markdown into token-aware chunks.
        Registers the document in PostgreSQL before emitting chunks.
        """
        while not self._shutdown.is_set():
            item = self._get(self._assembled_q)
            if item is None:
                break
            adoc: AssembledDoc = item
            try:
                self._update_stage(adoc.file_id, adoc.session_id, "chunking", 65)

                # Dedup check
                if self.rbac:
                    existing = self.rbac.find_doc_by_hash(
                        adoc.content_hash, adoc.doc_job.dept_id)
                    if existing:
                        logger.info("[Chunk] '%s' duplicate — skipping",
                                    adoc.doc_job.filename)
                        self._update_stage(adoc.file_id, adoc.session_id,
                                           "done", 100, note="duplicate_skipped")
                        continue

                    # Register document in PostgreSQL
                    doc_id = self.rbac.create_document(
                        file_name=adoc.doc_job.filename,
                        file_path=adoc.doc_job.filename,
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

                # Track chunk completion for this file
                with self._chunk_lock:
                    self._chunk_counter[adoc.file_id] = {
                        "received": 0, "total": total,
                        "doc_id": doc_id, "doc_job": adoc.doc_job,
                        "session_id": adoc.session_id,
                    }

                for idx, chunk in enumerate(raw_chunks):
                    self._chunk_q.put(ChunkItem(
                        file_id=adoc.file_id, session_id=adoc.session_id,
                        chunk_idx=idx, total_chunks=total,
                        content=chunk["content"], metadata=chunk["metadata"],
                        content_hash=adoc.content_hash, doc_job=adoc.doc_job,
                    ))

                logger.debug("[Chunk] '%s' → %d chunks", adoc.doc_job.filename, total)
            except Exception as e:
                logger.error("[Chunk] '%s' failed: %s",
                             adoc.doc_job.filename, e, exc_info=True)
                self._fail(adoc.file_id, adoc.session_id, str(e))

    # ── Stage 6: Cross-Document Embedding Batcher ─────────────────────────────

    def _embed_batcher_worker(self):
        """
        Accumulates ChunkItems from multiple documents into a single batch.
        One model.encode(batch=256) call → all embeddings in one GPU pass.
        Flushes when: batch full (256) OR timeout (1.5s) — whichever first.
        """
        embedder = MxbaiEmbedder()
        batch: list[ChunkItem] = []
        deadline = time.monotonic() + EMBED_TIMEOUT

        self._update_global_stage("embedding")

        while not self._shutdown.is_set():
            # Pull with short timeout so we can flush on deadline
            remaining = max(0.05, deadline - time.monotonic())
            try:
                item = self._chunk_q.get(timeout=remaining)
                if item is _STOP:
                    break
                batch.append(item)
            except Empty:
                pass   # Deadline hit — flush whatever we have

            flush = len(batch) >= EMBED_BATCH or time.monotonic() >= deadline

            if flush and batch:
                self._flush_embed_batch(batch, embedder)
                batch = []
                deadline = time.monotonic() + EMBED_TIMEOUT

        # Drain remaining
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
            self._store_q.put(EmbeddedItem(chunk_item=chunk_item, embedding=vector))

        logger.debug("[Embed] Flushed batch of %d chunks", len(batch))

    # ── Stage 7: Storage Writers ──────────────────────────────────────────────

    def _store_worker(self):
        """
        Writes each chunk + embedding to PostgreSQL.
        Tracks per-file completion; marks file as done when last chunk stored.
        """
        while not self._shutdown.is_set():
            item = self._get(self._store_q)
            if item is None:
                break
            ei: EmbeddedItem = item
            ci: ChunkItem    = ei.chunk_item

            try:
                if self.rbac:
                    with self._chunk_lock:
                        counter = self._chunk_counter.get(ci.file_id)
                    if counter is None:
                        continue

                    doc_id  = counter["doc_id"]
                    u_id    = ci.doc_job.upload_id if ci.doc_job.upload_type == "user" else None
                    a_id    = ci.doc_job.upload_id if ci.doc_job.upload_type == "admin" else None

                    chunk_id = self.rbac.add_chunk(
                        doc_id=doc_id, chunk_index=ci.chunk_idx,
                        chunk_text=ci.content,
                        chunk_token_count=len(ci.content.split()),
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

                # ── Check if this file is fully stored ───────────────────
                first_chunk = False
                done = False
                with self._chunk_lock:
                    if ci.file_id in self._chunk_counter:
                        if self._chunk_counter[ci.file_id]["received"] == 0:
                            first_chunk = True
                        self._chunk_counter[ci.file_id]["received"] += 1
                        rec   = self._chunk_counter[ci.file_id]["received"]
                        total = self._chunk_counter[ci.file_id]["total"]
                        if rec >= total:
                            done = True
                            self._chunk_counter.pop(ci.file_id)

                if first_chunk:
                    self._update_stage(ci.file_id, ci.session_id, "storing", 85)

                if done:
                    if self.rbac:
                        self.rbac.update_document_status(doc_id, "completed")
                        if ci.doc_job.upload_id:
                            self.rbac.update_upload_status(
                                ci.doc_job.upload_id, ci.doc_job.upload_type, "completed")
                    self._update_stage(ci.file_id, ci.session_id, "done", 100)
                    logger.info("[Store] '%s' → fully stored (%d chunks)",
                                ci.doc_job.filename, ci.total_chunks)

            except Exception as e:
                logger.error("[Store] chunk %d of '%s' failed: %s",
                             ci.chunk_idx, ci.doc_job.filename, e, exc_info=True)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get(self, q: Queue, timeout: float = 2.0):
        """Get from queue; return None on timeout or shutdown sentinel."""
        while not self._shutdown.is_set():
            try:
                item = q.get(timeout=timeout)
                if item is _STOP:
                    return None
                return item
            except Empty:
                continue
        return None

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

    def _update_global_stage(self, stage: str):
        pass   # Global metrics hook — extend if needed

    @staticmethod
    def _mps_available() -> bool:
        try:
            import torch
            return torch.backends.mps.is_available()
        except Exception:
            return False
