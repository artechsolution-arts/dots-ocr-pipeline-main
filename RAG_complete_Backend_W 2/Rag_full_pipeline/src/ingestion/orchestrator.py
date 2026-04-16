import time
import logging
import uuid
import hashlib
from typing import Optional, Dict, Any, List, Callable

from src.config import cfg
from src.models.schemas import PDFDoc, FileProgress
from src.ingestion.ocr.ocr_engine import HybridOCR
from src.ingestion.parsing.text_cleaner import TextCleaner
from src.ingestion.chunking.chunker import DocumentChunker
from src.ingestion.embedding.embedder import MxbaiEmbedder
# Note: In the current Rag_full_pipeline, observability might be missing or different.
# I'll keep the imports but they might need checking if they fail.
try:
    from src.observability import tracer, metrics
except ImportError:
    # Fallback dummy for observability if not present
    class Dummy:
        def __getattr__(self, name): return self
        def __call__(self, *args, **kwargs): return self
        def __enter__(self): return self
        def __exit__(self, *args): pass
    tracer = metrics = Dummy()

logger = logging.getLogger(__name__)

def retry_with_backoff(retries: int = 3, backoff_in_seconds: int = 2):
    """Simple retry decorator for transient failures in OCR/Embedding."""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        raise e
                    sleep = (backoff_in_seconds * (2 ** x))
                    logger.warning(f"Retrying {func.__name__} in {sleep}s due to: {e}")
                    time.sleep(sleep)
                    x += 1
        return wrapper
    return decorator

class IngestionOrchestrator:
    """
    Coordinates the document ingestion pipeline:
    Validation -> OCR -> Parsing -> Chunking -> Embedding -> Indexing
    Hardened with distributed fencing and granular progress tracking.
    """
    def __init__(self, rsm=None, rbac=None, storage=None):
        self.rsm = rsm
        self.rbac = rbac
        self.storage = storage
        
        # DotsOCR — sole OCR engine (VLM layout parser, HuggingFace backend)
        dots_ip   = cfg.dots_ocr_ip
        dots_port = cfg.dots_ocr_port

        self.ocr_engine = HybridOCR(
            ip=dots_ip,
            port=dots_port,
            model_name=cfg.dots_ocr_model,          # HuggingFace model ID, NOT Ollama
            use_hf=cfg.dots_ocr_use_hf,
            weights_path=cfg.dots_ocr_weights_path,
        )
            
        self.cleaner = TextCleaner()
        self.chunker = DocumentChunker(chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)
        self.embedder = MxbaiEmbedder()

    @retry_with_backoff(retries=2)
    def _safe_ocr(self, raw_bytes: bytes, output_dir: str = None):
        return self.ocr_engine.extract_text(raw_bytes, output_dir=output_dir)

    @retry_with_backoff(retries=2)
    def _safe_embed(self, texts: List[str]):
        return self.embedder.embed_batch(texts)

    def run_ingestion(self, raw_bytes: bytes, filename: str, user_id: str, dept_id: str, 
                      file_id: str, session_id: str, upload_type: str = "user",
                      upload_id: str = None) -> FileProgress:
        """
        Executes the full modular ingestion pipeline with fencing and task sets.
        """
        t0 = time.time()
        metrics.increment_counter("jobs_total")
        fp = FileProgress(file_id=file_id, session_id=session_id, filename=filename, size_kb=len(raw_bytes)/1024)
        
        # 0. Distributed Fence Check (Prevent race conditions in distributed worker pool)
        if self.rsm and not self.rsm.set_fence(file_id, owner=f"orchestrator-{file_id}"):
            logger.warning(f"File {file_id} is already being processed by another worker. Skipping.")
            fp.stage = "skipped"
            # Push skip status to Redis so the UI progress updates
            self.rsm.update_stage(file_id, session_id, "skipped", 100)
            return fp

        def _update_stage(stage, pct, **extra):
            fp.stage, fp.pct = stage, pct
            if self.rsm:
                self.rsm.update_stage(file_id, session_id, stage, pct, extra=extra if extra else None)

        with tracer.span("ingestion_orchestrator", {"filename": filename, "file_id": file_id}):
            try:
                # 1. Validation
                _update_stage("validating", 10)
                fp.started_at = time.time()
                content_hash = hashlib.sha256(raw_bytes).hexdigest()
                
                # Check for existing document hash to skip duplicates if config allows
                if self.rbac:
                    existing_id = self.rbac.find_doc_by_hash(content_hash, dept_id)
                    if existing_id:
                        logger.info(f"Document with hash {content_hash} already exists (ID: {existing_id}). Skipping.")
                        _update_stage("done", 100, note="duplicate_skipped")
                        return fp

                # 2. OCR / Extraction
                # Use UPLOAD_DIR from config instead of hardcoded if possible
                from src.config import UPLOAD_DIR
                extraction_dir = str(UPLOAD_DIR / f"extracted_{file_id}")
                _update_stage("ocr", 30)
                raw_text = self._safe_ocr(raw_bytes, output_dir=extraction_dir)
                
                # Guard: empty OCR output means either model is down or PDF has no content
                if not raw_text or not raw_text.strip():
                    raise ValueError(
                        f"OCR returned empty text. Check DotsOCR model "
                        f"(model={cfg.dots_ocr_model}, use_hf={cfg.dots_ocr_use_hf}) "
                        f"and verify PDF readability."
                    )

                # 3. Parsing / Cleaning
                _update_stage("parsing", 45)
                clean_text = self.cleaner.clean(raw_text)
                
                # ▶ STORAGE: Persist to SeaweedFS if available
                if self.storage:
                    import concurrent.futures
                    def _store_seaweed():
                        import asyncio
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            loop.run_until_complete(self.storage.store_uploaded_pdf(file_id, filename, raw_bytes))
                            loop.run_until_complete(self.storage.store_extracted_text(file_id, filename, {"raw": raw_text, "clean": clean_text}))
                        finally:
                            loop.close()
                    try:
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                            ex.submit(_store_seaweed).result(timeout=30)
                        logger.info(f"Document {file_id} persisted to SeaweedFS.")
                    except Exception as se:
                        logger.warning(f"SeaweedFS persistence failed (restricted mode): {se}")
                
                # 4. Data Object Creation
                # Estimate page count from double-newline separated sections
                estimated_pages = max(1, clean_text.count('\n\n') // 3 + 1)
                doc = PDFDoc(
                    filename=filename, 
                    raw_content=raw_bytes, 
                    extracted_text=clean_text, 
                    page_count=estimated_pages,
                    content_hash=content_hash, 
                    department_id=dept_id, 
                    uploaded_by=user_id
                )
                
                # 5. Chunking
                _update_stage("chunking", 60)
                chunks_raw = self.chunker.chunk_document(clean_text)
                fp.chunks = len(chunks_raw)
                
                # Initialize granular task tracking in Redis
                if self.rsm:
                    self.rsm.set_taskset(file_id, fp.chunks)

                # 6. Embedding
                _update_stage("embedding", 85)
                texts = [c["content"] for c in chunks_raw]
                embeddings = self._safe_embed(texts)
                
                # 7. Indexing / Storing (RBAC & Vector Store)
                _update_stage("storing", 95)
                if self.rbac:
                    # Determine source upload ID type (use registered upload_id, NOT file_id)
                    u_id = upload_id if upload_type == "user" else None
                    a_id = upload_id if upload_type == "admin" else None
                    
                    # a. Register the document metadata
                    # Note: Arguments might differ slightly in different project versions.
                    # I'll rely on the schema expected by the IngestionOrchestrator.
                    doc_id = self.rbac.create_document(
                        file_name=filename, 
                        file_path=filename, 
                        dept_id=dept_id, 
                        uploaded_by=user_id,
                        content_hash=content_hash,
                        page_count=doc.page_count,
                        ocr_used=True,
                        source_user_upload_id=u_id,
                        source_admin_upload_id=a_id
                    )
                    fp.doc_id = doc_id
                    
                    # b. Store chunks and embeddings
                    for i, (chunk_raw, embedding) in enumerate(zip(chunks_raw, embeddings)):
                        chunk_id = self.rbac.add_chunk(
                            doc_id=doc_id,
                            chunk_index=i,
                            chunk_text=chunk_raw["content"],
                            chunk_token_count=len(chunk_raw["content"].split()),
                            page_num=chunk_raw["metadata"].get("page", 0),
                            source_user_upload_id=u_id,
                            source_admin_upload_id=a_id
                        )
                        # Store vector  (keyword names must match RBACManager.store_embedding signature)
                        self.rbac.store_embedding(
                            chunk_id=chunk_id,
                            dept_id=dept_id,
                            embedding=embedding,
                            source_user_upload_id=u_id,
                            source_admin_upload_id=a_id
                        )
                        # Update granular progress
                        if self.rsm:
                            self.rsm.update_task_status(file_id, i, "completed")
                    
                    # c. Mark document as completed
                    self.rbac.update_document_status(doc_id, "completed")
                    if upload_id:
                        self.rbac.update_upload_status(upload_id, upload_type, "completed")

                # Finalize
                fp.finished_at = time.time()
                _update_stage("done", 100)
                metrics.increment_counter("jobs_success")
                metrics.observe_hist("processing_latency_seconds", time.time() - t0)
                
                return fp

            except Exception as e:
                metrics.increment_counter("jobs_failed")
                logger.error(f"Orchestration Error for {filename}: {e}", exc_info=True)
                _update_stage("error", 0, error=str(e))
                return fp
            finally:
                # Always clear the fence after processing (success or failure)
                if self.rsm:
                    self.rsm.clear_fence(file_id)
