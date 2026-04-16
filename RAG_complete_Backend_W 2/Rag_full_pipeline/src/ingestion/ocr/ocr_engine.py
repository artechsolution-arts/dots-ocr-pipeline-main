"""
DotsOCREngine
Single-engine OCR using DotsOCR (VLM-based layout parser).
Falls back to pypdf plain-text extraction when DotsOCR is unavailable.
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from src.config import cfg

# ── DotsOCR (Primary / Only Engine) ──────────────────────────────────────────
try:
    from dots_ocr import DotsOCRParser
    HAS_DOTSOCR = True
except ImportError:
    HAS_DOTSOCR = False

# ── pypdf — lightweight pure-Python fallback (no Paddle, no Tesseract) ────────
try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

logger = logging.getLogger(__name__)


class HybridOCR:
    """
    DotsOCR-only text extraction engine.

    Strategy
    ────────
    1. Run DotsOCR (VLM layout parser) → rich markdown output.
    2. If DotsOCR is unavailable or returns < 50 chars, fall back to pypdf
       plain-text extraction so the pipeline never stalls completely.
    3. No PaddleOCR dependency whatsoever.
    """

    def __init__(
        self,
        ip: str = "localhost",
        port: int = 8001,
        model_name: str = "rednote-hilab/dots.ocr",
        use_hf: bool = True,
        weights_path: str = "./weights/DotsOCR",
    ):
        self.model_name = model_name
        self.use_hf = use_hf
        self.weights_path = weights_path

        # ── Init DotsOCR ──────────────────────────────────────────────────────
        if HAS_DOTSOCR:
            try:
                logger.info(
                    "Initializing DotsOCR — model=%s, use_hf=%s, weights=%s, ip=%s:%d",
                    model_name, use_hf, weights_path, ip, port,
                )
                # In HF mode, we pass the local weights path as the 'model_name' 
                # so the parser can find the folder on disk.
                init_model = weights_path if use_hf else model_name
                
                self.dots_parser = DotsOCRParser(
                    ip=ip,
                    port=port,
                    model_name=init_model,
                    use_hf=use_hf,
                    num_thread=1 if use_hf else 64,
                )
                logger.info("✅ DotsOCR initialised (VLM Layout Engine, use_hf=%s)", use_hf)
            except Exception as e:
                self.dots_parser = None
                logger.error(
                    "❌ DotsOCR initialization FAILED: %s\n"
                    "   model_name=%s, use_hf=%s, weights_path=%s",
                    e, model_name, use_hf, weights_path, exc_info=True,
                )
        else:
            self.dots_parser = None
            logger.warning(
                "⚠️  DotsOCR not available "
                "(ensure dots_ocr/ package is on PYTHONPATH)"
            )

        if HAS_PYPDF:
            logger.info("✅ pypdf available as lightweight fallback")
        else:
            logger.warning("⚠️  pypdf not installed — fallback extraction disabled")

    # ─────────────────────────────────────────────────────────────────────────
    # Public API — called by IngestionOrchestrator
    # ─────────────────────────────────────────────────────────────────────────

    def extract_text(self, document_bytes: bytes, output_dir: Optional[str] = None) -> str:
        """
        Extract text from a PDF given its raw bytes.

        Returns
        ───────
        str  — Extracted / cleaned text (may be Markdown from DotsOCR).
        """
        if not document_bytes:
            logger.error("extract_text: received empty document bytes")
            return ""

        temp_dir = tempfile.mkdtemp(prefix="dotsocr_")
        temp_pdf  = os.path.join(temp_dir, "input.pdf")

        try:
            with open(temp_pdf, "wb") as fh:
                fh.write(document_bytes)

            # ── Step 1: DotsOCR (Primary) ─────────────────────────────────
            text = self._run_dotsocr(temp_pdf)

            # ── Step 2: pypdf fallback ────────────────────────────────────
            if not text or len(text.strip()) < 50:
                logger.warning(
                    "DotsOCR returned insufficient text (%d chars) → "
                    "falling back to pypdf extraction",
                    len(text.strip()) if text else 0,
                )
                text = self._run_pypdf(temp_pdf)

            # ── Optional: persist markdown/image assets ───────────────────
            if output_dir and text:
                self._persist_assets(temp_dir, output_dir)

            src = "DotsOCR" if self.dots_parser else "pypdf"
            logger.info(
                "✅ OCR extraction complete (%s) — %d characters extracted",
                src, len(text) if text else 0,
            )
            return text or ""

        except Exception as exc:
            logger.error("OCR pipeline failure: %s", exc, exc_info=True)
            return ""

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _run_dotsocr(self, pdf_path: str) -> str:
        """Run DotsOCR parser and collect its Markdown output files."""
        if not self.dots_parser:
            logger.warning("[DotsOCR] Parser not initialized — skipping OCR")
            return ""

        try:
            logger.info("[DotsOCR] Parsing file: %s (mode=%s, fitz_preprocess=%s)", 
                        pdf_path, cfg.dots_ocr_prompt_mode, cfg.dots_ocr_fitz_preprocess)
            results = self.dots_parser.parse_file(
                pdf_path, 
                prompt_mode=cfg.dots_ocr_prompt_mode,
                fitz_preprocess=cfg.dots_ocr_fitz_preprocess
            )

            pages = []
            for result in results:
                md_path = result.get("md_content_path")
                if md_path and os.path.exists(md_path):
                    with open(md_path, "r", encoding="utf-8", errors="replace") as fh:
                        pages.append(fh.read())

            output = "\n\n".join(pages)
            logger.info("[DotsOCR] Extracted %d chars from %d pages", len(output), len(pages))
            return output

        except Exception as exc:
            logger.error("[DotsOCR] Execution failed: %s", exc, exc_info=True)
            return ""

    def _run_pypdf(self, pdf_path: str) -> str:
        """Lightweight text extraction using pypdf (scanned PDFs yield empty strings)."""
        if not HAS_PYPDF:
            return ""

        try:
            reader = PdfReader(pdf_path)
            pages  = []
            for page in reader.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages.append(page_text)

            output = "\n\n".join(pages)
            logger.info("[pypdf fallback] Extracted %d chars from %d pages", len(output), len(pages))
            return output

        except Exception as exc:
            logger.warning("pypdf fallback failed: %s", exc)
            return ""

    def _persist_assets(self, temp_dir: str, output_dir: str) -> None:
        """Copy markdown and image assets from temp to the persistent output dir."""
        try:
            dest = Path(output_dir)
            dest.mkdir(parents=True, exist_ok=True)
            copied = 0
            for f in Path(temp_dir).rglob("*"):
                if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".md", ".json"):
                    shutil.copy2(str(f), str(dest / f.name))
                    copied += 1
            logger.info("Layout assets persisted to %s (%d files)", output_dir, copied)
        except Exception as exc:
            logger.warning("Failed to persist DotsOCR assets: %s", exc)
