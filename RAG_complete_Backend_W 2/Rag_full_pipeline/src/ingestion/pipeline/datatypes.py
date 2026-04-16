"""
Pipeline Datatypes
==================
Immutable data classes that flow between pipeline stages.

Flow:
  DocJob → [Preprocessing] → PageJob → [OCR] → PageResult
        → [Layout+Markdown] → PageMarkdown → [Assembly]
        → AssembledDoc → [Chunking] → ChunkItem (per chunk)
        → [EmbeddingBatcher] → EmbeddedItem → [Storage]
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from PIL import Image


# ── Stage 1 input ────────────────────────────────────────────────────────────

@dataclass
class DocJob:
    """A document submitted to the pipeline."""
    file_id:     str
    session_id:  str
    filename:    str
    raw_bytes:   bytes
    user_id:     str
    dept_id:     str
    upload_id:   Optional[str] = None
    upload_type: str = "user"


# ── Stage 1 → Stage 2 ────────────────────────────────────────────────────────

@dataclass
class PageJob:
    """One preprocessed page ready for OCR."""
    file_id:      str
    session_id:   str
    page_idx:     int
    total_pages:  int
    image:        Image.Image   # Enhanced PIL image for DotsOCR
    origin_image: Image.Image   # Original (pre-enhancement) for bbox mapping
    doc_job:      DocJob


# ── Stage 2 → Stage 3 ────────────────────────────────────────────────────────

@dataclass
class PageOCRResult:
    """Raw OCR JSON response for one page."""
    file_id:      str
    session_id:   str
    page_idx:     int
    total_pages:  int
    ocr_response: str           # Raw JSON string from DotsOCR model.generate()
    image:        Image.Image   # Needed for post_process_output bbox mapping
    origin_image: Image.Image
    doc_job:      DocJob
    error:        Optional[str] = None


# ── Stage 3 → Stage 4 ────────────────────────────────────────────────────────

@dataclass
class PageMarkdown:
    """Parsed layout + generated Markdown for one page."""
    file_id:     str
    session_id:  str
    page_idx:    int
    total_pages: int
    markdown:    str            # Final markdown text for this page
    doc_job:     DocJob
    error:       Optional[str] = None


# ── Stage 4 → Stage 5 ────────────────────────────────────────────────────────

@dataclass
class AssembledDoc:
    """All pages collected and assembled into one Markdown document."""
    file_id:      str
    session_id:   str
    markdown:     str
    page_count:   int
    content_hash: str
    doc_job:      DocJob


# ── Stage 5 → Stage 6 ────────────────────────────────────────────────────────

@dataclass
class ChunkItem:
    """One chunk from a document, ready for embedding."""
    file_id:      str
    session_id:   str
    chunk_idx:    int
    total_chunks: int
    content:      str
    metadata:     dict
    content_hash: str
    doc_job:      DocJob


# ── Stage 6 → Stage 7 ────────────────────────────────────────────────────────

@dataclass
class EmbeddedItem:
    """One chunk with its embedding vector, ready for storage."""
    chunk_item: ChunkItem
    embedding:  list[float]
