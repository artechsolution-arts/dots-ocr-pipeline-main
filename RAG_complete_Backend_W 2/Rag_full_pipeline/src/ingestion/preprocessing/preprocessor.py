"""
Preprocessing Stage  —  CPU  (OpenCV + PyMuPDF)
================================================
Converts raw PDF bytes → enhanced PIL Images, one per page.

Handles both document types automatically:
  Digital PDFs  → light sharpening only (text is already crisp)
  Scanned PDFs  → deskew + denoise (fastNlMeans) + CLAHE contrast boost

Detection: compares grayscale std-dev of first page.
  std < 60  → likely scanned  (low contrast, uniform grey noise)
  std ≥ 60  → likely digital  (high contrast, sharp edges)

DPI strategy:
  Digital : 200 DPI  (~11 MB/page) — model resizes to 1008×1008 anyway
  Scanned : 300 DPI  (~26 MB/page) — higher source res improves Lanczos resize
"""

from __future__ import annotations

import logging
from typing import Generator

import cv2
import fitz           # PyMuPDF
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# DPI constants
DPI_DIGITAL = 200
DPI_SCANNED = 300
# Grayscale std-dev threshold for scanned detection
SCANNED_STD_THRESHOLD = 60


class DocumentPreprocessor:
    """
    Stateless preprocessor — safe to call from multiple threads concurrently.
    """

    # ── Public API ────────────────────────────────────────────────────────────

    def stream_pages(
        self,
        pdf_bytes: bytes,
    ) -> Generator[tuple[int, int, Image.Image, Image.Image], None, None]:
        """
        Yield (page_idx, total_pages, enhanced_image, origin_image) for every page.

        Streams pages so OCR workers can start immediately while remaining
        pages are still being preprocessed — crucial for pipeline parallelism.
        """
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total = len(doc)

        # Detect scan type from first page (fast, single-page only)
        is_scanned = self._detect_scanned(doc[0])
        dpi = DPI_SCANNED if is_scanned else DPI_DIGITAL
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)

        logger.debug("PDF: %d pages, %s (DPI=%d)", total, "scanned" if is_scanned else "digital", dpi)

        for idx, page in enumerate(doc):
            pix    = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            origin = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            enhanced = self._enhance(origin, is_scanned)
            yield idx, total, enhanced, origin

        doc.close()

    # ── Scan detection ────────────────────────────────────────────────────────

    def _detect_scanned(self, page: fitz.Page) -> bool:
        """Quick scan: render first page at 72 DPI, check grayscale std-dev."""
        try:
            pix  = page.get_pixmap(matrix=fitz.Matrix(1, 1), colorspace=fitz.csGRAY)
            arr  = np.frombuffer(pix.samples, dtype=np.uint8)
            std  = float(arr.std())
            return std < SCANNED_STD_THRESHOLD
        except Exception:
            return False   # Default to digital if detection fails

    # ── Image enhancement ─────────────────────────────────────────────────────

    def _enhance(self, image: Image.Image, is_scanned: bool) -> Image.Image:
        arr = np.array(image)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        if is_scanned:
            bgr = self._process_scanned(bgr)
        else:
            bgr = self._process_digital(bgr)

        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    def _process_scanned(self, bgr: np.ndarray) -> np.ndarray:
        """Deskew → denoise → CLAHE contrast for scanned pages."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # 1. Deskew (Hough line detection, correct ≤ 5° tilt)
        bgr  = self._deskew(bgr, gray)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # 2. Denoise (fastNlMeans — good for scanner noise)
        gray = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

        # 3. CLAHE — adaptive contrast, prevents over-brightening
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)

        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def _process_digital(self, bgr: np.ndarray) -> np.ndarray:
        """Light unsharp-mask sharpening for digital PDFs (text already clean)."""
        kernel = np.array([[0, -0.5, 0],
                           [-0.5, 3, -0.5],
                           [0, -0.5, 0]], dtype=np.float32)
        return cv2.filter2D(bgr, -1, kernel)

    def _deskew(self, bgr: np.ndarray, gray: np.ndarray) -> np.ndarray:
        """Detect dominant line angle, rotate to horizontal."""
        try:
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180,
                threshold=80, minLineLength=80, maxLineGap=5,
            )
            if lines is None:
                return bgr

            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = x2 - x1
                if dx != 0:
                    angle = np.degrees(np.arctan2(y2 - y1, dx))
                    if -5.0 < angle < 5.0:   # Only correct small tilts
                        angles.append(angle)

            if not angles:
                return bgr

            tilt = float(np.median(angles))
            if abs(tilt) < 0.3:             # Skip trivial correction
                return bgr

            h, w = bgr.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), tilt, 1.0)
            return cv2.warpAffine(
                bgr, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
        except Exception as e:
            logger.debug("Deskew skipped: %s", e)
            return bgr
