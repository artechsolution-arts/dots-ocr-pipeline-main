# ═══════════════════════════════════════════════════════════════════════════════
#  RAG Complete Backend — Production Dockerfile
#  Hub image name : rag_complete_backend
#  OCR Engine     : DotsOCR (VLM-based) ONLY — no PaddleOCR
# ═══════════════════════════════════════════════════════════════════════════════

FROM python:3.10-slim

# ── Build-time environment ────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ── System libraries ──────────────────────────────────────────────────────────
# Only what DotsOCR + FastAPI actually need. No tesseract, no poppler-utils
# (DotsOCR converts pages internally using PyMuPDF — no poppler dependency).
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Image codec support for PIL / PyMuPDF
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    # PDF rendering fallback (poppler kept for pdf2image if needed)
    poppler-utils \
    # Compiler (some wheels need it)
    build-essential \
    gcc \
    # Health-check / Entrypoint wait
    curl \
    netcat-openbsd \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root app user (security hardening) ───────────────────────────────────
RUN groupadd -r appuser && useradd -r -g appuser -d /home/appuser -m appuser

WORKDIR /app

# ── PIP upgrade ───────────────────────────────────────────────────────────────
RUN pip install --upgrade pip setuptools wheel

# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — Core lightweight packages (cache-stable; rebuild rarely)
# ═══════════════════════════════════════════════════════════════════════════════
COPY requirements.txt .
RUN grep -vE 'torch|torchvision|transformers|sentence-transformers|qwen-vl-utils|modelscope|huggingface' \
    requirements.txt > req_core.txt && \
    pip install -r req_core.txt

# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — PyTorch CPU (avoids 2.6GB CUDA wheel downloads)
# ═══════════════════════════════════════════════════════════════════════════════
RUN pip install \
    torch==2.2.2+cpu \
    torchvision==0.17.2+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — Transformers + Embedding model
# ═══════════════════════════════════════════════════════════════════════════════
RUN pip install \
    transformers==4.51.3 \
    sentence-transformers>=2.7.0 \
    huggingface-hub>=0.30.0,\<1.0

# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 4 — DotsOCR support libs (VLM inference)
# ═══════════════════════════════════════════════════════════════════════════════
RUN pip install \
    qwen-vl-utils \
    openai>=1.0.0,\<2.0.0 \
    accelerate>=0.26.0 \
    modelscope>=1.18.0,\<2.0.0 \
    aiohttp>=3.8.0,\<4.0.0 \
    tqdm>=4.65.0,\<5.0.0

# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION CODE
# ═══════════════════════════════════════════════════════════════════════════════
COPY . .

# ── Runtime environment ───────────────────────────────────────────────────────
ENV PYTHONPATH="/app:/app/dots_ocr:/app/stg609_dots_ocr" \
    UPLOAD_DIR="/app/uploads" \
    HOME="/home/appuser" \
    HF_HOME="/app/weights" \
    DOTS_OCR_MODEL="rednote-hilab/dots.ocr" \
    DOTS_OCR_USE_HF="true" \
    DOTS_OCR_WEIGHTS="/app/weights/DotsOCR" \
    EMBEDDING_DEVICE="cpu"

# ── Directories & permissions ─────────────────────────────────────────────────
RUN mkdir -p /app/uploads /app/data /app/weights \
    && chown -R appuser:appuser /app /home/appuser \
    && chmod +x /usr/local/bin/docker-entrypoint.sh 2>/dev/null || true

# ── Entrypoint script ─────────────────────────────────────────────────────────
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh \
    && chown appuser:appuser /usr/local/bin/docker-entrypoint.sh

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

USER appuser

EXPOSE 8000

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["python", "main.py"]
