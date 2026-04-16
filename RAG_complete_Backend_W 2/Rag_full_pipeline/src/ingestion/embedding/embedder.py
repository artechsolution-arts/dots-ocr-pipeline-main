"""
MxbaiEmbedder
=============
Sentence-Transformers wrapper around mixedbread-ai/mxbai-embed-large-v1.

Key features for the M3 Ultra setup
-------------------------------------
- Batch size controlled by cfg.embedding_batch (default 128 with 512 GB RAM)
- show_progress_bar disabled in production (clutters logs)
- normalize_embeddings=True → cosine similarity in pgvector works correctly
- Automatic CPU fallback if the requested device is unavailable
"""

import logging
from typing import List

from src.config import cfg

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

logger = logging.getLogger(__name__)


class MxbaiEmbedder:

    def __init__(self, model_name: str = None, device: str = None, batch_size: int = None):
        self._model_name  = model_name  or getattr(cfg, "embedding_model",  "mixedbread-ai/mxbai-embed-large-v1")
        self._device      = device      or getattr(cfg, "embedding_device", "cpu")
        self._batch_size  = batch_size  or getattr(cfg, "embedding_batch",  128)
        self._embed_dim   = getattr(cfg, "embedding_dim", 1024)
        self.model        = self._load_model()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self):
        if SentenceTransformer is None:
            logger.error("sentence-transformers not installed")
            return None

        logger.info("Loading embedding model '%s' on device '%s' (batch_size=%d)",
                    self._model_name, self._device, self._batch_size)
        try:
            model = SentenceTransformer(self._model_name, device=self._device)
            logger.info("Embedding model loaded (dim=%d)", self._embed_dim)
            return model
        except Exception as e:
            logger.error("Failed to load embedding model on '%s': %s", self._device, e)
            if self._device != "cpu":
                logger.info("Retrying on CPU…")
                try:
                    model = SentenceTransformer(self._model_name, device="cpu")
                    self._device = "cpu"
                    logger.info("Embedding model loaded on CPU fallback")
                    return model
                except Exception as e2:
                    logger.error("CPU fallback also failed: %s", e2)
            return None

    # ── Public API ────────────────────────────────────────────────────────────

    def embed_text(self, text: str) -> List[float]:
        """Embed a single string. Returns a 1024-dim normalised vector."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of strings in batches.

        Uses cfg.embedding_batch chunks per forward pass — on an M3 Ultra CPU
        a batch of 128 short chunks takes ~1-2 s, keeping GPU idle time low.
        """
        if not self.model:
            logger.warning("Embedding model not loaded — returning zero vectors")
            return [[0.0] * self._embed_dim for _ in texts]

        if not texts:
            return []

        logger.debug("Embedding %d texts (batch_size=%d, device=%s)",
                     len(texts), self._batch_size, self._device)

        embeddings = self.model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.tolist()
