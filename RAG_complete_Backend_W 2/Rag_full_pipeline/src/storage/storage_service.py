"""
Storage Service
===============
High-level interface between the RAG pipeline and SeaweedFS.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any

from src.storage.seaweedfs_client import (
    SeaweedFSClient,
    chunk_key,
    processed_key,
    raw_pdf_key,
)

logger = logging.getLogger(__name__)


class StorageService:
    """Pipeline-aware wrapper around :class:`SeaweedFSClient`."""

    def __init__(self, client: SeaweedFSClient):
        self._client = client

    # ------------------------------------------------------------------
    # PDF lifecycle
    # ------------------------------------------------------------------

    async def store_uploaded_pdf(
        self,
        job_id: str,
        filename: str,
        data: bytes,
    ) -> str:
        """Persist a newly-uploaded PDF to object storage."""
        key = raw_pdf_key(job_id, filename)
        logger.info("[job=%s] Storing raw PDF → %s (%d bytes)", job_id, key, len(data))
        await self._client.upload_file(key, data, content_type="application/pdf")
        return key

    async def retrieve_pdf(self, job_id: str, filename: str) -> bytes:
        """Download the raw PDF bytes for a job from object storage."""
        key = raw_pdf_key(job_id, filename)
        logger.info("[job=%s] Retrieving raw PDF ← %s", job_id, key)
        return await self._client.download_file(key)

    async def store_pdf_from_path(
        self,
        job_id: str,
        local_path: str | Path,
    ) -> str:
        """Move a PDF from the local uploads/ temp directory to object storage."""
        local_path = Path(local_path)
        key = raw_pdf_key(job_id, local_path.name)
        logger.info("[job=%s] Moving local PDF %s → object storage %s", job_id, local_path, key)
        await self._client.upload_local_file(key, local_path)
        return key

    # ------------------------------------------------------------------
    # Processed artefacts
    # ------------------------------------------------------------------

    async def store_extracted_text(
        self,
        job_id: str,
        filename: str,
        extraction_result: dict[str, Any],
    ) -> str:
        """Serialise and store the text-extraction result JSON."""
        key = processed_key(job_id, filename)
        payload = json.dumps(extraction_result, ensure_ascii=False, indent=2).encode("utf-8")
        await self._client.upload_file(key, payload, content_type="application/json")
        logger.debug("[job=%s] Stored extraction artefact → %s", job_id, key)
        return key

    async def load_extracted_text(
        self,
        job_id: str,
        filename: str,
    ) -> dict[str, Any]:
        """Load a previously-stored text-extraction result."""
        key = processed_key(job_id, filename)
        raw = await self._client.download_file(key)
        return json.loads(raw.decode("utf-8"))

    async def store_chunk(
        self,
        job_id: str,
        chunk_index: int,
        chunk_data: dict[str, Any],
    ) -> str:
        """Persist a single chunk dict to object storage."""
        key = chunk_key(job_id, chunk_index)
        payload = json.dumps(chunk_data, ensure_ascii=False).encode("utf-8")
        await self._client.upload_file(key, payload, content_type="application/json")
        return key

    async def store_chunks_batch(
        self,
        job_id: str,
        chunks: list[dict[str, Any]],
    ) -> list[str]:
        """Persist all chunks for a job."""
        keys: list[str] = []
        for idx, chunk in enumerate(chunks):
            key = await self.store_chunk(job_id, idx, chunk)
            keys.append(key)
        logger.info("[job=%s] Stored %d chunks to object storage", job_id, len(chunks))
        return keys

    async def load_chunk(self, job_id: str, chunk_index: int) -> dict[str, Any]:
        """Load a single chunk by index."""
        key = chunk_key(job_id, chunk_index)
        raw = await self._client.download_file(key)
        return json.loads(raw.decode("utf-8"))

    # ------------------------------------------------------------------
    # Job cleanup
    # ------------------------------------------------------------------

    async def delete_job_artefacts(self, job_id: str) -> int:
        """Remove every object associated with job_id."""
        prefixes = [f"raw/{job_id}/", f"processed/{job_id}/", f"chunks/{job_id}/"]
        deleted = 0
        for prefix in prefixes:
            files = await self._client.list_files(prefix)
            for f in files:
                ok = await self._client.delete_file(f["name"])
                if ok:
                    deleted += 1
        logger.info("[job=%s] Deleted %d artefacts from object storage", job_id, deleted)
        return deleted

    # ------------------------------------------------------------------
    # URL helpers
    # ------------------------------------------------------------------

    def pdf_url(self, job_id: str, filename: str) -> str:
        return self._client.public_url(raw_pdf_key(job_id, filename))

    def processed_url(self, job_id: str, filename: str) -> str:
        return self._client.public_url(processed_key(job_id, filename))

    # ------------------------------------------------------------------
    # Health / listing
    # ------------------------------------------------------------------

    async def health(self) -> dict[str, Any]:
        ok = await self._client.health_check()
        return {"seaweedfs": "status_ok" if ok else "unreachable"}

    async def list_job_files(self, job_id: str) -> dict[str, list[dict]]:
        raw = await self._client.list_files(f"raw/{job_id}/")
        processed = await self._client.list_files(f"processed/{job_id}/")
        chunks = await self._client.list_files(f"chunks/{job_id}/")
        return {"raw": raw, "processed": processed, "chunks": chunks}
