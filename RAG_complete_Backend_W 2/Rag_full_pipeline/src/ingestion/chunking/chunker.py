import logging
import re
from typing import List, Dict
try:
    import tiktoken
except ImportError:
    tiktoken = None

logger = logging.getLogger(__name__)

class DocumentChunker:
    """
    Document-based chunker that splits Markdown text hierarchically based on
    headers while respecting token limits.

    Overlap
    -------
    After all chunks are created, chunk_overlap tokens from the end of each
    chunk are prepended to the start of the next chunk.  This ensures that
    context spanning a chunk boundary is represented in both adjacent chunks,
    which improves RAG retrieval recall near boundaries.
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 150):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        if tiktoken:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self.tokenizer = None
        else:
            self.tokenizer = None
        logger.info("Initialized DocumentChunker (size=%d, overlap=%d)",
                    chunk_size, chunk_overlap)

    def count_tokens(self, text: str) -> int:
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return len(text) // 4   # Fallback approximation

    def chunk_document(self, text: str) -> List[Dict[str, str]]:
        """
        Chunks Markdown-formatted document text into logical parts based on
        headers, then applies token-overlap between adjacent chunks.
        """
        if not text:
            return []

        # ── Split by Markdown headers ────────────────────────────────────────
        splits = re.split(r'(^#{1,6}\s+.*$)', text, flags=re.MULTILINE)

        raw_chunks: List[Dict] = []
        current_chunk    = ""
        current_metadata = {"type": "body", "heading": "Document Start"}

        if splits[0].strip():
            current_chunk = splits[0].strip()

        for i in range(1, len(splits), 2):
            header  = splits[i].strip()
            content = splits[i + 1].strip() if i + 1 < len(splits) else ""

            if current_chunk:
                raw_chunks.extend(self._split_large_text(current_chunk, current_metadata))

            current_chunk    = header + "\n\n" + content
            current_metadata = {"type": "section",
                                "heading": header.replace("#", "").strip()}

        if current_chunk:
            raw_chunks.extend(self._split_large_text(current_chunk, current_metadata))

        # ── Apply overlap ────────────────────────────────────────────────────
        if self.chunk_overlap > 0 and len(raw_chunks) > 1:
            raw_chunks = self._apply_overlap(raw_chunks)

        logger.info("Chunking complete — %d chunks", len(raw_chunks))
        return raw_chunks

    # ── Private helpers ───────────────────────────────────────────────────────

    def _split_large_text(self, text: str, metadata: dict) -> List[Dict[str, str]]:
        """Split text that exceeds chunk_size on paragraph boundaries."""
        if self.count_tokens(text) <= self.chunk_size:
            return [{"metadata": metadata.copy(), "content": text.strip()}]

        final_pieces: List[Dict] = []
        paragraphs   = text.split("\n\n")
        current_piece = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if self.count_tokens(current_piece) + self.count_tokens(para) > self.chunk_size:
                if current_piece:
                    final_pieces.append({"metadata": metadata.copy(),
                                         "content": current_piece.strip()})
                current_piece = para
            else:
                current_piece = (current_piece + "\n\n" + para) if current_piece else para

        if current_piece:
            final_pieces.append({"metadata": metadata.copy(),
                                  "content": current_piece.strip()})

        return final_pieces

    def _apply_overlap(self, chunks: List[Dict]) -> List[Dict]:
        """
        Prepend the last chunk_overlap tokens of chunk[i] to the start of
        chunk[i+1].  The metadata of each chunk is preserved as-is.
        """
        result = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = self._tail_tokens(chunks[i - 1]["content"], self.chunk_overlap)
            if tail:
                overlapped_content = tail + "\n\n" + chunks[i]["content"]
            else:
                overlapped_content = chunks[i]["content"]
            result.append({
                "metadata": chunks[i]["metadata"].copy(),
                "content":  overlapped_content,
            })
        return result

    def _tail_tokens(self, text: str, n_tokens: int) -> str:
        """Return the last n_tokens tokens of text decoded back to a string."""
        if not text:
            return ""
        if self.tokenizer:
            tokens = self.tokenizer.encode(text)
            if len(tokens) <= n_tokens:
                return text
            return self.tokenizer.decode(tokens[-n_tokens:])
        # Fallback: approximate by characters (4 chars ≈ 1 token)
        char_limit = n_tokens * 4
        return text[-char_limit:] if len(text) > char_limit else text
