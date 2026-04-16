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
    Document-based chunker that splits Markdown text hierarchically based on headers
    while respecting token limits. Developed for general long-form context.
    """
    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        if tiktoken:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self.tokenizer = None
        else:
            self.tokenizer = None
        logger.info(f"Initialized Document Chunker (size={chunk_size}, overlap={chunk_overlap})")

    def count_tokens(self, text: str) -> int:
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return len(text) // 4  # Fallback approximation

    def chunk_document(self, text: str) -> List[Dict[str, str]]:
        """
        Chunks Markdown-formatted document text into logical, hierarchical parts based on headers.
        """
        if not text:
            return []
            
        logger.info("Chunking document text based on Markdown headers...")
        
        # Split by markdown headers
        # Matches lines starting with 1 to 6 '#' followed by space
        # We capture the header so it remains inline with the split
        splits = re.split(r'(^#{1,6}\s+.*$)', text, flags=re.MULTILINE)
        
        chunks = []
        current_chunk = ""
        current_metadata = {"type": "body", "heading": "Document Start"}
        
        # The first split will be everything before the first header
        if splits[0].strip():
            current_chunk = splits[0].strip()
        
        # Iterate over the heading structures
        for i in range(1, len(splits), 2):
            header = splits[i].strip()
            content = splits[i+1].strip() if i+1 < len(splits) else ""
            
            # Flush the current chunk before starting the new header section
            if current_chunk:
                chunks.extend(self._split_large_text(current_chunk, current_metadata))
            
            # Reset for new section, inserting the header context back
            current_chunk = header + "\n\n" + content
            current_metadata = {"type": "section", "heading": header.replace("#", "").strip()}
            
        # Final flush
        if current_chunk:
            chunks.extend(self._split_large_text(current_chunk, current_metadata))
            
        logger.info(f"Chunking complete. Created {len(chunks)} document chunks.")
        return chunks

    def _split_large_text(self, text: str, metadata: dict) -> List[Dict[str, str]]:
        """Handles chunks that are still too large by splitting on paragraphs."""
        if self.count_tokens(text) <= self.chunk_size:
            return [{"metadata": metadata.copy(), "content": text.strip()}]
            
        # Recursive splitting logic
        final_pieces = []
        
        # Split by paragraph
        paragraphs = text.split("\n\n")
        current_piece = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            if self.count_tokens(current_piece) + self.count_tokens(para) > self.chunk_size:
                if current_piece:
                    final_pieces.append({"metadata": metadata.copy(), "content": current_piece.strip()})
                current_piece = para
            else:
                if current_piece:
                    current_piece += "\n\n" + para
                else:
                    current_piece = para
                    
        if current_piece:
            final_pieces.append({"metadata": metadata.copy(), "content": current_piece.strip()})
            
        return final_pieces
