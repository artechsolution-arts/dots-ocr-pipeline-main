import re
import unicodedata
import logging

logger = logging.getLogger(__name__)

class TextCleaner:
    """
    Handles normalizing and cleaning raw extracted text (like Markdown from DotsOCR).
    Optimized for heavily structured documents such as Invoices (80% of workload) 
    while remaining safe for general documents.
    """
    def __init__(self):
        # Handle zero-width spaces, isolated control characters, and common OCR ghosting
        self.junk_chars = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')
        
        # We target specific repeated noise structures without damaging Markdown
        self.excessive_newlines = re.compile(r'\n{3,}')
        
        # Standard page noise
        self.page_num_mark = re.compile(r'^\s*(Page|PAGE)\s*\d+\s*(of\s*\d+)?\s*$', re.MULTILINE | re.IGNORECASE)

    def clean(self, text: str) -> str:
        """
        Executes a robust parsing sequence tailored for document embedding workloads.
        Preserves Markdown structure (headers `#`, tables `|`, bold `**`) while stripping noise.
        """
        if not text:
            return ""
            
        logger.info("Parsing and cleaning document text format...")
        
        # 1. Unicode Normalization: Resolves ligatures (e.g., 'ﬁ' -> 'fi') and standardizes layout quotes
        text = unicodedata.normalize('NFKC', text)
        
        # 2. Strip destructive control characters that might break JSON parsers/Tokenizers
        text = self.junk_chars.sub('', text)
        
        # 3. Standardize carriage returns
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # 4. Filter empty page markers (e.g. "Page 1 of 12" on its own line)
        text = self.page_num_mark.sub('', text)
        
        # 5. Clean strange spacing within lines without breaking semantic tabular data
        lines = []
        for line in text.split('\n'):
            stripped = line.strip(" \t")
            if stripped:
                lines.append(stripped)
            else:
                lines.append("")
        text = '\n'.join(lines)
        
        # 6. Apply Invoice-specific parsing rules (since 80% are invoices)
        text = self.normalize_invoice_fields(text)
        
        # 7. Compress gigantic line-breaks (>3) into standard paragraph boundaries (double newline)
        text = self.excessive_newlines.sub('\n\n', text)
        
        return text.strip()

    def normalize_invoice_fields(self, text: str) -> str:
        """
        Special cleaning for invoice-like data (handling colons, tables, alignment noise).
        """
        # Ensure colons have a space after them if followed by text to aid tokenizer splitting
        # (e.g., "Total:100" -> "Total: 100")
        text = re.sub(r':([^\s\d])', r': \1', text)
        # Avoid separating decimal points but fix currency or colon attached to numbers
        text = re.sub(r':(\d)', r': \1', text)
        
        # Remove repeated dot patterns used for spacing in invoices (e.g., Description........Price)
        text = re.sub(r'\.{3,}', ' ', text)
        
        # Clean up repeated trailing hyphens or underscores used as fill lines
        text = re.sub(r'[_]{3,}', ' ', text)
        text = re.sub(r'[-]{4,}', '---', text) # Retain Markdown table boundaries but compress excessive lines
        
        return text
