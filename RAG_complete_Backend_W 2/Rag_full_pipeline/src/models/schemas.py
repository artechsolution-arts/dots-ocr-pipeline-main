import json, time, uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional
from src.config import PRIORITY_MAX_KB, LARGE_MIN_KB, RK_PRIORITY, RK_LARGE, RK_NORMAL

class FileStage(str, Enum):
    QUEUED     = "queued"
    VALIDATING = "validating"
    OCR        = "ocr"
    CHUNKING   = "chunking"
    EMBEDDING  = "embedding"
    STORING    = "storing"
    DONE       = "done"
    SKIPPED    = "skipped"
    ERROR      = "error"

TERMINAL_STAGES = {"done", "skipped", "error",
                   FileStage.DONE, FileStage.SKIPPED, FileStage.ERROR}

STAGE_PCT: Dict[str, int] = {
    "queued": 0, "validating": 10, "ocr": 25,
    "chunking": 45, "embedding": 65, "storing": 85,
    "done": 100, "skipped": 100, "error": 0,
}

@dataclass
class JobPayload:
    job_id:       str   = field(default_factory=lambda: str(uuid.uuid4()))
    session_id:   str   = ""
    file_id:      str   = ""
    filename:     str   = ""
    file_path:    str   = ""
    file_size_kb: float = 0.0
    user_id:      str   = ""
    dept_id:      str   = ""
    upload_type:  str   = "user"
    chat_id:      Optional[str] = None
    upload_id:    Optional[str] = None
    retry:        int   = 0
    enqueued_at:  float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(raw: bytes) -> "JobPayload":
        d = json.loads(raw)
        return JobPayload(**{k: v for k, v in d.items()
                             if k in JobPayload.__dataclass_fields__})

    def routing_key(self) -> str:
        if self.file_size_kb < PRIORITY_MAX_KB:
            return RK_PRIORITY
        if self.file_size_kb > LARGE_MIN_KB:
            return RK_LARGE
        return RK_NORMAL

@dataclass
class FileProgress:
    file_id:     str   = field(default_factory=lambda: str(uuid.uuid4()))
    session_id:  str   = ""
    filename:    str   = ""
    size_kb:     float = 0.0
    stage:       str   = FileStage.QUEUED
    pct:         int   = 0
    pages:       int   = 0
    chunks:      int   = 0
    doc_id:      Optional[str]   = None
    error:       Optional[str]   = None
    retry:       int   = 0
    started_at:  Optional[float] = None
    finished_at: Optional[float] = None

    @property
    def duration(self) -> Optional[float]:
        if self.started_at and self.finished_at:
            return round(self.finished_at - self.started_at, 2)
        return None

    def to_dict(self) -> dict:
        return {
            "file_id":    self.file_id,
            "session_id": self.session_id,
            "filename":   self.filename,
            "size_kb":    round(self.size_kb, 1),
            "stage":      self.stage if isinstance(self.stage, str) else self.stage.value,
            "pct":        self.pct,
            "pages":      self.pages,
            "chunks":     self.chunks,
            "doc_id":     self.doc_id,
            "error":      self.error,
            "retry":      self.retry,
            "started_at": self.started_at,
            "duration":   self.duration,
        }

    def to_redis_hash(self) -> dict:
        return {k: str(v) if v is not None else "" for k, v in self.to_dict().items()}

    @staticmethod
    def from_redis_hash(h: dict) -> "FileProgress":
        _i = lambda v: int(v)   if v else 0
        _f = lambda v: float(v) if v else 0.0
        _o = lambda v: v        if v else None
        return FileProgress(
            file_id    = h.get("file_id",    ""),
            session_id = h.get("session_id", ""),
            filename   = h.get("filename",   ""),
            size_kb    = _f(h.get("size_kb")),
            stage      = h.get("stage", FileStage.QUEUED),
            pct        = _i(h.get("pct")),
            pages      = _i(h.get("pages")),
            chunks     = _i(h.get("chunks")),
            doc_id     = _o(h.get("doc_id")),
            error      = _o(h.get("error")),
            retry      = _i(h.get("retry")),
            started_at = _f(h.get("started_at"))  or None,
            finished_at= _f(h.get("finished_at")) or None,
        )

    def finish_error(self, msg: str):
        self.stage = FileStage.ERROR
        self.error = msg
        self.pct   = 0
        self.finished_at = time.time()

@dataclass
class BatchSession:
    session_id:  str   = field(default_factory=lambda: str(uuid.uuid4()))
    user_id:     str   = ""
    dept_id:     str   = ""
    upload_type: str   = "user"
    total:       int   = 0
    created_at:  float = field(default_factory=time.time)
    status:      str   = "running"

    def to_redis_hash(self) -> dict:
        return {
            "session_id":  self.session_id,
            "user_id":     self.user_id,
            "dept_id":     self.dept_id,
            "upload_type": self.upload_type,
            "total":       str(self.total),
            "created_at":  str(self.created_at),
            "status":      self.status,
        }

    @staticmethod
    def from_redis_hash(h: dict) -> "BatchSession":
        return BatchSession(
            session_id  = h.get("session_id",  ""),
            user_id     = h.get("user_id",     ""),
            dept_id     = h.get("dept_id",     ""),
            upload_type = h.get("upload_type", "user"),
            total       = int(h.get("total",   0)),
            created_at  = float(h.get("created_at", time.time())),
            status      = h.get("status", "running"),
        )

@dataclass
class PDFDoc:
    doc_id:         str   = field(default_factory=lambda: str(uuid.uuid4()))
    filename:       str   = ""
    file_path:      str   = ""
    raw_content:    bytes = b""
    extracted_text: str   = ""
    page_count:     int   = 0
    content_hash:   str   = ""
    department_id:  Optional[str] = None
    uploaded_by:    Optional[str] = None
    metadata:       Dict[str, Any] = field(default_factory=dict)

@dataclass
class Chunk:
    chunk_id:    str = field(default_factory=lambda: str(uuid.uuid4()))
    doc_id:      str = ""
    text:        str = ""
    token_count: int = 0
    chunk_index: int = 0
    page_num:    int = 0
    metadata:    Dict[str, Any] = field(default_factory=dict)

@dataclass
class EmbeddedChunk:
    chunk:     Chunk
    embedding: List[float] = field(default_factory=list)

@dataclass
class RetrievedChunk:
    chunk:            Chunk
    similarity_score: float = 0.0
    rerank_score:     float = 0.0
    rank:             int   = 0
