import redis, json, time, logging
from typing import List, Optional, Tuple
from src.config import REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_PASSWORD, SESSION_TTL, FILE_TTL, WORKER_HB_TTL, DEDUP_TTL
from src.models.schemas import BatchSession, FileProgress, TERMINAL_STAGES

logger = logging.getLogger(__name__)

def _make_redis(decode: bool = True) -> redis.Redis:
    pool = redis.ConnectionPool(
        host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
        password=REDIS_PASSWORD,
        decode_responses=decode,
        max_connections=50,
        socket_timeout=5,
        socket_connect_timeout=5,
        health_check_interval=30,
    )
    return redis.Redis(connection_pool=pool)

class RK:
    STATS = "rag:stats:global"
    @staticmethod
    def session(sid: str)          -> str: return f"rag:session:{sid}"
    @staticmethod
    def session_files(sid: str)    -> str: return f"rag:session:{sid}:files"
    @staticmethod
    def file(fid: str)             -> str: return f"rag:file:{fid}"
    @staticmethod
    def progress_ch(sid: str)      -> str: return f"rag:progress:{sid}"
    @staticmethod
    def worker_hb(wid: str)        -> str: return f"rag:worker:{wid}:heartbeat"
    @staticmethod
    def rate(dept_id: str)         -> str: return f"rag:rate:{dept_id}"
    @staticmethod
    def dedup(content_hash: str)   -> str: return f"rag:dedup:{content_hash}"
    @staticmethod
    def fence(fid: str)           -> str: return f"rag:fence:{fid}"
    @staticmethod
    def taskset(fid: str)         -> str: return f"rag:taskset:{fid}"

class RedisStateManager:
    def __init__(self):
        self.r       = _make_redis(decode=True)
        self.r_bytes = _make_redis(decode=False)
        logger.info(f"[Redis] Connected {REDIS_HOST}:{REDIS_PORT}/db{REDIS_DB}")

    def ping(self) -> bool:
        try: return bool(self.r.ping())
        except redis.RedisError: return False

    def create_session(self, session: BatchSession) -> str:
        pipe = self.r.pipeline(transaction=True)
        pipe.hset(RK.session(session.session_id), mapping=session.to_redis_hash())
        pipe.expire(RK.session(session.session_id), SESSION_TTL)
        pipe.hset(RK.session_files(session.session_id), mapping={"_init": "1"})
        pipe.expire(RK.session_files(session.session_id), SESSION_TTL)
        pipe.execute()
        return session.session_id

    def get_session(self, session_id: str) -> Optional[BatchSession]:
        h = self.r.hgetall(RK.session(session_id))
        if not h or ("_init" in h and len(h) == 1): return None
        return BatchSession.from_redis_hash(h)

    def mark_session_complete(self, session_id: str):
        self.r.hset(RK.session(session_id), "status", "complete")

    def session_summary(self, session_id: str) -> Optional[dict]:
        session = self.get_session(session_id)
        if not session: return None
        files_map = self.r.hgetall(RK.session_files(session_id))
        file_ids  = [v for k, v in files_map.items() if k != "_init"]
        files = []
        done = skipped = errors = in_prog = total_chunks = 0
        for fid in file_ids:
            fp = self.get_file_progress(fid)
            if not fp: continue
            files.append(fp.to_dict())
            total_chunks += fp.chunks
            s = fp.stage
            if s == "done":    done    += 1
            elif s == "skipped": skipped += 1
            elif s == "error":   errors  += 1
            else:                in_prog += 1
        return {
            "session_id":   session_id, "total": session.total,
            "done": done, "skipped": skipped, "errors": errors,
            "in_progress": in_prog, "total_chunks": total_chunks,
            "status": session.status, "created_at": session.created_at, "files": files,
        }

    def list_sessions(self, limit: int = 20) -> List[dict]:
        sessions, cursor = [], 0
        while True:
            cursor, keys = self.r.scan(cursor, match="rag:session:*", count=100)
            for key in keys:
                if ":files" in key: continue
                h = self.r.hgetall(key)
                if h and "session_id" in h: sessions.append(h)
            if cursor == 0: break
        sessions.sort(key=lambda x: float(x.get("created_at", 0)), reverse=True)
        return sessions[:limit]

    def register_file(self, session_id: str, fp: FileProgress):
        pipe = self.r.pipeline(transaction=True)
        pipe.hset(RK.file(fp.file_id), mapping=fp.to_redis_hash())
        pipe.expire(RK.file(fp.file_id), FILE_TTL)
        pipe.hset(RK.session_files(session_id), fp.filename, fp.file_id)
        pipe.execute()

    def get_file_progress(self, file_id: str) -> Optional[FileProgress]:
        h = self.r.hgetall(RK.file(file_id))
        return FileProgress.from_redis_hash(h) if h else None

    def update_stage(self, file_id, session_id, stage, pct, extra=None, publish=True):
        updates = {"stage": stage, "pct": str(pct)}
        # Merge extra fields (e.g. started_at passed by stage workers)
        if extra:
            updates.update({k: str(v) if v is not None else "" for k, v in extra.items()})
        # Set started_at when actual processing begins (preprocessing stage),
        # overriding the upload-time placeholder stored at registration.
        if stage == "preprocessing" and "started_at" not in updates:
            updates["started_at"] = str(time.time())
        if stage in TERMINAL_STAGES:
            updates.setdefault("finished_at", str(time.time()))
        self.r.hset(RK.file(file_id), mapping=updates)
        if publish: self._publish(session_id, file_id)

    def _publish(self, session_id: str, file_id: str):
        fp = self.get_file_progress(file_id)
        if not fp: return
        event = {"type": "file_progress", "data": fp.to_dict()}
        try: self.r.publish(RK.progress_ch(session_id), json.dumps(event))
        except Exception: pass
        if fp.stage in ("done", "skipped", "error"): self._try_complete_session(session_id)

    def _try_complete_session(self, session_id: str):
        summary = self.session_summary(session_id)
        if summary and summary["in_progress"] == 0:
            self.mark_session_complete(session_id)
            event = {"type": "session_complete", "data": summary}
            try: self.r.publish(RK.progress_ch(session_id), json.dumps(event))
            except Exception: pass

    def subscribe_session(self, session_id: str, stop_event=None):
        """
        Generator that yields Redis pub/sub events for a session.

        stop_event: optional threading.Event — when set, the generator exits
        cleanly and releases the Redis pub/sub connection.  Always pass one
        from the SSE handler so the connection is freed on client disconnect.
        """
        summary = self.session_summary(session_id)
        if summary and summary.get("status") == "complete":
            yield {"type": "session_complete", "data": summary}
            return
        pub = self.r_bytes.pubsub(ignore_subscribe_messages=True)
        ch  = RK.progress_ch(session_id).encode()
        pub.subscribe(ch)
        try:
            deadline = time.time() + SESSION_TTL
            while time.time() < deadline:
                # Check stop_event frequently so client disconnects release
                # the Redis connection promptly (previously held for SESSION_TTL)
                if stop_event and stop_event.is_set():
                    break
                msg = pub.get_message(timeout=1)   # 1 s — responsive to stop_event
                if msg is None:
                    yield {"type": "ping"}
                    continue
                try:
                    data = json.loads(msg["data"])
                    yield data
                    if data.get("type") == "session_complete":
                        break
                except Exception:
                    continue
        finally:
            try:
                pub.unsubscribe(ch)
                pub.close()
            except Exception:
                pass

    def check_rate_limit(self, dept_id, limit=200, window_s=3600) -> Tuple[bool, int]:
        pipe = self.r.pipeline(transaction=True)
        pipe.incr(RK.rate(dept_id)); pipe.expire(RK.rate(dept_id), window_s)
        res = pipe.execute(); count = res[0]
        return count <= limit, count

    def set_dedup(self, content_hash, doc_id) -> bool:
        return self.r.set(RK.dedup(content_hash), doc_id, nx=True, ex=DEDUP_TTL) is True

    def check_dedup(self, content_hash) -> Optional[str]:
        return self.r.get(RK.dedup(content_hash))

    def incr_stat(self, field, by=1): self.r.hincrby(RK.STATS, field, by)

    def global_stats(self) -> dict:
        raw = self.r.hgetall(RK.STATS) or {}
        return {k: int(raw.get(k, 0)) for k in ["total_processed", "total_failed", "total_skipped"]}

    def worker_heartbeat(self, worker_id: str):
        self.r.setex(RK.worker_hb(worker_id), WORKER_HB_TTL, str(time.time()))

    def set_fence(self, file_id: str, owner: str, ttl: int = 600) -> bool:
        return bool(self.r.set(RK.fence(file_id), owner, nx=True, ex=ttl))

    def clear_fence(self, file_id: str):
        self.r.delete(RK.fence(file_id))

    def set_taskset(self, file_id: str, total_tasks: int, ttl: int = 3600):
        key = RK.taskset(file_id)
        self.r.hset(key, mapping={"total": str(total_tasks), "completed": "0"})
        self.r.expire(key, ttl)

    def update_task_status(self, file_id: str, task_index: int, status: str):
        key = RK.taskset(file_id)
        if status == "completed":
            self.r.hincrby(key, "completed", 1)
        self.r.hset(key, f"task:{task_index}", status)

    def active_workers(self) -> List[str]:
        workers, cursor = [], 0
        while True:
            cursor, keys = self.r.scan(cursor, match="rag:worker:*:heartbeat", count=100)
            for k in keys: workers.append(k.split(":")[2])
            if cursor == 0: break
        return workers

    def dashboard(self) -> dict:
        return {
            "active_workers": len(self.active_workers()),
            "worker_ids": self.active_workers(),
            "global_stats": self.global_stats(),
            "recent_sessions": self.list_sessions(limit=10),
        }

    def flush_all(self, confirm="") -> bool:
        if confirm != "YES_DELETE_ALL": return False
        cursor = 0; deleted = 0
        while True:
            cursor, keys = self.r.scan(cursor, match="rag:*", count=500)
            if keys: self.r.delete(*keys); deleted += len(keys)
            if cursor == 0: break
        return True
