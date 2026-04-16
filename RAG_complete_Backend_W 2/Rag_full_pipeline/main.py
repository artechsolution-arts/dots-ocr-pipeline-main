"""
On-Prem RAG Ingestion Service
==============================
Starts the FastAPI app, connects to all infrastructure, and launches the
worker pool for async PDF ingestion.

Services initialised on startup
--------------------------------
1. PostgreSQL + pgvector  — chunk & embedding storage
2. Redis                  — job state, SSE pub/sub, de-duplication fence
3. RabbitMQ               — async job queue (priority / normal / large queues)
4. SeaweedFS              — raw PDF + extracted markdown object storage
5. WorkerPool             — consumes RabbitMQ jobs → runs IngestionOrchestrator
"""

import logging
import uvicorn
import asyncio
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from src.config import cfg
from src.database.postgres_db import get_pg_pool, create_schema, RBACManager
from src.database.redis_db import RedisStateManager
from src.database.rabbitmq_broker import rabbit_connect, setup_topology
from src.services.rag_pipeline import RAGPipeline
from src.worker.pool import WorkerPool
from src.api.routes import create_router

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

_pool_pg  = None
_rsm      = None
_mq_conn  = None
_pipeline = None
_worker_pool = None
_ids      = {}


def _init_system_defaults(rbac: RBACManager) -> dict:
    """
    Ensure a System department and System user exist.
    These are used as the default scope for uploads that don't specify dept_id/user_id.
    Returns {"dept_default": <uuid>, "user_default": <uuid>}.
    """
    ids = {}

    # System department
    try:
        dept_id = rbac.create_department("System", "Default system department")
    except Exception:
        cur = rbac.conn.cursor()
        cur.execute("SELECT id FROM departments WHERE name='System'")
        row = cur.fetchone()
        dept_id = str(row[0]) if row else None
    ids["dept_default"] = dept_id

    # System user
    system_email = "system@internal.rag"
    try:
        user_id = rbac.create_user(
            system_email, "System", "no_hash_required", dept_id, is_super_admin=True
        )
    except Exception:
        cur = rbac.conn.cursor()
        cur.execute("SELECT id FROM users WHERE email=%s", (system_email,))
        row = cur.fetchone()
        user_id = str(row[0]) if row else None
    ids["user_default"] = user_id

    logger.info("System defaults — dept=%s user=%s", dept_id, user_id)
    return ids


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pool_pg, _rsm, _mq_conn, _pipeline, _worker_pool, _ids

    # 1. PostgreSQL ─────────────────────────────────────────────────────────────
    try:
        _pool_pg = get_pg_pool(minconn=1, maxconn=cfg.upload_workers + 5)
        conn = _pool_pg.getconn()
        conn.autocommit = True
        create_schema(conn)
        _ids = _init_system_defaults(RBACManager(conn))
        _pool_pg.putconn(conn)
        logger.info("PostgreSQL ready (pgvector schema applied)")
    except Exception as e:
        logger.error("PostgreSQL init failed: %s", e)

    # 2. Redis ──────────────────────────────────────────────────────────────────
    try:
        _rsm = RedisStateManager()
        logger.info("Redis connected")
    except Exception as e:
        logger.error("Redis init failed: %s", e)

    # 3. RabbitMQ ───────────────────────────────────────────────────────────────
    try:
        _mq_conn = rabbit_connect()
        setup_topology(_mq_conn)
        logger.info("RabbitMQ topology ready")
    except Exception as e:
        logger.error("RabbitMQ init failed: %s", e)

    # 4. SeaweedFS (non-blocking — falls back gracefully if unavailable) ────────
    _storage_service = None
    try:
        from src.storage import SeaweedFSClient, StorageService
        _sw_client = SeaweedFSClient(
            endpoint_url=cfg.SEAWEEDFS_S3_ENDPOINT,
            aws_access_key_id=cfg.SEAWEEDFS_ACCESS_KEY,
            aws_secret_access_key=cfg.SEAWEEDFS_SECRET_KEY,
            bucket=cfg.SEAWEEDFS_BUCKET,
        )
        _storage_service = StorageService(_sw_client)
        app.state.storage_service = _storage_service

        async def _check_seaweedfs():
            try:
                ok = await _sw_client.health_check()
                if ok:
                    logger.info("SeaweedFS healthy")
                else:
                    logger.warning("SeaweedFS unreachable — PDFs stored locally only")
            except Exception:
                logger.warning("SeaweedFS unreachable — PDFs stored locally only")

        asyncio.create_task(_check_seaweedfs())
    except Exception as e:
        logger.warning("SeaweedFS init skipped: %s", e)

    # 5. Pipeline — start stage workers (always, even in api-only mode) ─────────
    _pipeline = RAGPipeline(_pool_pg, _rsm, storage=_storage_service)
    run_type  = os.getenv("RUN_TYPE", "worker")

    if run_type == "worker":
        # Start all stage threads (preprocess, OCR, assemble, chunk, embed, store)
        _pipeline.start()
        logger.info("StagePipeline started (stage workers active)")
    else:
        logger.info("API-only mode — stage workers NOT started (RUN_TYPE=%s)", run_type)

    app.include_router(create_router(_rsm, _ids, _pipeline, _mq_conn))
    logger.info("Ingestion pipeline ready")

    # 6. RabbitMQ feeder pool (worker mode only) ─────────────────────────────
    if _mq_conn and run_type == "worker":
        # 3 feeders is enough to keep the _doc_q saturated; all heavy work
        # happens inside StagePipeline stage threads, not in feeders.
        _worker_pool = WorkerPool(_rsm, _pipeline, n=3)
        _worker_pool.start()
        logger.info("WorkerPool (RabbitMQ feeders) started")
    else:
        logger.info("Skipping RabbitMQ feeders (RUN_TYPE=%s)", run_type)

    yield  # ── application runs ─────────────────────────────────────────────

    # Shutdown
    if _worker_pool:
        _worker_pool.stop()
    if run_type == "worker":
        _pipeline.stop(timeout=30.0)
    if _mq_conn:
        _mq_conn.close()
    if _pool_pg:
        _pool_pg.closeall()
    logger.info("Shutdown complete")


app = FastAPI(
    title="RAG Ingestion API",
    description="On-prem document ingestion: PDF → OCR → Chunk → Embed → pgvector",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict to your query service origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Monitoring UI ──────────────────────────────────────────────────────────────
_UI_FILE = Path(__file__).parent / "src" / "static" / "index.html"

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_ui():
    """Serve the pipeline monitoring UI."""
    if _UI_FILE.exists():
        return HTMLResponse(_UI_FILE.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>UI not found — check src/static/index.html</h2>", status_code=404)

async def _run_worker_only():
    """
    Worker mode: initialise all services and run the stage pipeline +
    RabbitMQ feeders WITHOUT starting a web server.

    The API process (RUN_TYPE=api) owns port 8000.
    The worker process never touches uvicorn or any port.
    """
    global _pool_pg, _rsm, _mq_conn, _pipeline, _worker_pool, _ids

    # 1. PostgreSQL
    try:
        _pool_pg = get_pg_pool(minconn=1, maxconn=cfg.upload_workers + 5)
        conn = _pool_pg.getconn()
        conn.autocommit = True
        create_schema(conn)
        _ids = _init_system_defaults(RBACManager(conn))
        _pool_pg.putconn(conn)
        logger.info("PostgreSQL ready")
    except Exception as e:
        logger.error("PostgreSQL init failed: %s", e)

    # 2. Redis
    try:
        _rsm = RedisStateManager()
        logger.info("Redis connected")
    except Exception as e:
        logger.error("Redis init failed: %s", e)

    # 3. RabbitMQ
    try:
        _mq_conn = rabbit_connect()
        setup_topology(_mq_conn)
        logger.info("RabbitMQ topology ready")
    except Exception as e:
        logger.error("RabbitMQ init failed: %s", e)

    # 4. SeaweedFS
    _storage_service = None
    try:
        from src.storage import SeaweedFSClient, StorageService
        _sw_client = SeaweedFSClient(
            endpoint_url=cfg.SEAWEEDFS_S3_ENDPOINT,
            aws_access_key_id=cfg.SEAWEEDFS_ACCESS_KEY,
            aws_secret_access_key=cfg.SEAWEEDFS_SECRET_KEY,
            bucket=cfg.SEAWEEDFS_BUCKET,
        )
        _storage_service = StorageService(_sw_client)
        try:
            ok = await _sw_client.health_check()
            logger.info("SeaweedFS %s", "healthy" if ok else "unreachable — continuing without it")
        except Exception:
            logger.warning("SeaweedFS unreachable — continuing without it")
    except Exception as e:
        logger.warning("SeaweedFS init skipped: %s", e)

    # 5. Stage pipeline + workers
    _pipeline = RAGPipeline(_pool_pg, _rsm, storage=_storage_service)
    _pipeline.start()
    logger.info("StagePipeline started (all stage workers active)")

    # 6. RabbitMQ feeders
    if _mq_conn:
        _worker_pool = WorkerPool(_rsm, _pipeline, n=3)
        _worker_pool.start()
        logger.info("WorkerPool (RabbitMQ feeders) started")

    # Block until SIGINT / SIGTERM
    shutdown = asyncio.Event()
    import signal
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, shutdown.set)
        except (NotImplementedError, OSError):
            pass

    logger.info("Worker ready — listening for jobs (Ctrl+C to stop)")
    await shutdown.wait()

    # Graceful shutdown
    logger.info("Worker shutting down…")
    if _worker_pool:
        _worker_pool.stop()
    _pipeline.stop(timeout=30.0)
    if _mq_conn:
        try:
            _mq_conn.close()
        except Exception:
            pass
    if _pool_pg:
        _pool_pg.closeall()
    logger.info("Worker shutdown complete")


if __name__ == "__main__":
    run_type = os.getenv("RUN_TYPE", "worker")
    if run_type == "api":
        # API process: full FastAPI + uvicorn on port 8000
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    else:
        # Worker process: no web server — just the stage pipeline + RabbitMQ feeders
        asyncio.run(_run_worker_only())
