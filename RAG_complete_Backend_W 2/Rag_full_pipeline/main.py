import logging, uvicorn, asyncio, os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from src.config import cfg
from src.database.postgres_db import get_pg_connection, create_schema, RBACManager
from src.database.redis_db import RedisStateManager
from src.database.rabbitmq_broker import rabbit_connect, setup_topology
from src.services.rag_pipeline import RAGPipeline
from src.worker.pool import WorkerPool
from src.api.routes import create_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_pool_pg, _rsm, _mq_conn, _pipeline, _pool, _ids = None, None, None, None, None, {}

def seed_rbac(rbac: RBACManager):
    """Ensures a system-level master user exists and creates sample data."""
    ids = {}
    
    # 1. Create Default Department
    try:
        dept_id = rbac.create_department("System", "Base System Department")
        logger.info(f"Created System Department: {dept_id}")
    except Exception:
        cur = rbac.conn.cursor()
        cur.execute("SELECT id FROM departments WHERE name='System'")
        res = cur.fetchone()
        dept_id = str(res[0]) if res else None
    
    ids["dept_default"] = dept_id

    # 2. Create System User
    system_email = "system@internal.rag"
    try:
        user_id = rbac.create_user(system_email, "System Master", "no_hash_required", dept_id, True)
        logger.info(f"Created System User: {user_id}")
    except Exception:
        cur = rbac.conn.cursor()
        cur.execute("SELECT id FROM users WHERE email=%s", (system_email,))
        res = cur.fetchone()
        user_id = str(res[0]) if res else None

    ids["user_default"] = user_id

    # 3. Create sample departments and users for the UI
    sample_depts = [
        ("Administration", "Admin department"),
        ("QA", "Quality Assurance"),
        ("Plant", "Plant Operations"),
        ("Marketing", "Marketing department"),
        ("Sales", "Sales department"),
    ]
    sample_users = [
        ("admin@rag.local", "Admin", "Administration", True),
        ("qa@rag.local", "QA Specialist", "QA", False),
        ("plant@rag.local", "Plant Manager", "Plant", False),
        ("marketing@rag.local", "Marketing Lead", "Marketing", False),
        ("sales@rag.local", "Sales Executive", "Sales", False),
    ]
    dept_map = {}
    for name, desc in sample_depts:
        try:
            did = rbac.create_department(name, desc, None)
            dept_map[name] = did
            logger.info(f"Created dept '{name}': {did}")
        except Exception as e:
            logger.warning(f"Dept '{name}' already exists or error: {e}")
            cur = rbac.conn.cursor()
            cur.execute("SELECT id FROM departments WHERE name=%s", (name,))
            res = cur.fetchone()
            if res: dept_map[name] = str(res[0])
    for email, uname, dept_name, is_admin in sample_users:
        try:
            did = dept_map.get(dept_name)
            if did:
                uid = rbac.create_user(email, uname, "hashed_pw", did, is_admin)
                logger.info(f"Created user '{uname}': {uid}")
        except Exception as e:
            logger.warning(f"User '{uname}' already exists or error: {e}")

    return ids

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pool_pg, _rsm, _mq_conn, _pipeline, _pool, _ids
    
    # 1. Database & Schema
    try:
        from src.database.postgres_db import get_pg_pool
        _pool_pg = get_pg_pool(minconn=1, maxconn=cfg.upload_workers + 5)
        conn = _pool_pg.getconn()
        conn.autocommit = True
        create_schema(conn)
        _ids = seed_rbac(RBACManager(conn))
        _pool_pg.putconn(conn)
        logger.info(f"RBAC Seed IDs: {_ids}")
    except Exception as e:
        logger.error(f"PostgreSQL failed: {e}")

    # 2. Redis
    try:
        _rsm = RedisStateManager()
    except Exception as e:
        logger.error(f"Redis failed: {e}")

    # 3. RabbitMQ
    try:
        _mq_conn = rabbit_connect()
        setup_topology(_mq_conn)
    except Exception as e:
        logger.error(f"RabbitMQ failed: {e}")

    # 4. SeaweedFS Object Storage
    try:
        from src.storage import SeaweedFSClient, StorageService
        _sw_client = SeaweedFSClient(
            filer_url=cfg.SEAWEEDFS_FILER_URL,
            master_url=cfg.SEAWEEDFS_MASTER_URL,
            bucket=cfg.SEAWEEDFS_BUCKET,
        )
        _storage_service = StorageService(_sw_client)
        app.state.storage_service = _storage_service
        
        # Non-blocking health check
        async def _check():
            try:
                if await _sw_client.health_check():
                    logger.info("SeaweedFS connected and healthy ✓")
                else:
                    logger.warning("SeaweedFS unreachable - using local fallback")
            except Exception: logger.warning("SeaweedFS unreachable - using local fallback")
        asyncio.create_task(_check())
    except Exception as e:
        logger.error(f"SeaweedFS initialization failed: {e}")
        _storage_service = None

    # 5. Pipeline & Router
    _pipeline = RAGPipeline(_pool_pg, _rsm, storage=_storage_service)
    app.include_router(create_router(_rsm, _ids, _pipeline, _mq_conn))

    # 6. Worker Pool (Only start in Worker containers, not API)
    if _mq_conn and os.getenv("RUN_TYPE", "worker") == "worker":
        _pool = WorkerPool(_rsm, _pipeline, cfg.upload_workers)
        _pool.start()
        logger.info("Workers initialized")
    else:
        logger.info("Skipping WorkerPool initialization (RUN_TYPE is not 'worker')")
    
    yield
    
    if _pool: _pool.stop()
    if _mq_conn: _mq_conn.close()
    if _pool_pg: _pool_pg.closeall()
    if hasattr(app.state, "storage_service"):
        # Use simpler close to avoid event loop issues on exit
        logger.info("Closing storage client...")



app = FastAPI(title="RAG PDF Pipeline", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
