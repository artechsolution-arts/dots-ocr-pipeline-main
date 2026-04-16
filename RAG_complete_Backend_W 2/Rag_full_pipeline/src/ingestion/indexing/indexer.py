import logging
try:
    import psycopg2
except ImportError:
    psycopg2 = None

from src.config import EMBEDDING_DIM

logger = logging.getLogger(__name__)

class VectorIndexer:
    """
    Manages advanced vector indexing strategies using PG Vector.
    Specifically implements HNSW (Hierarchical Navigable Small World) indexes
    which provide superior recall and search speed compared to IVFFLAT.
    """
    def __init__(self, conn=None):
        self.conn = conn
        logger.info("Vector Indexer Initialized.")

    def _ensure_connection(self):
        if self.conn is None:
            raise ConnectionError("VectorIndexer requires an active database connection.")

    def create_hnsw_index(self):
        """
        Builds the HNSW index on the vector similarity embeddings.
        Using cosine similarity ops as mxbai embeddings are L2-normalized.
        """
        self._ensure_connection()
        try:
            with self.conn.cursor() as cur:
                logger.info("Setting up PGVector HNSW Index...")
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_emb_vector_hnsw 
                    ON embeddings 
                    USING hnsw (embedding vector_cosine_ops) 
                    WITH (m = 16, ef_construction = 64);
                """)
                # Only commit if not in autocommit mode
                try:
                    if not self.conn.autocommit:
                        self.conn.commit()
                except Exception:
                    pass  # autocommit mode — no explicit commit needed
                logger.info("HNSW index ready ✓")
                return True
        except Exception as e:
            logger.warning(f"HNSW index creation skipped: {e} (pgvector may not support HNSW in this version)")
            try:
                # Fallback: try IVFFlat index instead
                with self.conn.cursor() as cur:
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_emb_vector_ivfflat 
                        ON embeddings 
                        USING ivfflat (embedding vector_cosine_ops) 
                        WITH (lists = 100);
                    """)
                    try:
                        if not self.conn.autocommit:
                            self.conn.commit()
                    except Exception:
                        pass
                logger.info("IVFFlat fallback index created ✓")
            except Exception as e2:
                logger.warning(f"IVFFlat fallback also failed: {e2}. Vector search will use sequential scan.")
            return False

    def get_index_status(self) -> dict:
        """
        Retrieves the status and stats of the HNSW indices.
        """
        self._ensure_connection()
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT indexname, indexdef 
                    FROM pg_indexes 
                    WHERE tablename = 'embeddings' AND indexname LIKE 'idx_emb_vector%';
                """)
                indices = cur.fetchall()
                return {"active_indices": indices}
        except Exception as e:
            logger.error(f"Failed to query index stats: {e}")
            return {}
