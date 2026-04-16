#!/bin/sh
# ═══════════════════════════════════════════════════════════════════════════════
#  docker-entrypoint.sh — On-Prem RAG Ingestion Service
#
#  1. Sets CPU thread counts for optimal ARM64 / M3 Ultra performance
#  2. Waits for all infrastructure services to be reachable
#  3. Hands off to the CMD (python main.py)
# ═══════════════════════════════════════════════════════════════════════════════
set -e

# ── Thread tuning (inherited from docker-compose env, with safe defaults) ──────
# Each worker thread uses OMP_NUM_THREADS for intra-op parallelism.
# 6 workers x 4 OMP threads = 24 threads = M3 Ultra performance core count.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

echo "[entrypoint] Thread config: OMP=${OMP_NUM_THREADS} MKL=${MKL_NUM_THREADS} OPENBLAS=${OPENBLAS_NUM_THREADS}"
echo "[entrypoint] Workers=${UPLOAD_WORKERS:-?}  EmbedBatch=${EMBEDDING_BATCH_SIZE:-?}  Device=${EMBEDDING_DEVICE:-cpu}"

# ── Helper: wait for a TCP port to open ───────────────────────────────────────
wait_for() {
    HOST=$1
    PORT=$2
    NAME=$3
    echo "[entrypoint] Waiting for $NAME ($HOST:$PORT)..."
    attempts=0
    until nc -z "$HOST" "$PORT" 2>/dev/null; do
        attempts=$((attempts + 1))
        if [ "$attempts" -ge 90 ]; then
            echo "[entrypoint] ERROR: $NAME not reachable after 90 attempts. Exiting."
            exit 1
        fi
        sleep 2
    done
    echo "[entrypoint] $NAME ready ✓"
}

# ── Wait for required services ─────────────────────────────────────────────────
wait_for "${PG_HOST:-192.168.10.10}" "${PG_PORT:-5433}"    "PostgreSQL"
wait_for "${REDIS_HOST:-redis}"       "${REDIS_PORT:-6379}" "Redis"
wait_for "${RABBIT_HOST:-rabbitmq}"   "${RABBIT_PORT:-5672}" "RabbitMQ"

# SeaweedFS: non-fatal — pipeline continues if unavailable
SW_HOST=$(echo "${SEAWEEDFS_MASTER_URL:-http://192.168.10.10:9333}" | sed 's|http://||' | cut -d: -f1)
SW_PORT=$(echo "${SEAWEEDFS_MASTER_URL:-http://192.168.10.10:9333}" | sed 's|http://||' | cut -d: -f2)
if nc -z "$SW_HOST" "$SW_PORT" 2>/dev/null; then
    echo "[entrypoint] SeaweedFS ready ✓"
else
    echo "[entrypoint] WARNING: SeaweedFS not reachable -- PDFs stored locally only"
fi

echo "[entrypoint] All required services ready -- starting (RUN_TYPE=${RUN_TYPE:-worker})..."

# ── Hand off to CMD ────────────────────────────────────────────────────────────
exec "$@"
