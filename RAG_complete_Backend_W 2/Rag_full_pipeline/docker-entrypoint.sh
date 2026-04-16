#!/bin/sh
# ═══════════════════════════════════════════════════════════════════════════════
#  docker-entrypoint.sh — RAG Complete Backend
#  Waits for all infrastructure services before starting the app.
# ═══════════════════════════════════════════════════════════════════════════════
set -e

# ── Helper: wait for a TCP port to open ──────────────────────────────────────
wait_for() {
    HOST=$1
    PORT=$2
    NAME=$3
    echo "[entrypoint] Waiting for $NAME ($HOST:$PORT)..."
    attempts=0
    until nc -z "$HOST" "$PORT" 2>/dev/null; do
        attempts=$((attempts + 1))
        if [ "$attempts" -ge 60 ]; then
            echo "[entrypoint] ERROR: $NAME not reachable after 60 attempts. Exiting."
            exit 1
        fi
        sleep 2
    done
    echo "[entrypoint] $NAME is ready ✓"
}

# ── Wait for infrastructure ───────────────────────────────────────────────────
wait_for "${PG_HOST:-postgres}"     "${PG_PORT:-5432}"    "PostgreSQL"
wait_for "${REDIS_HOST:-redis}"     "${REDIS_PORT:-6379}" "Redis"
wait_for "${RABBIT_HOST:-rabbitmq}" "${RABBIT_PORT:-5672}" "RabbitMQ"

echo "[entrypoint] All infrastructure services ready — starting application..."

# ── Hand off to the CMD ───────────────────────────────────────────────────────
exec "$@"
