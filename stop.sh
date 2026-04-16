#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#  RAG Ingestion Pipeline — Stop Script
# ═══════════════════════════════════════════════════════════════════════════════
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_DIR="${REPO_ROOT}/.pids"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "\033[0;34m[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }

echo ""
info "Stopping RAG Ingestion Pipeline..."

# ── Stop native processes ──────────────────────────────────────────────────────
stop_pid_file() {
    local file="$1"
    local name="$2"
    if [ -f "$file" ]; then
        while IFS= read -r pid; do
            if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null && success "Stopped ${name} (pid=${pid})"
            fi
        done < "$file"
        rm -f "$file"
    else
        warn "${name} PID file not found (may already be stopped)"
    fi
}

stop_pid_file "${PID_DIR}/api.pid"     "API"
stop_pid_file "${PID_DIR}/workers.pid" "Workers"

# ── Stop Docker infra ─────────────────────────────────────────────────────────
info "Stopping Docker infrastructure (Redis + RabbitMQ)..."
cd "$REPO_ROOT"
docker compose down
success "Docker services stopped"

echo ""
echo -e "${GREEN}Pipeline stopped cleanly.${NC}"
echo ""
