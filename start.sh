#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#  RAG Ingestion Pipeline — Start Script
#  Mac Studio M3 Ultra · 80-core GPU · 512 GB unified memory
#
#  First run  : sets up venv, installs deps, downloads model weights (~7 GB)
#  Every run  : starts Redis + RabbitMQ in Docker, then API + Worker natively
#
#  Why native (not Docker) for api + worker:
#    Docker Desktop on macOS runs in a Linux VM with NO access to Apple Metal.
#    Native Python gets full MPS access → 5-8x faster OCR and embedding.
#
#  Usage:
#    chmod +x start.sh stop.sh
#    ./start.sh              # start pipeline (auto-setup on first run)
#    ./stop.sh               # clean shutdown
#    ./start.sh --workers 4  # override worker count
# ═══════════════════════════════════════════════════════════════════════════════
set -e

# ── Config ────────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${REPO_ROOT}/RAG_complete_Backend_W 2/Rag_full_pipeline"
ENV_FILE="${APP_DIR}/.env"
PID_DIR="${REPO_ROOT}/.pids"
LOG_DIR="${REPO_ROOT}/logs"
VENV_DIR="${REPO_ROOT}/venv"
WEIGHTS_DIR="${REPO_ROOT}/weights/DotsOCR"
REQUIREMENTS="${APP_DIR}/requirements.txt"

# Worker count — optimal for M3 Ultra MPS
WORKERS="${2:-3}"

# Parse --workers flag
while [[ $# -gt 0 ]]; do
    case $1 in
        --workers|-w) WORKERS="$2"; shift 2 ;;
        *) shift ;;
    esac
done

mkdir -p "$PID_DIR" "$LOG_DIR"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   RAG Ingestion Pipeline  —  M3 Ultra + MPS        ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════╝${NC}"
echo ""

# ── Hard prerequisites ────────────────────────────────────────────────────────
command -v docker  >/dev/null 2>&1 || { error "Docker not found. Install Docker Desktop for Mac."; exit 1; }
command -v python3 >/dev/null 2>&1 || { error "python3 not found. Run: brew install python@3.11"; exit 1; }

# ── Step 1: Python virtual environment ───────────────────────────────────────
if [ ! -f "${VENV_DIR}/bin/python3" ]; then
    info "Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
    success "Virtual environment created at ${VENV_DIR}"
else
    success "Virtual environment found"
fi

# All Python commands use the venv from here on
PYTHON="${VENV_DIR}/bin/python3"
PIP="${VENV_DIR}/bin/pip"

# ── Step 2: Install / verify dependencies ────────────────────────────────────
# Use a sentinel file so we only re-run pip when requirements.txt changes.
SENTINEL="${VENV_DIR}/.deps_installed"
if [ ! -f "$SENTINEL" ] || [ "$REQUIREMENTS" -nt "$SENTINEL" ]; then
    info "Installing Python dependencies (this takes a few minutes on first run)..."
    "$PIP" install --upgrade pip --quiet
    "$PIP" install -r "$REQUIREMENTS"
    touch "$SENTINEL"
    success "Dependencies installed"
else
    success "Dependencies up to date"
fi

# ── Step 3: Download DotsOCR model weights (first run only, ~7 GB) ───────────
if [ ! -d "$WEIGHTS_DIR" ] || [ -z "$(ls -A "$WEIGHTS_DIR" 2>/dev/null)" ]; then
    info "DotsOCR weights not found — downloading from HuggingFace (~7 GB)..."
    info "This is a one-time download. Go grab a coffee ☕"
    mkdir -p "$WEIGHTS_DIR"
    "$PYTHON" -c "
from huggingface_hub import snapshot_download
snapshot_download('rednote-hilab/dots.ocr', local_dir='${WEIGHTS_DIR}')
print('Download complete.')
"
    success "DotsOCR weights downloaded to ${WEIGHTS_DIR}"
else
    success "DotsOCR weights found at ${WEIGHTS_DIR}"
fi

# ── Step 4: Verify MPS (informational only) ───────────────────────────────────
"$PYTHON" -c "
import torch
if torch.backends.mps.is_available():
    print('  MPS available ✓  (80-core Apple GPU active)')
else:
    print('  MPS not available — falling back to CPU (much slower)')
" 2>/dev/null || true

# ── Step 5: Start Docker infrastructure ──────────────────────────────────────
info "Starting Redis + RabbitMQ in Docker..."
cd "$REPO_ROOT"
docker compose up redis rabbitmq -d --wait 2>/dev/null || docker compose up redis rabbitmq -d

# Wait for Redis
info "Waiting for Redis..."
until docker exec rag-redis redis-cli ping 2>/dev/null | grep -q PONG; do
    sleep 1
done
success "Redis ready"

# Wait for RabbitMQ
info "Waiting for RabbitMQ..."
for i in $(seq 1 30); do
    if docker exec rag-rabbitmq rabbitmq-diagnostics ping 2>/dev/null; then
        break
    fi
    sleep 2
done
success "RabbitMQ ready"

# ── Step 6: Set native environment ───────────────────────────────────────────
info "Configuring native Python environment (MPS)..."

# Load .env file
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

# Override with MPS-optimised values for native run
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/dots_ocr:${APP_DIR}"
export EMBEDDING_DEVICE="mps"
export EMBEDDING_BATCH_SIZE="256"
export UPLOAD_WORKERS="${WORKERS}"
export OMP_NUM_THREADS="8"          # Per-model CPU threads (M3 Ultra has 24 perf cores)
export MKL_NUM_THREADS="8"
export OPENBLAS_NUM_THREADS="8"
export TOKENIZERS_PARALLELISM="false"
export PYTORCH_ENABLE_MPS_FALLBACK="1"  # Fallback CPU for unsupported MPS ops
export LOG_LEVEL="${LOG_LEVEL:-INFO}"

# Redis + RabbitMQ are Docker containers — connect via localhost
export REDIS_HOST="localhost"
export RABBIT_HOST="localhost"

# Native paths (override Docker /app/* paths from .env)
export DOTS_OCR_WEIGHTS="${REPO_ROOT}/weights/DotsOCR"
export UPLOAD_DIR="${APP_DIR}/uploads"
mkdir -p "${UPLOAD_DIR}"

echo ""
info "Runtime config:"
echo "  Device      : MPS (Apple 80-core GPU)"
echo "  OCR threads : 3  (one DotsOCR model per thread, all on MPS)"
echo "  Preprocess  : 6  (CPU threads, OpenCV deskew/denoise/CLAHE)"
echo "  Embed batch : ${EMBEDDING_BATCH_SIZE}  (cross-doc batching, single GPU pass)"
echo "  OMP threads : ${OMP_NUM_THREADS}  (CPU intra-op, per model)"
echo "  PG          : ${PG_HOST}:${PG_PORT}/${PG_DATABASE}"
echo "  SeaweedFS   : ${SEAWEEDFS_S3_ENDPOINT}"
echo ""

# ── Step 7: Start FastAPI (api) natively ─────────────────────────────────────
info "Starting FastAPI API server (port 8000)..."

RUN_TYPE=api "$PYTHON" "${APP_DIR}/main.py" \
    > "${LOG_DIR}/api.log" 2>&1 &
API_PID=$!
echo "$API_PID" > "${PID_DIR}/api.pid"

sleep 3
if ! kill -0 "$API_PID" 2>/dev/null; then
    error "API failed to start. Check logs: ${LOG_DIR}/api.log"
    tail -30 "${LOG_DIR}/api.log"
    exit 1
fi
success "API running (pid=${API_PID})  →  http://localhost:8000"
success "Health check: curl http://localhost:8000/health"

# ── Step 8: Start Stage Pipeline worker (single process) ─────────────────────
#
#  The stage pipeline manages all concurrency internally:
#    Preprocess  : 6 CPU threads  (OpenCV)
#    OCR         : 3 MPS threads  (DotsOCR, one model per thread)
#    Assembler   : 1 CPU thread
#    Chunking    : 4 CPU threads
#    Embed       : 1 MPS thread   (256-chunk cross-doc batching)
#    Storage     : 4 IO  threads  (PostgreSQL)
#    MQ feeders  : 3 threads      (RabbitMQ consumer, no ML)
#
#  Total: 22 threads inside one Python process → full 80-core MPS utilisation.
#  No need to spawn multiple processes; MPS serialises command buffers anyway.
#
info "Starting ingestion worker (stage pipeline)..."

RUN_TYPE=worker "$PYTHON" "${APP_DIR}/main.py" \
    > "${LOG_DIR}/worker.log" 2>&1 &
W_PID=$!
echo "$W_PID" > "${PID_DIR}/workers.pid"
success "Worker started (pid=${W_PID})"

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   Pipeline is running!                             ║${NC}"
echo -e "${GREEN}╠════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║  API        : http://localhost:8000                ║${NC}"
echo -e "${GREEN}║  Health     : http://localhost:8000/health         ║${NC}"
echo -e "${GREEN}║  RabbitMQ   : http://localhost:15672 (guest/guest) ║${NC}"
echo -e "${GREEN}║  Stage      : 6 preprocess + 3 OCR + embed + store ║${NC}"
echo -e "${GREEN}╠════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║  Logs: ./logs/api.log  ./logs/worker.log           ║${NC}"
echo -e "${GREEN}║  Stop: ./stop.sh                                   ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════╝${NC}"
echo ""

# Keep script alive and tail logs
trap './stop.sh; exit 0' INT TERM
info "Tailing worker log (Ctrl+C to stop pipeline)..."
tail -f "${LOG_DIR}/worker.log"
