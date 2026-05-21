#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# HybridRAG — One-command setup
# Run: bash setup.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e  # exit on any error

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓ $1${NC}"; }
info() { echo -e "${BLUE}→ $1${NC}"; }
warn() { echo -e "${YELLOW}⚠ $1${NC}"; }
fail() { echo -e "${RED}✗ $1${NC}"; exit 1; }

echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "  HybridRAG Classifier — Environment Setup"
echo -e "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

# ── 1. Check prerequisites ────────────────────────────────────────────────────
info "Checking prerequisites..."

command -v python3 >/dev/null 2>&1 || fail "Python 3 not found. Install from https://python.org"
command -v docker >/dev/null 2>&1  || fail "Docker not found. Install from https://docker.com"
command -v docker compose >/dev/null 2>&1 || command -v docker-compose >/dev/null 2>&1 || fail "docker compose not found"
ok "Prerequisites OK (Python + Docker)"

# ── 2. Set up .env ────────────────────────────────────────────────────────────
if [ ! -f ".env" ]; then
  cp .env.example .env
  ok "Created .env from .env.example"
  warn "ACTION REQUIRED: Open .env and set your OPENAI_API_KEY before continuing"
  echo ""
  read -p "  Press Enter once you've added your API key to .env..."
else
  ok ".env already exists"
fi

# Verify Groq API key is set
if grep -q "your_groq_api_key_here" .env; then
  fail "GROQ_API_KEY is still set to the placeholder. Add your real key from https://console.groq.com/keys"
fi

# ── 3. Python virtual environment ─────────────────────────────────────────────
info "Setting up Python virtual environment..."
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
  ok "Created .venv"
fi

source .venv/bin/activate
pip install --quiet --upgrade pip
ok "Virtual environment activated"

# ── 4. Install dependencies ───────────────────────────────────────────────────
info "Installing Python dependencies (this may take a few minutes)..."
pip install --quiet -r requirements.txt
ok "Dependencies installed"

# ── 5. Start PostgreSQL + pgvector ────────────────────────────────────────────
info "Starting PostgreSQL + pgvector with Docker..."
docker compose up -d db

info "Waiting for database to be ready..."
until docker compose exec db pg_isready -U hybridrag -q 2>/dev/null; do
  printf "."
  sleep 1
done
echo ""
ok "Database is ready"

# ── 6. Download embedding model ───────────────────────────────────────────────
info "Downloading BGE embedding model (~1.3 GB, cached after first download)..."
python3 - <<'EOF'
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-large-en-v1.5")
print("  Model downloaded and cached.")
EOF
ok "Embedding model ready"

# ── 7. Ingest sample documents ────────────────────────────────────────────────
info "Ingesting 50 sample pharma MLR claims..."
python3 scripts/ingest_samples.py
ok "Sample data ingested"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "  Setup complete! 🎉"
echo -e "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  Start the API:   make api  (local)  or  make stack-up  (Docker)"
echo "  Run evals:       make eval"
echo "  All commands:    make help"
echo ""
