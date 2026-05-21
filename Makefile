.PHONY: help setup api stack-up stack-down eval eval-ragas eval-offline eval-synthetic ingest ingest-fda scrape-fda db-up db-down db-reset clean

help:
	@echo ""
	@echo "  HybridRAG — Available commands"
	@echo "  ──────────────────────────────────────────"
	@echo "  make setup            First-time setup (run this first)"
	@echo "  make stack-up         Start full stack: DB + API (Docker)"
	@echo "  make stack-down       Stop full stack"
	@echo "  make api              Start FastAPI server locally (port 8000)"
	@echo "  make ingest           Ingest sample documents (AI-generated)"
	@echo "  make scrape-fda       Scrape all FDA OPDP letters (full PDF parse)"
	@echo "  make scrape-fda-quick First 20 letters — good for testing"
	@echo "  make scrape-fda-year  Letters from 2024 only"
	@echo "  make ingest-fda       Ingest reviewed FDA data (run after scrape + review)"
	@echo "  make eval             Run standard eval (accuracy, F1, calibration, latency)"
	@echo "  make eval-ragas       Run full eval: RAGAS + standard metrics + slice breakdown"
	@echo "  make eval-offline     Run RAGAS eval in offline mode (no DB, no API keys needed)"
	@echo "  make eval-synthetic   Generate 40 synthetic cases + run full RAGAS eval"
	@echo "  make db-up            Start PostgreSQL container only"
	@echo "  make db-down          Stop PostgreSQL container"
	@echo "  make db-reset         Drop and recreate database (WARNING: deletes all data)"
	@echo "  make clean            Remove .venv and stop containers"
	@echo ""

setup:
	bash setup.sh

api:
	source .venv/bin/activate && uvicorn src.api.main:app --reload --port 8000

ingest:
	source .venv/bin/activate && python scripts/ingest_samples.py

scrape-fda:
	source .venv/bin/activate && pip install -q pdfplumber
	source .venv/bin/activate && python scripts/scrape_fda_opdp.py
	@echo ""
	@echo "  Review data/fda_opdp_raw.jsonl, then run: make ingest-fda"

scrape-fda-quick:
	source .venv/bin/activate && pip install -q pdfplumber
	source .venv/bin/activate && python scripts/scrape_fda_opdp.py --limit 20

scrape-fda-year:
	source .venv/bin/activate && pip install -q pdfplumber
	source .venv/bin/activate && python scripts/scrape_fda_opdp.py --year 2024

ingest-fda:
	@test -f data/fda_opdp_raw.jsonl || (echo "Run 'make scrape-fda' first" && exit 1)
	source .venv/bin/activate && python -c "from src.ingestion.ingest import ingest_from_jsonl; n = ingest_from_jsonl('data/fda_opdp_raw.jsonl'); print(f'Ingested {n} FDA documents')"

eval:
	source .venv/bin/activate && python -m src.eval.evaluate --data data/sample/test.jsonl

# Full RAGAS eval: all 5 metrics + standard metrics + slice breakdown (requires DB + OPENAI_API_KEY)
eval-ragas:
	@echo ""
	@echo "  Running RAGAS + standard eval on seed examples..."
	@echo "  Requires: DB running, GROQ_API_KEY set, OPENAI_API_KEY set (for RAGAS judge)"
	@echo ""
	source .venv/bin/activate && python -m src.eval.eval_ragas \
		--mode live \
		--data data/eval/seed_examples.jsonl \
		--output-dir data/eval/reports

# Offline eval: structure check with no DB or API keys (good for CI)
eval-offline:
	@echo ""
	@echo "  Running RAGAS eval in offline mode (mock data, no DB required)..."
	@echo ""
	source .venv/bin/activate && python -m src.eval.eval_ragas \
		--mode offline \
		--data data/eval/seed_examples.jsonl \
		--output-dir data/eval/reports

# Generate 40 synthetic cases from FDA data, then run full RAGAS eval
eval-synthetic:
	@echo ""
	@echo "  Generating synthetic eval cases from FDA data + running RAGAS eval..."
	@echo "  Requires: data/fda_opdp_raw.jsonl (run 'make scrape-fda' first)"
	@echo ""
	source .venv/bin/activate && python -m src.eval.eval_ragas \
		--mode live \
		--data data/eval/seed_examples.jsonl \
		--generate-synthetic \
		--n-synthetic 40 \
		--output-dir data/eval/reports

stack-up:
	docker compose up -d --build

stack-down:
	docker compose down

db-up:
	docker compose up -d db

db-down:
	docker compose down

db-reset:
	docker compose down -v
	docker compose up -d db
	@echo "Waiting for database..."
	@sleep 5
	@echo "Database reset complete."

clean:
	docker compose down
	rm -rf .venv
