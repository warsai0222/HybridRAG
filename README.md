# HybridRAG — Pharma Claim Compliance Classifier

A production-grade MLR (Medical-Legal-Regulatory) claim review system built on hybrid retrieval-augmented generation. Given a pharmaceutical marketing claim, it retrieves semantically similar claims from a reference library and uses GPT-4o to assign a compliance verdict — with chain-of-thought regulatory reasoning, confidence scoring, and automatic escalation for human review.

Built as a direct architectural mirror of [Solstice Health's Fact Check Assistant](https://solsticehealth.ai): a system that annotates biopharma marketing claims against a curated reference library to accelerate MLR review cycles.

## What It Does

**Input:** A pharmaceutical marketing claim  
**Output:** A compliance verdict, confidence score, and step-by-step regulatory reasoning

**Verdicts:**
| Label | Meaning |
|---|---|
| `supported` | Claim is directly backed by cited clinical evidence |
| `unsupported` | Claim overstates efficacy, uses absolute language, or has no evidence basis |
| `partially_supported` | Directionally correct but missing hedges (population, CI, p-value) |
| `false_balance` | Presents a serious safety risk as equivalent to a minor side effect |
| `needs_legal_review` | Off-label, comparative, biosimilarity, or pre-approval concerns |

## Architecture

```
Marketing Claim
       │
       ├── Dense Retrieval   (BGE-large embeddings + pgvector cosine search)
       ├── Sparse Retrieval  (BM25 keyword matching)
       │
       └── RRF Fusion        (Reciprocal Rank Fusion — weighted score merge)
                 │
                 └── Groq (Llama 3.3 70B)  — FDA/OPDP-aware MLR reasoning + verdict + confidence
                           │
                           ├── Compliance verdict (one of 5 labels)
                           ├── Confidence score (0–1)
                           ├── Chain-of-thought regulatory analysis
                           └── Human review flag (if confidence < 0.70)
```

The retrieval layer finds similar historical claims from the reference library. Groq's Llama 3.3 70B sees those as few-shot examples and reasons against them using FDA promotional guidelines (21 CFR Part 202) and OPDP standards. Claims below 0.70 confidence are automatically flagged for human MLR reviewer escalation.

## Quickstart

**Prerequisites:** Python 3.11+, Docker

```bash
# 1. Enter the project folder
cd HybridRAG

# 2. One-command setup (installs deps, starts DB, downloads BGE model, ingests sample claims)
bash setup.sh

# 3. Start the API
make api          # → http://localhost:8000/docs
```

## Usage

### Classify a claim via API
```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Drug X is the most effective treatment available for rheumatoid arthritis today."}'

# → { "label": "unsupported", "confidence": 0.91, "needs_human_review": false, "reasoning": "..." }
```

### Ingest your own reference claims
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"documents": [{"text": "In the Phase III trial, drug X reduced events by 34% vs placebo (p<0.001).", "label": "supported"}]}'
```

### Run the eval harness
```bash
make eval
# → Outputs accuracy, macro F1, per-label F1, calibration curve, and p95 latency
```

## Project Structure

```
HybridRAG/
├── src/
│   ├── config.py           # Settings (loaded from .env)
│   ├── database.py         # PostgreSQL + pgvector queries
│   ├── ingestion/
│   │   └── ingest.py       # Claim ingestion + BGE embedding
│   ├── retrieval/
│   │   ├── dense.py        # Dense retrieval (BGE + pgvector HNSW)
│   │   ├── sparse.py       # Sparse retrieval (BM25)
│   │   └── fusion.py       # Reciprocal Rank Fusion
│   ├── classifier/
│   │   └── classify.py     # GPT-4o MLR classification head (CoT + confidence)
│   ├── eval/
│   │   └── evaluate.py     # Eval harness (accuracy, F1, calibration, latency)
│   └── api/
│       └── main.py         # FastAPI app
├── scripts/
│   ├── init.sql            # Database schema
│   └── ingest_samples.py   # 50 sample pharma claims (10 per compliance label)
├── docker-compose.yml      # PostgreSQL + pgvector
├── requirements.txt
├── setup.sh                # One-command setup
└── Makefile                # Convenient commands
```

## Swapping the Reference Library

The reference library is fully pluggable. To load your own claims:

1. Update `LABELS` in `.env` to your compliance taxonomy
2. Replace sample documents in `scripts/ingest_samples.py`
3. Run `make db-reset && make ingest`

## Tech Stack

| Component | Technology |
|---|---|
| Embeddings | `BAAI/bge-large-en-v1.5` (sentence-transformers, 1024-dim) |
| Vector DB | PostgreSQL + pgvector (HNSW index, cosine similarity) |
| Sparse retrieval | BM25 (rank-bm25) |
| Score fusion | Reciprocal Rank Fusion (k=60) |
| LLM | Groq — Llama 3.3 70B (OpenAI-compatible client) |
| API | FastAPI |
| Eval | scikit-learn + RAGAS + custom harness |
