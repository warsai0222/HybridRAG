"""
Label Distribution Diagnostic
==============================
Shows label counts + percentages across:
  1. fda_opdp_raw.jsonl  (raw scraped data)
  2. seed_examples.jsonl (hand-crafted eval set)
  3. PostgreSQL documents table (live KB)

Run from the HybridRAG project root:
  python scripts/diagnose_label_skew.py

Flags severe imbalance (any label < 10% or > 50% of total).
"""

import json
import os
from collections import Counter
from pathlib import Path

# ── ANSI colors ────────────────────────────────────────────────────────────────
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

IMBALANCE_WARN   = 0.10   # < 10% → warning
IMBALANCE_SEVERE = 0.05   # <  5% → severe


def count_labels_jsonl(path: str | Path) -> Counter:
    counts: Counter = Counter()
    p = Path(path)
    if not p.exists():
        return counts
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                    # Support both field names used across different JSONL sources:
                    # 'label'          — fda_opdp_raw.jsonl, pi_claims_raw.jsonl
                    # 'expected_label' — seed_examples.jsonl (eval set)
                    label = rec.get("label") or rec.get("expected_label", "UNKNOWN")
                    counts[label] += 1
                except json.JSONDecodeError:
                    pass
    return counts


def count_labels_db() -> Counter | None:
    """Query the live PostgreSQL documents table."""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.database import get_db
        from sqlalchemy import text
        counts: Counter = Counter()
        with get_db() as db:
            rows = db.execute(text("SELECT label, COUNT(*) FROM documents GROUP BY label")).fetchall()
            for label, cnt in rows:
                counts[label] = cnt
        return counts
    except Exception as e:
        print(f"  DB query failed: {e}")
        return None


def print_distribution(title: str, counts: Counter, total_labels: list[str] | None = None) -> None:
    total = sum(counts.values())
    if total == 0:
        print(f"\n{BOLD}{title}{RESET}")
        print("  (no data)")
        return

    all_labels = sorted(set(list(counts.keys()) + (total_labels or [])))
    bar_max = 40

    print(f"\n{BOLD}{title}{RESET}  (n={total:,})")
    print("  " + "─" * 70)
    for label in all_labels:
        count = counts.get(label, 0)
        pct   = count / total if total > 0 else 0
        bar   = "█" * int(pct * bar_max)
        pct_str = f"{pct*100:5.1f}%"

        if pct < IMBALANCE_SEVERE:
            color = RED
            flag  = "  ← SEVERE imbalance"
        elif pct < IMBALANCE_WARN:
            color = YELLOW
            flag  = "  ← low representation"
        else:
            color = GREEN
            flag  = ""

        print(f"  {label:<25} {count:>5,}  {color}{pct_str}  {bar:<40}{RESET}{flag}")

    print()

    # Imbalance ratio
    if len(counts) >= 2:
        max_count = max(counts.values())
        min_label = min(counts, key=counts.get)
        min_count = counts[min_label]
        ratio = max_count / max(min_count, 1)
        color = RED if ratio > 10 else YELLOW if ratio > 5 else GREEN
        print(f"  Imbalance ratio (max/min): {color}{ratio:.1f}x{RESET}")
        if ratio > 5:
            print(f"  {RED}→ Classifier will be biased toward majority labels.{RESET}")
            print(f"    Add more '{min_label}' examples to balance the KB.")


def main() -> None:
    project_root = Path(__file__).parent.parent
    raw_jsonl  = project_root / "data" / "fda_opdp_raw.jsonl"
    seed_jsonl = project_root / "data" / "eval" / "seed_examples.jsonl"

    all_labels = [
        "supported", "partially_supported", "unsupported",
        "false_balance", "needs_legal_review", "insufficient_data",
    ]

    print(f"\n{'═' * 72}")
    print(f"  {BOLD}HybridRAG — Label Distribution Diagnostic{RESET}")
    print(f"{'═' * 72}")

    # ── 1. Raw scraped data ───────────────────────────────────────────────────
    raw_counts = count_labels_jsonl(raw_jsonl)
    if raw_counts:
        print_distribution("1. Scraped FDA Data  (fda_opdp_raw.jsonl)", raw_counts, all_labels)
    else:
        print(f"\n  {YELLOW}fda_opdp_raw.jsonl not found — run: make ingest-fda{RESET}")

    # ── 2. Seed eval set ─────────────────────────────────────────────────────
    seed_counts = count_labels_jsonl(seed_jsonl)
    if seed_counts:
        print_distribution("2. Seed Eval Set  (seed_examples.jsonl)", seed_counts, all_labels)
    else:
        print(f"\n  {YELLOW}seed_examples.jsonl not found{RESET}")

    # ── 3. Live DB ────────────────────────────────────────────────────────────
    print("\n3. Live Knowledge Base  (PostgreSQL documents table)")
    db_counts = count_labels_db()
    if db_counts is not None:
        print_distribution("   PostgreSQL — documents", db_counts, all_labels)
    else:
        print(f"  {YELLOW}DB unavailable — is the stack running? (make stack-up){RESET}")

    # ── 4. Fix recommendations ────────────────────────────────────────────────
    print(f"\n{'─' * 72}")
    print(f"  {BOLD}To improve KB balance:{RESET}")
    print("""
  Scrape more enforcement history (adds false_balance, unsupported, needs_legal_review):
    python scripts/scrape_fda_opdp.py --type untitled --years 2018-2021

  Add more compliant examples (adds partially_supported, supported):
    python scripts/ingest_pi_documents.py --skip-llm

  Re-ingest with caps after scraping:
    python scripts/run_ingestion_pipeline.py --input data/fda_opdp_raw.jsonl --skip-llm --max-per-label 100
    python scripts/run_ingestion_pipeline.py --input data/pi_claims_raw.jsonl --skip-llm --max-per-label 100

  Approve supported examples from review queue:
    python scripts/run_ingestion_pipeline.py --input data/pi_claims_review.jsonl --skip-llm
    python scripts/run_ingestion_pipeline.py --approve-all-pending

  Target: imbalance ratio < 3x, no active label < 10% of KB total
""")
    print(f"{'═' * 72}\n")


if __name__ == "__main__":
    main()
