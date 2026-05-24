"""
Ingestion Pipeline — Agent Harness Orchestrator
=================================================
Validates, deduplicates, and ingests any JSONL source into the knowledge base.
Designed to be the single entry point for all document ingestion.

Agent harness flow (LLM is called at most once per 10 borderline docs):
  1. Load JSONL                    — no API cost
  2. Rule-based validation         — no API cost
  3. Batch LLM for borderline      — 1 call per 10 docs (only if needed)
  4. Dedup by content_hash         — DB lookup, no API cost
  5. Ingest approved to DB         — BGE embed + Postgres insert
  6. Write review_queue to DB      — no API cost
  7. Report                        — prints summary table

Usage:
  # Ingest FDA enforcement letters (after running scraper)
  python scripts/run_ingestion_pipeline.py --input data/fda_opdp_raw.jsonl

  # Ingest PI documents (after running ingest_pi_documents.py)
  python scripts/run_ingestion_pipeline.py --input data/pi_claims_raw.jsonl

  # Dry run — validate but don't write to DB
  python scripts/run_ingestion_pipeline.py --input data/fda_opdp_raw.jsonl --dry-run

  # Skip LLM (rule-only validation, borderline → review queue)
  python scripts/run_ingestion_pipeline.py --input data/fda_opdp_raw.jsonl --skip-llm

  # Show what's currently in the review queue
  python scripts/run_ingestion_pipeline.py --show-review-queue

  # Approve a review_queue item by ID and ingest it
  python scripts/run_ingestion_pipeline.py --approve-review <id>

  # Run full pipeline: FDA scraper → PI fetch → validate → ingest
  python scripts/run_ingestion_pipeline.py --full-refresh
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path("data")


# ── Pipeline stages ────────────────────────────────────────────────────────────

def load_jsonl(path: str | Path) -> list[dict]:
    docs: list[dict] = []
    p = Path(path)
    if not p.exists():
        log.error(f"Input file not found: {p}")
        return []
    with open(p) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError as e:
                log.warning(f"  Skipping line {i}: {e}")
    log.info(f"Loaded {len(docs)} documents from {p}")
    return docs


def run_pipeline(
    input_path: str,
    dry_run: bool = False,
    skip_llm: bool = False,
    max_per_label: int | None = None,
) -> dict:
    """
    Full validate → dedup → ingest pipeline for one JSONL file.
    Returns summary counts dict.

    max_per_label: if set, caps how many docs of each label are ingested.
    Useful when one source (e.g. PI documents) overwhelms the KB with a
    single label. Docs beyond the cap are silently dropped — not rejected,
    not queued — they just aren't needed right now.
    """
    from src.ingestion.validate import validate_documents
    from src.ingestion.ingest import ingest_documents, ingest_from_jsonl
    from src.database import get_db, mark_url_ingested, insert_review_queue

    docs = load_jsonl(input_path)
    if not docs:
        return {"loaded": 0, "approved": 0, "review_queued": 0,
                "rejected": 0, "ingested": 0, "skipped": 0, "failed": 0}

    # Stage 1–4: validate (rule check + batched LLM for borderline)
    log.info("─" * 60)
    log.info("Stage 1-3: Rule-based validation...")
    result = validate_documents(docs, skip_llm=skip_llm)

    # Per-label cap — applied after validation, before ingestion
    # Keeps one source from overwhelming the KB with a single label
    if max_per_label is not None:
        from collections import Counter
        label_counts: Counter = Counter()
        capped: list[dict] = []
        dropped = 0
        for doc in result.approved:
            lbl = doc.get("label", "")
            if label_counts[lbl] < max_per_label:
                capped.append(doc)
                label_counts[lbl] += 1
            else:
                dropped += 1
        if dropped:
            log.info(
                f"Per-label cap ({max_per_label}): dropped {dropped} excess docs "
                f"— {len(capped)} remain for ingestion"
            )
        result.approved = capped

    log.info("─" * 60)
    log.info(f"Validation complete:")
    log.info(f"  Approved      : {len(result.approved)}")
    log.info(f"  Review queue  : {len(result.needs_review)}")
    log.info(f"  Rejected      : {len(result.rejected)}")

    if dry_run:
        log.info("─" * 60)
        log.info("DRY RUN — no writes to DB. Re-run without --dry-run to commit.")
        return {
            "loaded":        len(docs),
            "approved":      len(result.approved),
            "review_queued": len(result.needs_review),
            "rejected":      len(result.rejected),
            "ingested":      0,
            "skipped":       0,
            "failed":        0,
        }

    # Stage 5: ingest approved (with dedup built into ingest_documents)
    log.info("─" * 60)
    log.info("Stage 5: Ingesting approved documents...")
    ingest_counts = ingest_documents(result.approved)

    # Stage 6: write review queue to DB
    if result.needs_review:
        log.info(f"Stage 6: Writing {len(result.needs_review)} docs to review_queue...")
        with get_db() as db:
            for doc in result.needs_review:
                insert_review_queue(
                    db,
                    content=doc["text"],
                    label=doc["label"],
                    metadata=doc.get("metadata", {}),
                    content_hash=doc["_content_hash"],
                    reason=doc.get("_validation_note", "queued by pipeline"),
                )
        log.info("  Review queue updated")

    # Stage 7: mark source URLs as ingested
    log.info("Stage 7: Marking source URLs as ingested...")
    urls: dict[str, int] = {}
    for doc in result.approved:
        url = doc.get("metadata", {}).get("letter_url") or doc.get("metadata", {}).get("source_url", "")
        if url:
            urls[url] = urls.get(url, 0) + 1
    with get_db() as db:
        for url, count in urls.items():
            mark_url_ingested(db, source_url=url, doc_count=count, status="ingested")
    log.info(f"  Marked {len(urls)} source URLs as ingested")

    return {
        "loaded":        len(docs),
        "approved":      len(result.approved),
        "review_queued": len(result.needs_review),
        "rejected":      len(result.rejected),
        "ingested":      ingest_counts["ingested"],
        "skipped":       ingest_counts["skipped"],
        "failed":        ingest_counts["failed"],
    }


def show_review_queue() -> None:
    """Print all pending items in the review queue."""
    from src.database import get_db
    from sqlalchemy import text

    with get_db() as db:
        rows = db.execute(
            text("""
                SELECT id, label, reason, created_at,
                       LEFT(text, 120) AS preview
                FROM review_queue
                WHERE status = 'pending'
                ORDER BY created_at DESC
                LIMIT 50
            """)
        ).fetchall()

    if not rows:
        print("\n  Review queue is empty.\n")
        return

    print(f"\n{'─' * 72}")
    print(f"  Review Queue — {len(rows)} pending item(s)")
    print(f"{'─' * 72}")
    for row in rows:
        print(f"\n  [{row.id}] {row.label:<25} {str(row.created_at)[:16]}")
        print(f"       {row.reason}")
        print(f"       \"{row.preview}…\"")
    print(f"\n  To approve: python scripts/run_ingestion_pipeline.py --approve-review <id>")
    print(f"{'─' * 72}\n")


def approve_all_pending() -> None:
    """Approve and ingest every pending item in the review queue."""
    from src.database import get_db
    from src.ingestion.ingest import ingest_documents
    from sqlalchemy import text

    with get_db() as db:
        rows = db.execute(
            text("SELECT * FROM review_queue WHERE status = 'pending' ORDER BY id")
        ).fetchall()

    if not rows:
        print("\n  Review queue is empty — nothing to approve.\n")
        return

    print(f"\n  Approving {len(rows)} pending item(s)...")
    docs = [
        {"text": r.text, "label": r.label, "metadata": r.metadata,
         "_content_hash": r.content_hash}
        for r in rows
    ]
    counts = ingest_documents(docs)

    with get_db() as db:
        db.execute(
            text("UPDATE review_queue SET status='approved', reviewed_at=NOW() WHERE status='pending'")
        )

    print(f"  ✓ Ingested: {counts['ingested']}")
    print(f"  ↳ Skipped (already in DB): {counts['skipped']}")
    print(f"  ↳ Failed: {counts['failed']}\n")


def approve_review_item(item_id: int) -> None:
    """Move a review_queue item to the documents table (ingest it)."""
    from src.database import get_db, insert_review_queue
    from src.ingestion.ingest import ingest_documents
    from sqlalchemy import text

    with get_db() as db:
        row = db.execute(
            text("SELECT * FROM review_queue WHERE id = :id AND status = 'pending'"),
            {"id": item_id},
        ).fetchone()

    if not row:
        print(f"  No pending review item with ID {item_id}")
        return

    doc = {
        "text":           row.text,
        "label":          row.label,
        "metadata":       row.metadata,
        "_content_hash":  row.content_hash,
    }
    counts = ingest_documents([doc])

    if counts["ingested"] > 0:
        with get_db() as db:
            db.execute(
                text("""
                    UPDATE review_queue
                    SET status='approved', reviewed_at=NOW()
                    WHERE id = :id
                """),
                {"id": item_id},
            )
        print(f"\n  ✓ Item {item_id} approved and ingested.\n")
    else:
        print(f"\n  ⚠ Item {item_id} was already ingested (duplicate hash).\n")


def print_summary(counts: dict, input_path: str) -> None:
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    print("\n" + "═" * 64)
    print(f"  Ingestion Pipeline — Complete  ({now})")
    print("═" * 64)
    print(f"  Source           : {input_path}")
    print(f"  Loaded           : {counts['loaded']}")
    print(f"  ✓ Approved       : {counts['approved']}")
    print(f"  ✗ Rejected       : {counts['rejected']}")
    print(f"  ⚠ Review queued  : {counts['review_queued']}")
    print()
    print(f"  ↳ Ingested to DB : {counts['ingested']}")
    print(f"  ↳ Skipped (dedup): {counts['skipped']}")
    print(f"  ↳ Failed         : {counts['failed']}")
    print()
    if counts['review_queued'] > 0:
        print("  Review queue has items waiting — run:")
        print("    python scripts/run_ingestion_pipeline.py --show-review-queue")
    if counts['failed'] > 0:
        print(f"  ⚠ {counts['failed']} documents failed to ingest — check logs above")
    print("═" * 64 + "\n")


# ── Full refresh ───────────────────────────────────────────────────────────────

def full_refresh(skip_llm: bool = False) -> None:
    """
    Run the complete pipeline:
      1. Scrape new FDA enforcement letters (incremental — skips known URLs)
      2. Fetch PI documents for balanced labels
      3. Validate + ingest both sources
    """
    import subprocess

    log.info("═" * 60)
    log.info("FULL REFRESH — FDA enforcement letters + PI documents")
    log.info("═" * 60)

    # Step 1: scrape new FDA letters (scraper already has PDF cache)
    log.info("Step 1: Running FDA OPDP scraper (new letters only)...")
    subprocess.run(
        [sys.executable, "scripts/scrape_fda_opdp.py"],
        check=False,  # don't abort on scraper errors
    )

    # Step 2: fetch PI documents
    log.info("Step 2: Fetching FDA PI documents...")
    skip_flag = ["--skip-llm"] if skip_llm else []
    subprocess.run(
        [sys.executable, "scripts/ingest_pi_documents.py", "--limit", "100"] + skip_flag,
        check=False,
    )

    # Step 3: ingest both outputs
    for jsonl_path in [DATA_DIR / "fda_opdp_raw.jsonl", DATA_DIR / "pi_claims_raw.jsonl"]:
        if jsonl_path.exists():
            log.info(f"Step 3: Ingesting {jsonl_path}...")
            counts = run_pipeline(str(jsonl_path), skip_llm=skip_llm)
            print_summary(counts, str(jsonl_path))
        else:
            log.warning(f"  {jsonl_path} not found — skipping")

    log.info("Full refresh complete.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HybridRAG ingestion pipeline — validate, dedup, and ingest pharma claims"
    )
    parser.add_argument("--input",         type=str,  help="JSONL file to ingest")
    parser.add_argument("--dry-run",       action="store_true",
                        help="Validate only — don't write to DB")
    parser.add_argument("--skip-llm",      action="store_true",
                        help="Skip LLM validation (borderline docs → review queue)")
    parser.add_argument("--max-per-label", type=int,  default=None, metavar="N",
                        help="Cap ingestion at N docs per label (prevents one label dominating)")
    parser.add_argument("--show-review-queue", action="store_true",
                        help="Print pending items in the review queue")
    parser.add_argument("--approve-review",    type=int, metavar="ID",
                        help="Approve a single review queue item by ID and ingest it")
    parser.add_argument("--approve-all-pending", action="store_true",
                        help="Approve and ingest ALL pending review queue items at once")
    parser.add_argument("--full-refresh",  action="store_true",
                        help="Run scraper + PI fetch + ingest in one shot")
    args = parser.parse_args()

    if args.show_review_queue:
        show_review_queue()
        return

    if args.approve_review is not None:
        approve_review_item(args.approve_review)
        return

    if args.approve_all_pending:
        approve_all_pending()
        return

    if args.full_refresh:
        full_refresh(skip_llm=args.skip_llm)
        return

    if not args.input:
        parser.print_help()
        return

    counts = run_pipeline(
        input_path=args.input,
        dry_run=args.dry_run,
        skip_llm=args.skip_llm,
        max_per_label=args.max_per_label,
    )
    print_summary(counts, args.input)


if __name__ == "__main__":
    main()
