"""
FDA Prescribing Information (PI) Ingestion Script
===================================================
Pulls drug labels from the FDA OpenFDA Label API and extracts genuine
pharmaceutical marketing claims to balance the knowledge base.

Why this source?
  The FDA OPDP scraper only produces violation labels (false_balance,
  unsupported, needs_legal_review). The FDA Label API gives us FDA-approved
  prescribing information — the gold standard for what a *compliant* claim
  looks like. Clinical study results → 'supported'. Vague efficacy language
  without full stats → 'partially_supported'.

Source: https://api.fda.gov/drug/label.json
  - Public API, no key required
  - Returns structured JSON with sections (indications, clinical_studies, etc.)
  - Covers thousands of approved drugs across all therapeutic areas

Label assignment logic:
  supported           ← from clinical_studies section + has stats (p-value, CI, HR)
  partially_supported ← from indications_and_usage + missing full stats
  (false_balance, unsupported, needs_legal_review are NOT produced here —
   those come exclusively from FDA enforcement letters)

Output:
  data/pi_claims_raw.jsonl      ← validated approved claims (ready to ingest)
  data/pi_claims_review.jsonl   ← needs human confirmation before ingest
  data/pi_claims_rejected.jsonl ← dropped with reason

Usage:
  python scripts/ingest_pi_documents.py                    # default 100 labels
  python scripts/ingest_pi_documents.py --limit 50         # 50 labels
  python scripts/ingest_pi_documents.py --skip-llm         # rule-only validation
  python scripts/ingest_pi_documents.py --disease diabetes # filter by indication
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path

import requests

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
FDA_LABEL_API = "https://api.fda.gov/drug/label.json"
REQUEST_DELAY = 1.0   # seconds — be polite
API_PAGE_SIZE = 10    # labels per API call (max 100, but smaller = more diverse)

OUTPUT_DIR      = Path("data")
PI_RAW          = OUTPUT_DIR / "pi_claims_raw.jsonl"
PI_REVIEW       = OUTPUT_DIR / "pi_claims_review.jsonl"
PI_REJECTED     = OUTPUT_DIR / "pi_claims_rejected.jsonl"

# Sections in FDA labels and how they map to labels
# clinical_studies contains trial results → supported
# indications_and_usage contains approved use claims → partially_supported (often lacks stats)
SECTION_LABEL_MAP = {
    "clinical_studies":       "supported",
    "indications_and_usage":  "partially_supported",
}

# Regex that signals a claim has proper statistical backing → 'supported'
STATS_RE = re.compile(
    r"(?:"
    r"p\s*[<=>]\s*0\.\d+"          # p-value
    r"|(?:95|90|99)\s*%\s*ci"      # confidence interval
    r"|hazard\s+ratio"             # HR
    r"|odds\s+ratio"               # OR
    r"|\bhr\b\s*(?:=|of)\s*0\.\d+"
    r"|\d+(?:\.\d+)?\s*%\s+(?:vs|versus|compared)"  # "X% vs placebo"
    r"|randomized\s+controlled\s+trial"
    r"|phase\s+(?:2|3|ii|iii)"
    r")",
    re.IGNORECASE,
)

# Hedging language that signals partial support
HEDGE_RE = re.compile(
    r"\b(?:may|might|can|could|appears?\s+to|suggest[s]?|associated\s+with"
    r"|in\s+(?:some|certain)\s+patients?|in\s+clinical\s+studies?)\b",
    re.IGNORECASE,
)

# Sentence splitter — rough but good enough for PI text
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

# Boilerplate phrases common in PI sections that aren't claims
PI_BOILERPLATE_RE = re.compile(
    r"(?:"
    r"see\s+(full\s+)?prescribing\s+information"
    r"|the\s+recommended\s+dose\s+is"
    r"|dosage\s+and\s+administration"
    r"|refer\s+to"
    r"|table\s+\d+"
    r"|figure\s+\d+"
    r"|for\s+more\s+information"
    r"|not\s+all\s+adverse\s+reactions"
    r"|most\s+common\s+adverse\s+reactions"
    r")",
    re.IGNORECASE,
)

MIN_SENT_LEN = 40
MAX_SENT_LEN = 500


# ── FDA Label API ──────────────────────────────────────────────────────────────

def fetch_labels(limit: int = 100, search: str | None = None) -> list[dict]:
    """
    Fetch drug labels from the FDA OpenFDA Label API.
    Returns list of raw label dicts.
    """
    all_labels: list[dict] = []
    skip = 0

    while len(all_labels) < limit:
        batch = min(API_PAGE_SIZE, limit - len(all_labels))
        params: dict = {"limit": batch, "skip": skip}
        if search:
            params["search"] = search

        try:
            resp = requests.get(FDA_LABEL_API, params=params, timeout=20)
            if resp.status_code == 404:
                log.warning("No more results from FDA Label API")
                break
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if not results:
                break
            all_labels.extend(results)
            skip += len(results)
            log.info(f"  Fetched {len(all_labels)}/{limit} labels")
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            log.error(f"FDA Label API error: {e}")
            break

    return all_labels[:limit]


# ── Claim extraction ───────────────────────────────────────────────────────────

def _extract_section_text(label: dict, section_key: str) -> str:
    """
    Extract text from a label section.
    FDA labels store section text as a list of strings — join them.
    """
    raw = label.get(section_key, [])
    if isinstance(raw, list):
        return " ".join(raw)
    if isinstance(raw, str):
        return raw
    return ""


def _clean_pi_text(text: str) -> str:
    """Strip HTML tags, normalise whitespace, remove section markers."""
    text = re.sub(r"<[^>]+>", " ", text)             # strip HTML
    text = re.sub(r"\n+", " ", text)                  # flatten newlines
    text = re.sub(r"\s{2,}", " ", text)               # collapse spaces
    text = re.sub(r"\b\d+\s+INDICATIONS?\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d+\s+CLINICAL\s+STUDIES?\b", "", text, flags=re.IGNORECASE)
    return text.strip()


def _assign_label(sentence: str, section_key: str) -> str:
    """
    Assign compliance label based on section + sentence content.
    clinical_studies with stats → supported
    Everything else → partially_supported
    """
    if section_key == "clinical_studies" and STATS_RE.search(sentence):
        return "supported"
    return "partially_supported"


def extract_claims_from_label(label: dict) -> list[dict]:
    """
    Extract compliance-relevant claim sentences from an FDA drug label.
    Returns list of document dicts ready for validation.
    """
    # Pull drug name + application number for metadata
    openfda   = label.get("openfda", {})
    drug_name = (openfda.get("brand_name", ["Unknown"])[0]
                 if openfda.get("brand_name") else "Unknown")
    app_no    = (openfda.get("application_number", [""])[0]
                 if openfda.get("application_number") else "")
    set_id    = label.get("set_id", "")

    source_url = (
        f"https://labels.fda.gov/fdaLabel/ContentExtraction?"
        f"setId={set_id}" if set_id else "labels.fda.gov/internal"
    )

    records: list[dict] = []

    for section_key, base_label in SECTION_LABEL_MAP.items():
        raw_text = _extract_section_text(label, section_key)
        if not raw_text:
            continue

        clean_text = _clean_pi_text(raw_text)
        sentences  = SENT_RE.split(clean_text)

        for sent in sentences:
            sent = sent.strip()

            # Basic length filter
            if not (MIN_SENT_LEN <= len(sent) <= MAX_SENT_LEN):
                continue

            # Skip PI boilerplate
            if PI_BOILERPLATE_RE.search(sent):
                continue

            # Re-score label based on actual sentence content
            label_assigned = _assign_label(sent, section_key)

            records.append({
                "text":  sent,
                "label": label_assigned,
                "metadata": {
                    "source":       "fda_label_api",
                    "source_url":   source_url,
                    "drug":         drug_name,
                    "application":  app_no,
                    "set_id":       set_id,
                    "section":      section_key,
                    "has_stats":    bool(STATS_RE.search(sent)),
                    "has_hedging":  bool(HEDGE_RE.search(sent)),
                },
            })

    return records


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch FDA prescribing information and extract compliant claim examples"
    )
    parser.add_argument("--limit",    type=int,  default=100,
                        help="Max drug labels to fetch (default: 100)")
    parser.add_argument("--disease",  type=str,  default=None,
                        help="Filter labels by indication keyword (e.g. 'diabetes')")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip LLM validation — borderline docs go to review queue")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Lazy import to avoid circular dependency when running standalone
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.ingestion.validate import validate_documents

    # ── 1. Fetch labels from FDA API ──────────────────────────────────────────
    search = f"indications_and_usage:{args.disease}" if args.disease else None
    log.info(f"Fetching up to {args.limit} drug labels from FDA Label API...")
    labels = fetch_labels(limit=args.limit, search=search)
    log.info(f"Fetched {len(labels)} labels")

    # ── 2. Extract claims ─────────────────────────────────────────────────────
    all_docs: list[dict] = []
    for label in labels:
        docs = extract_claims_from_label(label)
        all_docs.extend(docs)

    log.info(f"Extracted {len(all_docs)} candidate claims from {len(labels)} labels")

    if not all_docs:
        log.error("No claims extracted — check API response structure")
        return

    # ── 3. Validate ───────────────────────────────────────────────────────────
    log.info("Running validation pipeline...")
    result = validate_documents(all_docs, skip_llm=args.skip_llm)

    # ── 4. Write outputs ──────────────────────────────────────────────────────
    def write_jsonl(path: Path, docs: list[dict]) -> None:
        with open(path, "w") as f:
            for doc in docs:
                # Strip internal validation keys before writing
                clean = {k: v for k, v in doc.items() if not k.startswith("_")}
                f.write(json.dumps(clean) + "\n")

    write_jsonl(PI_RAW,      result.approved)
    write_jsonl(PI_REVIEW,   result.needs_review)
    write_jsonl(PI_REJECTED, result.rejected)

    # ── 5. Summary ────────────────────────────────────────────────────────────
    from collections import Counter
    approved_labels = Counter(d["label"] for d in result.approved)
    review_labels   = Counter(d["label"] for d in result.needs_review)

    print("\n" + "─" * 64)
    print("  FDA PI Ingestion — Complete")
    print("─" * 64)
    print(f"  Labels fetched          : {len(labels)}")
    print(f"  Candidate claims        : {len(all_docs)}")
    print(f"  ✓ Approved (ready)      : {len(result.approved)}")
    print(f"  ⚠ Queued for review     : {len(result.needs_review)}")
    print(f"  ✗ Rejected              : {len(result.rejected)}")
    print()
    print("  Approved by label:")
    for lbl, cnt in sorted(approved_labels.items()):
        print(f"    {lbl:<25} {cnt}")
    if review_labels:
        print("  Review queue by label:")
        for lbl, cnt in sorted(review_labels.items()):
            print(f"    {lbl:<25} {cnt}")
    print()
    print(f"  Ready to ingest → {PI_RAW}")
    print(f"  Review needed   → {PI_REVIEW}")
    print(f"  Rejected log    → {PI_REJECTED}")
    print()
    print("  Next step:")
    print("    python scripts/run_ingestion_pipeline.py --input data/pi_claims_raw.jsonl")
    print("─" * 64 + "\n")


if __name__ == "__main__":
    main()
