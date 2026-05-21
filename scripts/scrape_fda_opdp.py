"""
FDA OPDP Enforcement Letter Scraper  (v2 — full PDF parsing)
=============================================================
Pulls every Untitled Letter from the FDA OPDP enforcement page, downloads
each letter as a PDF, extracts the text with pdfplumber, parses out the
specific marketing claims the FDA cited as violative, maps them to one of
the 5 HybridRAG compliance labels, and writes a review-ready JSONL file.

Output
------
  data/fda_opdp_raw.jsonl      ← review this before ingesting
  data/fda_opdp_skipped.jsonl  ← letters we couldn't parse
  data/pdf_cache/              ← cached PDFs (re-runs skip downloads)

Usage
-----
  python scripts/scrape_fda_opdp.py              # all letters
  python scripts/scrape_fda_opdp.py --limit 20   # first 20 letters
  python scripts/scrape_fda_opdp.py --year 2023  # one year only
  python scripts/scrape_fda_opdp.py --no-cache   # re-download everything

FDA letter structure (what we're parsing)
-----------------------------------------
  1. Header  — date, company, address, RE: drug / NDA number
  2. Intro   — describes the promotional material reviewed
  3. Background — approved indication
  4. Issue sections — one per violation type, each titled e.g.:
       "False or Misleading Efficacy Claims"
       "Omission of Risk Information"
       "Misleading Comparative Claims"
     Under each section the letter either:
       a) Lists bullet-point claims  (• "claim text")
       b) Quotes inline claims       ("claim text")
       c) Describes claims in prose  (no clean quote boundary)
  5. Conclusion / corrective action request

Label mapping (section title → label)
--------------------------------------
  false_balance        ← omission / risk / one-sided / fair balance
  needs_legal_review   ← off-label / superiority / comparative / unapproved
  unsupported          ← unsubstantiated / no evidence / false / misleading
  partially_supported  ← overstated / cherry-picked / out of context
  supported            ← fallback for compliant language (rarely appears)
"""

import argparse
import hashlib
import io
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pdfplumber
import requests
from bs4 import BeautifulSoup

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
BASE_URL = "https://www.fda.gov"

# Correct listing URL (verified May 2026)
LISTING_URL = (
    "https://www.fda.gov/drugs/warning-letters-and-notice-violation-letters-"
    "pharmaceutical-companies/untitled-letters"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

REQUEST_DELAY = 2.0   # seconds between requests — be polite to FDA servers

OUTPUT_DIR   = Path("data")
PDF_CACHE    = OUTPUT_DIR / "pdf_cache"
RAW_OUTPUT   = OUTPUT_DIR / "fda_opdp_raw.jsonl"
SKIP_OUTPUT  = OUTPUT_DIR / "fda_opdp_skipped.jsonl"

# ── Label rules ────────────────────────────────────────────────────────────────
# Each rule: (list of trigger keywords, label).
# Applied to the section header + surrounding paragraph text.
# Order matters — first match wins.

LABEL_RULES: list[tuple[list[str], str]] = [
    (
        [
            "omission of risk", "omit", "omits", "omitting", "omission",
            "risk information", "important safety information",
            "side effect", "adverse", "contraindication",
            "boxed warning", "black box warning",
            "minimizes", "minimizing", "downplays", "downplaying",
            "misleadingly omit", "fails to present", "fails to include",
            "one-sided", "one sided", "fair balance",
            "misleading safety", "safety concern",
        ],
        "false_balance",
    ),
    (
        [
            "off-label", "off label", "unapproved use", "unapproved indication",
            "not approved", "not fda-approved",
            "superiority", "superior to", "better than",
            "comparative claim", "comparative efficacy",
            "head-to-head", "head to head",
            "promotes for use", "promoting for use",
            "lacks substantial evidence for",
            "broadens the indication",
        ],
        "needs_legal_review",
    ),
    (
        [
            "unsubstantiated", "no substantial evidence",
            "lacks adequate", "no adequate and well-controlled",
            "not supported by", "no clinical evidence",
            "no evidence", "unsupported claim",
            "false or misleading", "false and misleading",
            "misleadingly implies", "misbranded",
            "no data", "lacks data",
        ],
        "unsupported",
    ),
    (
        [
            "overstat", "exaggerat",
            "cherry-pick", "cherry pick",
            "selective presentation", "selectively present",
            "out of context", "taken out of context",
            "inconsistent with", "not consistent with",
            "misrepresent", "distort",
            "misleading presentation", "misleading efficacy",
        ],
        "partially_supported",
    ),
]

# Section headers in FDA letters that signal a violation block
VIOLATION_SECTION_RE = re.compile(
    r"^(Issue[s]?|Violation[s]?|"
    r"False or Misleading|Unsubstantiated|Omission of Risk|"
    r"Misleading|Off-Label|Superiority|Comparative|Overstated|"
    r"Broadens|Lacks|No Substantial)",
    re.IGNORECASE | re.MULTILINE,
)

# ── HTTP session ───────────────────────────────────────────────────────────────

def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def fetch(session: requests.Session, url: str, retries: int = 3,
          stream: bool = False) -> requests.Response | None:
    """GET with retry + polite delay. Returns None on permanent failure."""
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=30, stream=stream)
            if resp.status_code == 200:
                return resp
            if resp.status_code in (404, 410):
                log.warning(f"  [HTTP {resp.status_code}] {url}")
                return None
            log.warning(f"  [HTTP {resp.status_code}] attempt {attempt+1}/{retries} — {url}")
        except requests.RequestException as e:
            log.warning(f"  Request error ({e}) — attempt {attempt+1}/{retries}")
        time.sleep(REQUEST_DELAY * (attempt + 1))
    log.error(f"  Giving up after {retries} attempts: {url}")
    return None


# ── Listing page ───────────────────────────────────────────────────────────────

def get_letter_index(session: requests.Session,
                     year_filter: int | None = None) -> list[dict]:
    """
    Scrape the OPDP untitled-letters listing page.
    Returns list of letter metadata dicts including the PDF download URL.

    The listing is a single HTML table with columns:
      Issued Date | Company / Individual | Product / Issue | Response | Close-Out
    Every untitled letter link is a PDF at https://www.fda.gov/media/<id>/download
    """
    log.info(f"Fetching listing page...")
    resp = fetch(session, LISTING_URL)
    if not resp:
        raise RuntimeError(f"Cannot reach FDA OPDP listing page: {LISTING_URL}")

    soup = BeautifulSoup(resp.text, "lxml")
    table = soup.find("table")
    if not table:
        raise RuntimeError("No table found on listing page — FDA may have changed layout")

    rows = table.find_all("tr")
    letters: list[dict] = []

    for row in rows[1:]:   # skip header row
        cells = row.find_all("td")
        if len(cells) < 3:
            continue

        # ── Table layout (verified from live FDA page, May 2026) ──────────────
        # cells[0] = Issued Date  (plain text, e.g. "4/8/2026")
        # cells[1] = Company / Individual  +  letter download links as sub-list
        # cells[2] = Product / Issue  (drug trade name + generic + route)
        # cells[3] = Response Letter
        # cells[4] = Close-Out Letter

        # ── Date ─────────────────────────────────────────────────────────────
        date_text = cells[0].get_text(strip=True)
        # Normalise "4/8/2026" → keep as-is; also handles "01/18/2024"
        if not re.search(r"\d{4}", date_text):
            date_text = ""

        # Year filter
        if year_filter and str(year_filter) not in date_text:
            continue

        # ── Letter PDF link — lives in cells[1] alongside company name ───────
        company_cell = cells[1]
        letter_link = None

        # Priority: link whose visible text says "Untitled Letter"
        for a in company_cell.find_all("a", href=True):
            link_text = a.get_text(strip=True).lower()
            title_attr = a.get("title", "").lower()
            href = a["href"]
            if "/media/" in href and (
                "untitled" in link_text or "untitled" in title_attr
            ):
                letter_link = urljoin(BASE_URL, href)
                break

        # Fallback: first /media/ download link in company cell
        if not letter_link:
            for a in company_cell.find_all("a", href=True):
                if "/media/" in a["href"] and "download" in a["href"]:
                    letter_link = urljoin(BASE_URL, a["href"])
                    break

        if not letter_link:
            log.debug(f"  No PDF link found: {date_text} {company_cell.get_text(strip=True)[:50]}")
            continue

        # ── Company name — first text node before the bullet list ─────────────
        # get_text gives us everything; take just the first non-empty line
        company_lines = [
            ln.strip() for ln in company_cell.get_text(separator="\n").splitlines()
            if ln.strip() and "Untitled Letter" not in ln and "Promotional" not in ln
        ]
        company = company_lines[0] if company_lines else ""

        # ── Drug name from cells[2] ───────────────────────────────────────────
        drug_raw = cells[2].get_text(separator=" ", strip=True)
        # Strip NDA/BLA/ANDA numbers
        drug = re.sub(r"(?:NDA|BLA|ANDA)\s*\d+[,\s]*", "", drug_raw).strip()
        # Keep just the trade name (UPPERCASE before the first parenthesis or "for")
        drug_trade = re.match(r"^([A-Z][A-Z0-9®™\s\-\/]+?)(?:\s*\(|\s+for\s|,|$)", drug)
        drug = drug_trade.group(1).strip() if drug_trade else drug[:60]

        letters.append({
            "pdf_url":     letter_link,
            "date":        date_text,
            "company":     company,
            "drug":        drug,
            "letter_type": "untitled_letter",
        })

    log.info(f"Found {len(letters)} untitled letters on listing page")
    return letters


# ── PDF download & cache ───────────────────────────────────────────────────────

def pdf_cache_path(url: str) -> Path:
    """Deterministic local path for a PDF URL."""
    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    # Try to get a readable filename from the URL
    path_part = urlparse(url).path.rstrip("/").split("/")[-1]
    name = f"{path_part}_{url_hash}.pdf" if path_part else f"{url_hash}.pdf"
    return PDF_CACHE / name


def get_pdf_bytes(session: requests.Session, url: str,
                  use_cache: bool = True) -> bytes | None:
    """
    Download a PDF, caching to disk so re-runs are instant.
    Returns raw bytes or None on failure.
    """
    cache_path = pdf_cache_path(url)

    if use_cache and cache_path.exists() and cache_path.stat().st_size > 1000:
        return cache_path.read_bytes()

    log.info(f"    Downloading PDF...")
    resp = fetch(session, url, stream=True)
    if not resp:
        return None

    data = resp.content
    if len(data) < 500:
        log.warning(f"    PDF too small ({len(data)} bytes) — skipping")
        return None

    PDF_CACHE.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(data)
    return data


# ── PDF text extraction ────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Use pdfplumber to extract text from a PDF.
    Joins all pages with double newlines.
    Normalizes ligatures and smart quotes that PDFs commonly mangle.
    """
    text_pages = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text(x_tolerance=2, y_tolerance=3)
            if page_text:
                text_pages.append(page_text)

    full_text = "\n\n".join(text_pages)

    # Normalize common PDF encoding artifacts
    replacements = {
        "’": "'",   # right single quotation mark
        "‘": "'",   # left single quotation mark
        "“": '"',   # left double quotation mark
        "”": '"',   # right double quotation mark
        "–": "-",   # en dash
        "—": "-",   # em dash
        "ﬁ": "fi",  # fi ligature
        "ﬂ": "fl",  # fl ligature
        "®": "®",
        "™": "™",
    }
    for bad, good in replacements.items():
        full_text = full_text.replace(bad, good)

    return full_text


# ── Claim extraction ───────────────────────────────────────────────────────────

# Patterns that signal the start of a violation section in FDA letters
SECTION_HEADER_RE = re.compile(
    r"^(?:Issue[s]?|Violation[s]?)"
    r"|False or Misleading"
    r"|Unsubstantiated"
    r"|Omission of Risk"
    r"|Misleading (?:Efficacy|Safety|Presentation|Comparative|Superiority)"
    r"|Off-Label"
    r"|Superiority Claim"
    r"|Lacks Substantial Evidence"
    r"|Overstated",
    re.IGNORECASE,
)

# Patterns that signal the END of a violation section
SECTION_END_RE = re.compile(
    r"^(?:Conclusion|Corrective Action|Request|Sincerely|Background|"
    r"cc:|Enclosure|OPDP requests)",
    re.IGNORECASE,
)

# Claim extraction — tries multiple strategies in order
BULLET_RE    = re.compile(r"^\s*[•\-\*]\s+(.+)", re.MULTILINE)
NUMBERED_RE  = re.compile(r"^\s*\d+\.\s+(.+)", re.MULTILINE)
QUOTED_RE    = re.compile(r'"([^"]{20,400}?)"')
INLINE_CLAIM_RE = re.compile(
    r'(?:statement[s]?|claim[s]?|representation[s]?|language|phrase[s]?|assertion[s]?)'
    r'\s*[:\-]\s*"([^"]{20,400}?)"',
    re.IGNORECASE,
)


def split_into_sections(text: str) -> list[dict]:
    """
    Split an FDA letter into labelled sections.
    Returns list of {header, body, label} dicts.
    Each section spans from one header line to the next.
    """
    lines = text.split("\n")
    sections: list[dict] = []
    current_header = "preamble"
    current_body: list[str] = []

    for line in lines:
        stripped = line.strip()

        if SECTION_HEADER_RE.match(stripped) and len(stripped) < 120:
            # Save the previous section
            if current_body:
                sections.append({
                    "header": current_header,
                    "body":   "\n".join(current_body),
                    "label":  assign_label(current_header, "\n".join(current_body)),
                })
            current_header = stripped
            current_body = []

        elif SECTION_END_RE.match(stripped):
            # Save current section and stop — everything after is boilerplate
            if current_body:
                sections.append({
                    "header": current_header,
                    "body":   "\n".join(current_body),
                    "label":  assign_label(current_header, "\n".join(current_body)),
                })
            break

        else:
            current_body.append(line)

    return sections


def extract_claims_from_section(header: str, body: str) -> list[str]:
    """
    Pull individual claim strings out of a violation section body.
    Tries four strategies in order; deduplicates across strategies.
    """
    claims: list[str] = []
    seen: set[str] = set()

    def add(text: str) -> None:
        text = text.strip().strip('"').strip()
        if len(text) < 20 or len(text) > 600:
            return
        norm = re.sub(r"\s+", " ", text.lower())
        if norm not in seen:
            seen.add(norm)
            claims.append(text)

    # Strategy 1: explicit bullet points
    for m in BULLET_RE.finditer(body):
        add(m.group(1))

    # Strategy 2: numbered lists
    for m in NUMBERED_RE.finditer(body):
        add(m.group(1))

    # Strategy 3: inline quotes
    for m in QUOTED_RE.finditer(body):
        add(m.group(1))

    # Strategy 4: "statement: X" / "claim: X" patterns
    for m in INLINE_CLAIM_RE.finditer(body):
        add(m.group(1))

    # Strategy 5: if nothing found, use the first substantive sentence
    # (captures letters that describe claims in prose without formatting)
    if not claims:
        sentences = re.split(r"(?<=[.!?])\s+", body.strip())
        for sent in sentences:
            sent = sent.strip()
            # Skip boilerplate intro sentences
            if any(kw in sent.lower() for kw in [
                "opdp", "fda has reviewed", "this letter", "we have reviewed",
                "the following", "enclosed", "please contact",
            ]):
                continue
            if 30 < len(sent) < 500:
                add(sent)
                break   # just the first good sentence in prose mode

    return claims


def assign_label(header: str, body: str) -> str:
    """
    Map a section header + body text to one of the 5 compliance labels.
    Priority-ordered keyword match; defaults to 'unsupported' in a violation
    section, 'supported' otherwise.
    """
    combined = (header + " " + body).lower()

    for keywords, label in LABEL_RULES:
        if any(kw in combined for kw in keywords):
            return label

    # If the section header looks like a violation → unsupported as fallback
    if SECTION_HEADER_RE.match(header.strip()):
        return "unsupported"

    return "supported"


def _truncate(text: str, max_chars: int = 300) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"


# ── Letter metadata extraction ─────────────────────────────────────────────────

def extract_re_line(text: str) -> str:
    """Pull the 'RE:' line from an FDA letter — usually contains drug + NDA."""
    m = re.search(r"RE:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    return m.group(1).strip() if m else ""


def extract_drug_name(text: str, fallback: str = "") -> str:
    """Extract the drug trade name from the RE: line or letter body."""
    re_line = extract_re_line(text)
    # Try to get all-caps trade name from RE: line
    m = re.search(r"\b([A-Z]{3,}[®™]?)\b", re_line)
    if m:
        return m.group(1)
    return fallback


# ── Main parser ────────────────────────────────────────────────────────────────

def parse_letter(session: requests.Session, meta: dict,
                 use_cache: bool = True) -> tuple[list[dict], str | None]:
    """
    Download + parse one FDA enforcement letter.
    Returns (list_of_records, error_message_or_None).
    """
    url = meta["pdf_url"]

    pdf_bytes = get_pdf_bytes(session, url, use_cache=use_cache)
    if not pdf_bytes:
        return [], "PDF download failed"

    try:
        text = extract_text_from_pdf(pdf_bytes)
    except Exception as e:
        return [], f"PDF extraction failed: {e}"

    if not text or len(text) < 200:
        return [], "Extracted text too short — likely a scanned/image PDF"

    # Improve drug name from actual letter text if table gave us nothing
    drug = meta.get("drug") or extract_drug_name(text) or "Unknown"

    sections = split_into_sections(text)

    # Only process sections that are actual violation sections
    violation_sections = [
        s for s in sections
        if SECTION_HEADER_RE.match(s["header"].strip()) and s["body"].strip()
    ]

    if not violation_sections:
        # Fallback: treat entire letter as one block
        violation_sections = [{
            "header": "Issue",
            "body":   text,
            "label":  assign_label("Issue", text),
        }]

    records: list[dict] = []
    for section in violation_sections:
        claims = extract_claims_from_section(section["header"], section["body"])
        for claim in claims:
            records.append({
                "text":  claim,
                "label": section["label"],
                "metadata": {
                    "source":            "fda_opdp",
                    "letter_type":       meta["letter_type"],
                    "letter_url":        url,
                    "drug":              drug,
                    "company":           meta.get("company", ""),
                    "date":              meta.get("date", ""),
                    "violation_section": _truncate(section["header"], 80),
                    "violation_context": _truncate(section["body"], 300),
                    "scraped_at":        datetime.utcnow().isoformat() + "Z",
                },
            })

    return records, None


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape FDA OPDP enforcement letters and extract compliance claims"
    )
    parser.add_argument("--limit",    type=int,  default=None,
                        help="Max letters to process")
    parser.add_argument("--year",     type=int,  default=None,
                        help="Filter to letters from this year (e.g. --year 2023)")
    parser.add_argument("--output",   type=str,  default=str(RAW_OUTPUT),
                        help=f"Output JSONL path (default: {RAW_OUTPUT})")
    parser.add_argument("--no-cache", action="store_true",
                        help="Re-download PDFs even if cached")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PDF_CACHE.mkdir(parents=True, exist_ok=True)
    out_path  = Path(args.output)
    use_cache = not args.no_cache

    session = make_session()

    # ── 1. Get letter index ───────────────────────────────────────────────────
    letters = get_letter_index(session, year_filter=args.year)
    if not letters:
        log.error("No letters found. Check listing URL or run with --year to narrow down.")
        return

    if args.limit:
        letters = letters[:args.limit]
        log.info(f"Limited to first {args.limit} letters")

    # ── 2. Process each letter ────────────────────────────────────────────────
    all_records: list[dict] = []
    skipped:     list[dict] = []
    label_counts: dict[str, int] = {}

    for i, meta in enumerate(letters, 1):
        log.info(f"[{i}/{len(letters)}] {meta['drug']:<20} {meta['date']}")
        time.sleep(REQUEST_DELAY)

        records, err = parse_letter(session, meta, use_cache=use_cache)

        if err:
            log.warning(f"  Skipped: {err}")
            skipped.append({**meta, "reason": err})
            continue

        if not records:
            log.warning("  No claims extracted")
            skipped.append({**meta, "reason": "no_claims_extracted"})
            continue

        log.info(f"  → {len(records)} claim(s) extracted")
        all_records.extend(records)
        for r in records:
            lbl = r["label"]
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

    # ── 3. Write output ───────────────────────────────────────────────────────
    with open(out_path, "w") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")

    with open(SKIP_OUTPUT, "w") as f:
        for s in skipped:
            f.write(json.dumps(s) + "\n")

    # ── 4. Summary ────────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  FDA OPDP Scraper — Complete")
    print("─" * 60)
    print(f"  Letters processed  : {len(letters)}")
    print(f"  Claims extracted   : {len(all_records)}")
    print(f"  Letters skipped    : {len(skipped)}")
    print(f"  PDFs cached in     : {PDF_CACHE}/")
    print()
    if label_counts:
        print("  Label distribution:")
        for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
            bar = "█" * min(count, 40)
            print(f"    {label:<25} {count:>4}  {bar}")
    print()
    print(f"  Output   → {out_path}")
    print(f"  Skipped  → {SKIP_OUTPUT}")
    print()
    print("  ⚠  REVIEW data/fda_opdp_raw.jsonl before ingesting.")
    print("  Delete or fix any rows where `text` is boilerplate, not a claim.")
    print("  Then run:  make db-reset && make ingest-fda")
    print("─" * 60)


if __name__ == "__main__":
    main()
