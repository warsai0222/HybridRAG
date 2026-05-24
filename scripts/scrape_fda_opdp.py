"""
FDA OPDP Enforcement Letter Scraper  (v3 — warning + untitled letters, multi-year)
====================================================================================
Pulls enforcement letters from the FDA OPDP page, downloads each as a PDF,
extracts the specific marketing claims the FDA cited as violative, maps them
to HybridRAG compliance labels, and writes a review-ready JSONL file.

Supports both letter types:
  Untitled Letters — less serious violations (omission of risk, misleading claims)
  Warning Letters  — more serious (off-label promotion, superiority, comparative)

Warning Letters are the primary source of 'needs_legal_review' examples because
they more frequently cite off-label use, unapproved indications, and superiority
claims that require legal/regulatory counsel.

Output
------
  data/fda_opdp_raw.jsonl         ← untitled letters output
  data/fda_warning_raw.jsonl      ← warning letters output
  data/fda_all_raw.jsonl          ← combined output (--type all)
  data/fda_opdp_skipped.jsonl     ← letters we couldn't parse
  data/pdf_cache/                 ← cached PDFs (re-runs skip downloads)

Usage
-----
  python scripts/scrape_fda_opdp.py                          # untitled, all years
  python scripts/scrape_fda_opdp.py --type warning           # warning letters only
  python scripts/scrape_fda_opdp.py --type all               # both letter types
  python scripts/scrape_fda_opdp.py --years 2020-2024        # specific year range
  python scripts/scrape_fda_opdp.py --type all --years 2019-2024  # full history
  python scripts/scrape_fda_opdp.py --limit 20               # first 20 letters
  python scripts/scrape_fda_opdp.py --no-cache               # re-download everything

Label mapping (section title → label)
--------------------------------------
  false_balance        ← omission / risk / one-sided / fair balance
  needs_legal_review   ← off-label / superiority / comparative / unapproved /
                         prior approval / unauthorized indication / broadens label
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

# Both listing URLs share the same table structure (verified May 2026)
LISTING_URLS = {
    "untitled": (
        "https://www.fda.gov/drugs/warning-letters-and-notice-violation-letters-"
        "pharmaceutical-companies/untitled-letters"
    ),
    "warning": (
        "https://www.fda.gov/drugs/warning-letters-and-notice-violation-letters-"
        "pharmaceutical-companies/warning-letters"
    ),
}

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
# Output files per letter type
RAW_OUTPUTS = {
    "untitled": OUTPUT_DIR / "fda_opdp_raw.jsonl",
    "warning":  OUTPUT_DIR / "fda_warning_raw.jsonl",
    "all":      OUTPUT_DIR / "fda_all_raw.jsonl",
}
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
            # Off-label / unapproved indication
            "off-label", "off label", "unapproved use", "unapproved indication",
            "not approved", "not fda-approved", "unapproved new drug",
            "promotes for use", "promoting for use", "promotes the use",
            "broadens the indication", "broadens the approved",
            "not included in the approved", "outside the approved indication",
            "unauthorized indication", "unauthorized use",
            "prior approval supplement",
            # Superiority / comparative — require head-to-head trial data
            "superiority", "superior to", "better than",
            "comparative claim", "comparative efficacy",
            "head-to-head", "head to head",
            "more effective than", "greater efficacy than",
            "outperforms", "preferred over",
            # Legal / regulatory / IP concerns
            "lacks substantial evidence for",
            "pre-approval", "pre-market", "pre-nda",
            "investigational", "not yet approved",
            "interchangeable", "biosimilar" , "interchangeability",
            "misbranding",                     # 21 USC 352 violation
            "new drug application" ,           # in context of unapproved promotion
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
            "misleadingly implies",
            "no data", "lacks data",
            "absolute claim", "eliminates", "cures", "guarantees",
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

def get_letter_index(
    session: requests.Session,
    listing_url: str,
    letter_type: str = "untitled_letter",
    year_filter: int | None = None,
) -> list[dict]:
    """
    Scrape an OPDP listing page (untitled or warning letters).
    Returns list of letter metadata dicts including the PDF download URL.

    The listing is a single HTML table with columns:
      Issued Date | Company / Individual | Product / Issue | Response | Close-Out
    Every letter link is a PDF at https://www.fda.gov/media/<id>/download
    """
    log.info(f"Fetching listing page: {listing_url}")
    resp = fetch(session, listing_url)
    if not resp:
        log.error(f"Cannot reach listing page (404 or network error): {listing_url}")
        log.error("If scraping warning letters, note that FDA OPDP issues very few")
        log.error("warning letters — most enforcement is via untitled letters.")
        log.error("Try: python scripts/scrape_fda_opdp.py --type untitled --years 2022-2025")
        return []

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
            "letter_type": letter_type,
        })

    log.info(f"Found {len(letters)} {letter_type}(s) on listing page")
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

def _parse_years(years_str: str) -> list[int]:
    """Parse '2019-2024' into [2019, 2020, 2021, 2022, 2023, 2024]."""
    if "-" in years_str:
        parts = years_str.split("-")
        start, end = int(parts[0]), int(parts[1])
        return list(range(start, end + 1))
    return [int(years_str)]


def _scrape_one_source(
    session: requests.Session,
    listing_url: str,
    letter_type: str,
    year_filter: int | None,
    limit: int | None,
    use_cache: bool,
) -> tuple[list[dict], list[dict], dict]:
    """Scrape one listing URL (untitled or warning). Returns (records, skipped, label_counts)."""
    letters = get_letter_index(
        session,
        listing_url=listing_url,
        letter_type=letter_type,
        year_filter=year_filter,
    )
    if not letters:
        log.warning(f"No letters found at {listing_url} (year={year_filter})")
        return [], [], {}

    if limit:
        letters = letters[:limit]
        log.info(f"Limited to first {limit} letters")

    all_records: list[dict] = []
    skipped:     list[dict] = []
    label_counts: dict[str, int] = {}

    for i, meta in enumerate(letters, 1):
        log.info(f"[{i}/{len(letters)}] {meta['drug']:<20} {meta['date']}  [{letter_type}]")
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

    return all_records, skipped, label_counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape FDA OPDP enforcement letters and extract compliance claims"
    )
    parser.add_argument(
        "--type", choices=["untitled", "warning", "all"], default="untitled",
        help="Letter type to scrape (default: untitled). "
             "'warning' targets off-label/superiority cases (needs_legal_review). "
             "'all' scrapes both."
    )
    parser.add_argument(
        "--years", type=str, default=None,
        help="Year or year range to scrape (e.g. '2023' or '2019-2024'). "
             "Omit to scrape all available years."
    )
    parser.add_argument("--limit",    type=int,  default=None,
                        help="Max letters per source URL (useful for testing)")
    parser.add_argument("--output",   type=str,  default=None,
                        help="Override output JSONL path")
    parser.add_argument("--no-cache", action="store_true",
                        help="Re-download PDFs even if cached")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PDF_CACHE.mkdir(parents=True, exist_ok=True)
    use_cache = not args.no_cache

    # Determine which letter types to scrape
    if args.type == "all":
        types_to_scrape = ["untitled", "warning"]
    else:
        types_to_scrape = [args.type]

    # Determine year(s) to scrape
    years = _parse_years(args.years) if args.years else [None]

    # Determine output path
    out_path = Path(args.output) if args.output else RAW_OUTPUTS[args.type]

    session = make_session()
    all_records:  list[dict]     = []
    all_skipped:  list[dict]     = []
    label_counts: dict[str, int] = {}

    for letter_type in types_to_scrape:
        listing_url = LISTING_URLS[letter_type]
        for year in years:
            year_label = str(year) if year else "all years"
            log.info(f"{'─'*55}")
            log.info(f"Scraping {letter_type} letters — {year_label}")
            log.info(f"{'─'*55}")

            records, skipped, counts = _scrape_one_source(
                session=session,
                listing_url=listing_url,
                letter_type=letter_type,
                year_filter=year,
                limit=args.limit,
                use_cache=use_cache,
            )
            all_records.extend(records)
            all_skipped.extend(skipped)
            for lbl, cnt in counts.items():
                label_counts[lbl] = label_counts.get(lbl, 0) + cnt

    # ── Write output ──────────────────────────────────────────────────────────
    # Append to existing file (not overwrite) so multiple runs accumulate data
    existing: list[dict] = []
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        existing.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        log.info(f"Appending to existing {out_path} ({len(existing)} existing records)")

    # Deduplicate by URL + claim text before writing
    seen_keys: set[str] = {
        f"{r['metadata']['letter_url']}::{r['text']}" for r in existing
    }
    new_records = [
        r for r in all_records
        if f"{r['metadata']['letter_url']}::{r['text']}" not in seen_keys
    ]
    log.info(f"New unique records: {len(new_records)} ({len(all_records) - len(new_records)} duplicates skipped)")

    with open(out_path, "w") as f:
        for record in existing + new_records:
            f.write(json.dumps(record) + "\n")

    with open(SKIP_OUTPUT, "a") as f:
        for s in all_skipped:
            f.write(json.dumps(s) + "\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_in_file = len(existing) + len(new_records)
    print("\n" + "─" * 60)
    print("  FDA OPDP Scraper — Complete")
    print("─" * 60)
    print(f"  Types scraped      : {', '.join(types_to_scrape)}")
    print(f"  Years scraped      : {args.years or 'all'}")
    print(f"  New claims found   : {len(new_records)}")
    print(f"  Duplicates skipped : {len(all_records) - len(new_records)}")
    print(f"  Letters skipped    : {len(all_skipped)}")
    print(f"  Total in output    : {total_in_file}")
    print(f"  PDFs cached in     : {PDF_CACHE}/")
    print()
    if label_counts:
        print("  New records — label distribution:")
        for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
            bar = "█" * min(count, 40)
            print(f"    {label:<25} {count:>4}  {bar}")
    print()
    print(f"  Output   → {out_path}")
    print(f"  Skipped  → {SKIP_OUTPUT}")
    print()
    print("  Next: run the ingestion pipeline to validate and load into DB:")
    print(f"    python scripts/run_ingestion_pipeline.py --input {out_path} --skip-llm")
    print("─" * 60)


if __name__ == "__main__":
    main()
