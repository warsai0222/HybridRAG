"""
API Security Layer
==================
Protects the HybridRAG API against:

  1. Prompt injection     — detects attempts to override or hijack LLM instructions
                            by scanning for known injection patterns (jailbreaks,
                            role-switching, instruction overrides, delimiter abuse)
  2. Jailbreak patterns   — "act as", "ignore previous", DAN prompts, etc.
  3. Oversized inputs     — hard cap at MAX_CLAIM_LENGTH chars (prevents token stuffing)
  4. Malformed ingest     — validates required fields, source domain allowlist,
                            label allowlist, and text length per document
  5. Control characters   — strips null bytes and non-printable chars that can
                            confuse tokenisers or bypass string-level checks
  6. Rate limiting        — per-IP throttle via SlowAPI (configurable in settings)

Usage:
  from src.api.security import sanitize_claim, validate_ingest_docs, limiter

  # In FastAPI handler:
  @app.post("/classify")
  @limiter.limit("30/minute")
  def classify_text(request: Request, body: ClassifyRequest):
      text = sanitize_claim(body.text)
      ...
"""

import logging
import re
import unicodedata
from typing import Any

from fastapi import HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)

# ── Rate limiter ───────────────────────────────────────────────────────────────
# Keyed by IP address.  Apply per-route with @limiter.limit("N/minute").
limiter = Limiter(key_func=get_remote_address)

# ── Hard limits ───────────────────────────────────────────────────────────────
MAX_CLAIM_LENGTH    = 2_000   # characters — a real pharma claim is never this long
MIN_CLAIM_LENGTH    = 10      # below this it's noise, not a claim
MAX_INGEST_BATCH    = 50      # documents per /ingest call
MAX_INGEST_DOC_LEN  = 2_000   # characters per document in an ingest batch

# ── Allowed labels (must stay in sync with src/config.py label_list) ──────────
ALLOWED_LABELS = {
    "supported",
    "partially_supported",
    "unsupported",
    "false_balance",
    "needs_legal_review",
    "insufficient_data",
}

# ── Trusted source domains for ingest ─────────────────────────────────────────
# Documents claiming to originate from outside these domains are rejected
# (mirrors the allowlist in src/ingestion/validate.py)
ALLOWED_SOURCE_DOMAINS = {
    "fda.gov",
    "labels.fda.gov",
    "api.fda.gov",
    "dailymed.nlm.nih.gov",
    "ncbi.nlm.nih.gov",
    "internal",
}


# ── Prompt injection patterns ─────────────────────────────────────────────────
# Each tuple is (human-readable name, compiled regex).
# All patterns are case-insensitive.  A match triggers a 400 rejection.
#
# Design note: we intentionally block at the API layer (before the prompt is
# built) rather than relying on the LLM to resist injection.  Belt + suspenders.

_RAW_INJECTION_PATTERNS: list[tuple[str, str]] = [
    # ── Classic instruction overrides ────────────────────────────────────────
    ("instruction_override",
     r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?|context)"),

    ("forget_instructions",
     r"(forget|disregard|discard|override|bypass|circumvent)\s+(your\s+)?(instructions?|rules?|guidelines?|system\s+prompt|training|context)"),

    ("new_instructions",
     r"(new|updated?|revised?|different|following)\s+instructions?\s*(are|follow|:)"),

    ("do_not_follow",
     r"do\s+not\s+follow\s+(your\s+)?(previous|prior|original|system)\s+(instructions?|prompt)"),

    # ── Role-switching / impersonation ───────────────────────────────────────
    ("act_as",
     r"\b(act|behave|respond|pretend|play|roleplay|simulate)\s+(as|like)\s+(a\s+)?(different|new|another|unrestricted|jailbroken|free|unfiltered|uncensored)"),

    ("you_are_now",
     r"you\s+are\s+now\s+(a\s+)?(different|new|another|unrestricted|jailbroken|an?\s+AI|an?\s+assistant|(?!a\s+pharma|a\s+regulatory))"),

    ("jailbreak_token",
     r"\b(DAN|JAILBREAK|STAN|DUDE|AIM|BasedGPT|DevMode|sudo\s+mode|god\s+mode|developer\s+mode)\b"),

    ("pretend_no_restrictions",
     r"(pretend|imagine|assume)\s+(you\s+have\s+no\s+(restrictions?|limits?|rules?|guidelines?|filters?))"),

    # ── System-level delimiter injection ────────────────────────────────────
    # Attackers inject fake turn markers to confuse the chat template
    ("system_marker",
     r"(\[SYSTEM\]|\[SYS\]|<\|system\|>|<system>|###\s*System\s*:|---\s*System\s*:|SYSTEM\s*PROMPT\s*:)"),

    ("turn_delimiter",
     r"(<\|im_start\|>|<\|im_end\|>|\[INST\]|\[/INST\]|<s>|</s>|<<SYS>>|<</SYS>>|Human\s*:\s*\n|Assistant\s*:\s*\n)"),

    ("chat_template_injection",
     r"(###\s*(Instruction|Response|Human|Assistant|Input|Output)\s*:)"),

    # ── Data exfiltration / meta-prompt probing ──────────────────────────────
    ("reveal_prompt",
     r"(repeat|print|show|output|reveal|display|return|tell\s+me)\s+(your|the|all|every)?\s*(system\s+prompt|instructions?|context|rules?|training\s+data|knowledge\s+base)"),

    ("what_are_instructions",
     r"what\s+(are|were)\s+(your|the)\s+(system\s+)?(instructions?|prompt|rules?|guidelines?)"),

    # ── Token / encoding abuse ───────────────────────────────────────────────
    ("null_injection",
     r"\x00"),   # null byte — filter before regex for clarity, but belt+suspenders

    ("excessive_special_chars",
     r"([<>\[\]{}|\\]{6,})"),   # ≥ 6 consecutive special chars — likely delimiter stuffing
]

# Compile once at module load — O(1) at request time
INJECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    (name, re.compile(pattern, re.IGNORECASE | re.UNICODE))
    for name, pattern in _RAW_INJECTION_PATTERNS
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _strip_control_chars(text: str) -> str:
    """
    Remove null bytes and non-printable control characters.

    Keeps: printable ASCII, Unicode letters/punctuation, standard whitespace
    (space, tab, newline, carriage return).
    Strips: null bytes, ESC sequences, BEL, BS, DEL, and other C0/C1 controls.
    """
    cleaned = []
    for ch in text:
        cat = unicodedata.category(ch)
        # Allow printable chars (not Cc = "other control") plus safe whitespace
        if cat != "Cc" or ch in ("\n", "\r", "\t"):
            cleaned.append(ch)
    return "".join(cleaned)


def _detect_injection(text: str) -> str | None:
    """
    Scan text against all injection patterns.
    Returns the pattern name if a match is found, None if clean.
    """
    for name, pattern in INJECTION_PATTERNS:
        if pattern.search(text):
            return name
    return None


def _check_domain(url: str) -> bool:
    """Return True if url's domain is in ALLOWED_SOURCE_DOMAINS."""
    if not url:
        return True   # no URL provided — pass (domain check doesn't apply)
    # Simple hostname extraction (no need for urllib overhead here)
    # Strip scheme
    host = re.sub(r"^https?://", "", url.lower())
    # Strip path
    host = host.split("/")[0].split("?")[0].split("#")[0]
    # Strip port
    host = host.split(":")[0]
    # Match exact domain OR any subdomain
    return any(
        host == allowed or host.endswith("." + allowed)
        for allowed in ALLOWED_SOURCE_DOMAINS
    )


# ── Public API ────────────────────────────────────────────────────────────────

def sanitize_claim(text: str) -> str:
    """
    Validate and sanitise a raw claim string from the /classify endpoint.

    Steps:
      1. Strip control characters
      2. Enforce length bounds (MIN_CLAIM_LENGTH .. MAX_CLAIM_LENGTH)
      3. Scan for prompt injection / jailbreak patterns

    Returns the cleaned text on success.
    Raises HTTPException(400) on any violation.

    The 400 responses are intentionally non-specific about *which* pattern
    matched to avoid giving attackers a roadmap for evasion.
    """
    if not isinstance(text, str):
        raise HTTPException(status_code=400, detail="Input must be a string.")

    # Step 1 — strip control characters
    clean = _strip_control_chars(text)

    # Step 2 — length bounds
    if len(clean) < MIN_CLAIM_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Input too short (minimum {MIN_CLAIM_LENGTH} characters)."
        )
    if len(clean) > MAX_CLAIM_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Input too long ({len(clean)} chars). "
                f"Maximum claim length is {MAX_CLAIM_LENGTH} characters."
            )
        )

    # Step 3 — injection detection
    matched = _detect_injection(clean)
    if matched:
        logger.warning(
            f"Prompt injection attempt blocked — pattern='{matched}' "
            f"input_prefix='{clean[:80]}...'"
        )
        raise HTTPException(
            status_code=400,
            detail=(
                "Input contains content that cannot be processed. "
                "Please submit a pharmaceutical marketing claim for compliance review."
            )
        )

    return clean


def validate_ingest_docs(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Validate a batch of documents submitted to /ingest.

    Checks per document:
      - Required fields: "text" (str), "label" (str)
      - Label must be in ALLOWED_LABELS
      - Text length within bounds
      - Source URL domain (if provided) must be in ALLOWED_SOURCE_DOMAINS
      - No prompt injection in the text field

    Returns the list of cleaned, validated documents.
    Raises HTTPException(400) on any violation (fails the whole batch —
    callers should pre-validate before submitting).
    """
    if not documents:
        raise HTTPException(status_code=400, detail="documents list is empty.")

    if len(documents) > MAX_INGEST_BATCH:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Batch too large ({len(documents)} docs). "
                f"Maximum is {MAX_INGEST_BATCH} documents per request."
            )
        )

    cleaned_docs: list[dict[str, Any]] = []

    for i, doc in enumerate(documents):
        idx = i + 1  # 1-based for human-readable error messages

        # ── Required fields ───────────────────────────────────────────────────
        if not isinstance(doc.get("text"), str) or not doc["text"].strip():
            raise HTTPException(
                status_code=400,
                detail=f"Document {idx}: 'text' field is required and must be a non-empty string."
            )
        if not isinstance(doc.get("label"), str) or not doc["label"].strip():
            raise HTTPException(
                status_code=400,
                detail=f"Document {idx}: 'label' field is required and must be a non-empty string."
            )

        # ── Label allowlist ───────────────────────────────────────────────────
        label = doc["label"].strip().lower()
        if label not in ALLOWED_LABELS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Document {idx}: label '{doc['label']}' is not valid. "
                    f"Allowed labels: {sorted(ALLOWED_LABELS)}"
                )
            )

        # ── Text sanitation ───────────────────────────────────────────────────
        clean_text = _strip_control_chars(doc["text"])

        if len(clean_text) < MIN_CLAIM_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Document {idx}: text too short (minimum {MIN_CLAIM_LENGTH} characters)."
            )
        if len(clean_text) > MAX_INGEST_DOC_LEN:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Document {idx}: text too long ({len(clean_text)} chars). "
                    f"Maximum is {MAX_INGEST_DOC_LEN} characters."
                )
            )

        # ── Injection check on text ───────────────────────────────────────────
        matched = _detect_injection(clean_text)
        if matched:
            logger.warning(
                f"Injection pattern in ingest doc {idx} — pattern='{matched}' "
                f"text_prefix='{clean_text[:80]}...'"
            )
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Document {idx}: text contains content that cannot be ingested. "
                    "Only verified pharmaceutical claim text is accepted."
                )
            )

        # ── Source URL domain check ───────────────────────────────────────────
        metadata = doc.get("metadata") or {}
        source_url = (
            metadata.get("source_url")
            or metadata.get("letter_url")
            or ""
        )
        if source_url and not _check_domain(source_url):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Document {idx}: source domain is not in the trusted allowlist. "
                    f"Accepted domains: {sorted(ALLOWED_SOURCE_DOMAINS)}"
                )
            )

        cleaned_docs.append({
            **doc,
            "text":  clean_text,
            "label": label,
        })

    return cleaned_docs
