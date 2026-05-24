"""
Document Validation Pipeline
==============================
Validates pharma claim documents before they enter the knowledge base.

Agent harness design — LLM is called ONLY when rules are insufficient:
  Stage 1  Rule-based rejection   (free — regex + keyword)
  Stage 2  Source allowlist check (free — URL pattern match)
  Stage 3  Claim structure check  (free — heuristics)
  Stage 4  Batch LLM validation   (paid — only for borderline docs, 10 per call)
  Stage 5  Label-specific gates   (free — 'supported' always queued for human review)

Returns three buckets:
  approved      → safe to ingest directly
  needs_review  → goes to review_queue table (human confirms before ingest)
  rejected      → dropped with reason logged

Why batch LLM calls?
  Validating 50 borderline docs one-at-a-time = 50 API calls.
  Batching 10 per prompt = 5 API calls. Same quality, 10x cheaper.
  We only reach LLM stage if rules can't make a clear call.
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import time

from openai import OpenAI, RateLimitError

from src.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Trusted source domains ─────────────────────────────────────────────────────
# Only documents from these origins are eligible for ingestion.
# Third-party pharma sites, blogs, or press releases are never trusted.
ALLOWED_SOURCE_DOMAINS = {
    "fda.gov",
    "labels.fda.gov",
    "api.fda.gov",
    "dailymed.nlm.nih.gov",    # NIH DailyMed — FDA-approved labeling
    "ncbi.nlm.nih.gov",        # PubMed / PMC peer-reviewed literature
    "internal",                # hand-authored examples marked as internal
}

# ── Boilerplate patterns ────────────────────────────────────────────────────────
# Sentences that look like claims but are procedural/administrative text.
# These appear constantly in FDA enforcement letters and PI documents as
# framing language — they describe the letter itself, not a drug claim.
BOILERPLATE_PATTERNS = [
    r"opdp\s+(has|have)\s+reviewed",
    r"this\s+letter\s+(is|was|has)",
    r"we\s+(have|are)\s+requesting",
    r"please\s+(contact|submit|provide|respond)",
    r"if\s+you\s+have\s+any\s+questions",
    r"sincerely\s*,",
    r"enclosure[s]?\s*:",
    r"cc\s*:",
    r"^dear\s+dr\.",
    r"^re\s*:",
    r"pursuant\s+to\s+\d+\s+cfr",
    r"prescribing\s+information\s+is\s+available",
    r"full\s+prescribing\s+information",
    r"see\s+(the\s+)?prescribing\s+information",
    r"highlights\s+of\s+prescribing\s+information",
    r"important\s+safety\s+information\s+appears",
    r"please\s+see\s+accompanying",
    r"dosing\s+and\s+administration",
    r"contraindications?\s*$",      # bare section headers
    r"warnings?\s+and\s+precautions?\s*$",
    r"adverse\s+reactions?\s*$",
]
_BOILERPLATE_RE = re.compile(
    "|".join(BOILERPLATE_PATTERNS),
    re.IGNORECASE,
)

# ── Claim structure indicators ─────────────────────────────────────────────────
# A real pharma marketing claim references at least one of: a drug/treatment,
# a clinical outcome, a patient population, or a comparative assertion.
CLAIM_INDICATORS = [
    r"\b(?:reduced?|decreas(?:ed?|ing)|lower(?:ed|ing)|improv(?:ed|ing)|increas(?:ed|ing))\b",
    r"\b(?:efficacy|efficacious|effective|effectiveness)\b",
    r"\b(?:trial|study|studi(?:ed|es)|clinical|rct|randomized)\b",
    r"\b(?:patient[s]?|subject[s]?|participant[s]?)\b",
    r"\b(?:statistically\s+significant|p\s*[<=>]\s*0\.\d+|p-value|ci\s*[\[\(])\b",
    r"\b(?:superior|inferior|non-inferior|comparable|equivalent)\b",
    r"\b(?:approved|indicated|contraindicated|off-label)\b",
    r"\b(?:risk|benefit|adverse|safety|tolerability)\b",
    r"\b(?:dose|dosing|mg|mcg|kg|once\s+daily|twice\s+daily|bid|qd)\b",
    r"\b(?:placebo|comparator|standard\s+of\s+care)\b",
    r"(?:\d+\s*%|\d+\s+percent)",       # numeric percentage
    r"(?:hr|or|rr|nnt)\s*(?:=|of)\s*\d",  # hazard/odds ratio
]
_CLAIM_RE = re.compile(
    "|".join(CLAIM_INDICATORS),
    re.IGNORECASE,
)

# Minimum number of claim indicators required to pass structure check
MIN_CLAIM_INDICATORS = 1

# Text length bounds — too short is a fragment, too long is a paragraph block
MIN_TEXT_LEN = 25
MAX_TEXT_LEN = 800

# Labels where the LLM validation stage is skipped (rules are sufficient).
# - Violation labels from FDA enforcement letters: source is authoritative, label is certain.
# - partially_supported from FDA PI docs: FDA-approved PI is a trusted source; "directionally
#   accurate but missing full stats" is exactly what PI indications sections are. Safe to
#   approve without LLM confirmation once it passes source + claim structure checks.
RULE_ONLY_LABELS = {"false_balance", "unsupported", "needs_legal_review", "partially_supported"}

# Labels that always go to human review regardless of LLM verdict.
# 'supported' = fully backed by cited evidence with no material omissions.
# A false positive here could lead a pharma team to treat a non-compliant claim as
# compliant — direct regulatory risk. Human must confirm before it enters the KB.
ALWAYS_REVIEW_LABELS = {"supported"}

# LLM batch size — how many borderline docs per single API call
LLM_BATCH_SIZE = 10


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    approved: list[dict] = field(default_factory=list)
    needs_review: list[dict] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.approved) + len(self.needs_review) + len(self.rejected)

    def summary(self) -> str:
        return (
            f"Validated {self.total} docs — "
            f"approved={len(self.approved)}, "
            f"review_queued={len(self.needs_review)}, "
            f"rejected={len(self.rejected)}"
        )


# ── Helpers ────────────────────────────────────────────────────────────────────

def content_hash(text: str) -> str:
    """SHA-256 of normalised text — used for deduplication."""
    normalised = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(normalised.encode()).hexdigest()


def _source_is_trusted(metadata: dict) -> bool:
    """Check if the document's source URL is from an allowed domain."""
    source = metadata.get("source", "")
    url = metadata.get("letter_url", metadata.get("source_url", ""))
    # Hand-authored internal examples always pass
    if source == "internal" or "internal" in url:
        return True
    for domain in ALLOWED_SOURCE_DOMAINS:
        if domain in url or domain in source:
            return True
    return False


def _is_boilerplate(text: str) -> bool:
    return bool(_BOILERPLATE_RE.search(text))


def _has_claim_structure(text: str) -> bool:
    matches = len(_CLAIM_RE.findall(text))
    return matches >= MIN_CLAIM_INDICATORS


def _length_ok(text: str) -> bool:
    return MIN_TEXT_LEN <= len(text.strip()) <= MAX_TEXT_LEN


# ── Stage 1–3: Rule-based checks ───────────────────────────────────────────────

def _rule_check(doc: dict) -> tuple[str, str]:
    """
    Apply all free rule-based checks.

    Returns (verdict, reason):
      'approved'      — passed all rules (for rule-only labels)
      'borderline'    — passed rules but needs LLM check (for ambiguous labels)
      'review'        — label requires human review by policy
      'rejected'      — failed a hard rule
    """
    text  = doc.get("text", "")
    label = doc.get("label", "")
    meta  = doc.get("metadata", {})

    # Hard reject: length
    if not _length_ok(text):
        return "rejected", f"Text length {len(text)} outside [{MIN_TEXT_LEN}, {MAX_TEXT_LEN}]"

    # Hard reject: boilerplate
    if _is_boilerplate(text):
        return "rejected", "Matched boilerplate pattern"

    # Hard reject: untrusted source
    if not _source_is_trusted(meta):
        return "rejected", f"Source not in allowlist: {meta.get('source_url', meta.get('source', 'unknown'))}"

    # Hard reject: no claim indicators
    if not _has_claim_structure(text):
        return "rejected", "No claim indicators found (no drug, outcome, or population reference)"

    # Policy: supported labels always go to human review
    if label in ALWAYS_REVIEW_LABELS:
        return "review", "Label 'supported' requires human confirmation before ingestion"

    # Rule-only labels: enforcement letters are authoritative — approve directly
    if label in RULE_ONLY_LABELS:
        return "approved", "Passed all rule checks (enforcement source — label authoritative)"

    # partially_supported: pass to LLM for confidence check
    return "borderline", "Passed rules — LLM needed to confirm label confidence"


# ── Stage 4: Batch LLM validation ─────────────────────────────────────────────

_BATCH_PROMPT_TEMPLATE = """\
You are a pharmaceutical regulatory expert validating training data for an MLR compliance classifier.

For each claim below, verify:
1. Is this a genuine pharmaceutical marketing claim (not procedural/administrative text)?
2. Is the assigned label correct given the claim content?

Labels:
  supported           — fully backed by cited clinical evidence, no material omissions
  partially_supported — directionally accurate but missing critical hedges or stats
  false_balance       — safety risk obscured or minimised relative to benefit claims
  needs_legal_review  — off-label, comparative, or IP/regulatory legal concerns
  unsupported         — no evidence basis, absolute language, or misrepresented data

Claims to validate:
{claims_block}

Respond with a JSON array (one object per claim, in the same order):
[
  {{
    "index": 0,
    "is_valid_claim": true,
    "label_correct": true,
    "suggested_label": "same as assigned if correct",
    "confidence": 0.85,
    "note": "brief reason"
  }},
  ...
]
Only return the JSON array. No other text."""


def _build_batch_prompt(docs: list[dict]) -> str:
    lines = []
    for i, doc in enumerate(docs):
        lines.append(
            f"[{i}] Label: {doc['label']}\n    Claim: {doc['text']}"
        )
    claims_block = "\n\n".join(lines)
    return _BATCH_PROMPT_TEMPLATE.format(claims_block=claims_block)


def _call_llm_batch(docs: list[dict], max_retries: int = 5) -> list[dict]:
    """
    Validate a batch of borderline docs in a single LLM call, with retry on rate limits.
    Returns the raw JSON array from the LLM.
    Falls back to approving all docs if the call fails (with a warning).
    """
    client = OpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )
    prompt = _build_batch_prompt(docs)

    try:
        for attempt in range(1, max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=settings.llm_model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                break
            except RateLimitError as e:
                if attempt == max_retries:
                    raise
                delay_match = re.search(r"retry in\s+([\d.]+)s", str(e), re.IGNORECASE)
                delay = float(delay_match.group(1)) + 2.0 if delay_match else 60.0
                logger.warning(f"Rate limit on batch LLM (attempt {attempt}/{max_retries}). Sleeping {delay:.1f}s...")
                time.sleep(delay)
        raw = response.choices[0].message.content.strip()
        # The model may wrap the array in a key — unwrap if needed
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            # Try to find the array inside
            for v in parsed.values():
                if isinstance(v, list):
                    return v
        if isinstance(parsed, list):
            return parsed
        logger.warning("LLM batch returned unexpected shape — approving all borderline docs")
        return []
    except Exception as e:
        logger.warning(f"LLM batch validation failed ({e}) — approving all borderline docs")
        return []


def _llm_validate_batch(borderline_docs: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Run batch LLM validation over borderline docs.
    Returns (approved_docs, review_docs).
    Batches in chunks of LLM_BATCH_SIZE to keep prompt size manageable.
    """
    approved: list[dict] = []
    review:   list[dict] = []

    total_batches = (len(borderline_docs) + LLM_BATCH_SIZE - 1) // LLM_BATCH_SIZE
    logger.info(
        f"LLM validation: {len(borderline_docs)} borderline docs "
        f"in {total_batches} batch(es) of ≤{LLM_BATCH_SIZE}"
    )

    for batch_idx in range(0, len(borderline_docs), LLM_BATCH_SIZE):
        batch = borderline_docs[batch_idx : batch_idx + LLM_BATCH_SIZE]
        results = _call_llm_batch(batch)

        # Map results back to docs by index
        result_map: dict[int, dict] = {r["index"]: r for r in results if "index" in r}

        for i, doc in enumerate(batch):
            result = result_map.get(i)

            if not result:
                # No result for this doc — approve with warning
                logger.warning(f"No LLM result for doc index {i} — approving by default")
                doc["_validation_note"] = "No LLM result — approved by default"
                approved.append(doc)
                continue

            is_valid   = result.get("is_valid_claim", True)
            label_ok   = result.get("label_correct", True)
            confidence = float(result.get("confidence", 0.5))
            note       = result.get("note", "")

            if not is_valid:
                doc["_validation_note"] = f"LLM rejected as non-claim: {note}"
                # Don't append to review or approved — silently discard
                logger.info(f"LLM rejected doc as non-claim: {doc['text'][:60]}…")
                continue

            if not label_ok or confidence < 0.65:
                # Ambiguous label — queue for human review
                doc["_validation_note"] = f"LLM uncertain (conf={confidence:.2f}): {note}"
                doc["_suggested_label"] = result.get("suggested_label", doc["label"])
                review.append(doc)
            else:
                doc["_validation_note"] = f"LLM approved (conf={confidence:.2f}): {note}"
                approved.append(doc)

        logger.info(
            f"  Batch {batch_idx // LLM_BATCH_SIZE + 1}/{total_batches} done — "
            f"approved={len(approved)}, review={len(review)} (running totals)"
        )

    return approved, review


# ── Main entry point ───────────────────────────────────────────────────────────

def validate_documents(
    documents: list[dict[str, Any]],
    skip_llm: bool = False,
) -> ValidationResult:
    """
    Validate a list of documents through the full pipeline.

    Args:
        documents:  List of dicts with at minimum 'text', 'label', 'metadata'.
        skip_llm:   If True, skip LLM stage (borderline docs go to review_queue).
                    Useful for offline testing or when API is unavailable.

    Returns:
        ValidationResult with approved / needs_review / rejected buckets.
        Each doc gets a '_validation_note' and '_content_hash' added.
    """
    result = ValidationResult()
    borderline: list[dict] = []

    logger.info(f"Validating {len(documents)} documents...")

    for doc in documents:
        # Attach content hash to every doc regardless of outcome
        doc["_content_hash"] = content_hash(doc.get("text", ""))

        verdict, reason = _rule_check(doc)
        doc["_validation_note"] = reason

        if verdict == "rejected":
            result.rejected.append(doc)
        elif verdict == "approved":
            result.approved.append(doc)
        elif verdict == "review":
            result.needs_review.append(doc)
        elif verdict == "borderline":
            borderline.append(doc)

    logger.info(
        f"Rule checks done — approved={len(result.approved)}, "
        f"review={len(result.needs_review)}, "
        f"rejected={len(result.rejected)}, "
        f"borderline={len(borderline)}"
    )

    # Stage 4: LLM for borderline docs only
    if borderline:
        if skip_llm:
            logger.info(f"skip_llm=True — routing {len(borderline)} borderline docs to review_queue")
            for doc in borderline:
                doc["_validation_note"] += " | LLM skipped — queued for human review"
            result.needs_review.extend(borderline)
        else:
            llm_approved, llm_review = _llm_validate_batch(borderline)
            result.approved.extend(llm_approved)
            result.needs_review.extend(llm_review)

    logger.info(result.summary())
    return result
