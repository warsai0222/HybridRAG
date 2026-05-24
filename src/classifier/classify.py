"""
LLM-based classification head with few-shot retrieval augmentation.

Domain: Pharmaceutical marketing claim compliance (MLR review).

Pipeline:
  1. Retrieve similar labeled claims via hybrid RAG (BGE dense + BM25 sparse + RRF fusion)
  2. Format retrieved examples as few-shot in-context evidence
  3. Ask Gemini 2.0 Flash to classify with explicit MLR regulatory reasoning
  4. Parse compliance verdict + confidence from structured JSON response
  5. Flag low-confidence predictions for human review
  6. Log every attempt — including failures — to classification_log
"""
import json
import logging
import re
import time
from dataclasses import dataclass

from openai import OpenAI, RateLimitError

from src.config import get_settings
from src.database import get_db, log_classification
from src.retrieval.fusion import retrieve_hybrid

logger = logging.getLogger(__name__)
settings = get_settings()

# Predictions below this threshold are flagged for human MLR review.
# Set at 0.7 (vs a generic 0.6) because pharma MLR errors carry regulatory
# and liability consequences — when in doubt, escalate to a human reviewer.
LOW_CONFIDENCE_THRESHOLD = 0.7


@dataclass
class ClassificationResult:
    label: str
    confidence: float
    rationale: str               # concise, reviewer-facing explanation
    retrieved_examples: list[dict]
    latency_ms: int
    needs_human_review: bool


def _build_prompt(query: str, examples: list[dict], labels: list[str]) -> str:
    """
    Build a few-shot MLR classification prompt.

    The retrieved documents act as in-context evidence — the LLM sees how
    similar claims were labeled and applies the same regulatory logic to the
    new claim. This is the core RAG pattern: retrieval + generation.

    insufficient_data is included as a valid label so the LLM has an explicit
    escape hatch when the retrieved examples are from an unrelated domain and
    it cannot ground its verdict in actual evidence.
    """
    # Exclude insufficient_data from the prompt definitions — it's the LLM's
    # escape hatch, not a normal compliance category. We describe it separately.
    classify_labels = [l for l in labels if l != "insufficient_data"]
    labels_str = ", ".join(classify_labels) + ", insufficient_data"

    examples_block = ""
    for i, ex in enumerate(examples, 1):
        examples_block += f"\nExample {i}:\n  Claim: {ex['text']}\n  Verdict: {ex['label']}\n"

    return f"""You are an expert pharmaceutical regulatory reviewer performing Medical-Legal-Regulatory (MLR) claim compliance analysis.

Your task is to classify a pharmaceutical marketing claim into exactly one of these compliance verdicts:
  {labels_str}

Verdict definitions:
  supported           — The claim is directly and fully backed by cited clinical data with ALL of the
                        following present: trial phase or study type, sample size or patient population,
                        specific effect size or endpoint result, statistical significance (p-value or CI),
                        and indication scope. If ANY of these elements is missing, use partially_supported.
                        Example: "In a Phase III RCT of 3,730 patients, drug X reduced CV death by 26%
                        vs placebo (HR 0.74; 95% CI 0.65-0.85; p<0.001)." ← supported

  partially_supported — The claim is directionally accurate but is MISSING at least one of: sample size,
                        effect size, p-value or confidence interval, patient population qualifier, or
                        indication scope. Vague efficacy language ("significantly improves", "reduces risk",
                        "demonstrated benefit") without accompanying statistics is always partially_supported.
                        Example: "Drug X significantly reduces cardiovascular risk in diabetic patients."
                        ← partially_supported (no trial data, no effect size, no p-value cited)
                        Example: "Drug X is indicated for adults with Type 2 diabetes." ← partially_supported
                        (approved indication language — directionally correct but no outcome evidence cited)

  unsupported         — The claim has no credible evidence basis, uses absolute language ("eliminates",
                        "cures", "the best", "guaranteed"), overstates efficacy beyond what trials showed,
                        or misrepresents data. No directional accuracy — the claim itself is wrong or fabricated.

  false_balance       — Safety information is presented in a way that obscures the true severity of risk.
                        A serious risk (black box warning, life-threatening adverse event, severe organ
                        toxicity, or rare but fatal adverse event) is omitted entirely, buried at the end,
                        minimised with hedging language, or juxtaposed with minor side effects (headache,
                        nausea) to make it seem equivalent. One-sided benefit claims with no mention of
                        contraindications or risk information also qualify.
                        Example: Ad emphasises "well tolerated, with mild side effects like headache"
                        while omitting a black box warning for hepatotoxicity. ← false_balance
                        Trigger question: Does the claim omit or downplay a serious known safety risk?

  needs_legal_review  — The claim raises concerns requiring legal or regulatory counsel: off-label use,
                        unapproved indications, comparative/superiority claims without head-to-head data,
                        IP or biosimilarity assertions, or pre-approval promotion.

  insufficient_data   — Use ONLY when the reference examples are from a clearly unrelated drug class,
                        therapeutic area, or regulatory context. Do NOT use as a fallback for uncertainty
                        — use confidence <0.7 instead and pick the closest label.

CRITICAL decision rule — supported vs partially_supported:
  Before assigning 'supported', verify the claim explicitly states ALL FIVE:
    ✓ Trial type or study design (Phase III, RCT, meta-analysis)
    ✓ Sample size or patient population
    ✓ Quantified outcome (%, HR, OR, absolute risk reduction)
    ✓ Statistical significance (p-value AND/OR confidence interval)
    ✓ Indication scope (which patients, which condition)
  If even one is absent → partially_supported. This is non-negotiable.

Regulatory context:
  - FDA promotional guidelines: claims must be fair, balanced, and not misleading (21 CFR Part 202).
  - OPDP standards require benefit claims be supported by substantial evidence.
  - Off-label promotion is prohibited; unapproved indications require legal review.
  - Comparative claims require head-to-head trial data using the same endpoints and patient population.

Reference claims from the compliance library (use these to ground your verdict):
{examples_block}

Claim to evaluate:
  {query}

Respond in this exact JSON format:
{{
  "rationale": "<2-3 sentence reviewer-facing explanation. Cite which reference examples informed your verdict. For supported vs partially_supported, explicitly state which of the 5 required elements are present or absent.>",
  "label": "<one of: {labels_str}>",
  "confidence": <float 0.0–1.0; be conservative — use ≥0.85 only when the verdict is unambiguous AND all evidence clearly points one way; use 0.70-0.84 for clear verdicts with minor uncertainty; use <0.70 when the claim sits between two verdicts or evidence is mixed; use 0.0 for insufficient_data. Overconfidence in MLR review is a compliance risk.>
}}"""


def _parse_retry_delay(error: RateLimitError, default: float = 60.0) -> float:
    """
    Extract the suggested retry delay from a RateLimitError message.
    Falls back to `default` seconds if no delay is found in the message.
    """
    match = re.search(r"retry in\s+([\d.]+)s", str(error), re.IGNORECASE)
    if match:
        return float(match.group(1)) + 2.0  # small buffer on top
    return default


def _call_llm(prompt: str, max_retries: int = 5) -> dict:
    """
    Call the LLM via the OpenAI-compatible client, with automatic retry on rate limits.

    Gemini's API is a drop-in replacement for OpenAI — same Python client,
    same interface, just a different base_url and api_key.
    response_format=json_object guarantees valid JSON output.
    Free tier: 1,500 req/day, 15 req/min (gemini-2.0-flash).

    On 429 rate limit errors, reads the suggested retry delay from the response
    and sleeps before retrying — so long-running evals complete unattended.
    """
    client = OpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=settings.llm_model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            break  # success — exit retry loop
        except RateLimitError as e:
            if attempt == max_retries:
                raise
            delay = _parse_retry_delay(e)
            logger.warning(
                f"Rate limit hit (attempt {attempt}/{max_retries}). "
                f"Sleeping {delay:.1f}s before retry..."
            )
            time.sleep(delay)

    raw = response.choices[0].message.content.strip()

    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not json_match:
        raise ValueError(f"No JSON found in LLM response: {raw}")

    return json.loads(json_match.group())


def classify(query: str, persist: bool = True) -> ClassificationResult:
    """
    Classify a pharma marketing claim using hybrid RAG + Gemini MLR reasoning.

    Args:
      query:   The claim text to evaluate.
      persist: If True, logs the result to classification_log (set False during eval).

    Returns:
      ClassificationResult with compliance verdict, confidence, rationale, and retrieved examples.

    Raises:
      Any exception from retrieval or LLM — but logs a failed status to the DB first.
    """
    start = time.time()

    try:
        # Step 1 — Retrieve similar labeled examples via hybrid RAG
        examples = retrieve_hybrid(query, top_k=settings.top_k_final)

        # Step 2 — Retrieval quality gate
        # If even the top-ranked result has a low RRF score, the knowledge base
        # doesn't contain sufficiently relevant precedents. Return insufficient_data
        # immediately rather than letting the LLM reason from poor evidence.
        top_rrf = examples[0]["rrf_score"] if examples else 0.0
        if not examples or top_rrf < settings.retrieval_quality_threshold:
            latency_ms = int((time.time() - start) * 1000)
            rationale = (
                f"No sufficiently relevant precedents found in the knowledge base "
                f"(best retrieval score: {top_rrf:.4f}, threshold: "
                f"{settings.retrieval_quality_threshold}). "
                f"This claim may be outside the current knowledge base domain. "
                f"Add relevant labeled examples and re-ingest to enable classification."
            )
            logger.warning(
                f"Insufficient retrieval quality for query='{query[:60]}' "
                f"(top_rrf={top_rrf:.4f} < threshold={settings.retrieval_quality_threshold})"
            )
            if persist:
                with get_db() as db:
                    log_classification(
                        db=db,
                        input_text=query,
                        predicted_label="insufficient_data",
                        confidence=0.0,
                        rationale=rationale,
                        retrieved_docs=[{k: v for k, v in ex.items() if k != "embedding"} for ex in examples],
                        latency_ms=latency_ms,
                        status="completed",
                    )
            return ClassificationResult(
                label="insufficient_data",
                confidence=0.0,
                rationale=rationale,
                retrieved_examples=examples,
                latency_ms=latency_ms,
                needs_human_review=True,
            )

        # Step 3 — Build prompt and call Gemini
        prompt = _build_prompt(query, examples, settings.label_list)
        response = _call_llm(prompt)

        # Step 4 — Parse and validate response
        label = response.get("label", "").strip()
        if label not in settings.label_list:
            logger.warning(f"LLM returned unexpected label '{label}' — defaulting to '{settings.label_list[0]}'")
            label = settings.label_list[0]

        confidence = float(response.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))  # clamp to [0, 1]
        rationale  = response.get("rationale", "")

        latency_ms    = int((time.time() - start) * 1000)
        needs_review  = confidence < LOW_CONFIDENCE_THRESHOLD

        # Step 5 — Persist successful classification
        if persist:
            serializable_examples = [
                {k: v for k, v in ex.items() if k != "embedding"}
                for ex in examples
            ]
            with get_db() as db:
                log_classification(
                    db=db,
                    input_text=query,
                    predicted_label=label,
                    confidence=confidence,
                    rationale=rationale,
                    retrieved_docs=serializable_examples,
                    latency_ms=latency_ms,
                    status="completed",
                )

        if needs_review:
            logger.warning(
                f"Low confidence: label='{label}', confidence={confidence:.2f} — flagged for human review"
            )

        return ClassificationResult(
            label=label,
            confidence=confidence,
            rationale=rationale,
            retrieved_examples=examples,
            latency_ms=latency_ms,
            needs_human_review=needs_review,
        )

    except Exception as e:
        latency_ms = int((time.time() - start) * 1000)
        logger.error(f"Classification failed for query='{query[:60]}': {e}")

        # Log the failure so it's visible in the DB without digging through logs
        if persist:
            try:
                with get_db() as db:
                    log_classification(
                        db=db,
                        input_text=query,
                        latency_ms=latency_ms,
                        status="failed",
                        error_message=str(e),
                    )
            except Exception as log_err:
                logger.error(f"Failed to log classification failure: {log_err}")

        raise
