"""
LLM-based classification head with few-shot retrieval augmentation.

Domain: Pharmaceutical marketing claim compliance (MLR review).

Pipeline:
  1. Retrieve similar labeled claims via hybrid RAG (BGE dense + BM25 sparse + RRF fusion)
  2. Format retrieved examples as few-shot in-context evidence
  3. Ask Groq (Llama 3.3 70B) to classify with explicit MLR regulatory reasoning
  4. Parse compliance verdict + confidence from structured JSON response
  5. Flag low-confidence predictions for human review
  6. Log every attempt — including failures — to classification_log
"""
import json
import logging
import re
import time
from dataclasses import dataclass

from openai import OpenAI

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
  supported           — The claim is directly and fully backed by cited clinical data, prescribing information, or peer-reviewed evidence. No material omissions.
  unsupported         — The claim has no credible evidence basis, overstates efficacy, misrepresents data, or uses absolute language ("eliminates", "cures", "best") without trial support.
  partially_supported — The claim is directionally accurate but missing critical hedges (patient population, effect size, statistical significance, confidence intervals, or indication scope).
  false_balance       — A serious safety risk (black box warning, rare life-threatening adverse event) is presented alongside a minor side effect in a way that obscures severity.
  needs_legal_review  — The claim raises IP, off-label use, comparative advertising, biosimilarity, interchangeability, or pre-approval promotion concerns requiring legal/regulatory counsel.
  insufficient_data   — Use ONLY when the reference examples below are from a clearly unrelated drug class, therapeutic area, or regulatory context and cannot meaningfully inform a verdict. Do NOT use this as a fallback for uncertainty — use confidence <0.7 instead.

Critical instruction: Your verdict must be grounded in the provided reference examples.
Do NOT classify from general pharmaceutical or FDA knowledge alone. If you cannot
connect your verdict to at least one of the reference examples, return insufficient_data.

Regulatory context:
  - FDA promotional guidelines: claims must be fair, balanced, and not misleading (21 CFR Part 202).
  - OPDP standards require benefit claims be supported by substantial evidence.
  - Off-label promotion is prohibited; unapproved indications require legal review.
  - Comparative claims require head-to-head trial data using the same endpoints and patient population.

Reference claims from the compliance library:
{examples_block}

Claim to evaluate:
  {query}

Respond in this exact JSON format:
{{
  "rationale": "<2-3 sentence reviewer-facing explanation. If classifying normally, cite which reference examples informed the verdict. If returning insufficient_data, explain why the reference examples are not relevant to this claim.>",
  "label": "<one of: {labels_str}>",
  "confidence": <float 0.0–1.0; use ≥0.85 only when verdict is unambiguous; use <0.7 when claim sits between two verdicts; use 0.0 for insufficient_data>
}}"""


def _call_llm(prompt: str) -> dict:
    """
    Call Groq (Llama 3.3 70B) via the OpenAI-compatible client.

    Groq's API is a drop-in replacement for OpenAI — same Python client,
    same interface, just a different base_url and api_key.
    response_format=json_object guarantees valid JSON output.
    """
    client = OpenAI(
        api_key=settings.groq_api_key,
        base_url=settings.groq_base_url,
    )

    response = client.chat.completions.create(
        model=settings.groq_model,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()

    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not json_match:
        raise ValueError(f"No JSON found in LLM response: {raw}")

    return json.loads(json_match.group())


def classify(query: str, persist: bool = True) -> ClassificationResult:
    """
    Classify a pharma marketing claim using hybrid RAG + Groq MLR reasoning.

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

        # Step 3 — Build prompt and call Groq
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
