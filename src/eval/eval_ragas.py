"""
RAGAS evaluation harness for HybridRAG.

Adapts RAGAS (designed for QA systems) to a classification pipeline:
  question  → "Classify this pharma claim for MLR compliance: {claim}"
  answer    → "{label}: {rationale}"  (what the system produced)
  contexts  → [text of each retrieved reference example]
  ground_truth → "{expected_label}: {expected_rationale}"

Metrics computed:
  - Faithfulness:       Is the rationale grounded in retrieved examples?
  - Answer Relevance:   Does the answer actually address the classification question?
  - Context Precision:  Are relevant examples ranked above irrelevant ones?
  - Context Recall:     Did retrieval surface the examples needed for the correct verdict?
  - Answer Similarity:  Does the produced verdict+rationale match ground truth semantically?

Also runs standard metrics (accuracy, F1, calibration, latency) from evaluate.py
and produces a combined JSON report + slice breakdown.

Run modes:
  --mode live    : calls the real classifier (requires running DB + API key)
  --mode offline : uses seed examples with mock contexts (for CI, no DB needed)

Usage:
  python -m src.eval.eval_ragas --mode live --data data/eval/seed_examples.jsonl
  python -m src.eval.eval_ragas --mode offline --data data/eval/seed_examples.jsonl
  python -m src.eval.eval_ragas --mode live --generate-synthetic --n-synthetic 40
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    answer_similarity,
    context_precision,
    context_recall,
    faithfulness,
)
from sklearn.metrics import accuracy_score, classification_report, f1_score

logger = logging.getLogger(__name__)

# ── Label set ────────────────────────────────────────────────────────────────
VALID_LABELS = {
    "supported",
    "unsupported",
    "partially_supported",
    "false_balance",
    "needs_legal_review",
    "insufficient_data",
}

# ── Slice definitions ─────────────────────────────────────────────────────────
# Each slice is a set of label values. We report RAGAS + F1 per slice so you
# can see which claim types the system struggles with.
SLICES = {
    "supported":          {"supported"},
    "unsupported":        {"unsupported"},
    "partially_supported": {"partially_supported"},
    "false_balance":      {"false_balance"},
    "needs_legal_review": {"needs_legal_review"},
    "insufficient_data":  {"insufficient_data"},
}


# ─────────────────────────────────────────────────────────────────────────────
# Live classification (calls the real system)
# ─────────────────────────────────────────────────────────────────────────────

def _run_live_classification(samples: list[dict]) -> list[dict]:
    """
    Classify each sample using the real HybridRAG pipeline.
    Returns a list of result dicts with all fields needed for RAGAS.
    """
    from src.classifier.classify import classify

    results = []
    for i, sample in enumerate(samples):
        claim = sample["claim"]
        logger.info(f"  [{i+1}/{len(samples)}] Classifying: {claim[:60]}...")

        t0 = time.time()
        result = classify(claim, persist=False)
        latency_ms = int((time.time() - t0) * 1000)

        # Format answer as "label: rationale" — this is what RAGAS will evaluate
        answer = f"{result.label}: {result.rationale}"

        # Extract text from retrieved examples as the context block
        contexts = [
            f"Claim: {ex.get('text', '')} | Label: {ex.get('label', '')} | "
            f"RRF Score: {ex.get('rrf_score', 0.0):.4f}"
            for ex in result.retrieved_examples
        ] or ["No relevant examples retrieved from knowledge base."]

        results.append({
            "id":              sample.get("id", f"sample_{i}"),
            "slice":           sample.get("slice", "unknown"),
            "claim":           claim,
            "question":        f"Classify this pharma marketing claim for MLR compliance: {claim}",
            "answer":          answer,
            "contexts":        contexts,
            "ground_truth":    sample["ground_truth"],
            "expected_label":  sample["expected_label"],
            "predicted_label": result.label,
            "confidence":      result.confidence,
            "latency_ms":      latency_ms,
            "needs_review":    result.needs_human_review,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Offline mode (no DB — uses seed examples with mock retrieved contexts)
# ─────────────────────────────────────────────────────────────────────────────

def _build_mock_contexts(sample: dict, all_samples: list[dict]) -> list[str]:
    """
    Build mock retrieved contexts for offline mode.
    Simulates what the retriever would return: other seed examples as "retrieved" docs.
    For same-label examples: treated as relevant (high precision scenario).
    For cross-label examples: treated as noise (tests precision degradation).
    """
    same_label = [
        f"Claim: {s['claim']} | Label: {s['expected_label']}"
        for s in all_samples
        if s["id"] != sample["id"] and s["expected_label"] == sample["expected_label"]
    ]
    other_label = [
        f"Claim: {s['claim']} | Label: {s['expected_label']}"
        for s in all_samples
        if s["id"] != sample["id"] and s["expected_label"] != sample["expected_label"]
    ]
    # Return 2 same-label + 1 different — realistic retrieval mix
    return (same_label[:2] + other_label[:1]) or [
        "No relevant examples available in offline mode."
    ]


def _run_offline_classification(samples: list[dict]) -> list[dict]:
    """
    Offline mode: use ground truth labels as "predicted" labels (no LLM call).
    This lets you test the RAGAS harness structure without a live system.
    Useful for CI checks and structure validation.
    """
    results = []
    for i, sample in enumerate(samples):
        # In offline mode, simulate a "perfect" system so RAGAS metrics
        # reflect the eval harness structure, not system quality
        answer = sample["ground_truth"]
        contexts = _build_mock_contexts(sample, samples)

        results.append({
            "id":              sample.get("id", f"sample_{i}"),
            "slice":           sample.get("slice", "unknown"),
            "claim":           sample["claim"],
            "question":        f"Classify this pharma marketing claim for MLR compliance: {sample['claim']}",
            "answer":          answer,
            "contexts":        contexts,
            "ground_truth":    sample["ground_truth"],
            "expected_label":  sample["expected_label"],
            "predicted_label": sample["expected_label"],   # perfect prediction in offline mode
            "confidence":      0.90,
            "latency_ms":      0,
            "needs_review":    False,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic testset generation via RAGAS
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_testset(n: int = 40, output_path: str = "data/eval/synthetic_examples.jsonl") -> list[dict]:
    """
    Use RAGAS TestsetGenerator to synthesize eval cases from the FDA data.
    Requires: live DB + embeddings. Generates simple, reasoning, and multi-context questions.

    The generated cases supplement the 10 manual seed examples — they expose
    failure modes you wouldn't think to test manually.
    """
    try:
        from langchain_community.document_loaders import JSONLoader
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas.testset.evolutions import multi_context, reasoning, simple
        from ragas.testset.generator import TestsetGenerator
    except ImportError as e:
        logger.error(f"Missing dependency for synthetic generation: {e}")
        logger.error("Install with: pip install langchain-community langchain-openai")
        return []

    logger.info(f"Generating {n} synthetic test cases from FDA data...")

    # Load FDA data from the ingested JSONL
    fda_path = Path("data/fda_opdp_raw.jsonl")
    if not fda_path.exists():
        logger.warning(f"FDA data not found at {fda_path}. Run 'make scrape-fda' first.")
        return []

    loader = JSONLoader(
        file_path=str(fda_path),
        jq_schema=".[]",
        text_content=False,
    )
    documents = loader.load()

    if not documents:
        logger.warning("No documents loaded from FDA data.")
        return []

    generator_llm = ChatOpenAI(model="gpt-4o-mini")   # cheap model for generation
    critic_llm    = ChatOpenAI(model="gpt-4o")         # stronger model for critique
    embeddings    = OpenAIEmbeddings()

    generator = TestsetGenerator.from_langchain(generator_llm, critic_llm, embeddings)

    testset = generator.generate_with_langchain_docs(
        documents,
        test_size=n,
        distributions={
            simple:        0.50,   # direct lookup questions
            reasoning:     0.25,   # inference-required questions
            multi_context: 0.25,   # multi-chunk synthesis questions
        },
    )

    df = testset.to_pandas()
    synthetic_samples = []

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for i, row in df.iterrows():
            sample = {
                "id":             f"synthetic_{i:03d}",
                "slice":          row.get("evolution_type", "synthetic"),
                "claim":          row["question"],
                "expected_label": "unknown",           # synthetic cases need human labeling
                "ground_truth":   row["ground_truth"],
            }
            synthetic_samples.append(sample)
            f.write(json.dumps(sample) + "\n")

    logger.info(f"Saved {len(synthetic_samples)} synthetic examples to {output_path}")
    logger.warning(
        "Synthetic examples have expected_label='unknown'. "
        "Review and label data/eval/synthetic_examples.jsonl before using for F1 eval."
    )
    return synthetic_samples


# ─────────────────────────────────────────────────────────────────────────────
# RAGAS metrics computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_ragas_metrics(results: list[dict]) -> dict:
    """
    Run all 5 RAGAS metrics on the result set.
    Filters to samples with known ground_truth for context_precision + context_recall.
    """
    dataset = Dataset.from_dict({
        "question":     [r["question"]  for r in results],
        "answer":       [r["answer"]    for r in results],
        "contexts":     [r["contexts"]  for r in results],
        "ground_truth": [r["ground_truth"] for r in results],
    })

    logger.info("Running RAGAS evaluation (this calls the LLM judge)...")

    ragas_result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_similarity,
        ],
    )

    return {
        "faithfulness":      round(ragas_result["faithfulness"],      4),
        "answer_relevancy":  round(ragas_result["answer_relevancy"],  4),
        "context_precision": round(ragas_result["context_precision"], 4),
        "context_recall":    round(ragas_result["context_recall"],    4),
        "answer_similarity": round(ragas_result["answer_similarity"], 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Standard classification metrics
# ─────────────────────────────────────────────────────────────────────────────

def _compute_standard_metrics(results: list[dict]) -> dict:
    """
    Accuracy, macro F1, per-class F1, calibration error, latency percentiles.
    Only runs on samples with known expected labels (skips synthetic/unknown).
    """
    labeled = [r for r in results if r["expected_label"] != "unknown"]
    if not labeled:
        return {}

    true_labels = [r["expected_label"]  for r in labeled]
    pred_labels = [r["predicted_label"] for r in labeled]
    confidences = [r["confidence"]      for r in labeled]
    latencies   = [r["latency_ms"]      for r in labeled]

    accuracy = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    report   = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)

    correctness = [1.0 if t == p else 0.0 for t, p in zip(true_labels, pred_labels)]
    calibration_error = sum(
        abs(c - corr) for c, corr in zip(confidences, correctness)
    ) / len(labeled)

    latency_series = pd.Series(latencies)

    return {
        "n_samples":           len(labeled),
        "accuracy":            round(accuracy, 4),
        "macro_f1":            round(macro_f1, 4),
        "calibration_error":   round(calibration_error, 4),
        "mean_confidence":     round(sum(confidences) / len(confidences), 4),
        "low_confidence_rate": round(sum(1 for c in confidences if c < 0.7) / len(confidences), 4),
        "latency_p50_ms":      int(latency_series.quantile(0.50)) if latencies else 0,
        "latency_p95_ms":      int(latency_series.quantile(0.95)) if latencies else 0,
        "per_class_f1": {
            label: round(v["f1-score"], 4)
            for label, v in report.items()
            if label not in ("accuracy", "macro avg", "weighted avg")
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Slice-based breakdown
# ─────────────────────────────────────────────────────────────────────────────

def _compute_slice_metrics(results: list[dict], ragas_df: pd.DataFrame | None = None) -> dict:
    """
    Break metrics down by slice (claim type / label category).
    For each slice: n_samples, accuracy, F1, mean confidence, mean RAGAS faithfulness.

    ragas_df: per-sample RAGAS scores if available (from ragas evaluate internals).
    """
    slice_metrics = {}

    for slice_name, label_set in SLICES.items():
        slice_results = [
            r for r in results
            if r.get("slice") == slice_name or r.get("expected_label") in label_set
        ]
        labeled = [r for r in slice_results if r["expected_label"] != "unknown"]

        if not labeled:
            continue

        true_labels = [r["expected_label"]  for r in labeled]
        pred_labels = [r["predicted_label"] for r in labeled]
        confidences = [r["confidence"]      for r in labeled]

        f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)

        slice_metrics[slice_name] = {
            "n_samples":       len(labeled),
            "accuracy":        round(accuracy_score(true_labels, pred_labels), 4),
            "f1":              round(f1, 4),
            "mean_confidence": round(sum(confidences) / len(confidences), 4),
            "needs_review_rate": round(
                sum(1 for r in labeled if r["needs_review"]) / len(labeled), 4
            ),
        }

    return slice_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Report output
# ─────────────────────────────────────────────────────────────────────────────

def _print_report(report: dict) -> None:
    """Pretty-print the combined eval report to stdout."""

    std  = report.get("standard_metrics", {})
    ragas = report.get("ragas_metrics", {})
    slices = report.get("slice_metrics", {})

    print("\n" + "═" * 58)
    print("  HybridRAG — Combined Eval Report")
    print(f"  {report['timestamp']}  |  mode: {report['mode']}")
    print("═" * 58)

    if std:
        print(f"\n  STANDARD METRICS  ({std.get('n_samples', 0)} labeled samples)")
        print(f"  {'Accuracy':<28} {std.get('accuracy', 0):.1%}")
        print(f"  {'Macro F1':<28} {std.get('macro_f1', 0):.1%}")
        print(f"  {'Calibration Error':<28} {std.get('calibration_error', 0):.4f}")
        print(f"  {'Mean Confidence':<28} {std.get('mean_confidence', 0):.1%}")
        print(f"  {'Low Confidence Rate':<28} {std.get('low_confidence_rate', 0):.1%}")
        print(f"  {'Latency p50 / p95':<28} {std.get('latency_p50_ms', 0)}ms / {std.get('latency_p95_ms', 0)}ms")

        if "per_class_f1" in std:
            print("\n  PER-CLASS F1:")
            for label, f1 in std["per_class_f1"].items():
                bar_len = int(f1 * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                flag = " ⚠" if f1 < 0.70 else ""
                print(f"  {label:<28} {bar} {f1:.1%}{flag}")

    if ragas:
        print(f"\n  RAGAS METRICS")
        thresholds = {
            "faithfulness":      (0.90, "hallucination gate"),
            "answer_relevancy":  (0.85, "on-topic gate"),
            "context_precision": (0.80, "retrieval ranking"),
            "context_recall":    (0.80, "retrieval coverage"),
            "answer_similarity": (0.85, "end-to-end quality"),
        }
        for metric, score in ragas.items():
            threshold, label = thresholds.get(metric, (0.80, ""))
            status = "✓" if score >= threshold else "✗"
            print(f"  {metric:<28} {score:.4f}  {status}  (target ≥{threshold:.2f}, {label})")

    if slices:
        print(f"\n  SLICE BREAKDOWN:")
        print(f"  {'Slice':<28} {'N':>4}  {'Acc':>6}  {'F1':>6}  {'Conf':>6}  {'Review%':>8}")
        print("  " + "─" * 56)
        for slice_name, m in slices.items():
            flag = " ⚠" if m["f1"] < 0.70 else ""
            print(
                f"  {slice_name:<28} {m['n_samples']:>4}  "
                f"{m['accuracy']:>5.1%}  {m['f1']:>5.1%}  "
                f"{m['mean_confidence']:>5.1%}  {m['needs_review_rate']:>7.1%}{flag}"
            )

    print("\n" + "═" * 58)
    out = report.get("output_path", "")
    if out:
        print(f"  Full report saved to: {out}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_ragas_eval(
    data_path: str,
    mode: str = "live",
    generate_synthetic: bool = False,
    n_synthetic: int = 40,
    output_dir: str = "data/eval/reports",
) -> dict:
    """
    Full evaluation pipeline:
      1. Load seed examples
      2. Optionally generate + append synthetic examples
      3. Classify all examples (live or offline)
      4. Compute RAGAS metrics
      5. Compute standard metrics
      6. Compute slice-level metrics
      7. Save JSON report + print summary

    Args:
      data_path:          Path to JSONL eval file (seed_examples.jsonl)
      mode:               'live' (real classifier) or 'offline' (mock, no DB)
      generate_synthetic: Whether to generate synthetic examples via RAGAS generator
      n_synthetic:        Number of synthetic examples to generate
      output_dir:         Directory to save JSON reports

    Returns:
      Combined report dict
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Eval data not found: {data_path}")

    # ── Step 1: Load seed examples ────────────────────────────────────────────
    samples = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    logger.info(f"Loaded {len(samples)} seed examples from {data_path}")

    # ── Step 2: Optionally generate synthetic examples ────────────────────────
    if generate_synthetic:
        synthetic_path = "data/eval/synthetic_examples.jsonl"
        synthetic = generate_synthetic_testset(n=n_synthetic, output_path=synthetic_path)
        if synthetic:
            samples.extend(synthetic)
            logger.info(f"Added {len(synthetic)} synthetic examples (total: {len(samples)})")

    # ── Step 3: Classify ──────────────────────────────────────────────────────
    logger.info(f"Running classification in '{mode}' mode...")
    if mode == "live":
        results = _run_live_classification(samples)
    else:
        results = _run_offline_classification(samples)

    # ── Step 4: RAGAS metrics ─────────────────────────────────────────────────
    ragas_metrics = {}
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key:
        ragas_metrics = _compute_ragas_metrics(results)
    else:
        logger.warning(
            "OPENAI_API_KEY not set. Skipping RAGAS metrics (they use GPT as judge). "
            "Set OPENAI_API_KEY in .env to enable RAGAS scoring."
        )

    # ── Step 5: Standard metrics ──────────────────────────────────────────────
    standard_metrics = _compute_standard_metrics(results)

    # ── Step 6: Slice metrics ─────────────────────────────────────────────────
    slice_metrics = _compute_slice_metrics(results)

    # ── Step 7: Assemble + save report ───────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"eval_{timestamp}_{mode}.json"

    report = {
        "timestamp":        datetime.now().isoformat(),
        "mode":             mode,
        "n_total_samples":  len(samples),
        "standard_metrics": standard_metrics,
        "ragas_metrics":    ragas_metrics,
        "slice_metrics":    slice_metrics,
        "output_path":      str(output_path),
        "per_sample_results": [
            {k: v for k, v in r.items() if k != "contexts"}  # omit long context text from report
            for r in results
        ],
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    _print_report(report)

    return report


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAGAS + standard eval harness for HybridRAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full live eval (requires DB + API keys):
  python -m src.eval.eval_ragas --mode live --data data/eval/seed_examples.jsonl

  # Offline structure check (no DB, no API keys):
  python -m src.eval.eval_ragas --mode offline --data data/eval/seed_examples.jsonl

  # Live + generate 40 synthetic examples:
  python -m src.eval.eval_ragas --mode live --generate-synthetic --n-synthetic 40
        """,
    )
    parser.add_argument(
        "--data", default="data/eval/seed_examples.jsonl",
        help="Path to JSONL eval file (default: data/eval/seed_examples.jsonl)"
    )
    parser.add_argument(
        "--mode", choices=["live", "offline"], default="live",
        help="'live' calls the real classifier; 'offline' uses mock data (default: live)"
    )
    parser.add_argument(
        "--generate-synthetic", action="store_true",
        help="Generate synthetic test cases from FDA data using RAGAS TestsetGenerator"
    )
    parser.add_argument(
        "--n-synthetic", type=int, default=40,
        help="Number of synthetic examples to generate (default: 40)"
    )
    parser.add_argument(
        "--output-dir", default="data/eval/reports",
        help="Directory to save JSON reports (default: data/eval/reports)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    run_ragas_eval(
        data_path=args.data,
        mode=args.mode,
        generate_synthetic=args.generate_synthetic,
        n_synthetic=args.n_synthetic,
        output_dir=args.output_dir,
    )
