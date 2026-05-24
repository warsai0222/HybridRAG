"""
Eval harness — measures accuracy, F1, confidence calibration, and latency.

Run: python -m src.eval.evaluate --data data/sample/test.jsonl
"""
import json
import logging
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.classifier.classify import classify

logger = logging.getLogger(__name__)


def run_eval(test_path: str, max_samples: int | None = None) -> dict:
    """
    Run evaluation on a JSONL test file.

    Each line: {"text": "...", "label": "..."}
    Labels must be valid MLR compliance verdicts.

    Returns a dict with accuracy, macro F1, per-class metrics, confidence
    calibration error, and latency percentiles.
    """
    test_path = Path(test_path)
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    samples = []
    with open(test_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    if max_samples:
        samples = samples[:max_samples]

    logger.info(f"Running eval on {len(samples)} samples from {test_path}")

    true_labels, pred_labels, confidences, latencies = [], [], [], []

    for i, sample in enumerate(samples):
        # Support both field naming conventions:
        # Standard ingest format:  {"text": "...", "label": "..."}
        # Seed eval format:        {"claim": "...", "expected_label": "..."}
        text       = sample.get("text") or sample.get("claim", "")
        true_label = sample.get("label") or sample.get("expected_label", "")
        result = classify(text, persist=False)
        true_labels.append(true_label)
        pred_labels.append(result.label)
        confidences.append(result.confidence)
        latencies.append(result.latency_ms)

        if (i + 1) % 10 == 0:
            logger.info(f"  {i + 1}/{len(samples)} evaluated...")

    # ── Core metrics ─────────────────────────────────────────────────────────
    accuracy  = accuracy_score(true_labels, pred_labels)
    macro_f1  = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    report    = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)

    # Confidence calibration: mean absolute error between confidence and binary correctness
    # A well-calibrated model that says 0.8 confidence should be right ~80% of the time.
    correctness = [1.0 if t == p else 0.0 for t, p in zip(true_labels, pred_labels)]
    calibration_error = sum(
        abs(c - corr) for c, corr in zip(confidences, correctness)
    ) / len(samples)

    latencies_ms = pd.Series(latencies)

    results = {
        "n_samples":          len(samples),
        "accuracy":           round(accuracy, 4),
        "macro_f1":           round(macro_f1, 4),
        "per_class": {
            label: {
                "precision": round(v["precision"], 4),
                "recall":    round(v["recall"], 4),
                "f1":        round(v["f1-score"], 4),
            }
            for label, v in report.items()
            if label not in ("accuracy", "macro avg", "weighted avg")
        },
        "mean_confidence":    round(sum(confidences) / len(confidences), 4),
        "calibration_error":  round(calibration_error, 4),
        "latency_p50_ms":     int(latencies_ms.quantile(0.50)),
        "latency_p95_ms":     int(latencies_ms.quantile(0.95)),
        "latency_p99_ms":     int(latencies_ms.quantile(0.99)),
        # Matches LOW_CONFIDENCE_THRESHOLD in classify.py
        "low_confidence_rate": round(
            sum(1 for c in confidences if c < 0.7) / len(confidences), 4
        ),
    }

    print("\n── HybridRAG Eval Results ───────────────────────")
    print(f"  Samples evaluated:  {results['n_samples']}")
    print(f"  Accuracy:           {results['accuracy']:.1%}")
    print(f"  Macro F1:           {results['macro_f1']:.1%}")
    print(f"  Calibration Error:  {results['calibration_error']:.4f}")
    print(f"  Mean Confidence:    {results['mean_confidence']:.1%}")
    print(f"  Low Conf Rate:      {results['low_confidence_rate']:.1%}  (< 0.7 threshold)")
    print(f"  Latency p50 / p95:  {results['latency_p50_ms']}ms / {results['latency_p95_ms']}ms")
    print("\n  Per-class F1:")
    for label, metrics in results["per_class"].items():
        print(f"    {label:<25} F1={metrics['f1']:.1%}  P={metrics['precision']:.1%}  R={metrics['recall']:.1%}")
    print("─────────────────────────────────────────────────\n")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate HybridRAG classifier")
    parser.add_argument("--data",  required=True,      help="Path to JSONL test file")
    parser.add_argument("--max",   type=int, default=None, help="Max samples to evaluate")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_eval(args.data, args.max)
