"""
Eval Set Builder
=================
Builds a larger, balanced eval set by sampling from real labeled data sources.
Combines existing seed examples with samples from scraped FDA data and PI docs.

Sources (in priority order):
  1. data/eval/seed_examples.jsonl     — hand-crafted, gold standard (keep all)
  2. data/fda_opdp_raw.jsonl           — real FDA enforcement letters (false_balance,
                                         unsupported, needs_legal_review)
  3. data/pi_claims_review.jsonl       — FDA-approved PI language (supported)
  4. data/pi_claims_raw.jsonl          — FDA PI indications (partially_supported)

Output:
  data/eval/eval_large.jsonl           — 60-80 examples, balanced across labels

Usage:
  python scripts/build_eval_set.py
  python scripts/build_eval_set.py --per-label 15 --output data/eval/eval_large.jsonl
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

DATA_DIR = Path("data")
EVAL_DIR = DATA_DIR / "eval"

# Target examples per label from non-seed sources
DEFAULT_PER_LABEL = 12

# Labels and which files to sample them from
LABEL_SOURCES = {
    "false_balance":       [DATA_DIR / "fda_opdp_raw.jsonl"],
    "unsupported":         [DATA_DIR / "fda_opdp_raw.jsonl"],
    "needs_legal_review":  [DATA_DIR / "fda_opdp_raw.jsonl"],
    "partially_supported": [DATA_DIR / "pi_claims_raw.jsonl"],
    "supported":           [DATA_DIR / "pi_claims_review.jsonl"],
}

# Field name mapping per source file
# seed_examples uses "claim" + "expected_label"
# all other files use "text" + "label"
def normalise(record: dict, source_file: Path) -> dict | None:
    """Normalise a record to standard {"text", "label", "source"} format."""
    if "seed_examples" in str(source_file):
        text  = record.get("claim", "")
        label = record.get("expected_label", "")
    else:
        text  = record.get("text", "")
        label = record.get("label", "")

    if not text or not label:
        return None
    return {
        "text":   text.strip(),
        "label":  label.strip(),
        "source": source_file.name,
    }


def load_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def build_eval_set(per_label: int = DEFAULT_PER_LABEL, seed: int = 42) -> list[dict]:
    random.seed(seed)
    eval_set: list[dict] = []
    seen_texts: set[str] = set()

    # ── Step 1: Keep all seed examples (gold standard) ────────────────────────
    seed_path = EVAL_DIR / "seed_examples.jsonl"
    for raw in load_jsonl(seed_path):
        norm = normalise(raw, seed_path)
        if norm and norm["text"] not in seen_texts:
            eval_set.append(norm)
            seen_texts.add(norm["text"])

    seed_counts = Counter(r["label"] for r in eval_set)
    print(f"Seed examples loaded: {len(eval_set)}")
    for lbl, cnt in seed_counts.items():
        print(f"  {lbl:<25} {cnt}")

    # ── Step 2: Sample from scraped sources per label ─────────────────────────
    print(f"\nSampling up to {per_label} additional examples per label...")

    for label, source_paths in LABEL_SOURCES.items():
        already_have = seed_counts.get(label, 0)
        need = per_label - already_have
        if need <= 0:
            print(f"  {label:<25} already have {already_have} from seed — skipping")
            continue

        # Gather all candidates for this label from all source files
        candidates: list[dict] = []
        for path in source_paths:
            for raw in load_jsonl(path):
                norm = normalise(raw, path)
                if (
                    norm
                    and norm["label"] == label
                    and norm["text"] not in seen_texts
                    and len(norm["text"]) >= 40
                    and len(norm["text"]) <= 500
                ):
                    candidates.append(norm)

        if not candidates:
            print(f"  {label:<25} no candidates found in source files")
            continue

        # Shuffle and pick — random sample ensures diversity
        random.shuffle(candidates)
        selected = candidates[:need]
        for rec in selected:
            eval_set.append(rec)
            seen_texts.add(rec["text"])

        print(f"  {label:<25} +{len(selected)} (had {already_have}, wanted {per_label})")

    return eval_set


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a balanced eval set from real FDA data")
    parser.add_argument("--per-label", type=int, default=DEFAULT_PER_LABEL,
                        help=f"Target examples per label (default: {DEFAULT_PER_LABEL})")
    parser.add_argument("--output",    type=str,
                        default=str(EVAL_DIR / "eval_large.jsonl"),
                        help="Output JSONL path")
    parser.add_argument("--seed",      type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output)

    eval_set = build_eval_set(per_label=args.per_label, seed=args.seed)

    # Write output
    with open(out_path, "w") as f:
        for rec in eval_set:
            f.write(json.dumps(rec) + "\n")

    # Summary
    final_counts = Counter(r["label"] for r in eval_set)
    total = len(eval_set)
    max_count = max(final_counts.values()) if final_counts else 1
    min_count = min(final_counts.values()) if final_counts else 1

    print(f"\n{'─' * 60}")
    print(f"  Eval Set Built — {out_path}")
    print(f"{'─' * 60}")
    print(f"  Total examples : {total}")
    print(f"  Imbalance ratio: {max_count/max(min_count,1):.1f}x")
    print()
    print(f"  {'Label':<25} {'Count':>5}  {'%':>6}")
    print(f"  {'─'*25} {'─'*5}  {'─'*6}")
    for label, count in sorted(final_counts.items()):
        pct = count / total * 100
        bar = "█" * int(pct / 3)
        print(f"  {label:<25} {count:>5}  {pct:>5.1f}%  {bar}")
    print()
    print(f"  Run eval:")
    print(f"    python -m src.eval.evaluate --data {out_path}")
    print(f"{'─' * 60}\n")


if __name__ == "__main__":
    main()
