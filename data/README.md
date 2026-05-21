# HybridRAG — Data Directory

## FDA OPDP Real-World Data Pipeline

This folder holds training data for the HybridRAG classifier.

---

### Workflow

```
make scrape-fda
    ↓
Review data/fda_opdp_raw.jsonl manually
    ↓
Delete or fix bad rows
    ↓
make db-reset   (wipe old AI-generated sample data)
make ingest-fda (seed with real FDA claims)
```

---

### Files

| File | What it is |
|---|---|
| `fda_opdp_raw.jsonl` | Raw scraper output — **review before ingesting** |
| `fda_opdp_skipped.jsonl` | Letters the scraper couldn't parse |
| `sample/test.jsonl` | Hand-curated eval set (don't delete) |

---

### JSONL format (each line)

```json
{
  "text": "The only once-daily treatment proven to reduce exacerbations by 60%",
  "label": "unsupported",
  "metadata": {
    "source": "fda_opdp",
    "letter_type": "untitled_letter",
    "letter_url": "https://www.fda.gov/...",
    "drug": "Exampla",
    "company": "Pharma Corp",
    "date": "March 14, 2022",
    "violation_context": "...lacks substantial evidence for the 60% claim...",
    "scraped_at": "2026-05-18T..."
  }
}
```

---

### Labels

| Label | Meaning |
|---|---|
| `supported` | Claim has cited clinical data / FDA-approved language |
| `unsupported` | Unsubstantiated — no adequate evidence cited |
| `partially_supported` | Some data exists but claim is overstated |
| `false_balance` | Risk info omitted / one-sided presentation |
| `needs_legal_review` | Off-label, superiority, or comparative claims |

---

### Review tips

1. Open `fda_opdp_raw.jsonl` in VS Code (JSON Lines format)
2. Check that `text` looks like an actual marketing claim, not boilerplate
3. Check that `label` matches the `violation_context` summary
4. Delete any rows where `text` is clearly procedural/legal language (not a claim)
5. Aim for ~100–200 high-quality rows; quality > quantity for few-shot RAG

---

### Running the scraper with options

```bash
# Full scrape (all letters)
make scrape-fda

# Quick test — first 30 letters only
make scrape-fda-quick

# Filter by year
source .venv/bin/activate
python scripts/scrape_fda_opdp.py --year 2023

# Custom output path
python scripts/scrape_fda_opdp.py --output data/my_subset.jsonl
```
