# I Built a Pharma Compliance Classifier on Real FDA Data — Here's How It Works and What I Learned

When I started building HybridRAG, I thought the hard part would be the LLM prompt. It wasn't. The hard part was retrieval — and understanding why a single retrieval strategy always leaves something on the table.

Here's the full breakdown of what I built, the decisions I made, and the lessons I'd carry into the next project.

---

## What It Does

HybridRAG is a Medical-Legal-Regulatory (MLR) claim compliance classifier. You give it a pharmaceutical marketing claim — something a drug company might put in an ad or a pamphlet — and it tells you whether the claim is supported by clinical evidence, unsupported, partially supported, or raises legal/safety red flags.

It outputs a six-way verdict:

| Label | What it means |
|---|---|
| `supported` | Fully backed by cited clinical data |
| `unsupported` | Overstates efficacy, no evidence basis |
| `partially_supported` | Directionally correct but missing key hedges |
| `false_balance` | Buries a serious safety risk next to a minor one |
| `needs_legal_review` | Off-label, comparative, or biosimilarity concerns |
| `insufficient_data` | Outside the knowledge base domain |

This mirrors what real MLR review teams do manually before every promotional piece gets approved. Companies like Solstice Health are building software to automate exactly this. I wanted to understand how.

---

## Stage 1: The Data Foundation

The classifier is only as good as its reference library. You can't classify a claim in isolation — you need precedents. So the first question was: where do I get labeled examples of compliant and non-compliant pharma claims?

The answer: FDA's Office of Prescription Drug Promotion (OPDP) publishes every untitled and warning letter they send to pharmaceutical companies. These are real enforcement actions — real claims that real companies submitted, and real explanations of why the FDA found them violating.

I built a scraper using `pdfplumber` to pull and parse these PDFs, extract the violating claims, and assign labels based on the violation type described in the letter.

```python
# scripts/scrape_fda_opdp.py — core extraction loop
for letter in opdp_letters:
    pdf_text = extract_pdf_text(letter.url)
    claims = extract_claims_from_letter(pdf_text)
    label = assign_label_from_violation_type(pdf_text)
```

The key insight here: FDA enforcement letters are an underused corpus. They're free, authoritative, and structured around exactly the compliance taxonomy I needed. No manual labeling required.

---

## Stage 2: Hybrid Retrieval

This is the core architectural decision of the project — and the thing I'd explain in every interview.

Most RAG implementations use only dense retrieval: embed the query, find the nearest vectors, retrieve the top-k. It works. But it has a blind spot.

Dense retrieval understands *meaning*. "Reduces cardiac events" and "lowers heart attack risk" score high together even though they share no words. But dense retrieval struggles with *specificity* — exact drug names, statistical terms, specific trial identifiers. If you ask about "FORXIGA HR 0.74 p<0.001" and the database has exactly that string, dense retrieval might not rank it first.

BM25 (sparse retrieval) is the opposite. It's pure keyword matching — if the string is there, it finds it. But it's blind to semantics. "Reduces cardiac events" and "lowers heart attack risk" score zero overlap.

The solution is to run both and fuse the results.

```python
# src/retrieval/fusion.py
def retrieve_hybrid(query: str, top_k: int = 5) -> list[dict]:
    dense_results  = retrieve_dense(query,  top_k=10)  # semantic
    sparse_results = retrieve_sparse(query, top_k=10)  # keyword
    return reciprocal_rank_fusion(dense_results, sparse_results, top_k=top_k)
```

But here's where I made my first mistake: I tried to fuse by averaging the scores.

It doesn't work. Dense scores are cosine similarity values between 0 and 1. BM25 scores are term-frequency weighted values that can be arbitrarily large. Averaging them is mathematically meaningless.

The fix is Reciprocal Rank Fusion (RRF), from a 2009 paper by Cormack et al.:

```
RRF(d) = Σ [ 1 / (k + rank(d)) ]
```

Instead of comparing scores, you compare *ranks*. A document ranked #1 by both retrievers gets `1/(60+1) + 1/(60+1) ≈ 0.033`. The `k=60` constant smooths the contribution of top vs. lower ranks. The result is scale-invariant — it doesn't matter that dense and sparse scores live in different spaces.

This is why you can't just swap in a different retriever without rethinking fusion. The fusion strategy is load-bearing.

---

## Stage 3: Classification

Once retrieval returns the top-5 most relevant precedents, they become few-shot context for the LLM. The model sees:

- The claim to evaluate
- 5 similar historical claims from the reference library, each with its label
- Explicit FDA regulatory context (21 CFR Part 202, OPDP standards)

```python
# The retrieved examples become the few-shot evidence block
examples_block = ""
for i, ex in enumerate(examples, 1):
    examples_block += f"\nExample {i}:\n  Claim: {ex['text']}\n  Verdict: {ex['label']}\n"
```

The instruction to the model is explicit: *your verdict must be grounded in the provided reference examples*. If the retriever couldn't surface relevant precedents, return `insufficient_data` rather than reasoning from general knowledge. This is the retrieval quality gate — a pre-LLM check that rejects low-confidence retrievals before they become hallucinated verdicts.

I use Groq's Llama 3.3 70B via the OpenAI-compatible client. It's free, fast, and handles the structured JSON output format consistently.

---

## Stage 4: Evaluation

I built two eval layers:

**Standard eval** — accuracy, macro F1, per-class F1, confidence calibration, latency percentiles. Run with `make eval`.

**RAGAS eval** — adapts the RAGAS framework (designed for QA systems) to a classifier. The claim becomes the question, the label + rationale becomes the answer, and the retrieved examples become the contexts. This gives you five additional signals: faithfulness, answer relevancy, context precision, context recall, and answer similarity.

Baseline results on 10 seed examples:

- Accuracy: 50%
- Macro F1: 46.3%
- Latency p50/p95: 1064ms / 5301ms

50% sounds bad. It isn't shocking — the retriever is pulling from FDA enforcement letters (mostly violation examples), which skews toward `unsupported` and `partially_supported`. The `supported` class has high recall but terrible precision because the model over-predicts it when the retriever doesn't return strong negative precedents. The fix is a more balanced knowledge base, not a better model.

This is the core lesson from running eval: **the numbers tell you what's wrong, not just how wrong you are**. 50% accuracy + `supported` precision of 28.6% tells you exactly where to look. Without eval, you're guessing.

---

## Lessons Learned

**1. Retrieval is the product.** The LLM is the easy part — swap models freely. The retrieval pipeline is what makes or breaks the system. Every design decision about embedding models, index types, fusion strategies, and quality gates compounds here.

**2. Run eval early, not at the end.** I set up the eval harness at the start of Week 2, before I had good results. That was right. Eval tells you where to invest next — you need it before you start optimizing, not after.

**3. `insufficient_data` is not a cop-out.** Adding an explicit escape hatch for out-of-domain queries made the classifier more honest and more useful. A wrong confident answer is worse than a correct "I don't know."

**4. BM25 is underrated in the era of embeddings.** I've seen RAG implementations that skip sparse retrieval entirely. The hybrid approach consistently outperforms pure dense on exact-match queries — drug names, statistical values, trial identifiers — exactly the things that matter in pharma compliance.

**5. Eval frameworks need domain adaptation.** RAGAS assumes a QA structure. Adapting it to a classifier required remapping the schema (claim → question, label+rationale → answer, retrieved examples → contexts). The metrics are still meaningful after adaptation — but you have to do the translation work yourself.

---

## What's Next

Week 3 is about retrieval tuning — rebalancing the knowledge base, adding more `supported` class examples, and measuring the F1 improvement. After that: PRISM, a multimodal research intelligence agent.

The repo is live: [github.com/warsai0222/HybridRAG](https://github.com/warsai0222/HybridRAG)

If you're building RAG systems or working on AI for regulated industries, I'd genuinely like to hear what you're seeing. Drop a comment or connect.

