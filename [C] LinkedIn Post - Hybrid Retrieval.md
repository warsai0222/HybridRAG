Most RAG systems only use one type of retrieval.

That's leaving a lot on the table.

Here's why I use both dense embeddings AND BM25 in my RAG pipeline — and why averaging their scores was my first mistake:

---

**Dense retrieval** (BGE embeddings + cosine similarity) understands meaning.

"Reduces cardiac events" and "lowers heart attack risk" score high together — even though they share zero words.

Great for semantic similarity. But it struggles when the query is specific: exact drug names, trial identifiers, statistical values like "HR 0.74, p<0.001".

---

**BM25** (sparse retrieval) nails keywords.

If the string is in the database, it finds it. No ambiguity.

But it's blind to semantics. "Reduces cardiac events" and "lowers heart attack risk"? Zero overlap. Different words = different documents.

---

**The fix: run both, then fuse the results.**

But here's where I went wrong at first — I tried to average the scores.

Dense scores are cosine similarities: values between 0 and 1.
BM25 scores are term-frequency weighted: arbitrarily large.

Averaging them is mathematically meaningless. You're combining two completely different scales.

---

**The right approach: Reciprocal Rank Fusion (RRF)**

Instead of combining scores, you combine *ranks*:

RRF(d) = Σ [ 1 / (k + rank(d)) ]

A document ranked #1 by both retrievers scores ~0.033. One ranked #10 by both scores ~0.029. Scale-invariant, robust, and empirically outperforms weighted score fusion on standard benchmarks.

The k=60 constant comes from the original 2009 paper. It smooths the contribution of top ranks — without it, rank-1 documents dominate everything.

---

I built this into HybridRAG — a pharma MLR compliance classifier that retrieves similar FDA enforcement precedents to classify marketing claims.

The retrieval pipeline runs both, fuses with RRF, and applies a quality gate before the LLM ever sees the results. If the top retrieved result scores below a threshold, the system returns "insufficient data" rather than letting the LLM hallucinate a verdict.

Repo is live if you want to dig into the implementation:
github.com/warsai0222/HybridRAG

What retrieval strategies are you using in your RAG systems?

