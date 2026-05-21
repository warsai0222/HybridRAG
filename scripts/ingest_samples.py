"""
Ingest 50 sample pharmaceutical marketing claims to seed the database.

Each claim is labelled with an MLR (Medical-Legal-Regulatory) compliance verdict:
  supported             — claim is directly and fully backed by cited evidence
  unsupported           — claim has no credible evidence basis or overstates data
  partially_supported   — claim is directionally correct but hedges are missing
  false_balance         — claim presents minor risks and major risks as equivalent
  needs_legal_review    — claim requires IP, liability, or regulatory counsel sign-off

Run: python scripts/ingest_samples.py
"""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingestion.ingest import ingest_documents

SAMPLE_DOCUMENTS = [
    # ── supported ─────────────────────────────────────────────────────────────
    # Claims that are directly, fully backed by cited clinical evidence
    {
        "text": "In the Phase III CLARITY trial (N=1,200), drug X reduced primary endpoint events by 34% vs placebo (p<0.001, 95% CI 22–45%).",
        "label": "supported"
    },
    {
        "text": "Drug Y demonstrated a statistically significant reduction in HbA1c of 1.8 percentage points from baseline at 24 weeks in adults with type 2 diabetes (p<0.001).",
        "label": "supported"
    },
    {
        "text": "The most common adverse events in clinical trials were headache (12%), nausea (9%), and fatigue (7%), consistent with the prescribing information.",
        "label": "supported"
    },
    {
        "text": "Drug Z is indicated for the treatment of moderate-to-severe plaque psoriasis in adults who are candidates for systemic therapy, per FDA approval granted March 2023.",
        "label": "supported"
    },
    {
        "text": "Treatment with drug A resulted in complete remission in 58% of patients at Week 12, as reported in the pivotal ANCHOR study published in NEJM.",
        "label": "supported"
    },
    {
        "text": "Drug B is contraindicated in patients with severe hepatic impairment (Child-Pugh Class C) based on pharmacokinetic data showing a 4-fold increase in AUC.",
        "label": "supported"
    },
    {
        "text": "In a 52-week open-label extension study, 73% of patients who responded at Week 12 maintained response through one year of treatment with drug C.",
        "label": "supported"
    },
    {
        "text": "Patients treated with drug D showed a 41% reduction in annualized relapse rate compared to interferon beta-1a 44 mcg in the head-to-head TRANSFORM trial.",
        "label": "supported"
    },
    {
        "text": "Drug E's safety profile was evaluated in over 3,000 patients across three randomized controlled trials with a median exposure of 18 months.",
        "label": "supported"
    },
    {
        "text": "At the recommended dose of 10 mg once daily, drug F achieved peak plasma concentration within 2 hours and has a half-life of approximately 14 hours.",
        "label": "supported"
    },

    # ── unsupported ───────────────────────────────────────────────────────────
    # Claims with no credible evidence basis or that overstate available data
    {
        "text": "Drug X is the most effective treatment available for rheumatoid arthritis today.",
        "label": "unsupported"
    },
    {
        "text": "Patients who switch to drug Y see immediate and complete symptom relief within 24 hours.",
        "label": "unsupported"
    },
    {
        "text": "Drug Z has been proven to prevent the progression of Alzheimer's disease in all patient populations.",
        "label": "unsupported"
    },
    {
        "text": "Nine out of ten physicians prefer prescribing drug A over any competitor in its class.",
        "label": "unsupported"
    },
    {
        "text": "Drug B eliminates the risk of cardiovascular events in high-risk patients.",
        "label": "unsupported"
    },
    {
        "text": "Drug C cures chronic migraine permanently with a single course of treatment.",
        "label": "unsupported"
    },
    {
        "text": "Patients experience zero side effects with drug D compared to other treatments in the category.",
        "label": "unsupported"
    },
    {
        "text": "Drug E is superior to all biologics currently on the market for treatment of moderate Crohn's disease.",
        "label": "unsupported"
    },
    {
        "text": "Drug F has been shown to reverse liver fibrosis in 100% of cases within 6 months.",
        "label": "unsupported"
    },
    {
        "text": "Taking drug G once a week provides complete protection against viral reactivation.",
        "label": "unsupported"
    },

    # ── partially_supported ───────────────────────────────────────────────────
    # Directionally correct claims that are missing important hedges or context
    {
        "text": "Drug X reduces inflammation markers in patients with lupus.",
        "label": "partially_supported"
    },
    {
        "text": "Clinical studies suggest drug Y may improve cognitive function in patients with mild impairment.",
        "label": "partially_supported"
    },
    {
        "text": "Patients on drug Z reported improved quality of life scores after 12 weeks of treatment.",
        "label": "partially_supported"
    },
    {
        "text": "Drug A has shown promising results in reducing tumor size in certain cancer subtypes.",
        "label": "partially_supported"
    },
    {
        "text": "Drug B demonstrated a reduction in pain scores in patients with osteoarthritis in clinical trials.",
        "label": "partially_supported"
    },
    {
        "text": "In studies, drug C was associated with lower rates of hospitalization compared to standard of care.",
        "label": "partially_supported"
    },
    {
        "text": "Drug D helped patients achieve better blood pressure control over a 6-month period.",
        "label": "partially_supported"
    },
    {
        "text": "Researchers observed that drug E may reduce the frequency of atrial fibrillation episodes.",
        "label": "partially_supported"
    },
    {
        "text": "Drug F has been linked to improvements in bone mineral density in postmenopausal women.",
        "label": "partially_supported"
    },
    {
        "text": "Evidence suggests drug G can lower LDL cholesterol levels when used alongside dietary changes.",
        "label": "partially_supported"
    },

    # ── false_balance ─────────────────────────────────────────────────────────
    # Claims that present minor and major risks as equivalent, distorting the risk profile
    {
        "text": "Drug X may cause headache or, rarely, multi-organ failure — side effects vary by patient.",
        "label": "false_balance"
    },
    {
        "text": "Like all medications, drug Y carries some risks — from mild dry mouth to potential liver toxicity.",
        "label": "false_balance"
    },
    {
        "text": "Patients may experience minor injection-site reactions or, in some cases, anaphylaxis with drug Z.",
        "label": "false_balance"
    },
    {
        "text": "Drug A's side effects include fatigue and, on occasion, serious cardiovascular events — similar to other drugs in its class.",
        "label": "false_balance"
    },
    {
        "text": "Drug B has been associated with nausea and, in rare cases, Stevens-Johnson syndrome — typical for this therapeutic area.",
        "label": "false_balance"
    },
    {
        "text": "Among users of drug C, some report mild rash while others have experienced immune-mediated hepatitis — every patient responds differently.",
        "label": "false_balance"
    },
    {
        "text": "Drug D carries the same risk level as OTC ibuprofen, though prescribing information includes a black box warning for GI perforation.",
        "label": "false_balance"
    },
    {
        "text": "Side effects of drug E range from occasional dizziness to rare cases of progressive multifocal leukoencephalopathy — a manageable spectrum.",
        "label": "false_balance"
    },
    {
        "text": "Drug F's tolerability profile includes mild insomnia and, uncommonly, suicidal ideation — comparable to other antidepressants on the market.",
        "label": "false_balance"
    },
    {
        "text": "Patients on drug G may notice mild weight gain or, in extreme cases, pancreatitis — outcomes that depend on individual health status.",
        "label": "false_balance"
    },

    # ── needs_legal_review ────────────────────────────────────────────────────
    # Claims requiring IP, liability, off-label, or regulatory counsel before approval
    {
        "text": "Drug X is the only FDA-approved treatment for this indication, making it the gold standard of care.",
        "label": "needs_legal_review"
    },
    {
        "text": "Data from our ongoing Phase II trial suggests drug Y may also benefit pediatric patients, though this use is not yet approved.",
        "label": "needs_legal_review"
    },
    {
        "text": "Drug Z outperforms competitor brand B in every head-to-head measure, based on our internal analysis.",
        "label": "needs_legal_review"
    },
    {
        "text": "Our proprietary formulation of drug A is protected by a patent that expires in 2038, ensuring no generics during this period.",
        "label": "needs_legal_review"
    },
    {
        "text": "Drug B has been used off-label by oncologists to treat solid tumors, with encouraging results in case series.",
        "label": "needs_legal_review"
    },
    {
        "text": "We guarantee that patients on drug C will not experience the cardiovascular events seen with competitor drug D.",
        "label": "needs_legal_review"
    },
    {
        "text": "Drug E is currently under FDA priority review for a new indication in treatment-resistant depression.",
        "label": "needs_legal_review"
    },
    {
        "text": "Physicians can prescribe drug F at doses above the label recommendation for patients who do not respond to standard dosing.",
        "label": "needs_legal_review"
    },
    {
        "text": "Drug G is biosimilar to brand X and has been confirmed interchangeable by the FDA — substitution is automatic at the pharmacy.",
        "label": "needs_legal_review"
    },
    {
        "text": "Based on preclinical models, drug H may prevent cancer recurrence — a benefit we expect to confirm in our upcoming trial.",
        "label": "needs_legal_review"
    },
]


if __name__ == "__main__":
    print(f"Ingesting {len(SAMPLE_DOCUMENTS)} pharma claim compliance samples...")
    n = ingest_documents(SAMPLE_DOCUMENTS)
    print(f"Done — {n} documents ingested successfully.")
    print("\nDistribution:")
    from collections import Counter
    dist = Counter(d["label"] for d in SAMPLE_DOCUMENTS)
    for label, count in sorted(dist.items()):
        print(f"  {label}: {count}")
