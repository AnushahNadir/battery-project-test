# Paper Scope — Battery AI Co-Scientist
<!-- Updated: 2026-04-16 -->

## Problem

Battery RUL models output point estimates, but real deployment requires trustworthiness and decision reliability. A model that says "50 cycles remaining" without any confidence bound, risk signal, or audit logic is not safely actionable.

## Research Question

Can an uncertainty-aware, auditable pipeline improve the reliability of battery RUL prediction?

## Working Title

**An Uncertainty-Aware AI Co-Scientist for Reliable Battery RUL Prediction**

## One-Sentence Thesis

A reliable battery RUL system should not only predict remaining life, but also state how uncertain it is, estimate near-term failure risk, and expose explicit decision logic for accepting or flagging predictions.

## Primary Contribution

A staged AI co-scientist system combining:
1. Point prediction (XGBoost + TCN + statistical baseline)
2. Calibrated uncertainty (conformal prediction, per temperature group)
3. Near-term failure risk (survival/hazard, 20-cycle horizon)
4. Explicit audit logic (deterministic Reliability Audit Layer: PASS / WARNING / FAIL)

---

## Required in Final Paper

- Data pipeline (NASA + CALCE, cycle feature generation, RUL labelling)
- Feature generation
- XGBoost (primary predictor)
- TCN (sequence comparison model)
- Statistical baseline (weak reference)
- Conformal uncertainty (90% coverage target)
- Hazard/failure-risk estimate (20-cycle horizon)
- Reliability Audit Layer (deterministic PASS / WARNING / FAIL)
- Per-battery evaluation (battery-level split, battery-level verdict)

## Optional — Appendix Only

- Counterfactuals
- Reasoning text / degradation hypotheses
- Dashboard screenshots (one only, as demonstration artifact)

## Excluded Unless Mature

- TF-IDF RAG
- FAISS RAG
- LLM-based explanation module
- Any module that cannot be tied to a measurable reliability metric in Results

### Exact rule for RAG

If the RAG is incomplete, unstable, hard to evaluate, or not tied to a metric in the Results section, it must be removed from the paper body. Code can stay in the repo. No claims in the manuscript.

Safe compromise line for Discussion/Future Work:
> "An evidence retrieval module is under active development but was excluded from the core reliability evaluation."

---

## The Paper Story

**Not:** "we trained XGBoost and TCN and added some extra modules."

**Yes:**
1. Battery RUL prediction alone is insufficient for real decision support.
2. A trustworthy system must separate: prediction, uncertainty, risk, and audit.
3. We implement this as a staged pipeline.
4. We show that each added reliability component improves practical trustworthiness, even if pure RMSE improvement is not the only outcome.

---

## Biggest Danger

> You accidentally write a paper about a complicated codebase instead of a paper about reliable battery prediction.

Every section must answer: *"Does this help prove the system is more trustworthy?"*

If not, cut it.
