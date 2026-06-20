"""Fix remaining items in Battery_Project_Report.docx + generate F5 scatter."""
from docx import Document
from docx.shared import Pt
from docx.oxml.ns import qn
import copy, json
from pathlib import Path

doc = Document("Battery_Project_Report.docx")
P = doc.paragraphs

def set_para(para, text):
    for run in para.runs:
        run.text = ""
    if para.runs:
        para.runs[0].text = text
    else:
        para.add_run(text)

# ── Fix [41]: "Three different types" → honest two-model framing ──────────
set_para(P[41],
    "Three types of model are trained: a statistical exponential-fade baseline, "
    "XGBoost (gradient boosting), and a TCN/MLP sequence model. After training, "
    "inverse-RMSE ensemble weights are computed. Any model whose RMSE exceeds "
    "1.5x the best model RMSE is automatically excluded. On this dataset, the "
    "statistical baseline is always excluded (multi-seed mean RMSE 319.66 cycles, "
    "highly unstable), leaving a two-model ensemble: XGBoost (weight 0.557) + "
    "TCN/MLP (weight 0.443)."
)
print("[41] fixed: three-model -> two-model intro")

# ── Fix [193]: CV/test ratio still says 2.2 ──────────────────────────────
set_para(P[193],
    "CV / test RMSE ratio: Ratio > 1.5 triggers a WARNING. "
    "5-fold CV mean RMSE = 17.64 cycles (seed 42); multi-seed test RMSE = 44.74 cycles. "
    "Ratio (multi-seed test / CV) = 2.54 -> WARNING. "
    "Ratio (seed-42 test / CV) = 1.23 -> borderline PASS. "
    "The discrepancy arises because seed 42 produced an easier-than-average test split."
)
print("[193] fixed: CV ratio 2.2 -> 2.54 (multi-seed)")

# ── Rename section 14 to 15 (Key Equations Reference) ────────────────────
for p in P:
    if "14. Key Equations Reference" in p.text:
        set_para(p, "15. Key Equations Reference")
        print("Renamed: 14. Key Equations -> 15.")
        break

# ── Insert Limitations section before Key Equations ───────────────────────
# Find the last non-empty paragraph to append after
# We append at the end of doc body then XML-insert before equations ref

lim_heading = "14. Limitations and Honest Assessment"
lim_items = [
    "1. Single chemistry, single dataset. All 34 cells are 18650 lithium-ion cells cycled "
    "in controlled lab conditions (NASA PCoE, Saha & Goebel 2007). Generalization to "
    "pouch cells, LFP chemistry, or field-degraded batteries is unproven.",

    "2. Cross-battery RMSE is substantially higher than within-battery benchmarks. "
    "Multi-seed mean RMSE = 44.74 +/- 8.38 cycles vs sub-5 cycle results in "
    "Catelani (2021) and Zhao (2022) -- but those methods observe early cycles of "
    "the same target cell. Our task is strictly harder: entirely unseen cells at test time.",

    "3. cycle_index dominance creates a deployment concern. Ablation shows that "
    "removing cycle_index costs +7.10 RMSE cycles (the single largest ablation effect). "
    "The model relies heavily on knowing how far into its life a battery is. "
    "Deployment requires that the cycle count of the target battery is known. "
    "In scenarios where early-cycle data is unavailable, performance will degrade further.",

    "4. Conformal coverage guarantee is void under distribution shift. Split-conformal "
    "prediction provides marginal coverage only when calibration and test data are "
    "exchangeable. PSI analysis shows RED-level drift on 5/10 features (PSI 1.01-2.57). "
    "Cold-group empirical coverage (74.5%) falls below the 80% target as a direct "
    "consequence. The guarantee is a bound, not a promise, under the exchangeability assumption.",

    "5. Survival analysis is severely underpowered. Only 5 EOL events were observed in "
    "the 6 test batteries (event rate 0.78%). No quantitative survival metrics (C-index, "
    "Brier score) are reported. The survival component is an exploratory system feature "
    "and cannot be validated with this event count.",

    "6. Anomaly detection has no ground truth. The 32 flagged cycles are identified by "
    "unsupervised residual thresholding. Without ground-truth anomaly labels, precision "
    "and recall cannot be computed. Flagged cycles may reflect genuine anomalies or "
    "simply batteries with high intrinsic prediction difficulty (e.g., B0052 cold group).",

    "7. RAG explanations are not human-evaluated. The automated rubric (hedging, "
    "mechanistic reasoning, source grounding, overconfidence) passes 6/6 pre-computed "
    "explanations, but no human rater study was conducted. The quality of generated "
    "explanations for novel battery conditions is unknown.",
]

# Append new paragraphs to doc
new_heading_p = doc.add_paragraph(lim_heading)
try:
    new_heading_p.style = doc.styles["Heading 1"]
except:
    pass

new_paras = [new_heading_p]
for item in lim_items:
    np_ = doc.add_paragraph(item)
    new_paras.append(np_)

# XML-insert: move new paragraphs to be before the Key Equations heading
body = doc.element.body
all_body_paras = body.findall(qn("w:p"))

# Find the Key Equations paragraph in body XML
target_p_el = None
for el in all_body_paras:
    texts = "".join(
        r.text or ""
        for r in el.iter(qn("w:t"))
    )
    if "15. Key Equations Reference" in texts or "Key Equations Reference" in texts:
        target_p_el = el
        break

if target_p_el is not None:
    for np_ in reversed(new_paras):
        target_p_el.addprevious(np_._p)
    print(f"Inserted Limitations section ({len(new_paras)} paragraphs) before Key Equations")
else:
    print("Key Equations heading not found in body XML; Limitations appended at end")

doc.save("Battery_Project_Report.docx")
print("Battery_Project_Report.docx saved.")
