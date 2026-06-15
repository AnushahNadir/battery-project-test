"""Generate Streamlit guide document in English and Korean."""
from pathlib import Path
from docx import Document
from docx.shared import Pt, RGBColor, Cm, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

OUT = Path(__file__).resolve().parents[1] / "docs_output"
OUT.mkdir(exist_ok=True)

DARK_BLUE = RGBColor(0x1A, 0x3A, 0x5C)
MID_BLUE  = RGBColor(0x2E, 0x6D, 0xA4)
ORANGE    = RGBColor(0xC0, 0x50, 0x00)
GREEN     = RGBColor(0x1A, 0x6B, 0x3C)
GRAY      = RGBColor(0x44, 0x44, 0x44)

def add_rule(doc):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(1)
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement('w:pBdr')
    bot = OxmlElement('w:bottom')
    bot.set(qn('w:val'), 'single')
    bot.set(qn('w:sz'), '6')
    bot.set(qn('w:space'), '1')
    bot.set(qn('w:color'), '2E6DA4')
    pBdr.append(bot)
    pPr.append(pBdr)

def add_h1(doc, text):
    h = doc.add_heading(text, level=1)
    h.paragraph_format.space_before = Pt(20)
    h.paragraph_format.space_after = Pt(4)
    for run in h.runs:
        run.font.color.rgb = DARK_BLUE
    add_rule(doc)

def add_h2(doc, text):
    h = doc.add_heading(text, level=2)
    h.paragraph_format.space_before = Pt(12)
    h.paragraph_format.space_after = Pt(4)
    for run in h.runs:
        run.font.color.rgb = MID_BLUE

def add_h3(doc, text):
    h = doc.add_heading(text, level=3)
    h.paragraph_format.space_before = Pt(8)
    h.paragraph_format.space_after = Pt(3)
    for run in h.runs:
        run.font.color.rgb = ORANGE

def add_body(doc, text):
    p = doc.add_paragraph(text)
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(3)
    for run in p.runs:
        run.font.size = Pt(11)
        run.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)

def add_bullet(doc, text, level=0):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.left_indent = Inches(0.25 * (level + 1))
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(text)
    run.font.size = Pt(11)

def add_code(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.left_indent = Inches(0.4)
    r = p.add_run(text)
    r.font.name = 'Courier New'
    r.font.size = Pt(10)
    r.font.color.rgb = RGBColor(0x33, 0x33, 0x99)

def add_note(doc, text, note_type="note"):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.left_indent = Inches(0.3)
    if note_type == "warning":
        label, color = "WARNING: ", ORANGE
    elif note_type == "tip":
        label, color = "TIP: ", GREEN
    else:
        label, color = "NOTE: ", MID_BLUE
    rl = p.add_run(label)
    rl.font.bold = True
    rl.font.color.rgb = color
    rl.font.size = Pt(11)
    rt = p.add_run(text)
    rt.font.size = Pt(11)
    rt.font.color.rgb = GRAY
    rt.font.italic = True


def build_en(doc):
    # Cover
    tp = doc.add_paragraph()
    tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    tr = tp.add_run("Battery AI Co-Scientist")
    tr.font.size = Pt(30)
    tr.font.bold = True
    tr.font.color.rgb = DARK_BLUE

    sp = doc.add_paragraph()
    sp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sr = sp.add_run("Streamlit Dashboard — Complete User Guide & Technical Reference")
    sr.font.size = Pt(15)
    sr.font.color.rgb = MID_BLUE

    doc.add_paragraph()
    mp = doc.add_paragraph()
    mp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    mr = mp.add_run("dashboard/app.py  |  Port 8501  |  Requires: data/processed/cycle_features_with_rul.csv")
    mr.font.size = Pt(11)
    mr.font.color.rgb = GRAY
    doc.add_page_break()

    # ── OVERVIEW ─────────────────────────────────────────────────────────────
    add_h1(doc, "1. Overview")
    add_body(doc,
        "The Battery AI Co-Scientist dashboard is a single-page interactive web application built "
        "with Streamlit. It provides a unified interface to every component of the battery RUL "
        "prediction pipeline: model predictions, uncertainty quantification, risk assessment, "
        "anomaly detection, feature importance, counterfactual analysis, supervisor auditing, "
        "and AI-generated natural language explanations.")
    add_body(doc,
        "The dashboard is designed to serve two audiences: battery engineers who need actionable "
        "health monitoring information, and researchers who want to inspect model behavior in detail.")

    add_h2(doc, "1.1 How to Launch")
    add_body(doc, "Step 1: Ensure the processed data file exists:")
    add_code(doc, "python -m src.modeling.run_full_pipeline")
    add_body(doc, "Step 2: Start the dashboard:")
    add_code(doc, "python -m streamlit run dashboard/app.py")
    add_body(doc, "Step 3: Open in your browser:")
    add_code(doc, "http://localhost:8501")
    add_note(doc,
        "If port 8501 is already in use, specify a different port: "
        "python -m streamlit run dashboard/app.py --server.port 8502", "tip")

    add_h2(doc, "1.2 Technical Requirements")
    add_bullet(doc, "Python 3.9+ with all dependencies from requirements.txt installed")
    add_bullet(doc, "data/processed/cycle_features_with_rul.csv must exist (generated by run_full_pipeline)")
    add_bullet(doc, "trained_models/ directory must contain model artifacts")
    add_bullet(doc, "configs/pipeline.yaml must exist (contains all thresholds)")
    add_bullet(doc, "For AI Explanation: NVIDIA GPU with CUDA (RTX 3060 or better recommended)")
    add_bullet(doc, "Internet connection on first run (to download Gemma model weights from HuggingFace)")

    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    add_h1(doc, "2. Left Sidebar — Battery Selector")
    add_body(doc,
        "The left sidebar contains the battery selection dropdown. All battery IDs found in "
        "data/processed/cycle_features_with_rul.csv are listed with their temperature group label "
        "(e.g., 'B0005 (cold)', 'B0047 (room)', 'B0025 (hot)').")
    add_body(doc,
        "Selecting a battery immediately loads all of its cycle data and triggers all downstream "
        "computations. The dashboard re-renders automatically.")
    add_note(doc,
        "If the sidebar shows 'No battery data found', the data file is missing. "
        "Run: python -m src.modeling.run_full_pipeline", "warning")

    add_h2(doc, "2.1 Battery Label Format")
    add_bullet(doc, "Format: [Battery ID] ([temperature group])")
    add_bullet(doc, "Temperature groups: cold (4 deg C), room (24 deg C), hot (44 deg C)")
    add_bullet(doc, "Example: 'B0047 (room)' means battery B0047 operated at room temperature")
    add_note(doc,
        "Batteries B0049-B0056 are flagged as anomalous in pipeline.yaml. They are still "
        "selectable in the dashboard and will show their anomaly flags prominently.", "note")

    # ── SECTION 1: RUL CHART ─────────────────────────────────────────────────
    add_h1(doc, "3. Section 1 — RUL Prediction Chart")
    add_body(doc,
        "The main chart is an interactive Plotly figure showing the full degradation history "
        "of the selected battery from cycle 1 to the most recent observed cycle.")

    add_h2(doc, "3.1 What Is Shown")
    add_bullet(doc, "Blue solid line (left y-axis): Actual measured discharge capacity in Ah over all observed cycles")
    add_bullet(doc, "Orange solid line (right y-axis): Model's predicted RUL at each cycle (in cycles remaining)")
    add_bullet(doc, "Shaded orange/yellow band: 90% conformal prediction interval around the RUL prediction")
    add_bullet(doc, "Red dashed horizontal line: EoL threshold at 1.6 Ah (80% of nominal 2.0 Ah)")
    add_bullet(doc, "Vertical dashed line (if present): Current/last observed cycle marker")

    add_h2(doc, "3.2 How to Interact")
    add_bullet(doc, "Hover over any point: tooltip shows exact cycle number, capacity, and RUL value")
    add_bullet(doc, "Click and drag: zoom into a specific cycle range")
    add_bullet(doc, "Double-click: reset zoom to full view")
    add_bullet(doc, "Camera icon (top right): download as PNG image")
    add_bullet(doc, "Legend items: click to show/hide individual traces")

    add_h2(doc, "3.3 Interpretation Guide")
    add_bullet(doc, "Steeper orange slope = faster predicted RUL decline = approaching EoL faster")
    add_bullet(doc, "Narrow shaded band = high model confidence (in-distribution battery)")
    add_bullet(doc, "Wide shaded band = lower model confidence (unusual battery or temperature)")
    add_bullet(doc, "If the blue line crosses the red dashed line = battery has reached EoL")
    add_note(doc,
        "The two y-axes (capacity left, RUL right) have different scales. The capacity axis "
        "shows absolute Ah; the RUL axis shows cycles remaining.", "note")

    # ── SECTION 2: UNCERTAINTY ───────────────────────────────────────────────
    add_h1(doc, "4. Section 2 — Uncertainty Panel")
    add_body(doc,
        "Below the main chart, four metric cards and a bar chart display the uncertainty "
        "decomposition for the battery's most recently observed cycle.")

    add_h2(doc, "4.1 Metric Cards")
    add_bullet(doc, "Point Estimate: The ensemble's weighted mean RUL prediction for the current cycle")
    add_bullet(doc, "Lower Bound (5th pct): Pessimistic end of the 90% conformal interval — worst-case RUL")
    add_bullet(doc, "Upper Bound (95th pct): Optimistic end of the 90% conformal interval — best-case RUL")
    add_bullet(doc, "CI Width: Upper bound minus lower bound, in cycles. Narrower = more confident prediction")

    add_h2(doc, "4.2 Uncertainty Decomposition Bar Chart")
    add_body(doc,
        "A horizontal bar chart breaks total uncertainty into two components:")
    add_bullet(doc, "Aleatoric uncertainty: Inherent randomness in the battery degradation process and measurement noise. Cannot be reduced with more data.")
    add_bullet(doc, "Epistemic uncertainty: Model uncertainty due to limited training data or out-of-distribution inputs. Can be reduced with more calibration data.")
    add_body(doc,
        "For in-distribution batteries (room temperature, smooth degradation), epistemic uncertainty "
        "is small. For out-of-distribution batteries (cold temperature, first-ever battery of a new "
        "type), epistemic uncertainty dominates and the CI is wide.")

    add_h2(doc, "4.3 Interpreting CI Width")
    add_bullet(doc, "< 30 cycles: High confidence. Battery is well within training distribution.")
    add_bullet(doc, "30-60 cycles: Moderate confidence. Normal for most room-temperature batteries.")
    add_bullet(doc, "60-100 cycles: Lower confidence. Battery may be out-of-distribution or in a non-standard temperature group.")
    add_bullet(doc, "> 100 cycles: Low confidence. LOBO safety factor may be inflating the interval. Collect more calibration data for this battery type.")

    # ── SECTION 3: RISK ──────────────────────────────────────────────────────
    add_h1(doc, "5. Section 3 — Risk and Survival Analysis")
    add_body(doc,
        "This section shows the output of the discrete-time hazard model: the probability that "
        "the battery fails within the next 20 cycles, and the full survival curve S(t).")

    add_h2(doc, "5.1 Failure Probability Gauge")
    add_body(doc,
        "A semicircular gauge chart shows P(failure within 20 cycles) as a percentage. "
        "The gauge is color-coded:")
    add_bullet(doc, "GREEN zone (0-30%): LOW risk. Battery has a less than 30% chance of failing in the next 20 cycles.")
    add_bullet(doc, "AMBER zone (30-70%): MEDIUM risk. Monitor closely. Consider scheduling inspection.")
    add_bullet(doc, "RED zone (70-100%): HIGH risk. High probability of failure within 20 cycles. Prioritize for review.")
    add_note(doc,
        "Thresholds are configurable in pipeline.yaml (risk.high_threshold = 0.70, "
        "risk.medium_threshold = 0.30).", "tip")

    add_h2(doc, "5.2 Risk Label")
    add_body(doc,
        "A large colored badge below the gauge shows LOW (green), MEDIUM (amber), or HIGH (red). "
        "This is the operational risk classification for the current cycle.")

    add_h2(doc, "5.3 Survival Curve S(t)")
    add_body(doc,
        "A line chart shows S(t) = P(battery survives beyond t cycles from now) for t = 1 to 50. "
        "S(0) = 1.0 (certain survival at current cycle) and S(t) decreases monotonically. "
        "The value at t=20 corresponds to the failure probability shown in the gauge: "
        "P(fail within 20 cycles) = 1 - S(20).")
    add_bullet(doc, "Steep decline in S(t): rapid deterioration expected over the next few cycles")
    add_bullet(doc, "Flat S(t): battery is expected to remain healthy for the near-term horizon")

    # ── SECTION 4: ANOMALY ───────────────────────────────────────────────────
    add_h1(doc, "6. Section 4 — Anomaly Detection")
    add_body(doc,
        "The anomaly detection section shows whether the battery's current capacity trajectory "
        "deviates significantly from the expected degradation behavior modeled during training.")

    add_h2(doc, "6.1 Anomaly Score Card")
    add_body(doc,
        "A large metric card shows the current anomaly residual score (0-100) with a color badge:")
    add_bullet(doc, "GREEN badge 'NORMAL': Score is below the 95th percentile training residual threshold. Battery degradation is consistent with the expected trajectory.")
    add_bullet(doc, "RED badge 'ANOMALY': Score exceeds the threshold. The battery is deviating significantly from expected behavior.")
    add_body(doc,
        "The anomaly score is computed as: |actual_capacity - expected_capacity| normalized to a "
        "0-100 scale based on training data residual distribution.")

    add_h2(doc, "6.2 Anomaly Score Time Series")
    add_body(doc,
        "A line chart shows the anomaly score at every observed cycle. A horizontal red dashed "
        "line marks the 95th percentile threshold. Cycles where the score exceeds the threshold "
        "are highlighted.")
    add_bullet(doc, "Consistently high scores throughout: battery has persistent anomalous behavior (check if it is one of B0049-B0056)")
    add_bullet(doc, "Score spikes at specific cycles: transient events (measurement errors, charge protocol changes)")
    add_bullet(doc, "Score gradually increasing: early warning of accelerating degradation")

    add_h2(doc, "6.3 Important Notes on Anomaly Detection")
    add_note(doc,
        "An ANOMALY flag does NOT mean the battery is about to fail. It means the degradation "
        "pattern is unusual relative to the training distribution. Combine with the risk score "
        "for a full picture.", "warning")
    add_note(doc,
        "Batteries B0049-B0056 will frequently show ANOMALY flags because their degradation "
        "patterns are inherently irregular. This is expected behavior.", "note")

    # ── SECTION 5: SHAP ──────────────────────────────────────────────────────
    add_h1(doc, "7. Section 5 — Feature Importance (SHAP)")
    add_body(doc,
        "The feature importance section explains which input features most influenced the "
        "RUL prediction for the currently selected cycle of the selected battery.")

    add_h2(doc, "7.1 SHAP Waterfall Chart")
    add_body(doc,
        "The waterfall chart shows the 10 most influential features for the current prediction. "
        "Each bar represents one feature's SHAP value — its contribution to the prediction "
        "relative to the average prediction across all training samples.")
    add_bullet(doc, "Bar pointing RIGHT (positive SHAP): This feature's value pushed the RUL prediction HIGHER than average")
    add_bullet(doc, "Bar pointing LEFT (negative SHAP): This feature's value pushed the RUL prediction LOWER than average")
    add_bullet(doc, "Bar length: magnitude of the contribution in cycles")
    add_bullet(doc, "E[f(x)] at the bottom: the baseline (average) prediction before feature adjustments")
    add_bullet(doc, "f(x) at the top: the final prediction after all feature contributions")

    add_h2(doc, "7.2 Typical Feature Patterns")
    add_bullet(doc, "capacity_ahr: Usually the largest contributor. High capacity -> large positive SHAP (more cycles remaining).")
    add_bullet(doc, "capacity_fade_rate: Usually negative SHAP for fast-degrading batteries. Very negative value means accelerating fade.")
    add_bullet(doc, "rolling_mean_5: Smoothed trend. Consistently declining -> negative SHAP.")
    add_bullet(doc, "cycle_number: Later cycles -> fewer remaining -> negative contribution to RUL.")
    add_bullet(doc, "temperature_c: Cold temperatures increase uncertainty (wider CI) and may decrease RUL prediction due to lithium plating risk.")

    add_h2(doc, "7.3 Counterfactual Analysis Panel")
    add_body(doc,
        "A separate panel allows sensitivity analysis on any feature:")
    add_numbered_step(doc, "Select a feature from the dropdown (e.g., temperature_c)")
    add_numbered_step(doc, "Enter a hypothetical value (e.g., 24.0 to simulate room temperature instead of 4.0)")
    add_numbered_step(doc, "Click 'Run Counterfactual'")
    add_numbered_step(doc, "The panel shows: Original RUL prediction vs. Counterfactual RUL prediction and the difference in cycles")
    add_note(doc,
        "Counterfactual results are model predictions, not physical simulations. They show how "
        "sensitive the model is to a feature, not the true physical effect of changing that condition.", "warning")


def add_numbered_step(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.4)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(text)
    r.font.size = Pt(11)

    # ── SECTION 6: SUPERVISOR ────────────────────────────────────────────────

add_h1_fn = add_h1
def build_en_part2(doc):
    add_h1(doc, "8. Section 6 — Supervisor Audit")
    add_body(doc,
        "The supervisor audit is an automated quality gate that evaluates whether the current "
        "prediction batch meets the reliability standards required for operational use.")

    add_h2(doc, "8.1 Verdict Categories")
    add_bullet(doc, "PASS (green): All three quality criteria are met. Predictions are trustworthy for operational use.")
    add_bullet(doc, "CONDITIONAL PASS (amber): One or more criteria are marginal (between pass and auto-approve thresholds). Manual review recommended before acting on predictions.")
    add_bullet(doc, "FAIL (red): One or more criteria are clearly failed. Do not rely on this batch of predictions. Investigate root cause.")

    add_h2(doc, "8.2 Quality Criteria")
    add_body(doc, "The audit checks three criteria:")

    add_h3(doc, "Criterion 1: Model Confidence Score")
    add_body(doc,
        "The model's internal confidence in the current prediction, computed from the ratio of "
        "epistemic to total uncertainty. Thresholds:")
    add_bullet(doc, "Score >= 0.85: Auto-approve (high confidence, no review needed)")
    add_bullet(doc, "Score 0.70-0.85: Pass (acceptable confidence)")
    add_bullet(doc, "Score < 0.70: Fail (insufficient confidence)")
    add_code(doc, "supervisor.low_confidence_accept_threshold = 0.70  # in pipeline.yaml")
    add_code(doc, "gating.review_threshold = 0.85                     # in pipeline.yaml")

    add_h3(doc, "Criterion 2: RMSE on Temperature Group")
    add_body(doc,
        "The RMSE computed on test-set batteries of the same temperature group as the selected "
        "battery. This reflects how well the model generalizes to batteries similar to the one "
        "being evaluated. Threshold:")
    add_bullet(doc, "RMSE <= 100 cycles: Pass")
    add_bullet(doc, "RMSE > 100 cycles: Fail")
    add_code(doc, "supervisor.rmse_pass_threshold = 100.0  # in pipeline.yaml")

    add_h3(doc, "Criterion 3: CV-RMSE Ratio")
    add_body(doc,
        "The coefficient of variation of RMSE — RMSE divided by mean RUL. This normalizes "
        "RMSE for batteries at different life stages. Threshold:")
    add_bullet(doc, "CV-RMSE <= 1.5: Pass (relative error is acceptable)")
    add_bullet(doc, "CV-RMSE > 1.5: Warning/Fail (relative error is too high)")
    add_code(doc, "supervisor.cv_rmse_ratio_warn_threshold = 1.5  # in pipeline.yaml")

    add_h2(doc, "8.3 Reasoning Text")
    add_body(doc,
        "Below the verdict card, a text box provides a plain-language explanation of the verdict. "
        "For a PASS, it confirms which criteria were met and at what values. For a CONDITIONAL PASS "
        "or FAIL, it identifies which criterion failed and what the observed value was, helping "
        "engineers diagnose whether to retrain, recalibrate, or investigate data quality.")

    # ── SECTION 7: RAG ───────────────────────────────────────────────────────
    add_h1(doc, "9. Section 7 — AI Explanation (RAG Engine)")
    add_body(doc,
        "The AI Explanation section allows users to ask natural language questions about any "
        "aspect of the selected battery's current health, risk, or model predictions. Answers "
        "are generated by the local RAG engine and grounded in both domain literature and the "
        "battery's specific pipeline data.")

    add_h2(doc, "9.1 How to Use")
    add_body(doc, "Step 1: Type a question in the text input box. Examples:")
    add_bullet(doc, "'Why is the risk HIGH for this battery?'")
    add_bullet(doc, "'What does the wide confidence interval indicate?'")
    add_bullet(doc, "'What is causing the anomaly flag at this cycle?'")
    add_bullet(doc, "'How does this battery compare to a healthy room-temperature battery?'")
    add_bullet(doc, "'What does a failure probability of 0.73 mean in practice?'")
    add_body(doc, "Step 2: Click 'Generate AI Summary'")
    add_body(doc, "Step 3: Wait 30-90 seconds for generation (first call may take up to 3 minutes to load the model)")
    add_body(doc, "Step 4: Read the answer. Sources are cited inline as [Source 1], [Source 2], etc.")

    add_h2(doc, "9.2 What the RAG Engine Does Internally")
    add_body(doc, "The following sequence runs when you click 'Generate AI Summary':")
    add_bullet(doc, "1. Your query is encoded into a 384-dimensional embedding vector using SentenceTransformer (all-MiniLM-L6-v2)")
    add_bullet(doc, "2. ChromaDB performs a cosine similarity search over all stored document chunks in the knowledge base")
    add_bullet(doc, "3. The top-3 most relevant chunks are retrieved (from data/knowledge_base/*.txt and any indexed PDFs)")
    add_bullet(doc, "4. A structured prompt is assembled containing: system instructions with 9 grounding rules + retrieved chunks labeled [Source 1], [Source 2], [Source 3] + battery pipeline data (RUL, CI, risk score, anomaly score, SHAP top features, supervisor verdict)")
    add_bullet(doc, "5. The prompt is sent to Gemma 3-4B-IT (running locally on GPU, 4-bit NF4 quantized)")
    add_bullet(doc, "6. Gemma generates a 3-5 sentence answer using greedy decoding (max 180 new tokens)")
    add_bullet(doc, "7. The answer is displayed with source file names listed below")

    add_h2(doc, "9.3 The 9 Grounding Rules")
    add_body(doc,
        "The prompt enforces 9 rules that constrain the LLM to produce safe, grounded answers:")
    add_bullet(doc, "Rule 1: Use pipeline data values for this battery's specific numbers. Never substitute numbers from reference documents.")
    add_bullet(doc, "Rule 2: Use reference documents only for physical interpretation, mechanisms, and context.")
    add_bullet(doc, "Rule 3: Cite inline as [Source N] for mechanistic claims supported by retrieved documents.")
    add_bullet(doc, "Rule 4: If retrieved evidence is insufficient, say 'The available evidence does not support a conclusion.' Do not fabricate.")
    add_bullet(doc, "Rule 5: Do NOT claim specific causal mechanisms (lithium plating, SEI cracking) unless a retrieved source directly supports that diagnosis.")
    add_bullet(doc, "Rule 6: Distinguish model signals (predicted RUL, risk score) from physical facts.")
    add_bullet(doc, "Rule 7: If pipeline data shows contradictory signals, flag the contradiction explicitly.")
    add_bullet(doc, "Rule 8: Do NOT recommend prescriptive actions (replace, discard) unless a retrieved source supports that action.")
    add_bullet(doc, "Rule 9: Answer in 3-5 sentences. Be concise and technical.")

    add_h2(doc, "9.4 Performance and Hardware Requirements")
    add_kv_doc(doc, "First call time", "2-4 minutes (loading Gemma model weights into VRAM)")
    add_kv_doc(doc, "Subsequent calls", "30-90 seconds (model already in VRAM)")
    add_kv_doc(doc, "GPU requirement", "NVIDIA GPU with CUDA, minimum 8 GB VRAM (RTX 3060 = 12 GB, sufficient)")
    add_kv_doc(doc, "CPU fallback", "Not supported — will raise RuntimeError if CUDA unavailable")
    add_kv_doc(doc, "Model", "google/gemma-3-4b-it, 4-bit NF4 quantized via bitsandbytes")
    add_kv_doc(doc, "Local weights path", "models/gemma-3-4b-it/ (auto-downloads from HuggingFace if absent)")
    add_note(doc,
        "If generation fails with 'Repo id must use alphanumeric chars', the local model path "
        "does not exist. The system will automatically fall back to downloading from HuggingFace. "
        "Ensure you have an internet connection on first run.", "warning")

    # ── SECTION 8: TECHNICAL ARCH ────────────────────────────────────────────
    add_h1(doc, "10. Technical Architecture")

    add_h2(doc, "10.1 Framework and Libraries")
    add_kv_doc(doc, "Frontend", "Streamlit 1.x")
    add_kv_doc(doc, "Charts", "Plotly Express + Plotly Graph Objects")
    add_kv_doc(doc, "ML models", "scikit-learn, XGBoost, PyTorch (TCN)")
    add_kv_doc(doc, "Uncertainty", "scipy, custom conformal calibration")
    add_kv_doc(doc, "RAG retrieval", "sentence-transformers, chromadb")
    add_kv_doc(doc, "RAG generation", "transformers, bitsandbytes (Gemma 3-4B-IT)")
    add_kv_doc(doc, "Config", "PyYAML + Pydantic")

    add_h2(doc, "10.2 Caching Strategy")
    add_body(doc, "The dashboard uses two Streamlit caching decorators:")
    add_bullet(doc, "@st.cache_data: Applied to data loading functions. Caches the processed CSV to avoid re-reading on every interaction. TTL: Streamlit default (5 minutes).")
    add_bullet(doc, "@st.cache_resource: Applied to model loading functions. Caches model objects (XGBoost, TCN, RAG) for the full session lifetime. Models are loaded once and reused across all battery selections.")

    add_h2(doc, "10.3 BASE Path Resolution")
    add_body(doc,
        "The dashboard resolves all file paths relative to the app.py file location using:")
    add_code(doc, "BASE = Path(__file__).resolve().parent.parent")
    add_body(doc,
        "This ensures the dashboard works correctly regardless of what directory Streamlit is "
        "launched from. All data files, model files, and config files are referenced as "
        "BASE / 'data/...' etc.")

    add_h2(doc, "10.4 Config Integration")
    add_body(doc,
        "All thresholds and parameters are read from configs/pipeline.yaml at startup. "
        "The dashboard does not hardcode any magic numbers. Key values loaded:")
    add_bullet(doc, "risk.high_threshold (0.70), risk.medium_threshold (0.30): risk label boundaries")
    add_bullet(doc, "anomaly.capacity_floor_factor (0.50): minimum capacity for anomaly detection")
    add_bullet(doc, "conformal.coverage (0.90): target coverage for prediction intervals")
    add_bullet(doc, "gating.low_threshold (0.60), gating.review_threshold (0.85): supervisor thresholds")

    # ── TROUBLESHOOTING ───────────────────────────────────────────────────────
    add_h1(doc, "11. Troubleshooting")

    add_h2(doc, "11.1 Common Errors and Fixes")

    add_h3(doc, "Error: 'No battery data found'")
    add_body(doc, "Cause: data/processed/cycle_features_with_rul.csv does not exist.")
    add_body(doc, "Fix:")
    add_code(doc, "python -m src.modeling.run_full_pipeline")
    add_body(doc, "This runs the full pipeline and generates the required file. Takes 5-15 minutes.")

    add_h3(doc, "Error: 'Explanation failed: Repo id must use alphanumeric chars'")
    add_body(doc, "Cause: The local Gemma model weights are not in models/gemma-3-4b-it/")
    add_body(doc, "Fix: The system automatically falls back to HuggingFace download. Ensure internet connection. On first run, the download may take 10-20 minutes (approximately 2.5 GB).")
    add_note(doc, "After download, weights are cached in the HuggingFace cache directory. Subsequent runs load locally.", "tip")

    add_h3(doc, "Error: 'Address already in use' / Port 8501 occupied")
    add_body(doc, "Cause: Another Streamlit or Python process is using port 8501.")
    add_body(doc, "Fix option 1 — Use a different port:")
    add_code(doc, "python -m streamlit run dashboard/app.py --server.port 8502")
    add_body(doc, "Fix option 2 — Kill all Python processes (Windows PowerShell):")
    add_code(doc, "Get-Process python* | Stop-Process -Force")

    add_h3(doc, "Error: 'CUDA not available'")
    add_body(doc, "Cause: PyTorch was installed without CUDA support, or no NVIDIA GPU is present.")
    add_body(doc, "Fix: Reinstall PyTorch with CUDA:")
    add_code(doc, "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
    add_note(doc, "The RAG AI Explanation requires CUDA. All other dashboard sections work on CPU.", "note")

    add_h3(doc, "Dashboard loads but shows all zeros or empty charts")
    add_body(doc, "Cause: Stale Streamlit cache from a previous failed run.")
    add_body(doc, "Fix: Clear the cache from the Streamlit menu (top right '...' -> Clear cache) and refresh the page.")

    add_h2(doc, "11.2 Performance Tips")
    add_bullet(doc, "The dashboard is fastest when all models are already in @st.cache_resource — avoid restarting Streamlit unnecessarily")
    add_bullet(doc, "For RAG generation, the RTX 3060 (12 GB VRAM) handles Gemma 3-4B-IT in 4-bit mode with 30-90 second generation time")
    add_bullet(doc, "If you need faster responses, reduce max_new_tokens in local_rag.py (currently 180)")
    add_bullet(doc, "The knowledge base is indexed on first run and cached in data/vector_db/. Adding new PDFs to data/papers/ requires calling reindex() to rebuild the index")


def add_kv_doc(doc, key, value):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.left_indent = Inches(0.2)
    rk = p.add_run(key + ": ")
    rk.bold = True
    rk.font.color.rgb = DARK_BLUE
    rv = p.add_run(value)
    rv.font.color.rgb = GRAY


def build_ko(doc):
    tp = doc.add_paragraph()
    tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    tr = tp.add_run("Battery AI Co-Scientist")
    tr.font.size = Pt(30)
    tr.font.bold = True
    tr.font.color.rgb = DARK_BLUE

    sp = doc.add_paragraph()
    sp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sr = sp.add_run("Streamlit 대시보드 - 완전한 사용자 가이드 및 기술 참조")
    sr.font.size = Pt(16)
    sr.font.color.rgb = MID_BLUE
    doc.add_page_break()

    add_h1(doc, "1. 개요")
    add_body(doc,
        "Battery AI Co-Scientist 대시보드는 Streamlit으로 구축된 단일 페이지 인터랙티브 웹 애플리케이션입니다. "
        "배터리 RUL 예측 파이프라인의 모든 구성 요소에 대한 통합 인터페이스를 제공합니다: 모델 예측, "
        "불확실성 정량화, 위험 평가, 이상 탐지, 특성 중요도, 반사실적 분석, 감독자 감사, "
        "AI 생성 자연어 설명.")

    add_h2(doc, "1.1 실행 방법")
    add_body(doc, "1단계: 처리된 데이터 파일이 존재하는지 확인:")
    add_code(doc, "python -m src.modeling.run_full_pipeline")
    add_body(doc, "2단계: 대시보드 시작:")
    add_code(doc, "python -m streamlit run dashboard/app.py")
    add_body(doc, "3단계: 브라우저에서 열기:")
    add_code(doc, "http://localhost:8501")
    add_note(doc, "포트 8501이 이미 사용 중이라면: python -m streamlit run dashboard/app.py --server.port 8502", "tip")

    add_h2(doc, "1.2 기술 요구사항")
    add_bullet(doc, "requirements.txt의 모든 의존성이 설치된 Python 3.9+")
    add_bullet(doc, "data/processed/cycle_features_with_rul.csv 존재 (run_full_pipeline으로 생성)")
    add_bullet(doc, "trained_models/ 디렉토리에 모델 아티팩트 존재")
    add_bullet(doc, "AI 설명을 위한 NVIDIA GPU (RTX 3060 이상 권장)")
    add_bullet(doc, "첫 실행 시 인터넷 연결 (HuggingFace에서 Gemma 모델 가중치 다운로드)")

    add_h1(doc, "2. 왼쪽 사이드바 - 배터리 선택기")
    add_body(doc,
        "왼쪽 사이드바에는 배터리 선택 드롭다운이 있습니다. "
        "data/processed/cycle_features_with_rul.csv에 있는 모든 배터리 ID가 온도 그룹 레이블과 함께 나열됩니다 "
        "(예: 'B0005 (cold)', 'B0047 (room)', 'B0025 (hot)').")
    add_note(doc,
        "사이드바에 '배터리 데이터를 찾을 수 없음'이 표시되면 데이터 파일이 없습니다. "
        "python -m src.modeling.run_full_pipeline 실행", "warning")

    add_h1(doc, "3. 섹션 1 - RUL 예측 차트")
    add_body(doc,
        "메인 차트는 선택된 배터리의 사이클 1부터 가장 최근 관측 사이클까지 전체 열화 이력을 "
        "보여주는 인터랙티브 Plotly 그림입니다.")

    add_h2(doc, "3.1 표시 내용")
    add_bullet(doc, "파란 실선 (왼쪽 y축): 모든 관측 사이클에서 실제 측정된 방전 용량 (Ah)")
    add_bullet(doc, "주황 실선 (오른쪽 y축): 각 사이클에서 모델이 예측한 RUL (남은 사이클)")
    add_bullet(doc, "음영 밴드: RUL 예측 주변의 90% 순응형 예측 구간")
    add_bullet(doc, "빨간 점선: EoL 임계값 1.6 Ah (공칭 2.0 Ah의 80%)")

    add_h2(doc, "3.2 인터랙션 방법")
    add_bullet(doc, "임의 지점 호버: 툴팁에서 정확한 사이클 번호, 용량, RUL 값 확인")
    add_bullet(doc, "클릭 및 드래그: 특정 사이클 범위로 확대")
    add_bullet(doc, "더블 클릭: 전체 보기로 확대 초기화")
    add_bullet(doc, "카메라 아이콘: PNG 이미지로 다운로드")

    add_h1(doc, "4. 섹션 2 - 불확실성 패널")
    add_body(doc,
        "메인 차트 아래, 네 개의 메트릭 카드와 막대 차트가 배터리의 가장 최근 관측 사이클에 대한 "
        "불확실성 분해를 표시합니다.")

    add_h2(doc, "4.1 메트릭 카드")
    add_bullet(doc, "점 추정값: 현재 사이클에 대한 앙상블의 가중 평균 RUL 예측")
    add_bullet(doc, "하한 (5 백분위수): 90% 순응형 구간의 비관적 끝 (최악의 경우 RUL)")
    add_bullet(doc, "상한 (95 백분위수): 90% 순응형 구간의 낙관적 끝 (최선의 경우 RUL)")
    add_bullet(doc, "CI 너비: 상한 - 하한 (사이클 단위). 더 좁을수록 더 확신 있는 예측")

    add_h2(doc, "4.2 불확실성 분해 막대 차트")
    add_bullet(doc, "우발적 불확실성: 배터리 열화 과정과 측정 노이즈의 고유한 무작위성. 더 많은 데이터로도 줄일 수 없음.")
    add_bullet(doc, "인식적 불확실성: 제한된 훈련 데이터 또는 분포 외 입력으로 인한 모델 불확실성. 더 많은 보정 데이터로 줄일 수 있음.")

    add_h1(doc, "5. 섹션 3 - 위험 및 생존 분석")
    add_body(doc,
        "이 섹션은 이산 시간 위험 모델의 출력을 보여줍니다: 배터리가 다음 20 사이클 내에 고장날 확률과 "
        "전체 생존 곡선 S(t).")

    add_h2(doc, "5.1 고장 확률 게이지")
    add_bullet(doc, "녹색 구역 (0-30%): 낮은 위험. 다음 20 사이클 내 고장 확률 30% 미만.")
    add_bullet(doc, "황색 구역 (30-70%): 중간 위험. 면밀히 모니터링하고 점검 예약 고려.")
    add_bullet(doc, "빨간 구역 (70-100%): 높은 위험. 20 사이클 내 고장 높은 확률. 검토 우선순위 지정.")
    add_note(doc, "임계값은 pipeline.yaml에서 설정 가능: risk.high_threshold=0.70, risk.medium_threshold=0.30", "tip")

    add_h1(doc, "6. 섹션 4 - 이상 탐지")
    add_body(doc,
        "이상 탐지 섹션은 배터리의 현재 용량 궤적이 훈련 중 모델링된 예상 열화 동작에서 "
        "크게 벗어나는지 여부를 보여줍니다.")

    add_h2(doc, "6.1 이상 점수 카드")
    add_bullet(doc, "녹색 배지 'NORMAL': 점수가 95 백분위수 훈련 잔차 임계값 미만. 배터리 열화가 예상 궤적과 일치.")
    add_bullet(doc, "빨간 배지 'ANOMALY': 점수가 임계값 초과. 배터리가 예상 동작에서 크게 벗어남.")
    add_note(doc,
        "ANOMALY 플래그는 배터리가 곧 고장난다는 의미가 아닙니다. 열화 패턴이 훈련 분포에 비해 "
        "비정상적이라는 의미입니다. 전체 그림을 보려면 위험 점수와 결합하세요.", "warning")

    add_h1(doc, "7. 섹션 5 - 특성 중요도 (SHAP)")
    add_body(doc,
        "특성 중요도 섹션은 선택된 배터리의 현재 선택된 사이클에 대해 어떤 입력 특성이 "
        "RUL 예측에 가장 크게 영향을 미쳤는지 설명합니다.")

    add_h2(doc, "7.1 SHAP 워터폴 차트 해석")
    add_bullet(doc, "오른쪽 막대 (양의 SHAP): 이 특성 값이 평균보다 RUL 예측을 더 높게 만듦")
    add_bullet(doc, "왼쪽 막대 (음의 SHAP): 이 특성 값이 평균보다 RUL 예측을 더 낮게 만듦")
    add_bullet(doc, "막대 길이: 사이클 단위의 기여 크기")

    add_h2(doc, "7.2 반사실적 분석 패널")
    add_body(doc, "임의 특성에 대한 민감도 분석:")
    add_bullet(doc, "드롭다운에서 특성 선택 (예: temperature_c)")
    add_bullet(doc, "가상 값 입력 (예: 4.0 대신 24.0 입력하여 실온 시뮬레이션)")
    add_bullet(doc, "'반사실적 실행' 클릭")
    add_bullet(doc, "원래 RUL 예측 대 반사실적 RUL 예측 및 사이클 차이 확인")

    add_h1(doc, "8. 섹션 6 - 감독자 감사")
    add_body(doc,
        "감독자 감사는 현재 예측 배치가 운영 사용에 필요한 신뢰성 기준을 충족하는지 평가하는 "
        "자동화된 품질 게이트입니다.")

    add_h2(doc, "8.1 판정 범주")
    add_bullet(doc, "합격 (녹색): 세 가지 품질 기준 모두 충족. 운영 사용에 신뢰할 수 있는 예측.")
    add_bullet(doc, "조건부 합격 (황색): 하나 이상의 기준이 한계적. 예측에 따라 행동하기 전에 수동 검토 권장.")
    add_bullet(doc, "불합격 (빨간색): 하나 이상의 기준이 명확히 실패. 이 예측 배치에 의존하지 마세요.")

    add_h2(doc, "8.2 품질 기준 (pipeline.yaml에서 설정 가능)")
    add_bullet(doc, "모델 신뢰도 점수: >= 0.85 자동 승인, 0.70-0.85 합격, < 0.70 불합격")
    add_bullet(doc, "RMSE: <= 100 사이클 합격")
    add_bullet(doc, "CV-RMSE 비율: <= 1.5 합격")

    add_h1(doc, "9. 섹션 7 - AI 설명 (RAG 엔진)")
    add_body(doc,
        "AI 설명 섹션은 사용자가 선택된 배터리의 현재 건강, 위험, 모델 예측에 관한 "
        "자연어 질문을 할 수 있게 합니다.")

    add_h2(doc, "9.1 사용 방법")
    add_bullet(doc, "텍스트 입력 상자에 질문 입력: '이 배터리의 위험이 높은 이유는?' 또는 '넓은 신뢰 구간이 의미하는 것은?'")
    add_bullet(doc, "'AI 요약 생성' 클릭")
    add_bullet(doc, "생성 대기 30-90초 (첫 번째 호출은 최대 3분 소요)")
    add_bullet(doc, "답변 읽기. 출처가 [출처 1], [출처 2] 등으로 인용됨")

    add_h2(doc, "9.2 RAG 엔진이 내부적으로 하는 작업")
    add_bullet(doc, "1. SentenceTransformer로 쿼리를 384차원 임베딩 벡터로 인코딩")
    add_bullet(doc, "2. ChromaDB가 지식 베이스의 모든 저장된 문서 청크에 대해 코사인 유사도 검색")
    add_bullet(doc, "3. 상위 3개 가장 관련성 높은 청크 검색")
    add_bullet(doc, "4. 구조화된 프롬프트 조합: 9개 규칙 시스템 지침 + 검색된 청크 + 배터리 파이프라인 데이터")
    add_bullet(doc, "5. Gemma 3-4B-IT (GPU에서 로컬 실행, 4비트 NF4 양자화)에 프롬프트 전달")
    add_bullet(doc, "6. Gemma가 탐욕적 디코딩으로 3-5문장 답변 생성 (최대 180 새 토큰)")

    add_h2(doc, "9.3 성능 및 하드웨어 요구사항")
    add_kv_doc(doc, "첫 번째 호출 시간", "2-4분 (Gemma 모델 가중치를 VRAM에 로드)")
    add_kv_doc(doc, "이후 호출", "30-90초 (모델이 이미 VRAM에 있음)")
    add_kv_doc(doc, "GPU 요구사항", "CUDA를 지원하는 NVIDIA GPU, 최소 8 GB VRAM (RTX 3060 = 12 GB 충분)")
    add_kv_doc(doc, "모델", "google/gemma-3-4b-it, bitsandbytes를 통한 4비트 NF4 양자화")

    add_h1(doc, "10. 기술 아키텍처")

    add_h2(doc, "10.1 프레임워크 및 라이브러리")
    add_kv_doc(doc, "프론트엔드", "Streamlit 1.x")
    add_kv_doc(doc, "차트", "Plotly Express + Plotly Graph Objects")
    add_kv_doc(doc, "ML 모델", "scikit-learn, XGBoost, PyTorch (TCN)")
    add_kv_doc(doc, "RAG 검색", "sentence-transformers, chromadb")
    add_kv_doc(doc, "RAG 생성", "transformers, bitsandbytes (Gemma 3-4B-IT)")

    add_h2(doc, "10.2 캐싱 전략")
    add_bullet(doc, "@st.cache_data: 데이터 로딩 함수에 적용. 처리된 CSV를 캐시하여 매 상호작용마다 재읽기 방지.")
    add_bullet(doc, "@st.cache_resource: 모델 로딩 함수에 적용. 모델 객체를 전체 세션 동안 캐시. 모든 배터리 선택에서 재사용.")

    add_h1(doc, "11. 문제 해결")

    add_h3(doc, "오류: '배터리 데이터를 찾을 수 없음'")
    add_body(doc, "원인: data/processed/cycle_features_with_rul.csv가 없음")
    add_code(doc, "python -m src.modeling.run_full_pipeline")

    add_h3(doc, "오류: 'Explanation failed: Repo id must use alphanumeric chars'")
    add_body(doc, "원인: 로컬 Gemma 모델 가중치가 models/gemma-3-4b-it/에 없음")
    add_body(doc, "해결: 시스템이 자동으로 HuggingFace 다운로드로 대체됩니다. 인터넷 연결 확인. 첫 실행 시 약 2.5 GB 다운로드 (10-20분).")

    add_h3(doc, "오류: '포트 8501 이미 사용 중'")
    add_body(doc, "PowerShell에서 모든 Python 프로세스 종료:")
    add_code(doc, "Get-Process python* | Stop-Process -Force")

    add_h3(doc, "오류: 'CUDA를 사용할 수 없음'")
    add_body(doc, "PyTorch를 CUDA로 재설치:")
    add_code(doc, "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
    add_note(doc, "RAG AI 설명은 CUDA가 필요합니다. 다른 모든 대시보드 섹션은 CPU에서 작동합니다.", "note")


def make_guide_en():
    doc = Document()
    for sec in doc.sections:
        sec.top_margin = Cm(2.5)
        sec.bottom_margin = Cm(2.5)
        sec.left_margin = Cm(2.8)
        sec.right_margin = Cm(2.5)
    build_en(doc)
    build_en_part2(doc)
    path = OUT / "Streamlit_Guide_English_Detailed.docx"
    doc.save(str(path))
    print(f"  Saved: {path}")


def make_guide_ko():
    doc = Document()
    for sec in doc.sections:
        sec.top_margin = Cm(2.5)
        sec.bottom_margin = Cm(2.5)
        sec.left_margin = Cm(2.8)
        sec.right_margin = Cm(2.5)
    build_ko(doc)
    path = OUT / "Streamlit_Guide_Korean_Detailed.docx"
    doc.save(str(path))
    print(f"  Saved: {path}")


if __name__ == "__main__":
    print("Building Streamlit Guide (English)...")
    make_guide_en()
    print("Building Streamlit Guide (Korean)...")
    make_guide_ko()
    print("Done.")
