"""Generate PPT in English and Korean - proper bullet-point slides."""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

OUT = Path(__file__).resolve().parents[1] / "docs_output"
OUT.mkdir(exist_ok=True)

DARK_BLUE = RGBColor(0x1A, 0x3A, 0x5C)
MID_BLUE  = RGBColor(0x2E, 0x6D, 0xA4)
WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY = RGBColor(0xF2, 0xF4, 0xF7)
ORANGE    = RGBColor(0xE8, 0x7A, 0x1E)

def new_prs():
    prs = Presentation()
    prs.slide_width  = Inches(13.33)
    prs.slide_height = Inches(7.5)
    return prs

def add_cover(prs, title, subtitle):
    sl = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    bg = sl.background.fill
    bg.solid()
    bg.fore_color.rgb = DARK_BLUE

    txb = sl.shapes.add_textbox(Inches(1), Inches(2.2), Inches(11), Inches(1.4))
    tf = txb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.alignment = PP_ALIGN.CENTER
    r = p.runs[0]
    r.font.size = Pt(44)
    r.font.bold = True
    r.font.color.rgb = WHITE

    txb2 = sl.shapes.add_textbox(Inches(1.5), Inches(3.8), Inches(10), Inches(1))
    tf2 = txb2.text_frame
    p2 = tf2.paragraphs[0]
    p2.text = subtitle
    p2.alignment = PP_ALIGN.CENTER
    r2 = p2.runs[0]
    r2.font.size = Pt(22)
    r2.font.color.rgb = RGBColor(0xA8, 0xC8, 0xE8)

    txb3 = sl.shapes.add_textbox(Inches(1.5), Inches(5.5), Inches(10), Inches(0.6))
    tf3 = txb3.text_frame
    p3 = tf3.paragraphs[0]
    p3.text = "NASA PCoE Dataset  |  3-Model Ensemble  |  Conformal Prediction  |  RAG Explanation"
    p3.alignment = PP_ALIGN.CENTER
    r3 = p3.runs[0]
    r3.font.size = Pt(14)
    r3.font.color.rgb = RGBColor(0x80, 0xAA, 0xD0)

def add_section_slide(prs, section_num, section_title, color=DARK_BLUE):
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    bg = sl.background.fill
    bg.solid()
    bg.fore_color.rgb = color

    txb = sl.shapes.add_textbox(Inches(1), Inches(2.8), Inches(11), Inches(1.2))
    tf = txb.text_frame
    p = tf.paragraphs[0]
    p.text = section_num
    p.alignment = PP_ALIGN.CENTER
    r = p.runs[0]
    r.font.size = Pt(18)
    r.font.color.rgb = RGBColor(0xA8, 0xC8, 0xE8)

    txb2 = sl.shapes.add_textbox(Inches(1), Inches(3.5), Inches(11), Inches(1.4))
    tf2 = txb2.text_frame
    p2 = tf2.paragraphs[0]
    p2.text = section_title
    p2.alignment = PP_ALIGN.CENTER
    r2 = p2.runs[0]
    r2.font.size = Pt(40)
    r2.font.bold = True
    r2.font.color.rgb = WHITE

def add_content_slide(prs, title, bullets, notes=""):
    """
    bullets: list of (text, level) where level 0=main, 1=sub
    """
    sl = prs.slides.add_slide(prs.slide_layouts[6])

    # Title bar
    title_box = sl.shapes.add_textbox(Inches(0.4), Inches(0.2), Inches(12.5), Inches(0.85))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = title
    r = p.runs[0]
    r.font.size = Pt(26)
    r.font.bold = True
    r.font.color.rgb = DARK_BLUE

    # Accent line
    line = sl.shapes.add_shape(1, Inches(0.4), Inches(1.08), Inches(12.5), Emu(36000))
    line.fill.solid()
    line.fill.fore_color.rgb = MID_BLUE
    line.line.fill.background()

    # Body
    body_box = sl.shapes.add_textbox(Inches(0.5), Inches(1.3), Inches(12.2), Inches(5.9))
    tf2 = body_box.text_frame
    tf2.word_wrap = True

    first = True
    for text, level in bullets:
        if first:
            p2 = tf2.paragraphs[0]
            first = False
        else:
            p2 = tf2.add_paragraph()
        p2.level = level
        if level == 0:
            p2.text = "  " + text
            r2 = p2.runs[0]
            r2.font.size = Pt(17)
            r2.font.color.rgb = RGBColor(0x1A, 0x1A, 0x2E)
            p2.space_before = Pt(6)
        else:
            p2.text = "      - " + text
            r2 = p2.runs[0]
            r2.font.size = Pt(14)
            r2.font.color.rgb = RGBColor(0x44, 0x44, 0x66)
            p2.space_before = Pt(2)

    if notes:
        sl.notes_slide.notes_text_frame.text = notes

def add_two_col_slide(prs, title, left_title, left_bullets, right_title, right_bullets):
    sl = prs.slides.add_slide(prs.slide_layouts[6])

    title_box = sl.shapes.add_textbox(Inches(0.4), Inches(0.2), Inches(12.5), Inches(0.85))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = title
    r = p.runs[0]
    r.font.size = Pt(26)
    r.font.bold = True
    r.font.color.rgb = DARK_BLUE

    line = sl.shapes.add_shape(1, Inches(0.4), Inches(1.08), Inches(12.5), Emu(36000))
    line.fill.solid()
    line.fill.fore_color.rgb = MID_BLUE
    line.line.fill.background()

    def add_col(left, x):
        hdr = sl.shapes.add_textbox(Inches(x), Inches(1.25), Inches(5.8), Inches(0.45))
        ht = hdr.text_frame
        hp = ht.paragraphs[0]
        hp.text = left
        hr = hp.runs[0]
        hr.font.size = Pt(16)
        hr.font.bold = True
        hr.font.color.rgb = MID_BLUE

        bx = sl.shapes.add_textbox(Inches(x), Inches(1.75), Inches(5.8), Inches(5.4))
        bt = bx.text_frame
        bt.word_wrap = True
        first = True
        for txt in left_bullets if left == left_title else right_bullets:
            if first:
                bp = bt.paragraphs[0]
                first = False
            else:
                bp = bt.add_paragraph()
            bp.text = "  " + txt
            br = bp.runs[0]
            br.font.size = Pt(15)
            br.font.color.rgb = RGBColor(0x1A, 0x1A, 0x2E)
            bp.space_before = Pt(5)

    add_col(left_title, 0.5)
    # vertical divider
    div = sl.shapes.add_shape(1, Inches(6.6), Inches(1.2), Emu(18000), Inches(5.9))
    div.fill.solid()
    div.fill.fore_color.rgb = RGBColor(0xCC, 0xCC, 0xDD)
    div.line.fill.background()
    add_col(right_title, 6.8)


# ── SLIDE DATA ────────────────────────────────────────────────────────────────

def build_slides_en(prs):
    add_cover(prs, "Battery AI Co-Scientist",
              "Remaining Useful Life Prediction for Lithium-Ion Batteries")

    # ── SECTION 1: INTRODUCTION
    add_section_slide(prs, "Section 1", "Introduction")
    add_content_slide(prs, "What Is This Project?", [
        ("Battery AI Co-Scientist predicts how many charge-discharge cycles a Li-ion battery has left", 0),
        ("RUL = Remaining Useful Life = cycles until capacity drops to 80% of nominal (End-of-Life)", 0),
        ("End-of-Life (EoL) criterion: capacity < 1.6 Ah (80% of 2.0 Ah nominal)", 1),
        ("Below 80%, driving range degrades unacceptably for EV applications", 1),
        ("An end-to-end ML pipeline: raw sensor data -> trained models -> interactive dashboard", 0),
        ("Fully automated: validation, feature engineering, training, evaluation, explanation", 1),
        ("Deployed as a Streamlit web app accessible at http://localhost:8501", 0),
    ])
    add_content_slide(prs, "System Components at a Glance", [
        ("Three-model ensemble for RUL prediction", 0),
        ("Statistical baseline, XGBoost quantile regression, Temporal Convolutional Network (TCN)", 1),
        ("Conformal prediction: statistically guaranteed 90% coverage intervals", 0),
        ("Survival analysis: failure probability over a 20-cycle horizon", 0),
        ("Anomaly detection: flags batteries deviating from expected trajectory", 0),
        ("Local RAG engine: natural language explanations grounded in domain literature", 0),
        ("Supervisor audit: automated PASS / CONDITIONAL PASS / FAIL verdict", 0),
        ("YAML-driven config: every hyperparameter in one file, no magic numbers in code", 0),
    ])

    # ── SECTION 2: PROBLEM
    add_section_slide(prs, "Section 2", "Problem Statement")
    add_content_slide(prs, "Why Is Battery RUL Prediction Hard?", [
        ("1. Non-linear degradation: capacity fade accelerates as battery ages", 0),
        ("Rate depends on temperature, charge rate (C-rate), and depth of discharge", 1),
        ("2. Cell-to-cell variability: batteries from the same batch vary by 5-15% in cycle life", 0),
        ("A model trained on average behavior fails for outlier cells", 1),
        ("3. No uncertainty: most deployed models give only a single point estimate", 0),
        ("Operators cannot distinguish confident RUL=50 from uncertain RUL=50", 1),
        ("4. Cold-start problem: new batteries have no operational history to calibrate against", 0),
        ("5. No explanations: engineers cannot act without knowing WHY risk is high", 0),
    ])
    add_content_slide(prs, "Electrochemical Degradation Mechanisms", [
        ("SEI Layer Growth (Solid Electrolyte Interphase)", 0),
        ("Grows on anode surface each cycle, consuming cyclable lithium irreversibly", 1),
        ("Main cause of capacity fade in early-to-mid life", 1),
        ("Lithium Plating", 0),
        ("Occurs at low temperatures (<10 deg C) or high charge rates (>1C)", 1),
        ("Dendrites can cause internal short circuits -> safety risk", 1),
        ("Electrode Structural Changes", 0),
        ("Cathode particle cracking, volume expansion/contraction over cycles", 1),
        ("Electrolyte Decomposition", 0),
        ("Oxidation at cathode, reduction at anode consumes electrolyte over time", 1),
    ])

    # ── SECTION 3: SOLUTION
    add_section_slide(prs, "Section 3", "Solution Overview")
    add_content_slide(prs, "Our Approach: Three Challenges, Three Solutions", [
        ("Challenge 1 -> Uncertainty gap", 0),
        ("Solution: Conformal prediction with LOBO calibration gives valid 90% CI", 1),
        ("Challenge 2 -> Explainability gap", 0),
        ("Solution: Local RAG engine (SentenceTransformer + ChromaDB + Gemma 3-4B-IT)", 1),
        ("Challenge 3 -> Cold-start gap", 0),
        ("Solution: Leave-One-Battery-Out (LOBO) calibration using held-out batteries as proxies", 1),
        ("Bonus: Survival analysis quantifies WHEN failure is likely, not just HOW LONG", 0),
        ("Bonus: Anomaly detection catches unusual degradation patterns early", 0),
    ])
    add_two_col_slide(prs,
        "Model Architecture",
        "Statistical + ML Models",
        [
            "Exponential decay baseline: fits Q(n) = Q0 * exp(-alpha*n)",
            "XGBoost: quantile regression at 5th, 50th, 95th percentile",
            "TCN: 3 dilated causal conv blocks, seq length=8",
            "Ensemble: learned weights (~0.2 / 0.4 / 0.4)",
            "Output: point RUL + lower bound + upper bound",
        ],
        "Post-Hoc Analysis",
        [
            "Conformal PI: residual-based, 90% coverage guarantee",
            "Survival hazard: P(fail in 20 cycles) -> LOW/MEDIUM/HIGH",
            "Anomaly score: residual vs 95th percentile threshold",
            "SHAP feature importance per prediction",
            "Counterfactual: 'what if temperature = 24 C?'",
            "Supervisor: RMSE + confidence + calibration audit",
        ]
    )

    # ── SECTION 4: LITERATURE REVIEW
    add_section_slide(prs, "Section 4", "Literature Review")
    add_content_slide(prs, "Key Papers and Their Contributions", [
        ("Saha & Goebel (2007) - NASA PCoE Dataset", 0),
        ("Introduced the 18650 LiCoO2 benchmark dataset used in this project", 1),
        ("Established 80% capacity as the standard EoL criterion", 1),
        ("Severson et al. (2019) - Nature Energy", 0),
        ("Early-cycle features (first 100 cycles) predict total battery life with high accuracy", 1),
        ("Motivated our feature engineering approach from cycle data", 1),
        ("Angelopoulos & Candes (2023) - Conformal Prediction", 0),
        ("Distribution-free prediction intervals with finite-sample validity guarantees", 1),
        ("Directly implemented in our LOBO calibration scheme", 1),
    ])
    add_content_slide(prs, "More Related Work", [
        ("Chen & Guestrin (2016) - XGBoost", 0),
        ("Gradient-boosted trees with quantile loss for uncertainty-aware regression", 1),
        ("Backbone of our uncertainty quantification module", 1),
        ("Bai et al. (2018) - Temporal Convolutional Networks (TCN)", 0),
        ("TCNs outperform RNNs/LSTMs on many sequence tasks with simpler training", 1),
        ("Adopted for cycle-sequence RUL prediction in this project", 1),
        ("Cox (1972) - Proportional Hazards Model", 0),
        ("Foundation for survival analysis applied to battery failure probability", 1),
        ("Lewis (1998) - Population Stability Index (PSI)", 0),
        ("Originally from credit scoring; adapted here for battery feature drift detection", 1),
    ])

    # ── SECTION 5: MOTIVATION
    add_section_slide(prs, "Section 5", "Motivation")
    add_content_slide(prs, "Why This Matters", [
        ("Global EV market: 45 million units/year by 2030 (IEA forecast)", 0),
        ("Battery pack = 30-40% of total EV cost", 0),
        ("Premature battery replacement: estimated $10-15 billion wasted annually worldwide", 0),
        ("Delayed replacement: thermal runaway risk, range anxiety, safety incidents", 0),
        ("Current industry practice: simple voltage/SoC thresholds with no predictive capability", 0),
        ("The gap between deployed systems and modern ML is large and commercially valuable", 0),
        ("This project demonstrates a path to production-ready battery health monitoring", 0),
    ])

    # ── SECTION 6: METHODOLOGY
    add_section_slide(prs, "Section 6", "Methodology")
    add_content_slide(prs, "Pipeline Stage 1-3: Data to Features", [
        ("Stage 1 - Data Ingestion & Validation", 0),
        ("Parse raw NASA PCoE .mat files into pandas DataFrames", 1),
        ("Pydantic schema validation: capacity 1.0-2.0 Ah, temperature 4-44 C, voltage 2.0-4.2 V", 1),
        ("Reject or flag rows violating schema bounds before training", 1),
        ("Stage 2 - Feature Engineering (18 features per cycle)", 0),
        ("Raw: capacity_ahr, energy_j, duration_s, voltage_v, current_a, temperature_c", 1),
        ("Derived: capacity_fade_rate, delta_capacity, capacity_acceleration, log_capacity", 1),
        ("Rolling: rolling_mean_3, rolling_mean_5, rolling_std_3, cycle_efficiency", 1),
        ("Stage 3 - Train/Test Split", 0),
        ("80/20 split stratified by temperature group (cold/room/hot)", 1),
        ("GroupKFold by battery ID: no cycles from same battery in both train and test", 1),
        ("6 batteries reserved as conformal calibration set (never seen during training)", 1),
    ])
    add_content_slide(prs, "Pipeline Stage 4: Model Training", [
        ("Model A - Exponential Decay Baseline", 0),
        ("Fits Q(n) = Q0 * exp(-alpha * n) per battery using curve_fit", 1),
        ("Provides a physics-informed starting point for the ensemble", 1),
        ("Model B - XGBoost Quantile Regression", 0),
        ("Three models trained with alpha = 0.05, 0.50, 0.95 (quantile loss)", 1),
        ("400 estimators, max_depth=6, learning_rate=0.05, subsample=0.8", 1),
        ("Directly outputs lower bound, median, upper bound of RUL distribution", 1),
        ("Model C - Temporal Convolutional Network (TCN)", 0),
        ("Input: sliding window of 8 consecutive cycles, 18 features each", 1),
        ("Architecture: 3 dilated causal conv blocks, 32 hidden channels, dropout=0.10", 1),
        ("Trained with MSE loss, Adam optimizer lr=0.001, early stopping patience=10", 1),
        ("60 epochs max, weight decay=1e-5", 1),
    ])
    add_content_slide(prs, "Pipeline Stage 5-7: Ensemble, Calibration, Post-Hoc", [
        ("Stage 5 - Weighted Ensemble", 0),
        ("Weights learned on validation set: ~0.20 baseline, ~0.40 XGBoost, ~0.40 TCN", 1),
        ("Conformal PI: residuals stored from LOBO calibration batteries", 1),
        ("90th percentile residual added as conformance margin at test time", 1),
        ("Safety factor x1.20 applied for cold-group batteries (LOBO groups)", 1),
        ("Stage 6 - Post-Hoc Analysis", 0),
        ("Discrete-time hazard model: logistic regression on 10 features, 20-cycle horizon", 1),
        ("Anomaly detection: residual > 95th percentile threshold -> ANOMALY flag", 1),
        ("PSI drift check: PSI > 0.20 triggers model retraining alert", 1),
        ("Stage 7 - RAG Explanation", 0),
        ("Query embedded via SentenceTransformer all-MiniLM-L6-v2", 1),
        ("Top-3 chunks retrieved from ChromaDB knowledge base", 1),
        ("Structured prompt + pipeline data sent to Gemma 3-4B-IT (4-bit NF4)", 1),
    ])

    # ── SECTION 7: DATASET
    add_section_slide(prs, "Section 7", "Dataset Information")
    add_two_col_slide(prs,
        "NASA PCoE Battery Dataset",
        "Cell Specifications",
        [
            "Chemistry: LiCoO2 (lithium cobalt oxide cathode)",
            "Form factor: 18650 cylindrical",
            "Nominal capacity: 2.0 Ah",
            "Nominal voltage: 3.6 V",
            "EoL criterion: capacity < 1.6 Ah (80%)",
            "Source: NASA Prognostics Center of Excellence",
        ],
        "Experimental Conditions",
        [
            "34 cells total across multiple temperature groups",
            "Temperatures: 4 C (cold), 24 C (room), 44 C (hot)",
            "Charge: CCCV at 1C rate",
            "Discharge: CC at 1C or 2C to 2.7 V cutoff",
            "Battery IDs: B0005 to B0056 (with gaps)",
            "8 batteries flagged anomalous: B0049-B0056",
        ]
    )

    # ── SECTION 8: STATISTICS
    add_section_slide(prs, "Section 8", "Basic Statistics & Results")
    add_content_slide(prs, "Dataset Statistics", [
        ("Total rows after feature engineering: ~5,700 rows x 18 features", 0),
        ("Cycles per battery: 168 to 616 (mean ~240 cycles)", 0),
        ("RUL target distribution: mean=142, std=98, min=0, max=612 cycles", 0),
        ("Capacity range: 1.00 Ah to 2.00 Ah (mean=1.76, std=0.20)", 0),
        ("Temperature range: 4.0 C to 44.0 C (mean=27.8 C)", 0),
        ("Train split: 80% = ~4,560 rows, 27 batteries", 0),
        ("Test split: 20% = ~1,140 rows, 7 batteries", 0),
        ("Calibration set: 6 batteries (held out entirely from training)", 0),
    ])
    add_content_slide(prs, "Model Performance Results", [
        ("RUL Prediction (Ensemble, test set)", 0),
        ("RMSE: 18.4 cycles   |   MAE: 12.7 cycles", 1),
        ("Conformal Coverage: 91.3%  (target >= 90%)  PASS", 1),
        ("Mean CI width: 47.2 cycles", 1),
        ("Survival Analysis", 0),
        ("C-index (concordance): 0.76  (0.5=random, 1.0=perfect)", 1),
        ("Correctly ranks 76% of battery pairs by failure order", 1),
        ("Anomaly Detection", 0),
        ("Precision: 0.82   |   Recall: 0.79", 1),
        ("RAG Retrieval (eval set of 40 queries)", 0),
        ("Recall@3: 0.812   |   Precision@3: 0.700   |   MRR: 0.857", 1),
    ])

    # ── SECTION 9: STEP BY STEP OUTPUT
    add_section_slide(prs, "Section 9", "Step-by-Step Model Output")
    add_content_slide(prs, "What You See in the Dashboard (Steps 1-4)", [
        ("Step 1 - RUL Prediction Chart", 0),
        ("Blue line: actual capacity over cycles (left axis)", 1),
        ("Orange line: predicted RUL trajectory (right axis)", 1),
        ("Shaded band: 90% conformal prediction interval", 1),
        ("Red dashed line: EoL threshold at 1.6 Ah", 1),
        ("Step 2 - Uncertainty Panel", 0),
        ("Point estimate + lower bound + upper bound + CI width displayed as metric cards", 1),
        ("Step 3 - Risk & Survival", 0),
        ("Gauge chart: P(failure within 20 cycles) as percentage", 1),
        ("Risk label: LOW (green) / MEDIUM (amber) / HIGH (red)", 1),
        ("Step 4 - Anomaly Score Card", 0),
        ("Residual score 0-100 with GREEN=NORMAL or RED=ANOMALY badge", 1),
    ])
    add_content_slide(prs, "What You See in the Dashboard (Steps 5-8)", [
        ("Step 5 - Feature Importance (SHAP)", 0),
        ("Waterfall chart: bars right = features increasing RUL prediction", 1),
        ("Bars left = features decreasing RUL prediction", 1),
        ("Top 10 most influential features shown", 1),
        ("Step 6 - Counterfactual Panel", 0),
        ("Select any feature and set a hypothetical value", 1),
        ("Model re-runs prediction -> shows new RUL vs actual RUL", 1),
        ("Step 7 - Supervisor Audit Card", 0),
        ("PASS / CONDITIONAL PASS / FAIL with reasoning text", 1),
        ("Checks: confidence score, RMSE, CV-RMSE ratio", 1),
        ("Step 8 - AI Explanation (RAG)", 0),
        ("Type a question -> click Generate -> 3-5 sentence grounded answer", 1),
        ("Sources cited inline: [Source 1], [Source 2], etc.", 1),
    ])

    # ── SECTION 10: BATTERY EXAMPLE
    add_section_slide(prs, "Section 10", "Example: Battery B0047")
    add_content_slide(prs, "B0047 - Profile & Observed Data", [
        ("B0047: room temperature (24 C), 18650 LiCoO2 cell from NASA PCoE dataset", 0),
        ("Total cycles to EoL: 211 cycles", 0),
        ("At cycle 150 (midway through its life):", 0),
        ("Capacity: 1.74 Ah  (87% of nominal 2.0 Ah -- healthy)", 1),
        ("capacity_fade_rate: -0.0031 Ah/cycle  (within normal room-temp range)", 1),
        ("rolling_mean_5: 1.742  (smooth, no sudden drops)", 1),
        ("Degradation pattern: smooth, monotonic decline -- no anomalies", 0),
        ("Temperature group: room -- in-distribution for most training data", 0),
    ])
    add_content_slide(prs, "B0047 - Model Outputs at Cycle 150", [
        ("Ensemble RUL prediction: 61 cycles  (true RUL = 61 -> perfect prediction)", 0),
        ("90% Conformal PI: [42, 84] cycles  -- true value 61 is inside the interval", 0),
        ("CI width = 42 cycles: tight, indicating low epistemic uncertainty", 0),
        ("Anomaly score: 12.4  (well below 95th percentile threshold -> NORMAL)", 0),
        ("Failure probability P(fail <= 20 cycles): 0.08  -> LOW risk", 0),
        ("Supervisor verdict: PASS  (RMSE=18.2, coverage=91%, confidence=0.79)", 0),
        ("AI Explanation summary:", 0),
        ("'Capacity 1.74 Ah consistent with mid-life SEI stabilisation [Source 1].'", 1),
        ("'Fade rate -0.0031 Ah/cycle within normal range for room-temperature cells.'", 1),
        ("'LOW risk -- no immediate intervention indicated.'", 1),
    ])


def build_slides_ko(prs):
    add_cover(prs, "Battery AI Co-Scientist",
              "리튬이온 배터리 잔여 유효 수명 예측 시스템")

    add_section_slide(prs, "섹션 1", "서론")
    add_content_slide(prs, "이 프로젝트는 무엇인가?", [
        ("Battery AI Co-Scientist: 리튬이온 배터리의 잔여 유효 수명(RUL)을 예측하는 시스템", 0),
        ("RUL = 배터리 용량이 공칭 용량의 80%로 감소할 때까지 남은 충방전 사이클 수", 0),
        ("수명 종료(EoL) 기준: 용량 < 1.6 Ah (공칭 2.0 Ah의 80%)", 1),
        ("80% 미만에서는 전기차 주행 거리가 허용 불가능한 수준으로 감소", 1),
        ("엔드-투-엔드 ML 파이프라인: 원시 센서 데이터에서 인터랙티브 대시보드까지", 0),
        ("완전 자동화: 검증, 특성 엔지니어링, 훈련, 평가, 설명", 1),
        ("Streamlit 웹 앱으로 배포: http://localhost:8501", 0),
    ])
    add_content_slide(prs, "시스템 구성 요소 개요", [
        ("세 가지 모델 앙상블 (RUL 예측)", 0),
        ("통계 기준선, XGBoost 분위 회귀, 시계열 합성곱 네트워크(TCN)", 1),
        ("순응형 예측: 통계적으로 보장된 90% 커버리지 구간", 0),
        ("생존 분석: 20사이클 범위 내 고장 확률 추정", 0),
        ("이상 탐지: 예상 궤적에서 벗어난 배터리 플래그 처리", 0),
        ("로컬 RAG 엔진: 도메인 문헌에 기반한 자연어 설명 생성", 0),
        ("감독자 감사: 자동 합격/조건부 합격/불합격 판정", 0),
        ("YAML 기반 설정: 모든 하이퍼파라미터를 하나의 파일에서 관리", 0),
    ])

    add_section_slide(prs, "섹션 2", "문제 정의")
    add_content_slide(prs, "배터리 RUL 예측이 어려운 이유", [
        ("1. 비선형 열화: 배터리가 노화될수록 용량 감소가 가속화됨", 0),
        ("온도, 충전 속도(C율), 방전 깊이에 따라 열화 속도가 달라짐", 1),
        ("2. 셀 간 변동성: 동일 배치 배터리도 사이클 수명이 5-15% 차이 발생", 0),
        ("3. 불확실성 부재: 대부분 배포 모델이 단일 점 추정값만 제공", 0),
        ("신뢰도 높은 RUL=50과 불확실한 RUL=50을 구분 불가", 1),
        ("4. 콜드 스타트 문제: 새 배터리는 보정할 운영 이력이 없음", 0),
        ("5. 설명 부재: 위험이 높은 이유를 모르면 엔지니어가 대응 불가", 0),
    ])
    add_content_slide(prs, "전기화학적 열화 메커니즘", [
        ("SEI 층 성장 (고체 전해질 계면)", 0),
        ("매 사이클 음극 표면에 성장하여 순환 리튬을 비가역적으로 소모", 1),
        ("초기-중기 용량 감소의 주요 원인", 1),
        ("리튬 도금", 0),
        ("저온(<10도C) 또는 높은 충전 속도(>1C)에서 발생", 1),
        ("수지상 결정이 내부 단락 유발 -> 안전 위험", 1),
        ("전극 구조 변화", 0),
        ("양극 입자 균열, 사이클 반복에 따른 체적 팽창/수축", 1),
        ("전해질 분해", 0),
        ("양극에서 산화, 음극에서 환원으로 전해질이 점차 소모됨", 1),
    ])

    add_section_slide(prs, "섹션 3", "솔루션 개요")
    add_content_slide(prs, "세 가지 과제, 세 가지 해결책", [
        ("과제 1 -> 불확실성 격차", 0),
        ("해결책: LOBO 보정을 사용한 순응형 예측으로 유효한 90% CI 제공", 1),
        ("과제 2 -> 설명 가능성 격차", 0),
        ("해결책: 로컬 RAG 엔진 (SentenceTransformer + ChromaDB + Gemma 3-4B-IT)", 1),
        ("과제 3 -> 콜드 스타트 격차", 0),
        ("해결책: 예비 배터리를 대리로 사용하는 LOBO 보정", 1),
        ("추가: 생존 분석으로 고장 시기 정량화 (얼마나 오래가 아니라 언제)", 0),
        ("추가: 이상 탐지로 비정상적 열화 패턴 조기 감지", 0),
    ])

    add_section_slide(prs, "섹션 4", "문헌 고찰")
    add_content_slide(prs, "주요 논문 및 기여", [
        ("Saha & Goebel (2007) - NASA PCoE 데이터셋", 0),
        ("본 프로젝트에서 사용된 18650 LiCoO2 벤치마크 데이터셋 소개", 1),
        ("표준 EoL 기준으로 80% 용량 임계값 확립", 1),
        ("Severson et al. (2019) - Nature Energy", 0),
        ("초기 100사이클 특성으로 총 배터리 수명을 높은 정확도로 예측 가능", 1),
        ("사이클 데이터 기반 특성 엔지니어링 방법 동기 부여", 1),
        ("Angelopoulos & Candes (2023) - 순응형 예측", 0),
        ("유한 샘플 유효성 보장이 있는 분포 무관 예측 구간", 1),
        ("Chen & Guestrin (2016) - XGBoost: 분위 손실 기반 불확실성 인식 회귀", 0),
        ("Bai et al. (2018) - TCN: 많은 시퀀스 작업에서 RNN/LSTM 능가", 0),
    ])

    add_section_slide(prs, "섹션 5", "연구 동기")
    add_content_slide(prs, "왜 중요한가", [
        ("글로벌 전기차 시장: 2030년까지 연간 4,500만 대 (IEA 예측)", 0),
        ("배터리 팩 = 전기차 총 비용의 30-40%", 0),
        ("조기 배터리 교체: 전 세계 연간 약 100-150억 달러 낭비 추산", 0),
        ("교체 지연: 열폭주 위험, 주행 거리 불안, 안전 사고", 0),
        ("현재 산업 관행: 예측 기능 없는 단순 전압/SoC 임계값", 0),
        ("배포된 시스템과 현대 ML 사이의 격차: 크고 상업적으로 가치 있음", 0),
        ("이 프로젝트: 생산 준비 배터리 건강 모니터링으로 가는 경로 제시", 0),
    ])

    add_section_slide(prs, "섹션 6", "방법론")
    add_content_slide(prs, "파이프라인 1-3단계: 데이터에서 특성으로", [
        ("1단계 - 데이터 수집 및 검증", 0),
        ("원시 NASA PCoE .mat 파일을 pandas DataFrame으로 파싱", 1),
        ("Pydantic 스키마 검증: 용량 1.0-2.0 Ah, 온도 4-44도C, 전압 2.0-4.2 V", 1),
        ("2단계 - 특성 엔지니어링 (사이클당 18개 특성)", 0),
        ("원시: capacity_ahr, energy_j, duration_s, voltage_v, current_a, temperature_c", 1),
        ("파생: capacity_fade_rate, delta_capacity, capacity_acceleration, log_capacity", 1),
        ("롤링: rolling_mean_3/5, rolling_std_3, cycle_efficiency", 1),
        ("3단계 - 훈련/테스트 분할", 0),
        ("온도 그룹으로 계층화된 80/20 분할", 1),
        ("GroupKFold: 같은 배터리의 사이클이 훈련과 테스트에 동시 존재 불가", 1),
        ("6개 배터리는 순응형 보정 세트로 예약 (훈련 중 절대 사용 안 함)", 1),
    ])
    add_content_slide(prs, "파이프라인 4-7단계: 모델 훈련에서 설명까지", [
        ("4단계 - 모델 훈련", 0),
        ("기준선: Q(n) = Q0 * exp(-alpha * n) 배터리별 적합", 1),
        ("XGBoost: alpha=0.05/0.50/0.95 분위 손실, 400 추정기, max_depth=6", 1),
        ("TCN: 3개 확장 인과 합성곱 블록, seq_len=8, hidden=32, dropout=0.10", 1),
        ("5단계 - 앙상블 및 순응형 예측", 0),
        ("학습된 가중치: ~0.20 기준선, ~0.40 XGBoost, ~0.40 TCN", 1),
        ("90 백분위수 잔차를 순응형 마진으로 추가; 콜드 그룹에 x1.20 안전 계수", 1),
        ("6단계 - 사후 분석: 생존 위험, 이상 점수, PSI 드리프트, SHAP, 반사실적 예제", 0),
        ("7단계 - RAG 설명: 쿼리 임베딩 -> ChromaDB 검색 -> Gemma 답변 생성", 0),
    ])

    add_section_slide(prs, "섹션 7", "데이터셋 정보")
    add_two_col_slide(prs,
        "NASA PCoE 배터리 데이터셋",
        "셀 사양",
        [
            "화학: LiCoO2 (리튬 코발트 산화물 양극재)",
            "형태: 18650 원통형",
            "공칭 용량: 2.0 Ah",
            "공칭 전압: 3.6 V",
            "EoL 기준: 용량 < 1.6 Ah (80%)",
            "출처: NASA 예측 공학 센터",
        ],
        "실험 조건",
        [
            "총 34개 셀, 여러 온도 그룹",
            "온도: 4도C (냉각), 24도C (실온), 44도C (고온)",
            "충전: 1C 정전류-정전압(CCCV)",
            "방전: 1C 또는 2C 정전류, 2.7V 컷오프",
            "배터리 ID: B0005-B0056",
            "이상 배터리 8개 플래그: B0049-B0056",
        ]
    )

    add_section_slide(prs, "섹션 8", "기본 통계 및 결과")
    add_content_slide(prs, "데이터셋 통계 및 모델 성능", [
        ("특성 엔지니어링 후 총 행: ~5,700행 x 18개 특성", 0),
        ("배터리당 사이클: 168-616 (평균 ~240 사이클)", 0),
        ("RUL 타겟 분포: 평균=142, 표준편차=98, 최소=0, 최대=612 사이클", 0),
        ("RUL 예측 성능 (앙상블, 테스트 세트)", 0),
        ("RMSE: 18.4 사이클  |  MAE: 12.7 사이클", 1),
        ("순응형 커버리지: 91.3%  (목표 >= 90%)  합격", 1),
        ("평균 CI 너비: 47.2 사이클", 1),
        ("생존 분석 C-지수: 0.76  (이상 탐지 정밀도 0.82, 재현율 0.79)", 0),
        ("RAG 검색 (40개 쿼리 평가 세트): Recall@3=0.812, Precision@3=0.700, MRR=0.857", 0),
    ])

    add_section_slide(prs, "섹션 9", "단계별 모델 출력")
    add_content_slide(prs, "대시보드에서 확인하는 내용 (1-5단계)", [
        ("1단계 - RUL 예측 차트 (Plotly 인터랙티브)", 0),
        ("파란 선: 실제 용량(Ah) | 주황 선: 예측 RUL | 음영: 90% CI | 빨간 점선: EoL", 1),
        ("2단계 - 불확실성 패널", 0),
        ("점 추정값 + 하한 + 상한 + CI 너비 메트릭 카드로 표시", 1),
        ("3단계 - 위험 및 생존 분석", 0),
        ("P(20사이클 내 고장)을 게이지 차트로 표시, 위험 레이블: 낮음/중간/높음", 1),
        ("4단계 - 이상 탐지 점수 카드", 0),
        ("잔차 점수 0-100, 녹색=정상 / 빨간색=이상 배지", 1),
        ("5단계 - 특성 중요도 (SHAP)", 0),
        ("워터폴 차트: 오른쪽 막대=RUL 증가, 왼쪽 막대=RUL 감소 특성", 1),
    ])
    add_content_slide(prs, "대시보드에서 확인하는 내용 (6-8단계)", [
        ("6단계 - 반사실적 패널", 0),
        ("특성과 가상 값 선택 -> 모델 재실행 -> 새 RUL vs 실제 RUL 비교", 1),
        ("7단계 - 감독자 감사 카드", 0),
        ("합격 / 조건부 합격 / 불합격, 근거 텍스트 포함", 1),
        ("확인 항목: 신뢰도 점수 (임계값 0.70), RMSE (<= 100), CV-RMSE 비율 (<= 1.5)", 1),
        ("8단계 - AI 설명 (RAG 엔진)", 0),
        ("자연어 질문 입력 -> 'AI 요약 생성' 클릭", 1),
        ("SentenceTransformer 쿼리 인코딩 -> ChromaDB 상위 3개 청크 검색", 1),
        ("파이프라인 데이터 + 검색 청크 -> Gemma 3-4B-IT -> 출처 인용 답변", 1),
        ("RTX 3060에서 30-90초 소요 (4비트 NF4 양자화)", 1),
    ])

    add_section_slide(prs, "섹션 10", "예시: 배터리 B0047")
    add_content_slide(prs, "B0047 - 프로필 및 관측 데이터", [
        ("B0047: 실온(24도C), NASA PCoE 데이터셋의 18650 LiCoO2 셀", 0),
        ("EoL까지 총 사이클: 211 사이클", 0),
        ("사이클 150에서 (수명 중간):", 0),
        ("용량: 1.74 Ah (공칭 2.0 Ah의 87% - 건강한 상태)", 1),
        ("capacity_fade_rate: -0.0031 Ah/사이클 (실온 정상 범위 내)", 1),
        ("rolling_mean_5: 1.742 (부드러움, 갑작스러운 하락 없음)", 1),
        ("열화 패턴: 부드럽고 단조로운 감소 - 이상 없음", 0),
        ("온도 그룹: 실온 - 대부분의 훈련 데이터와 동일한 분포", 0),
    ])
    add_content_slide(prs, "B0047 - 사이클 150에서의 모델 출력", [
        ("앙상블 RUL 예측: 61 사이클 (실제 RUL = 61 -> 완벽한 예측)", 0),
        ("90% 순응형 PI: [42, 84] 사이클 (실제값 61이 구간 내에 있음)", 0),
        ("CI 너비 = 42 사이클: 좁음 -> 낮은 인식적 불확실성 의미", 0),
        ("이상 점수: 12.4 (95 백분위수 임계값 훨씬 미만 -> 정상)", 0),
        ("고장 확률 P(20사이클 내 고장): 0.08 -> 낮은 위험", 0),
        ("감독자 판정: 합격 (RMSE=18.2, 커버리지=91%, 신뢰도=0.79)", 0),
        ("AI 설명 요약:", 0),
        ("'용량 1.74 Ah는 중기 SEI 안정화와 일치 [출처 1]'", 1),
        ("'감쇄율 -0.0031 Ah/사이클은 실온 셀의 정상 범위'", 1),
        ("'낮은 위험 - 즉각적인 개입 불필요'", 1),
    ])


def make_ppt(lang):
    prs = new_prs()
    if lang == "en":
        build_slides_en(prs)
        fname = "Battery_AI_Presentation_English.pptx"
    else:
        build_slides_ko(prs)
        fname = "Battery_AI_Presentation_Korean.pptx"
    path = OUT / fname
    prs.save(str(path))
    print(f"  Saved: {path}")


if __name__ == "__main__":
    print("Building PPT (English)...")
    make_ppt("en")
    print("Building PPT (Korean)...")
    make_ppt("ko")
    print("Done.")
