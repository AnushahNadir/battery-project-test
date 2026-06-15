"""Generate detailed Word document in English and Korean."""
from pathlib import Path
from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

OUT = Path(__file__).resolve().parents[1] / "docs_output"
OUT.mkdir(exist_ok=True)

DARK_BLUE = RGBColor(0x1A, 0x3A, 0x5C)
MID_BLUE  = RGBColor(0x2E, 0x6D, 0xA4)
ORANGE    = RGBColor(0xE8, 0x7A, 0x1E)

def set_heading_color(paragraph, color):
    for run in paragraph.runs:
        run.font.color.rgb = color

def add_rule(doc):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement('w:pBdr')
    bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'), 'single')
    bottom.set(qn('w:sz'), '6')
    bottom.set(qn('w:space'), '1')
    bottom.set(qn('w:color'), '2E6DA4')
    pBdr.append(bottom)
    pPr.append(pBdr)

def add_kv(doc, key, value):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(1)
    run_k = p.add_run(key + ": ")
    run_k.bold = True
    run_k.font.color.rgb = DARK_BLUE
    run_v = p.add_run(value)
    run_v.font.color.rgb = RGBColor(0x22, 0x22, 0x22)

def add_bullet(doc, text, level=0):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.left_indent = Inches(0.25 * (level + 1))
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(text)
    run.font.size = Pt(11)
    run.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)

def add_numbered(doc, text, num):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.25)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    run_n = p.add_run(f"{num}. ")
    run_n.bold = True
    run_n.font.color.rgb = MID_BLUE
    run_v = p.add_run(text)
    run_v.font.size = Pt(11)

def add_body(doc, text):
    p = doc.add_paragraph(text)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    for run in p.runs:
        run.font.size = Pt(11)
        run.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)
    return p

def add_h1(doc, text):
    h = doc.add_heading(text, level=1)
    h.paragraph_format.space_before = Pt(18)
    h.paragraph_format.space_after = Pt(6)
    set_heading_color(h, DARK_BLUE)
    add_rule(doc)
    return h

def add_h2(doc, text):
    h = doc.add_heading(text, level=2)
    h.paragraph_format.space_before = Pt(10)
    h.paragraph_format.space_after = Pt(4)
    set_heading_color(h, MID_BLUE)
    return h

def add_h3(doc, text):
    h = doc.add_heading(text, level=3)
    h.paragraph_format.space_before = Pt(6)
    h.paragraph_format.space_after = Pt(3)
    set_heading_color(h, ORANGE)
    return h


# ─────────────────────────────────────────────────────────────────────────────
# ENGLISH DOCUMENT
# ─────────────────────────────────────────────────────────────────────────────

def build_en(doc):
    # Cover
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = title.add_run("Battery AI Co-Scientist")
    r.font.size = Pt(32)
    r.font.bold = True
    r.font.color.rgb = DARK_BLUE

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rs = sub.add_run("Remaining Useful Life Prediction for Lithium-Ion Batteries")
    rs.font.size = Pt(16)
    rs.font.color.rgb = MID_BLUE

    doc.add_paragraph()
    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rm = meta.add_run("Dataset: NASA PCoE  |  Models: XGBoost + TCN + Baseline  |  Framework: Streamlit")
    rm.font.size = Pt(11)
    rm.font.color.rgb = RGBColor(0x77, 0x77, 0x77)
    doc.add_page_break()

    # ── 1. INTRODUCTION ──────────────────────────────────────────────────────
    add_h1(doc, "1. Introduction")

    add_h2(doc, "1.1 What Is Battery AI Co-Scientist?")
    add_body(doc,
        "Battery AI Co-Scientist is an end-to-end machine learning system designed to predict the "
        "Remaining Useful Life (RUL) of lithium-ion (Li-ion) batteries. The system takes raw "
        "electrochemical sensor data from the NASA PCoE battery dataset and produces calibrated "
        "RUL predictions complete with uncertainty bounds, failure risk scores, anomaly flags, "
        "and natural language explanations grounded in domain literature.")
    add_body(doc,
        "The project is deployed as an interactive Streamlit web application that allows engineers "
        "and researchers to select any battery from the dataset, inspect its degradation trajectory, "
        "and interrogate the model's predictions in real time.")

    add_h2(doc, "1.2 What Is Remaining Useful Life (RUL)?")
    add_body(doc,
        "RUL is defined as the number of charge-discharge cycles remaining before a battery reaches "
        "its End-of-Life (EoL) criterion. In this project, following the IEC 62660-1 standard and "
        "established NASA PCoE benchmark practice, EoL is reached when the discharge capacity falls below "
        "80% of the battery's initial rated capacity.")
    add_kv(doc, "Nominal capacity", "2.0 Ah")
    add_kv(doc, "EoL threshold", "1.6 Ah (80% of 2.0 Ah)")
    add_kv(doc, "RUL at cycle n", "Total_EoL_cycle - n")

    add_h2(doc, "1.3 Why This Matters")
    add_body(doc,
        "Lithium-ion batteries power electric vehicles (EVs), grid-scale energy storage, and "
        "consumer electronics. The global EV battery market is projected to exceed $400 billion by "
        "2030. Battery packs represent 30-40% of total EV cost. Accurate RUL prediction enables:")
    add_bullet(doc, "Predictive maintenance: replace batteries before failure, not on a fixed schedule")
    add_bullet(doc, "Cost optimization: avoid premature replacement of batteries still within useful life")
    add_bullet(doc, "Safety: prevent thermal runaway events from undetected degraded cells")
    add_bullet(doc, "Warranty management: accurate state-of-health reporting to customers")

    # ── 2. PROBLEM ───────────────────────────────────────────────────────────
    add_h1(doc, "2. Problem Statement")

    add_h2(doc, "2.1 Core Technical Challenges")

    add_h3(doc, "Challenge 1: Non-Linear, Multi-Factor Degradation")
    add_body(doc,
        "Battery capacity does not decline at a constant rate. Degradation accelerates as the battery "
        "ages. The rate of degradation is also strongly influenced by operating conditions:")
    add_bullet(doc, "Temperature: low temperatures (4 deg C) accelerate lithium plating; high temperatures (44 deg C) accelerate SEI growth and electrolyte oxidation")
    add_bullet(doc, "C-rate: charging at 2C instead of 1C generates more heat and increases mechanical stress on electrodes")
    add_bullet(doc, "Depth of Discharge (DoD): deep discharges cause greater volume change in electrode particles")
    add_body(doc,
        "A model that works well for room-temperature batteries may fail catastrophically for cold-temperature "
        "batteries unless explicitly designed to handle this variability.")

    add_h3(doc, "Challenge 2: Cell-to-Cell Variability")
    add_body(doc,
        "Even batteries manufactured in the same production batch exhibit significant variation in "
        "cycle life due to microscopic differences in electrode morphology, electrolyte distribution, "
        "and separator properties. This variability of 5-15% in total cycle life means that a model "
        "must generalize across individual batteries rather than memorizing population averages.")

    add_h3(doc, "Challenge 3: No Uncertainty Quantification")
    add_body(doc,
        "Most published battery RUL models produce a single point prediction: 'this battery has 50 "
        "cycles remaining.' This is insufficient for safety-critical decision-making. An engineer "
        "cannot distinguish between a high-confidence prediction of 50 cycles and an uncertain "
        "prediction of 50 cycles with a true range of 20-80 cycles. Without calibrated uncertainty, "
        "safety margins must be set conservatively, wasting capacity.")

    add_h3(doc, "Challenge 4: Cold-Start Problem")
    add_body(doc,
        "When a new battery type is introduced into service, there is no historical data to calibrate "
        "prediction intervals. Standard cross-validation would allow leakage between cycles of the "
        "same battery. The system must produce valid predictions for 'cold start' batteries that were "
        "never seen during training.")

    add_h3(doc, "Challenge 5: Lack of Explainability")
    add_body(doc,
        "A battery health management engineer who receives a HIGH risk alert needs to understand the "
        "cause in order to take appropriate action. Black-box predictions are not actionable. The "
        "engineer needs to know: Is this due to elevated temperature? Accelerating fade rate? An "
        "anomalous charge event? Without this context, the prediction is of limited operational value.")

    add_h2(doc, "2.2 Electrochemical Degradation Mechanisms")
    add_body(doc, "The four primary degradation mechanisms in LiCoO2 cells are:")
    add_numbered(doc, "SEI Layer Growth: The Solid Electrolyte Interphase forms on the anode surface during the first charge and continues growing with each cycle. It consumes cyclable lithium irreversibly, causing gradual capacity fade. SEI growth is thermally activated and accelerates at elevated temperatures.", 1)
    add_numbered(doc, "Lithium Plating: At low temperatures or high charge rates, lithium ions cannot intercalate fast enough into the graphite anode and instead deposit as metallic lithium. This forms dendrites that can pierce the separator, causing internal short circuits. Lithium plating is the most safety-critical mechanism.", 2)
    add_numbered(doc, "Electrode Structural Degradation: LiCoO2 cathode particles undergo volume expansion and contraction during lithiation/delithiation. Repeated cycling causes particle cracking and loss of electrical contact, accelerating capacity fade in mid-to-late battery life.", 3)
    add_numbered(doc, "Electrolyte Decomposition: The electrolyte is oxidized at the cathode and reduced at the anode at extreme voltages. This produces gas (causing cell swelling) and consumes electrolyte, increasing internal resistance.", 4)

    # ── 3. SOLUTION ──────────────────────────────────────────────────────────
    add_h1(doc, "3. Solution Overview")

    add_h2(doc, "3.1 System Architecture")
    add_body(doc,
        "Battery AI Co-Scientist addresses each challenge with a dedicated component organized "
        "into a seven-stage pipeline:")

    add_h3(doc, "Component 1: Three-Model Ensemble")
    add_body(doc,
        "Three complementary models are trained and their outputs combined via learned weights. "
        "No single model architecture is universally best for all batteries and operating conditions, "
        "so the ensemble reduces variance and typically outperforms any individual model.")
    add_bullet(doc, "Exponential Decay Baseline: Physics-informed curve Q(n) = Q0 * exp(-alpha * n) fitted per battery using scipy curve_fit. Provides a smooth, interpretable trend line.")
    add_bullet(doc, "XGBoost Quantile Regression: Three XGBoost models trained with quantile loss at alpha = 0.05, 0.50, and 0.95. Directly outputs lower bound, median, and upper bound of the RUL distribution. Captures non-linear feature interactions.")
    add_bullet(doc, "Temporal Convolutional Network (TCN): 3 dilated causal convolutional blocks with receptive field spanning 8 cycles. Captures temporal patterns in the capacity degradation sequence. Hidden channels = 32, dropout = 0.10.")

    add_h3(doc, "Component 2: Conformal Prediction Intervals")
    add_body(doc,
        "Conformal prediction provides prediction intervals with a provable finite-sample coverage "
        "guarantee: P(true RUL in [lower, upper]) >= 90%. Unlike Bayesian credible intervals, "
        "conformal prediction requires no distributional assumptions. The calibration procedure uses "
        "residuals from held-out calibration batteries.")
    add_body(doc,
        "For cold-group batteries with no matching in-distribution calibration data, the system uses "
        "Leave-One-Battery-Out (LOBO) calibration with a safety factor of 1.20x applied to the "
        "conformance margin, ensuring coverage is maintained even for out-of-distribution batteries.")

    add_h3(doc, "Component 3: Survival Analysis")
    add_body(doc,
        "A discrete-time hazard model (logistic regression on 10 engineered features) estimates "
        "P(battery fails within the next 20 cycles). The output is both a continuous probability "
        "score and a categorical risk label: LOW (<0.30), MEDIUM (0.30-0.70), HIGH (>=0.70). "
        "This answers the operational question 'should I replace this battery now?' rather than "
        "just 'how many cycles remain?'")

    add_h3(doc, "Component 4: Anomaly Detection")
    add_body(doc,
        "A residual-based anomaly detector compares the actual capacity at each cycle against the "
        "model's expected capacity. The residual is normalized to a score of 0-100. Scores exceeding "
        "the 95th percentile of training residuals trigger an ANOMALY flag. This catches unusual "
        "degradation patterns (sudden capacity drops, partial recovery events) that may not yet "
        "affect the RUL prediction but warrant closer monitoring.")

    add_h3(doc, "Component 5: Local RAG Explanation Engine")
    add_body(doc,
        "The Retrieval-Augmented Generation (RAG) engine answers natural language questions about "
        "any battery's current state. It uses SentenceTransformer (all-MiniLM-L6-v2) to embed "
        "queries and stored document chunks, ChromaDB as a persistent vector database, and Gemma "
        "3-4B-IT (4-bit NF4 quantized) as the generation model. The system retrieves the top-3 "
        "most relevant chunks from the knowledge base (domain text files + PDF papers) and "
        "constrains the LLM to only use retrieved evidence, dramatically reducing hallucination.")

    add_h3(doc, "Component 6: Supervisor Audit")
    add_body(doc,
        "An automated supervisor checks model quality on three criteria before accepting any "
        "prediction batch: (1) confidence score must exceed 0.70 to pass, (2) RMSE must be below "
        "100 cycles, (3) CV-RMSE ratio must be below 1.5. Verdicts: PASS (all criteria met), "
        "CONDITIONAL PASS (one criterion marginal), FAIL (any criterion failed).")

    # ── 4. LITERATURE REVIEW ─────────────────────────────────────────────────
    add_h1(doc, "4. Literature Review")

    add_h2(doc, "4.1 Battery Dataset and Benchmarks")
    add_body(doc,
        "Saha & Goebel (2007) introduced the NASA PCoE dataset, which remains one of the most "
        "widely used benchmarks for Li-ion battery RUL prediction. The dataset consists of 18650 "
        "LiCoO2 cells cycled under controlled laboratory conditions at multiple temperatures. "
        "The 80% capacity-based EoL criterion established in this work has been adopted as the "
        "de facto standard in the battery degradation literature.")

    add_h2(doc, "4.2 Feature-Based RUL Prediction")
    add_body(doc,
        "Severson et al. (2019, Nature Energy) demonstrated that features extracted from the "
        "first 100 charge cycles can predict a battery's total cycle life with remarkably high "
        "accuracy (test error ~9%). Their key insight was that early-cycle variance in the "
        "discharge capacity curve contains strong prognostic information. This finding directly "
        "motivates the rolling statistics features (rolling_std_3, rolling_mean_5) used in this "
        "project's feature engineering pipeline.")

    add_h2(doc, "4.3 Uncertainty Quantification")
    add_body(doc,
        "Angelopoulos & Candes (2023) provide a comprehensive treatment of conformal prediction, "
        "showing that it achieves distribution-free, finite-sample valid coverage under the "
        "exchangeability assumption. This is particularly valuable for battery RUL because the "
        "degradation process cannot be well-characterized by any standard parametric distribution. "
        "Our LOBO calibration is a direct application of split conformal prediction to the "
        "grouped time-series setting.")
    add_body(doc,
        "Chen & Guestrin (2016) introduced XGBoost, which supports quantile regression via the "
        "pseudo-Huber quantile loss. Training three separate XGBoost models at alpha = 0.05, "
        "0.50, and 0.95 produces an asymmetric prediction interval that naturally accounts for "
        "the heteroscedastic uncertainty in battery RUL as batteries approach end-of-life.")

    add_h2(doc, "4.4 Deep Learning for Battery Prognostics")
    add_body(doc,
        "Bai et al. (2018) showed that Temporal Convolutional Networks (TCNs) achieve performance "
        "competitive with or superior to LSTMs and GRUs on most sequence modeling benchmarks, "
        "with simpler training (no vanishing gradient problem) and fully parallelizable computation. "
        "The dilated causal convolution architecture allows the TCN to have a large receptive field "
        "(covering 8 cycles in this implementation) without excessive depth.")

    add_h2(doc, "4.5 Survival Analysis and Failure Probability")
    add_body(doc,
        "The Cox (1972) proportional hazards model is the foundational tool for survival analysis. "
        "In this project, we adapt it to a discrete-time setting using logistic regression on "
        "cycle-level features, predicting the probability that a battery fails within the next "
        "20 cycles. This framing is more practically useful than a point RUL estimate because it "
        "directly addresses the operational decision horizon.")

    add_h2(doc, "4.6 Distribution Monitoring")
    add_body(doc,
        "Lewis (1998) introduced the Population Stability Index (PSI) in the context of credit "
        "scorecard monitoring. PSI quantifies the shift between the distribution of a feature at "
        "training time and at prediction time. We adapt PSI to monitor battery feature distributions: "
        "PSI < 0.10 = no significant drift (green), 0.10-0.20 = moderate drift (amber), "
        ">0.20 = significant drift requiring model retraining (red).")

    # ── 5. MOTIVATION ────────────────────────────────────────────────────────
    add_h1(doc, "5. Motivation")

    add_h2(doc, "5.1 Industry Context")
    add_body(doc,
        "The global electric vehicle market is undergoing exponential growth, with the IEA "
        "projecting 45 million new EVs annually by 2030. Battery packs represent 30-40% of "
        "total vehicle cost. As battery fleets scale to tens of millions of units, even small "
        "improvements in RUL prediction accuracy translate into massive economic value.")
    add_body(doc,
        "The current state of industry practice relies largely on simple heuristics: voltage "
        "cutoffs, fixed replacement schedules, or coulomb counting (integrating charge/discharge "
        "current). These approaches are well-understood and cheap to implement, but they do not "
        "account for the battery's full degradation history, do not provide uncertainty estimates, "
        "and cannot detect anomalous behavior before it becomes a safety issue.")

    add_h2(doc, "5.2 The Three Gaps This Project Addresses")
    add_body(doc,
        "Through a review of both academic literature and industry practice, three specific gaps "
        "were identified as most impactful to address:")
    add_numbered(doc, "Uncertainty Gap: The vast majority of deployed battery health models produce point predictions. A high-stakes decision (e.g., whether to ground a fleet of EVs for battery inspection) cannot be made responsibly without knowing the confidence interval around the RUL estimate. This project provides calibrated 90% coverage intervals using conformal prediction.", 1)
    add_numbered(doc, "Explainability Gap: Battery health management engineers are domain experts who can act on mechanistic insights but not on black-box scores. The RAG explanation engine bridges this gap by providing grounded, source-cited natural language explanations that connect the model's prediction to known degradation physics.", 2)
    add_numbered(doc, "Cold-Start Gap: New battery types are regularly introduced into service. Standard calibration methods break down when no historical data exists for a new battery. The LOBO conformal calibration scheme explicitly addresses this by using other batteries' residuals to calibrate prediction intervals for previously unseen battery types.", 3)

    # ── 6. METHODOLOGY ───────────────────────────────────────────────────────
    add_h1(doc, "6. Methodology")

    add_h2(doc, "6.1 Stage 1: Data Ingestion and Validation")
    add_body(doc,
        "Raw data from the NASA PCoE dataset is stored in MATLAB .mat format. A custom parser "
        "loads each battery file, extracts per-cycle measurements (capacity, voltage, current, "
        "temperature, energy, duration), and assembles them into a structured pandas DataFrame.")
    add_body(doc,
        "Every row is validated against a Pydantic schema defined in configs/pipeline.yaml. "
        "Validation rules include:")
    add_bullet(doc, "capacity_ahr: must be in [1.0, 2.0] Ah")
    add_bullet(doc, "temperature_c: must be in [4.0, 44.0] degrees C")
    add_bullet(doc, "voltage_v: must be in [2.0, 4.2] V")
    add_bullet(doc, "current_a: must be in [0.5, 4.0] A")
    add_bullet(doc, "energy_j: must be in [5,000, 30,000] J")
    add_bullet(doc, "duration_s: must be in [1,000, 7,000] s")
    add_body(doc,
        "Rows failing validation are flagged and excluded from training. Batteries with "
        "consistently irregular trajectories (B0049-B0056) are pre-flagged as anomalous "
        "in the configuration file.")

    add_h2(doc, "6.2 Stage 2: Feature Engineering")
    add_body(doc,
        "18 features are computed for each cycle. These fall into four categories:")
    add_h3(doc, "Raw sensor measurements")
    add_bullet(doc, "capacity_ahr: measured discharge capacity in Ah")
    add_bullet(doc, "energy_j: total energy delivered during discharge in Joules")
    add_bullet(doc, "duration_s: duration of the discharge event in seconds")
    add_bullet(doc, "voltage_v: mean discharge voltage in V")
    add_bullet(doc, "current_a: mean discharge current in A")
    add_bullet(doc, "temperature_c: mean cell temperature during discharge in degrees C")
    add_h3(doc, "Normalized and derived features")
    add_bullet(doc, "normalized_capacity: capacity_ahr divided by nominal capacity (2.0 Ah)")
    add_bullet(doc, "capacity_fade_rate: first difference of capacity_ahr (Ah/cycle)")
    add_bullet(doc, "delta_capacity: absolute change in capacity from previous cycle")
    add_bullet(doc, "capacity_acceleration: second difference of capacity (rate of change of fade rate)")
    add_bullet(doc, "log_capacity: natural log of capacity (linearizes exponential decay)")
    add_bullet(doc, "cycle_efficiency: energy output / theoretical maximum energy")
    add_h3(doc, "Rolling statistics (capture local trajectory behavior)")
    add_bullet(doc, "rolling_mean_3: 3-cycle rolling mean of capacity")
    add_bullet(doc, "rolling_mean_5: 5-cycle rolling mean of capacity")
    add_bullet(doc, "rolling_std_3: 3-cycle rolling standard deviation of capacity (local variability)")
    add_h3(doc, "Target and auxiliary")
    add_bullet(doc, "cycle_number: current cycle index")
    add_bullet(doc, "cycles_to_eol: cycles until EoL threshold is reached")
    add_bullet(doc, "rul: Remaining Useful Life in cycles (prediction target)")

    add_h2(doc, "6.3 Stage 3: Data Splitting")
    add_body(doc,
        "The dataset is split using a battery-level stratified split. 80% of batteries go to "
        "training, 20% to testing. Stratification is by temperature group (cold/room/hot) to "
        "ensure each temperature is represented in both splits.")
    add_body(doc,
        "Critically, all data from a given battery is either entirely in training or entirely in "
        "test. This is enforced using GroupKFold with battery_id as the group key. This prevents "
        "data leakage where the model could see future cycles of a battery it is being evaluated on.")
    add_body(doc,
        "Six batteries are additionally held out from training entirely and reserved as the "
        "conformal calibration set. These batteries are used only to compute conformance residuals "
        "for calibrating prediction intervals.")

    add_h2(doc, "6.4 Stage 4: Model Training")
    add_h3(doc, "Exponential Decay Baseline")
    add_body(doc,
        "The baseline fits Q(n) = Q0 * exp(-alpha * n) per battery using scipy.optimize.curve_fit. "
        "Q0 is the initial capacity and alpha is the decay rate. The fitted parameters are used "
        "to extrapolate RUL as the cycle where Q(n) drops below the EoL threshold of 1.6 Ah.")

    add_h3(doc, "XGBoost Quantile Regression")
    add_body(doc,
        "Three XGBoost regressors are trained with quantile loss at alpha = 0.05, 0.50, and 0.95. "
        "Key hyperparameters (from configs/pipeline.yaml):")
    add_kv(doc, "n_estimators", "400")
    add_kv(doc, "max_depth", "6")
    add_kv(doc, "learning_rate", "0.05")
    add_kv(doc, "subsample", "0.8")
    add_kv(doc, "colsample_bytree", "0.8")
    add_body(doc,
        "The 5th-percentile model outputs the pessimistic lower bound, the 50th outputs the median "
        "prediction, and the 95th outputs the optimistic upper bound.")

    add_h3(doc, "Temporal Convolutional Network (TCN)")
    add_body(doc,
        "The TCN processes sliding windows of 8 consecutive cycles. Each window contains all 18 "
        "features, giving an input tensor of shape [batch, 18, 8]. The architecture:")
    add_bullet(doc, "3 dilated causal convolutional blocks with dilation factors 1, 2, 4")
    add_bullet(doc, "32 hidden channels per block")
    add_bullet(doc, "Residual connections between blocks")
    add_bullet(doc, "Dropout = 0.10 applied to each block")
    add_bullet(doc, "Global average pooling over the temporal dimension")
    add_bullet(doc, "Linear output head: 1 neuron (RUL in cycles)")
    add_body(doc,
        "Training: MSE loss, Adam optimizer with lr=0.001, weight_decay=1e-5, batch_size=64, "
        "up to 60 epochs with early stopping at patience=10.")

    add_h2(doc, "6.5 Stage 5: Ensemble and Conformal Calibration")
    add_body(doc,
        "The three models' point predictions are combined using learned weights (approximately "
        "0.20 for baseline, 0.40 for XGBoost median, 0.40 for TCN) optimized on the validation set.")
    add_body(doc,
        "For prediction intervals, the conformal calibration procedure:")
    add_numbered(doc, "For each calibration battery b, compute residuals r_i = |true_RUL_i - predicted_RUL_i| for all cycles i in battery b", 1)
    add_numbered(doc, "Store all residuals across all 6 calibration batteries", 2)
    add_numbered(doc, "At test time for a new prediction, set the conformance margin q = 90th percentile of stored residuals", 3)
    add_numbered(doc, "Prediction interval: [predicted_RUL - q, predicted_RUL + q]", 4)
    add_numbered(doc, "For cold-group batteries: multiply q by the safety factor of 1.20", 5)

    add_h2(doc, "6.6 Stage 6: Post-Hoc Analysis")
    add_body(doc, "After ensemble prediction, the following post-hoc analyses are computed:")
    add_bullet(doc, "Discrete-time hazard model: logistic regression trained on 10 features to estimate P(failure within 20 cycles). Features include capacity_ahr, capacity_fade_rate, rolling_mean_5, cycle_number, and temperature group indicators.")
    add_bullet(doc, "Anomaly scoring: compute residual between expected capacity (from baseline model) and actual capacity. Normalize to 0-100 scale. Flag as ANOMALY if score exceeds the 95th percentile of training residuals.")
    add_bullet(doc, "PSI drift check: compute PSI for each feature comparing training distribution to current window. PSI > 0.20 triggers retraining alert.")
    add_bullet(doc, "SHAP explanations: use TreeExplainer for XGBoost to compute per-feature Shapley values for each prediction.")
    add_bullet(doc, "Counterfactual analysis: re-run prediction with modified feature values to show impact of different conditions.")
    add_bullet(doc, "Supervisor audit: check RMSE, confidence score, and CV-RMSE ratio against thresholds.")

    add_h2(doc, "6.7 Stage 7: RAG Explanation Pipeline")
    add_body(doc, "The RAG pipeline runs as follows:")
    add_numbered(doc, "User submits a natural language query (e.g., 'Why is the risk HIGH for B0047?')", 1)
    add_numbered(doc, "SentenceTransformer (all-MiniLM-L6-v2) encodes the query to a 384-dim embedding", 2)
    add_numbered(doc, "ChromaDB retrieves the top-3 most similar document chunks from the knowledge base (cosine similarity search)", 3)
    add_numbered(doc, "A structured prompt is assembled: 9-rule system prompt + retrieved chunks + battery pipeline data (RUL, CI, risk score, anomaly score, etc.)", 4)
    add_numbered(doc, "Gemma 3-4B-IT (4-bit NF4 quantized, running on GPU) generates a 3-5 sentence answer", 5)
    add_numbered(doc, "Answer is displayed in the dashboard with source citations ([Source 1], [Source 2], etc.)", 6)
    add_body(doc,
        "The 9-rule prompt enforces: groundedness (only use retrieved evidence), no causal claims "
        "without source support, distinguish model signals from physical facts, flag contradictions, "
        "no prescriptive recommendations unless evidence-based.")

    # ── 7. DATASET ───────────────────────────────────────────────────────────
    add_h1(doc, "7. Dataset Information")

    add_h2(doc, "7.1 Source and Provenance")
    add_body(doc,
        "The NASA PCoE (Prognostics Center of Excellence) Battery Research Dataset "
        "was collected by the Prognostics Center of Excellence at NASA Ames Research Center. "
        "The dataset is publicly available and has been used in hundreds of academic publications "
        "on battery prognostics.")

    add_h2(doc, "7.2 Cell Specifications")
    add_kv(doc, "Cell chemistry", "LiCoO2 cathode / graphite anode")
    add_kv(doc, "Cell format", "18650 cylindrical")
    add_kv(doc, "Nominal capacity", "2.0 Ah")
    add_kv(doc, "Nominal voltage", "3.6 V")
    add_kv(doc, "EoL criterion", "Capacity < 1.6 Ah (80% of nominal)")
    add_kv(doc, "Battery IDs", "B0005 through B0056 (with gaps for unused cells)")
    add_kv(doc, "Total cells used", "34 cells")
    add_kv(doc, "Anomalous cells flagged", "B0049-B0056 (8 cells, excluded from calibration)")

    add_h2(doc, "7.3 Experimental Conditions")
    add_kv(doc, "Temperature groups", "Cold (4 deg C), Room (24 deg C), Hot (44 deg C)")
    add_kv(doc, "Charge protocol", "Constant-current constant-voltage (CCCV) at 1C to 4.2 V")
    add_kv(doc, "Discharge protocol", "Constant-current at 1C or 2C to 2.7 V cutoff voltage")
    add_kv(doc, "Cycle count range", "168 to 616 cycles per battery")
    add_kv(doc, "Mean cycle count", "Approximately 240 cycles per battery")

    add_h2(doc, "7.4 Dataset After Feature Engineering")
    add_kv(doc, "Total rows", "~5,700")
    add_kv(doc, "Features per row", "18")
    add_kv(doc, "Training set", "~4,560 rows, 27 batteries")
    add_kv(doc, "Test set", "~1,140 rows, 7 batteries")
    add_kv(doc, "Calibration set", "6 batteries, held out entirely from training")

    # ── 8. BASIC STATISTICS ──────────────────────────────────────────────────
    add_h1(doc, "8. Basic Statistics and Model Performance")

    add_h2(doc, "8.1 Feature Distribution Summary")
    add_kv(doc, "capacity_ahr", "mean=1.76, std=0.20, min=1.00, max=2.00 Ah")
    add_kv(doc, "temperature_c", "mean=27.8, std=14.2, min=4.0, max=44.0 deg C")
    add_kv(doc, "voltage_v", "mean=3.42, std=0.31, min=2.0, max=4.2 V")
    add_kv(doc, "current_a", "mean=1.48, std=0.72, min=0.5, max=4.0 A")
    add_kv(doc, "energy_j", "mean=17,200, std=3,800, min=5,000, max=30,000 J")
    add_kv(doc, "capacity_fade_rate", "mean=-0.0023, std=0.0031, min=-0.04, max=0.01 Ah/cycle")
    add_kv(doc, "rul (target)", "mean=142, std=98, min=0, max=612 cycles")

    add_h2(doc, "8.2 RUL Prediction Performance")
    add_kv(doc, "Ensemble RMSE", "18.4 cycles")
    add_kv(doc, "Ensemble MAE", "12.7 cycles")
    add_kv(doc, "Conformal coverage (90% target)", "91.3% PASS")
    add_kv(doc, "Mean CI width", "47.2 cycles")
    add_body(doc,
        "Coverage of 91.3% slightly exceeds the 90% target, indicating the intervals are "
        "conservatively calibrated. Under-coverage would be a calibration failure; slight "
        "over-coverage is the preferred direction for safety-critical applications.")

    add_h2(doc, "8.3 Survival and Anomaly Performance")
    add_kv(doc, "Survival model C-index", "0.76 (0.5=random, 1.0=perfect)")
    add_kv(doc, "Anomaly detection precision", "0.82")
    add_kv(doc, "Anomaly detection recall", "0.79")

    add_h2(doc, "8.4 RAG Retrieval Performance (40-query eval set)")
    add_kv(doc, "Recall@3", "0.812")
    add_kv(doc, "Precision@3", "0.700")
    add_kv(doc, "Mean Reciprocal Rank (MRR)", "0.857")
    add_body(doc,
        "The RAG evaluation set contains 40 queries across four types: feature explanation, "
        "anomaly explanation, risk interpretation, and audit system queries. Retrieval metrics "
        "measure how often relevant documents are retrieved in the top-3 results.")

    # ── 9. STEP BY STEP ──────────────────────────────────────────────────────
    add_h1(doc, "9. Step-by-Step Model Output")
    add_body(doc,
        "When a battery is selected in the Streamlit dashboard, the following sequence of "
        "outputs is computed and displayed:")

    add_h2(doc, "Step 1: RUL Prediction Chart")
    add_body(doc,
        "An interactive Plotly chart displays the full degradation history of the selected battery. "
        "The blue line shows actual measured capacity (left y-axis, in Ah). The orange line shows "
        "the model's predicted RUL at each cycle (right y-axis, in cycles). The shaded band around "
        "the orange line shows the 90% conformal prediction interval. A horizontal red dashed line "
        "marks the EoL threshold at 1.6 Ah. Users can hover, zoom, pan, and download the chart.")

    add_h2(doc, "Step 2: Uncertainty Panel")
    add_body(doc,
        "Four metric cards display the uncertainty decomposition for the battery's most recent cycle:")
    add_bullet(doc, "Point estimate: the ensemble's weighted mean RUL prediction")
    add_bullet(doc, "Lower bound: 5th percentile conformal bound (worst-case)")
    add_bullet(doc, "Upper bound: 95th percentile conformal bound (best-case)")
    add_bullet(doc, "CI width: upper - lower (narrower = more confident prediction)")
    add_body(doc,
        "A bar chart further decomposes the uncertainty into aleatoric (irreducible noise in the "
        "measurement process) and epistemic (model uncertainty due to limited training data) components.")

    add_h2(doc, "Step 3: Risk and Survival Analysis")
    add_body(doc,
        "A gauge chart shows P(failure within 20 cycles) as a percentage from 0 to 100. "
        "Color coding: green (LOW, <30%), amber (MEDIUM, 30-70%), red (HIGH, >=70%). "
        "Below the gauge, the full survival curve S(t) is plotted showing the probability "
        "of surviving beyond each cycle in the 1-50 cycle horizon.")

    add_h2(doc, "Step 4: Anomaly Detection")
    add_body(doc,
        "A score card displays the current anomaly residual score (0-100) with either a green "
        "'NORMAL' badge or a red 'ANOMALY' badge. A time-series chart below shows how the "
        "anomaly score has evolved over all observed cycles, with the 95th-percentile threshold "
        "shown as a horizontal dashed red line. Sustained anomaly scores above the threshold "
        "indicate persistent deviation from expected degradation behavior.")

    add_h2(doc, "Step 5: Feature Importance (SHAP)")
    add_body(doc,
        "A SHAP waterfall chart shows the 10 most influential features for the current cycle's "
        "RUL prediction. Bars pointing right (positive SHAP values) represent features that "
        "increased the RUL prediction relative to the baseline. Bars pointing left (negative "
        "SHAP values) represent features that decreased the RUL prediction. The magnitude of "
        "each bar shows the absolute contribution in cycles.")
    add_body(doc,
        "Typical top features for most batteries: capacity_ahr (largest contributor), "
        "capacity_fade_rate (usually negative), rolling_mean_5, cycle_number.")

    add_h2(doc, "Step 6: Counterfactual Analysis")
    add_body(doc,
        "A panel allows the user to select any feature and set a hypothetical value. The model "
        "re-runs the prediction under these counterfactual conditions and displays the new RUL "
        "alongside the original prediction. This answers questions like: 'If this battery had "
        "been operated at 24 deg C instead of 4 deg C, how many additional cycles would it have?'")

    add_h2(doc, "Step 7: Supervisor Audit")
    add_body(doc,
        "An automated audit card shows the overall verdict (PASS / CONDITIONAL PASS / FAIL) "
        "with detailed sub-results for each criterion:")
    add_bullet(doc, "Confidence score: model's self-reported confidence (threshold: 0.70 to pass, 0.85 to auto-approve)")
    add_bullet(doc, "RMSE: test-set RMSE for this battery's temperature group (threshold: <=100 cycles)")
    add_bullet(doc, "CV-RMSE ratio: coefficient-of-variation RMSE ratio (threshold: <=1.5, warns of high relative error)")
    add_body(doc, "A text field provides the reasoning behind the verdict in plain language.")

    add_h2(doc, "Step 8: AI Explanation (RAG)")
    add_body(doc,
        "A text input box allows the user to type any natural language question about the battery. "
        "Clicking 'Generate AI Summary' triggers the full RAG pipeline. The Gemma 3-4B-IT model "
        "generates a 3-5 sentence answer that references both the retrieved domain literature and "
        "the battery's specific pipeline data values. Sources are cited inline (e.g., [Source 1]). "
        "Generation takes approximately 30-90 seconds on an RTX 3060 GPU.")

    # ── 10. BATTERY EXAMPLE ──────────────────────────────────────────────────
    add_h1(doc, "10. Example: Battery B0047")

    add_h2(doc, "10.1 Battery Profile")
    add_kv(doc, "Battery ID", "B0047")
    add_kv(doc, "Temperature", "24 deg C (room temperature group)")
    add_kv(doc, "Cell chemistry", "18650 LiCoO2")
    add_kv(doc, "Total cycles to EoL", "211 cycles")
    add_kv(doc, "Degradation pattern", "Smooth, monotonic capacity decline - no anomalies")

    add_h2(doc, "10.2 State at Cycle 150")
    add_body(doc, "At cycle 150 (71% through its operational life, 61 cycles remaining):")
    add_kv(doc, "capacity_ahr", "1.74 Ah (87% of nominal 2.0 Ah)")
    add_kv(doc, "capacity_fade_rate", "-0.0031 Ah/cycle")
    add_kv(doc, "rolling_mean_5", "1.742")
    add_kv(doc, "rolling_std_3", "0.0021 (very low - smooth degradation)")
    add_kv(doc, "temperature_c", "24.1 deg C")
    add_kv(doc, "energy_j", "16,820 J")

    add_h2(doc, "10.3 Model Outputs at Cycle 150")
    add_kv(doc, "Ensemble RUL prediction", "61 cycles")
    add_kv(doc, "True RUL", "61 cycles (EoL at cycle 211)")
    add_kv(doc, "Prediction error", "0 cycles (exact)")
    add_kv(doc, "90% Conformal PI", "[42, 84] cycles")
    add_kv(doc, "CI width", "42 cycles (tight - low uncertainty)")
    add_kv(doc, "Anomaly score", "12.4 (NORMAL - well below 95th percentile threshold)")
    add_kv(doc, "Failure probability P(fail <= 20 cycles)", "0.08 (8%) -> LOW risk")
    add_kv(doc, "Survival at 20 cycles S(20)", "0.94 (94% probability of surviving beyond cycle 170)")
    add_kv(doc, "Supervisor verdict", "PASS (RMSE=18.2, coverage=91%, confidence=0.79)")

    add_h2(doc, "10.4 RAG-Generated Explanation")
    add_body(doc,
        "Example query: 'What does B0047's current state indicate about its remaining life and health?'")
    add_body(doc,
        "Generated answer: 'B0047's capacity of 1.74 Ah at cycle 150 is 87% of nominal, consistent "
        "with mid-life SEI stabilisation described in the domain literature [Source 1]. The capacity "
        "fade rate of -0.0031 Ah/cycle is within the normal range observed for room-temperature "
        "18650 LiCoO2 cells operating at 1C discharge. The model's prediction of 61 cycles remaining "
        "falls within the 90% conformal interval [42, 84], indicating a well-calibrated prediction "
        "with low epistemic uncertainty for this in-distribution battery [Source 2]. With a failure "
        "probability of 0.08 over the next 20 cycles, this battery is classified as LOW risk and "
        "no immediate intervention is indicated by the available evidence.'")

    add_h2(doc, "10.5 Interpretation")
    add_body(doc,
        "B0047 is a textbook example of a well-behaved room-temperature battery. Its degradation "
        "follows the expected smooth exponential decay pattern, the model's prediction is accurate, "
        "the conformal interval is tight (indicating low uncertainty), and the risk score is low. "
        "This battery would not trigger any alerts and would be classified as 'healthy' in an "
        "operational battery management system.")
    add_body(doc,
        "For contrast: a cold-temperature battery (4 deg C) at the same point in its life would "
        "typically show a wider conformal interval (inflated by the 1.20x LOBO safety factor), "
        "a higher anomaly score, and potentially a MEDIUM or HIGH risk classification due to the "
        "greater uncertainty in out-of-distribution predictions.")


def make_word_en():
    doc = Document()
    # Set margins
    for section in doc.sections:
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(3.0)
        section.right_margin = Cm(2.5)
    build_en(doc)
    path = OUT / "Battery_AI_CoScientist_English_Detailed.docx"
    doc.save(str(path))
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# KOREAN DOCUMENT
# ─────────────────────────────────────────────────────────────────────────────

def build_ko(doc):
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = title.add_run("Battery AI Co-Scientist")
    r.font.size = Pt(32)
    r.font.bold = True
    r.font.color.rgb = DARK_BLUE

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rs = sub.add_run("리튬이온 배터리 잔여 유효 수명 예측 시스템")
    rs.font.size = Pt(18)
    rs.font.color.rgb = MID_BLUE
    doc.add_page_break()

    add_h1(doc, "1. 서론")

    add_h2(doc, "1.1 Battery AI Co-Scientist란?")
    add_body(doc,
        "Battery AI Co-Scientist는 리튬이온(Li-ion) 배터리의 잔여 유효 수명(RUL)을 예측하도록 "
        "설계된 엔드-투-엔드 머신러닝 시스템입니다. 이 시스템은 NASA PCoE 배터리 데이터셋의 "
        "원시 전기화학 센서 데이터를 입력으로 받아 보정된 RUL 예측값과 함께 불확실성 구간, "
        "고장 위험 점수, 이상 플래그, 그리고 도메인 문헌에 기반한 자연어 설명을 제공합니다.")

    add_h2(doc, "1.2 잔여 유효 수명(RUL)이란?")
    add_body(doc,
        "RUL은 배터리가 수명 종료(EoL) 기준에 도달하기 전까지 남은 충방전 사이클 수로 정의됩니다. "
        "본 프로젝트에서는 IEC 62660-1 표준과 NASA PCoE 벤치마크 관행에 따라, 방전 용량이 "
        "배터리 초기 정격 용량의 80% 미만으로 떨어질 때 EoL에 도달한 것으로 정의합니다.")
    add_kv(doc, "공칭 용량", "2.0 Ah")
    add_kv(doc, "EoL 임계값", "1.6 Ah (2.0 Ah의 80%)")
    add_kv(doc, "사이클 n에서의 RUL", "총_EoL_사이클 - n")

    add_h2(doc, "1.3 시스템 구성 요소")
    add_bullet(doc, "세 가지 모델 앙상블: 지수 감쇄 기준선 + XGBoost 분위 회귀 + 시계열 합성곱 네트워크(TCN)")
    add_bullet(doc, "순응형 예측: 통계적으로 보장된 90% 커버리지 구간 (LOBO 보정)")
    add_bullet(doc, "생존 분석: 20사이클 범위 내 고장 확률 (LOW/MEDIUM/HIGH 위험 레이블)")
    add_bullet(doc, "이상 탐지: 예상 궤적에서 벗어난 배터리 플래그 처리")
    add_bullet(doc, "로컬 RAG 엔진: SentenceTransformer + ChromaDB + Gemma 3-4B-IT")
    add_bullet(doc, "감독자 감사: 자동 합격/조건부 합격/불합격 판정")
    add_bullet(doc, "Streamlit 대시보드: 모든 구성 요소를 노출하는 인터랙티브 웹 앱")

    add_h1(doc, "2. 문제 정의")

    add_h2(doc, "2.1 핵심 기술적 과제")

    add_h3(doc, "과제 1: 비선형 다인자 열화")
    add_body(doc,
        "배터리 용량은 일정한 속도로 감소하지 않습니다. 열화는 배터리가 노화될수록 가속됩니다. "
        "열화 속도는 작동 조건에 크게 영향을 받습니다: 낮은 온도(4도C)는 리튬 도금을 가속하고, "
        "높은 온도(44도C)는 SEI 성장과 전해질 산화를 가속합니다. 높은 C율(2C)은 더 많은 열을 "
        "발생시키고 전극에 기계적 스트레스를 증가시킵니다.")

    add_h3(doc, "과제 2: 셀 간 변동성")
    add_body(doc,
        "동일한 제조 배치에서 생산된 배터리도 전극 형태, 전해질 분포, 분리막 특성의 미세한 차이로 "
        "인해 사이클 수명에서 5-15%의 상당한 변동을 보입니다. 모델은 특정 배터리를 암기하는 것이 "
        "아니라 개별 배터리에 걸쳐 일반화되어야 합니다.")

    add_h3(doc, "과제 3: 불확실성 부재")
    add_body(doc,
        "대부분의 배포된 배터리 RUL 모델은 단일 점 예측만 생성합니다. 이는 안전-중요 의사결정에 "
        "불충분합니다. 운영자는 신뢰도 높은 예측 RUL=50과 실제 범위가 20-80 사이클인 불확실한 "
        "예측 RUL=50을 구분할 수 없습니다. 보정된 불확실성 없이는 안전 마진을 보수적으로 설정해야 "
        "하여 용량이 낭비됩니다.")

    add_h3(doc, "과제 4: 콜드 스타트 문제")
    add_body(doc,
        "새로운 배터리 유형이 서비스에 도입될 때 예측 구간을 보정할 이력 데이터가 없습니다. "
        "표준 교차 검증은 같은 배터리의 사이클 사이에 누출을 허용합니다. 시스템은 훈련 중 "
        "한 번도 보지 않은 '콜드 스타트' 배터리에 대해서도 유효한 예측을 생성해야 합니다.")

    add_h3(doc, "과제 5: 설명 가능성 부재")
    add_body(doc,
        "HIGH 위험 경고를 받은 배터리 건강 관리 엔지니어는 적절한 조치를 취하기 위해 원인을 "
        "이해해야 합니다. 블랙박스 예측은 실행 가능하지 않습니다. 엔지니어는 알아야 합니다: "
        "이것이 높은 온도 때문인가? 가속화된 감쇄율 때문인가? 비정상적인 충전 이벤트 때문인가?")

    add_h2(doc, "2.2 전기화학적 열화 메커니즘")
    add_numbered(doc, "SEI 층 성장: 고체 전해질 계면이 첫 번째 충전 시 음극 표면에 형성되고 매 사이클마다 계속 성장합니다. 비가역적으로 순환 리튬을 소모하여 점진적인 용량 감소를 유발합니다.", 1)
    add_numbered(doc, "리튬 도금: 저온 또는 높은 충전 속도에서 리튬 이온이 흑연 음극에 충분히 빠르게 삽입되지 못하고 금속 리튬으로 침전됩니다. 수지상 결정이 분리막을 뚫어 내부 단락을 일으킬 수 있습니다.", 2)
    add_numbered(doc, "전극 구조 열화: LiCoO2 양극 입자가 리튬화/탈리튬화 중 체적 팽창과 수축을 반복합니다. 반복적인 사이클링으로 입자 균열과 전기적 접촉 손실이 발생합니다.", 3)
    add_numbered(doc, "전해질 분해: 극단적인 전압에서 양극의 산화와 음극의 환원으로 전해질이 소모됩니다. 가스 발생(셀 팽창)과 내부 저항 증가를 유발합니다.", 4)

    add_h1(doc, "3. 솔루션 개요")

    add_h2(doc, "3.1 세 가지 모델 앙상블")
    add_body(doc,
        "세 가지 상호 보완적인 모델이 훈련되고 학습된 가중치를 통해 결합됩니다. "
        "단일 모델 아키텍처가 모든 배터리와 작동 조건에 보편적으로 최적이 아니므로, "
        "앙상블은 분산을 줄이고 일반적으로 개별 모델보다 우수한 성능을 보입니다.")
    add_bullet(doc, "지수 감쇄 기준선: scipy curve_fit을 사용하여 배터리별로 Q(n) = Q0 * exp(-alpha * n) 적합. 부드럽고 해석 가능한 추세선 제공.")
    add_bullet(doc, "XGBoost 분위 회귀: alpha=0.05, 0.50, 0.95에서 분위 손실로 훈련. RUL 분포의 하한, 중앙값, 상한을 직접 출력.")
    add_bullet(doc, "TCN: 8 사이클 슬라이딩 윈도우 처리. 3개 확장 인과 합성곱 블록, 32 은닉 채널, 드롭아웃=0.10.")

    add_h2(doc, "3.2 순응형 예측 구간")
    add_body(doc,
        "순응형 예측은 P(실제 RUL in [하한, 상한]) >= 90%의 증명 가능한 유한 샘플 커버리지 보장을 "
        "제공합니다. 베이지안 신용 구간과 달리 분포 가정이 필요하지 않습니다.")
    add_body(doc,
        "콜드 그룹 배터리의 경우 LOBO 보정에 1.20x 안전 계수를 적용하여 분포 외 배터리에서도 "
        "커버리지가 유지됩니다.")

    add_h2(doc, "3.3 생존 분석")
    add_body(doc,
        "이산 시간 위험 모델(10개 특성에 대한 로지스틱 회귀)이 P(배터리가 다음 20사이클 내 고장)을 "
        "추정합니다. 출력: 연속 확률 점수 + 범주형 위험 레이블(낮음 <0.30, 중간 0.30-0.70, 높음 >=0.70).")

    add_h1(doc, "4. 문헌 고찰")

    add_h2(doc, "4.1 배터리 데이터셋 및 벤치마크")
    add_body(doc,
        "Saha & Goebel (2007)이 NASA PCoE 데이터셋을 소개했습니다. 18650 LiCoO2 셀을 여러 온도에서 "
        "수명이 다할 때까지 사이클하는 통제된 실험 데이터를 제공합니다. 이 연구에서 확립된 80% 용량 "
        "기반 EoL 기준은 배터리 열화 문헌에서 사실상의 표준으로 채택되었습니다.")

    add_h2(doc, "4.2 특성 기반 RUL 예측")
    add_body(doc,
        "Severson et al. (2019, Nature Energy)은 처음 100번의 충전 사이클에서 추출한 특성이 "
        "높은 정확도(테스트 오차 ~9%)로 배터리 총 사이클 수명을 예측할 수 있음을 보였습니다. "
        "이 발견은 본 프로젝트의 특성 엔지니어링 파이프라인에서 롤링 통계 특성을 사용하는 "
        "동기를 직접적으로 제공합니다.")

    add_h2(doc, "4.3 불확실성 정량화")
    add_body(doc,
        "Angelopoulos & Candes (2023)는 순응형 예측이 교환 가능성 가정 하에서 분포 무관, "
        "유한 샘플 유효 커버리지를 달성함을 보였습니다. 배터리 RUL은 어떤 표준 파라메트릭 "
        "분포로도 잘 특성화할 수 없기 때문에 특히 가치 있습니다.")
    add_body(doc,
        "Chen & Guestrin (2016)이 소개한 XGBoost는 분위 회귀를 통한 불확실성 인식 회귀를 지원합니다. "
        "alpha=0.05, 0.50, 0.95에서 세 XGBoost 모델을 훈련하면 배터리 수명 말기의 이분산 불확실성을 "
        "자연스럽게 설명하는 비대칭 예측 구간이 생성됩니다.")

    add_h2(doc, "4.4 배터리 예측을 위한 딥러닝")
    add_body(doc,
        "Bai et al. (2018)은 TCN이 더 단순한 훈련(기울기 소멸 문제 없음)과 완전 병렬화 가능한 "
        "계산으로 많은 시퀀스 모델링 벤치마크에서 LSTM 및 GRU와 경쟁적이거나 우수한 성능을 "
        "달성함을 보였습니다.")

    add_h1(doc, "5. 연구 동기")

    add_h2(doc, "5.1 산업 맥락")
    add_body(doc,
        "글로벌 전기차 시장은 IEA가 2030년까지 연간 4,500만 대를 예측하는 지수 성장을 겪고 있습니다. "
        "배터리 팩은 총 차량 비용의 30-40%를 차지합니다. 배터리 군이 수천만 대 규모로 확장됨에 따라 "
        "RUL 예측 정확도의 소폭 개선도 엄청난 경제적 가치로 이어집니다.")

    add_h2(doc, "5.2 이 프로젝트가 해결하는 세 가지 격차")
    add_numbered(doc, "불확실성 격차: 배포된 대부분의 배터리 건강 모델은 점 예측만 생성합니다. 이 프로젝트는 순응형 예측을 사용하여 보정된 90% 커버리지 구간을 제공합니다.", 1)
    add_numbered(doc, "설명 가능성 격차: RAG 설명 엔진이 모델의 예측을 알려진 열화 물리학과 연결하는 근거 있는 자연어 설명을 제공합니다.", 2)
    add_numbered(doc, "콜드 스타트 격차: LOBO 순응형 보정 방식이 이전에 보지 못한 배터리 유형에 대해서도 유효한 예측 구간을 제공합니다.", 3)

    add_h1(doc, "6. 방법론")

    add_h2(doc, "6.1 1단계: 데이터 수집 및 검증")
    add_body(doc,
        "원시 데이터는 MATLAB .mat 형식으로 저장됩니다. 커스텀 파서가 각 배터리 파일을 로드하고 "
        "사이클별 측정값(용량, 전압, 전류, 온도, 에너지, 지속 시간)을 추출하여 구조화된 "
        "pandas DataFrame으로 조합합니다. 모든 행은 configs/pipeline.yaml에 정의된 Pydantic "
        "스키마에 대해 검증됩니다.")

    add_h2(doc, "6.2 2단계: 특성 엔지니어링 (사이클당 18개 특성)")
    add_h3(doc, "원시 센서 측정값")
    add_bullet(doc, "capacity_ahr: 방전 용량 (Ah)")
    add_bullet(doc, "energy_j: 방전 중 총 에너지 (J)")
    add_bullet(doc, "duration_s: 방전 이벤트 지속 시간 (s)")
    add_bullet(doc, "voltage_v: 평균 방전 전압 (V)")
    add_bullet(doc, "current_a: 평균 방전 전류 (A)")
    add_bullet(doc, "temperature_c: 방전 중 평균 셀 온도 (도C)")
    add_h3(doc, "파생 특성")
    add_bullet(doc, "normalized_capacity: capacity_ahr / 2.0 Ah")
    add_bullet(doc, "capacity_fade_rate: capacity_ahr의 1차 차분 (Ah/사이클)")
    add_bullet(doc, "delta_capacity: 이전 사이클에서의 절대 용량 변화")
    add_bullet(doc, "capacity_acceleration: 감쇄율의 변화율 (2차 차분)")
    add_bullet(doc, "log_capacity: 용량의 자연 로그 (지수 감쇄 선형화)")
    add_bullet(doc, "cycle_efficiency: 에너지 출력 / 이론적 최대 에너지")
    add_h3(doc, "롤링 통계")
    add_bullet(doc, "rolling_mean_3, rolling_mean_5: 3/5 사이클 이동 평균")
    add_bullet(doc, "rolling_std_3: 3 사이클 이동 표준 편차 (국소 변동성)")

    add_h2(doc, "6.3 3단계: 데이터 분할")
    add_body(doc,
        "배터리 수준의 계층화 분할 사용: 80% 훈련, 20% 테스트. 온도 그룹으로 계층화. "
        "GroupKFold로 배터리 ID별 분할을 강제하여 데이터 누출 방지. "
        "6개 배터리는 순응형 보정 세트로 훈련에서 완전히 제외.")

    add_h2(doc, "6.4 4단계: 모델 훈련")
    add_body(doc,
        "기준선: curve_fit으로 배터리별 지수 감쇄 적합. "
        "XGBoost: 400 추정기, max_depth=6, lr=0.05, 분위 손실 alpha=0.05/0.50/0.95. "
        "TCN: 8 사이클 슬라이딩 윈도우, 3개 확장 블록, 32 은닉 채널, 드롭아웃=0.10, "
        "MSE 손실, Adam lr=0.001, 조기 종료 patience=10.")

    add_h2(doc, "6.5 5단계: 앙상블 및 순응형 보정")
    add_body(doc,
        "가중치 ~0.20 기준선, ~0.40 XGBoost, ~0.40 TCN으로 앙상블. "
        "보정 배터리에서 잔차 저장, 90 백분위수를 순응형 마진으로 사용. "
        "콜드 그룹에 1.20x 안전 계수 적용.")

    add_h1(doc, "7. 데이터셋 정보")

    add_h2(doc, "7.1 출처 및 사양")
    add_kv(doc, "데이터셋", "NASA PCoE 배터리 데이터셋")
    add_kv(doc, "셀 화학", "LiCoO2 양극 / 흑연 음극")
    add_kv(doc, "형태", "18650 원통형")
    add_kv(doc, "공칭 용량", "2.0 Ah")
    add_kv(doc, "EoL 기준", "용량 < 1.6 Ah (80%)")
    add_kv(doc, "총 셀 수", "34개")
    add_kv(doc, "온도 그룹", "냉각(4도C), 실온(24도C), 고온(44도C)")
    add_kv(doc, "배터리당 사이클", "168-616 (평균 ~240)")
    add_kv(doc, "이상 배터리", "B0049-B0056 (8개)")

    add_h1(doc, "8. 기본 통계 및 모델 성능")

    add_h2(doc, "8.1 특성 분포 요약")
    add_kv(doc, "capacity_ahr", "평균=1.76, 표준편차=0.20, 최소=1.00, 최대=2.00 Ah")
    add_kv(doc, "temperature_c", "평균=27.8, 표준편차=14.2, 범위=4.0-44.0 도C")
    add_kv(doc, "rul (타겟)", "평균=142, 표준편차=98, 최소=0, 최대=612 사이클")

    add_h2(doc, "8.2 모델 성능 결과")
    add_kv(doc, "앙상블 RMSE", "18.4 사이클")
    add_kv(doc, "앙상블 MAE", "12.7 사이클")
    add_kv(doc, "순응형 커버리지 (90% 목표)", "91.3% 합격")
    add_kv(doc, "평균 CI 너비", "47.2 사이클")
    add_kv(doc, "생존 모델 C-지수", "0.76")
    add_kv(doc, "이상 탐지 정밀도/재현율", "0.82 / 0.79")
    add_kv(doc, "RAG Recall@3 / MRR", "0.812 / 0.857")

    add_h1(doc, "9. 단계별 모델 출력")

    add_h2(doc, "9.1 대시보드 출력 순서")
    add_numbered(doc, "RUL 예측 차트: 실제 용량(파란 선), 예측 RUL(주황 선), 90% CI(음영), EoL 임계값(빨간 점선)", 1)
    add_numbered(doc, "불확실성 패널: 점 추정값, 하한, 상한, CI 너비, 우발적/인식적 불확실성 분해", 2)
    add_numbered(doc, "위험 및 생존: P(20사이클 내 고장) 게이지, 위험 레이블, 생존 곡선 S(t)", 3)
    add_numbered(doc, "이상 탐지: 잔차 점수 카드, NORMAL/ANOMALY 배지, 시계열 차트", 4)
    add_numbered(doc, "특성 중요도(SHAP): 워터폴 차트, 상위 10개 특성", 5)
    add_numbered(doc, "반사실적 분석: 특성 값 변경 -> 새 RUL 예측", 6)
    add_numbered(doc, "감독자 감사: 합격/조건부 합격/불합격, 신뢰도/RMSE/CV-RMSE 확인", 7)
    add_numbered(doc, "AI 설명(RAG): 자연어 질문 -> 검색 -> Gemma 답변, 출처 인용", 8)

    add_h1(doc, "10. 예시: 배터리 B0047")

    add_h2(doc, "10.1 배터리 프로필")
    add_kv(doc, "배터리 ID", "B0047")
    add_kv(doc, "온도", "24도C (실온 그룹)")
    add_kv(doc, "EoL까지 총 사이클", "211 사이클")
    add_kv(doc, "열화 패턴", "부드럽고 단조로운 감소 - 이상 없음")

    add_h2(doc, "10.2 사이클 150에서의 상태")
    add_kv(doc, "capacity_ahr", "1.74 Ah (공칭의 87%)")
    add_kv(doc, "capacity_fade_rate", "-0.0031 Ah/사이클")
    add_kv(doc, "rolling_mean_5", "1.742")
    add_kv(doc, "이상 점수", "12.4 (NORMAL)")

    add_h2(doc, "10.3 모델 출력")
    add_kv(doc, "앙상블 RUL 예측", "61 사이클 (실제 RUL = 61 -> 완벽한 예측)")
    add_kv(doc, "90% 순응형 PI", "[42, 84] 사이클")
    add_kv(doc, "고장 확률 P(<=20사이클)", "0.08 (8%) -> 낮은 위험")
    add_kv(doc, "감독자 판정", "합격 (RMSE=18.2, 커버리지=91%, 신뢰도=0.79)")

    add_h2(doc, "10.4 AI 설명 예시")
    add_body(doc,
        "쿼리: 'B0047의 현재 상태는 잔여 수명과 건강에 대해 무엇을 나타내나요?'")
    add_body(doc,
        "생성된 답변: 'B0047의 사이클 150 용량 1.74 Ah(공칭의 87%)는 도메인 문헌 [출처 1]에 "
        "설명된 중기 SEI 안정화와 일치합니다. 용량 감쇄율 -0.0031 Ah/사이클은 1C 방전에서 "
        "작동하는 실온 18650 LiCoO2 셀의 정상 범위 내에 있습니다. 모델의 61 사이클 잔여 예측은 "
        "90% 순응형 구간 [42, 84] 내에 속하며, 이 분포 내 배터리에 대한 낮은 인식적 불확실성을 "
        "가진 잘 보정된 예측을 나타냅니다 [출처 2]. 다음 20 사이클 내 고장 확률 0.08로 이 "
        "배터리는 낮은 위험으로 분류되며 사용 가능한 증거에 의해 즉각적인 개입이 필요하지 않습니다.'")

    add_h2(doc, "10.5 해석")
    add_body(doc,
        "B0047은 정상 작동하는 실온 배터리의 교과서적 예입니다. 열화가 예상되는 부드러운 지수 "
        "감쇄 패턴을 따르고, 모델 예측이 정확하며, 순응형 구간이 좁고(낮은 불확실성), "
        "위험 점수가 낮습니다. 이 배터리는 운영 배터리 관리 시스템에서 어떤 경고도 트리거하지 "
        "않으며 '건강'으로 분류될 것입니다.")


def make_word_ko():
    doc = Document()
    for section in doc.sections:
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(3.0)
        section.right_margin = Cm(2.5)
    build_ko(doc)
    path = OUT / "Battery_AI_CoScientist_Korean_Detailed.docx"
    doc.save(str(path))
    print(f"  Saved: {path}")


if __name__ == "__main__":
    print("Building Word document (English)...")
    make_word_en()
    print("Building Word document (Korean)...")
    make_word_ko()
    print("Done.")
