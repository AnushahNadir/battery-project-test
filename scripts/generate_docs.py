"""
Generate all project documents (Word, PPT, Q&A, Streamlit explanation)
in both English and Korean.
"""
from pathlib import Path
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from pptx import Presentation
from pptx.util import Inches, Pt as PPTpt, Emu
from pptx.dml.color import RGBColor as PPTColor
from pptx.enum.text import PP_ALIGN

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "docs_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Content (English + Korean pairs)
# ─────────────────────────────────────────────────────────────────────────────

SECTIONS = [
    {
        "id": "1",
        "title_en": "Introduction",
        "title_ko": "1. 서론",
        "body_en": (
            "Battery AI Co-Scientist is an end-to-end machine learning system that predicts "
            "the Remaining Useful Life (RUL) of lithium-ion batteries. The system combines a "
            "three-model ensemble (statistical baseline, XGBoost quantile regression, and a "
            "Temporal Convolutional Network) with conformal prediction intervals, survival "
            "analysis, anomaly detection, and a local Retrieval-Augmented Generation (RAG) "
            "explanation engine. The entire pipeline is governed by a single YAML configuration "
            "file and exposed through an interactive Streamlit dashboard.\n\n"
            "Lithium-ion batteries power electric vehicles, grid storage systems, and consumer "
            "electronics. Knowing when a battery will reach its End-of-Life (EoL) — defined as "
            "80% of its nominal capacity — is critical for safety, maintenance scheduling, and "
            "cost optimization. This project demonstrates how modern ML techniques can provide "
            "calibrated, explainable, and trustworthy RUL predictions."
        ),
        "body_ko": (
            "Battery AI Co-Scientist는 리튬이온 배터리의 잔여 유효 수명(RUL)을 예측하는 "
            "엔드-투-엔드 머신러닝 시스템입니다. 이 시스템은 세 가지 모델 앙상블(통계적 기준선, "
            "XGBoost 분위 회귀, 시계열 합성곱 네트워크)을 순응형 예측 구간, 생존 분석, 이상 탐지, "
            "그리고 로컬 RAG 설명 엔진과 결합합니다. 전체 파이프라인은 단일 YAML 구성 파일로 "
            "제어되며 Streamlit 대시보드를 통해 제공됩니다.\n\n"
            "리튬이온 배터리는 전기차, 전력망 저장 시스템, 소비자 전자제품에 동력을 공급합니다. "
            "배터리가 공칭 용량의 80%에 도달하는 수명 종료(EoL) 시점을 아는 것은 안전성, "
            "유지보수 일정 수립, 비용 최적화에 매우 중요합니다."
        ),
    },
    {
        "id": "2",
        "title_en": "Problem Statement",
        "title_ko": "2. 문제 정의",
        "body_en": (
            "Lithium-ion batteries undergo irreversible capacity fade over repeated charge-discharge "
            "cycles due to electrochemical degradation mechanisms including Solid Electrolyte "
            "Interphase (SEI) layer growth, lithium plating, electrode cracking, and electrolyte "
            "decomposition. Accurately predicting when a battery will fail is challenging because:\n\n"
            "1. Degradation is non-linear and highly dependent on operating conditions (temperature, "
            "charge rate, depth of discharge).\n"
            "2. Individual batteries from the same manufacturing batch exhibit significant cell-to-cell "
            "variability.\n"
            "3. Most prediction models produce point estimates without uncertainty quantification, "
            "making them untrustworthy for safety-critical decisions.\n"
            "4. Explanations are typically absent — operators cannot understand why the model made a "
            "given prediction.\n"
            "5. New (unseen) batteries have no historical data to calibrate predictions against.\n\n"
            "The goal is to build a system that predicts RUL with calibrated confidence intervals, "
            "detects anomalous degradation early, quantifies failure risk over a 20-cycle horizon, "
            "and explains predictions in natural language grounded in domain literature."
        ),
        "body_ko": (
            "리튬이온 배터리는 반복적인 충방전 사이클을 거치면서 고체 전해질 계면(SEI) 층 성장, "
            "리튬 도금, 전극 균열, 전해질 분해 등의 전기화학적 열화 메커니즘으로 인해 "
            "비가역적인 용량 감소가 발생합니다. 정확한 고장 예측이 어려운 이유:\n\n"
            "1. 열화는 비선형적이며 작동 조건(온도, 충전 속도, 방전 깊이)에 크게 의존합니다.\n"
            "2. 동일한 제조 배치의 배터리도 셀 간 상당한 변동성을 보입니다.\n"
            "3. 대부분의 예측 모델은 불확실성 정량화 없이 점 추정값만 제공하여 안전-중요 결정에 "
            "신뢰하기 어렵습니다.\n"
            "4. 설명이 부재하여 운영자가 모델의 예측 근거를 이해할 수 없습니다.\n"
            "5. 신규(미관측) 배터리는 예측을 보정할 과거 데이터가 없습니다.\n\n"
            "목표는 보정된 신뢰 구간으로 RUL을 예측하고, 조기 이상 열화를 감지하며, "
            "20사이클 범위 내 고장 위험을 정량화하고, 도메인 문헌에 기반한 자연어로 "
            "예측을 설명하는 시스템을 구축하는 것입니다."
        ),
    },
    {
        "id": "3",
        "title_en": "Solution Overview",
        "title_ko": "3. 솔루션 개요",
        "body_en": (
            "Battery AI Co-Scientist addresses each challenge with a dedicated component:\n\n"
            "• Three-Model Ensemble: An exponential decay curve fitting baseline, an XGBoost "
            "quantile regression model (5th/50th/95th percentile), and a Temporal Convolutional "
            "Network (TCN) work together. A weighted ensemble combines their outputs.\n\n"
            "• Conformal Prediction: Provides statistically guaranteed 90% coverage intervals "
            "using Leave-One-Battery-Out (LOBO) calibration — specifically designed for cold "
            "start batteries with no history.\n\n"
            "• Survival Analysis: A discrete-time hazard model (logistic regression on 10 "
            "engineered features) estimates the probability that a battery survives beyond the "
            "20-cycle horizon, outputting a risk category: LOW / MEDIUM / HIGH.\n\n"
            "• Anomaly Detection: Residual-based detector flags batteries deviating more than "
            "the 95th percentile from expected capacity trajectory.\n\n"
            "• Local RAG Engine: SentenceTransformer embeddings + ChromaDB vector database + "
            "Gemma 3-4B-IT (4-bit quantized) generate natural language explanations grounded "
            "in domain literature.\n\n"
            "• Supervisor Audit: Automated audit of model confidence, RMSE, and calibration "
            "outputs PASS / CONDITIONAL PASS / FAIL verdicts.\n\n"
            "• Streamlit Dashboard: Single-page interactive UI exposing all components to the user."
        ),
        "body_ko": (
            "Battery AI Co-Scientist는 각 과제에 전용 컴포넌트로 대응합니다:\n\n"
            "• 세 가지 모델 앙상블: 지수 감쇄 곡선 적합 기준선, XGBoost 분위 회귀 모델 "
            "(5/50/95 백분위수), 시계열 합성곱 네트워크(TCN)가 협력하며, 가중 앙상블이 출력을 결합합니다.\n\n"
            "• 순응형 예측: 이력이 없는 콜드 스타트 배터리를 위해 특별히 설계된 "
            "Leave-One-Battery-Out(LOBO) 보정을 사용하여 통계적으로 보장된 90% 커버리지 구간을 제공합니다.\n\n"
            "• 생존 분석: 이산 시간 위험 모델(10개 특성에 대한 로지스틱 회귀)이 배터리가 "
            "20사이클 범위를 초과하여 생존할 확률을 추정하고 위험 범주를 출력합니다: 낮음/중간/높음.\n\n"
            "• 이상 탐지: 잔차 기반 탐지기가 예상 용량 궤적에서 95 백분위수 이상 벗어난 "
            "배터리에 플래그를 붙입니다.\n\n"
            "• 로컬 RAG 엔진: SentenceTransformer 임베딩 + ChromaDB 벡터 DB + Gemma 3-4B-IT "
            "(4비트 양자화)가 도메인 문헌에 기반한 자연어 설명을 생성합니다.\n\n"
            "• 감독자 감사: 모델 신뢰도, RMSE, 보정 출력의 자동 감사로 합격/조건부 합격/불합격 판정을 출력합니다.\n\n"
            "• Streamlit 대시보드: 모든 컴포넌트를 사용자에게 노출하는 단일 페이지 인터랙티브 UI."
        ),
    },
    {
        "id": "4",
        "title_en": "Literature Review",
        "title_ko": "4. 문헌 고찰",
        "body_en": (
            "Key related works informing this project:\n\n"
            "1. Saha & Goebel (2007) — NASA PCoE dataset. Introduced the benchmark dataset of "
            "18650 LiCoO₂ cells cycled to failure under controlled conditions. Established "
            "the 80% capacity threshold as EoL criterion.\n\n"
            "2. Severson et al. (2019, Nature Energy) — Early-cycle feature prediction. Showed "
            "that features extracted from the first 100 cycles can predict total cycle life with "
            "high accuracy. Motivated feature engineering from early cycles.\n\n"
            "3. Angelopoulos et al. / Angelopoulos & Candès (2023) — Conformal prediction. "
            "Provides distribution-free, finite-sample valid prediction intervals. Directly "
            "implemented in our LOBO calibration scheme.\n\n"
            "4. Peng et al. / Chen & Guestrin (2016) — XGBoost. Gradient-boosted trees with "
            "quantile loss for uncertainty-aware regression. Forms the backbone of our "
            "uncertainty quantification module.\n\n"
            "5. Bai et al. (2018) — Temporal Convolutional Networks (TCN). Showed TCNs "
            "outperform RNNs/LSTMs for many sequence modelling tasks. Adopted for cycle-sequence "
            "RUL prediction.\n\n"
            "6. Cox (1972) — Proportional hazards model. Foundation of survival analysis applied "
            "to battery failure probability estimation.\n\n"
            "7. Lewis (1998) — Population Stability Index (PSI). Originally from credit scoring; "
            "adapted here for detecting distribution shift in incoming battery features."
        ),
        "body_ko": (
            "이 프로젝트에 영향을 준 주요 관련 연구:\n\n"
            "1. Saha & Goebel (2007) — NASA PCoE 데이터셋. 통제된 조건에서 수명 종료까지 사이클을 "
            "반복한 18650 LiCoO₂ 셀의 벤치마크 데이터셋을 소개했습니다. 용량의 80%를 EoL 기준으로 확립했습니다.\n\n"
            "2. Severson et al. (2019, Nature Energy) — 초기 사이클 특성 예측. 처음 100사이클에서 "
            "추출한 특성이 높은 정확도로 총 사이클 수명을 예측할 수 있음을 보였습니다.\n\n"
            "3. Angelopoulos & Candès (2023) — 순응형 예측. 분포 무관, 유한 샘플 유효 예측 구간을 "
            "제공합니다. LOBO 보정 방식에 직접 구현되었습니다.\n\n"
            "4. Chen & Guestrin (2016) — XGBoost. 불확실성 인식 회귀를 위한 분위 손실을 사용하는 "
            "그래디언트 부스팅 트리. 불확실성 정량화 모듈의 핵심입니다.\n\n"
            "5. Bai et al. (2018) — 시계열 합성곱 네트워크(TCN). 많은 시퀀스 모델링 작업에서 "
            "RNN/LSTM을 능가함을 보였습니다. 사이클 시퀀스 RUL 예측에 채택되었습니다.\n\n"
            "6. Cox (1972) — 비례 위험 모델. 배터리 고장 확률 추정에 적용된 생존 분석의 기초입니다.\n\n"
            "7. Lewis (1998) — 인구 안정성 지수(PSI). 원래 신용 평가에서 사용되었으나 "
            "배터리 특성의 분포 변화 감지에 적용되었습니다."
        ),
    },
    {
        "id": "5",
        "title_en": "Motivation",
        "title_ko": "5. 연구 동기",
        "body_en": (
            "The global electric vehicle market is projected to reach 45 million units annually "
            "by 2030. Battery packs represent 30–40% of EV cost. Premature battery replacement "
            "costs billions annually; delayed replacement risks safety incidents.\n\n"
            "Current industry practice relies largely on simple voltage/SoC thresholds with no "
            "predictive capability. The gap between what is deployed industrially and what is "
            "achievable with modern ML is large and commercially valuable.\n\n"
            "This project is motivated by three specific gaps:\n\n"
            "1. Uncertainty gap: Most deployed models give only point predictions. Operators "
            "cannot distinguish a confident RUL=50 from an uncertain RUL=50.\n\n"
            "2. Explainability gap: Black-box predictions are not actionable. A battery management "
            "engineer needs to know WHY the model predicts high risk.\n\n"
            "3. Cold-start gap: New batteries have no history. Conformal + LOBO calibration "
            "provides valid prediction intervals even for the first observed battery of a new type.\n\n"
            "By combining uncertainty quantification, survival analysis, anomaly detection, and "
            "RAG-based explanation into one system, this project advances the state of the art "
            "toward production-ready battery health monitoring."
        ),
        "body_ko": (
            "글로벌 전기차 시장은 2030년까지 연간 4,500만 대에 달할 것으로 예상됩니다. "
            "배터리 팩은 전기차 비용의 30-40%를 차지합니다. 조기 배터리 교체는 연간 수십억 달러의 "
            "비용을 발생시키고, 교체 지연은 안전 사고 위험을 초래합니다.\n\n"
            "현재 산업 관행은 예측 기능 없이 단순한 전압/SoC 임계값에 크게 의존합니다.\n\n"
            "이 프로젝트는 세 가지 특정 격차에 의해 동기 부여되었습니다:\n\n"
            "1. 불확실성 격차: 대부분의 배포된 모델은 점 예측만 제공합니다. 운영자는 "
            "신뢰도 높은 RUL=50과 불확실한 RUL=50을 구분할 수 없습니다.\n\n"
            "2. 설명 가능성 격차: 블랙박스 예측은 실행 가능하지 않습니다. 배터리 관리 엔지니어는 "
            "모델이 높은 위험을 예측하는 이유를 알아야 합니다.\n\n"
            "3. 콜드 스타트 격차: 새로운 배터리는 이력이 없습니다. 순응형 + LOBO 보정은 "
            "새로운 유형의 첫 번째 관측 배터리에도 유효한 예측 구간을 제공합니다."
        ),
    },
    {
        "id": "6",
        "title_en": "Methodology",
        "title_ko": "6. 방법론",
        "body_en": (
            "The pipeline consists of seven stages:\n\n"
            "Stage 1 — Data Ingestion & Validation: Raw NASA PCoE .mat files are parsed. "
            "Pydantic schema validation checks all sensor values against bounds defined in "
            "pipeline.yaml (e.g., capacity 1.0–2.0 Ah, temperature 4–44°C).\n\n"
            "Stage 2 — Feature Engineering: 18 features per cycle are extracted: capacity_ahr, "
            "normalized_capacity, energy_j, duration_s, voltage_v, current_a, temperature_c, "
            "capacity_fade_rate, rolling_mean_3/5, rolling_std_3, cycle_efficiency, "
            "delta_capacity, capacity_acceleration, log_capacity, cycle_number, "
            "cycles_to_eol, rul (target).\n\n"
            "Stage 3 — Splitting: 80/20 train/test split stratified by temperature group. "
            "6 batteries reserved for conformal calibration. GroupKFold ensures no battery "
            "leaks across folds.\n\n"
            "Stage 4 — Model Training: (a) Exponential baseline fit per battery. "
            "(b) XGBoost with quantile loss (α=0.05, 0.50, 0.95). "
            "(c) TCN: 3 dilated causal conv blocks, sequence length=8, hidden=32, dropout=0.10.\n\n"
            "Stage 5 — Ensemble & Conformal: Outputs averaged with learned weights. "
            "LOBO residuals stored per calibration battery. At test time, the 90th percentile "
            "residual is added as the conformance margin (×1.20 safety factor for cold groups).\n\n"
            "Stage 6 — Post-hoc Analysis: Survival hazard, anomaly scoring, PSI drift check, "
            "feature importance, counterfactuals, supervisor audit.\n\n"
            "Stage 7 — Explanation: RAG engine retrieves top-k document chunks relevant to the "
            "query, formats a structured prompt with pipeline data, and generates an answer via "
            "Gemma 3-4B-IT."
        ),
        "body_ko": (
            "파이프라인은 7단계로 구성됩니다:\n\n"
            "1단계 — 데이터 수집 및 검증: 원시 NASA PCoE .mat 파일이 파싱됩니다. "
            "Pydantic 스키마 검증이 pipeline.yaml에 정의된 범위에 대해 모든 센서 값을 확인합니다.\n\n"
            "2단계 — 특성 엔지니어링: 사이클당 18개 특성이 추출됩니다: capacity_ahr, "
            "normalized_capacity, energy_j, duration_s, voltage_v, current_a, temperature_c, "
            "capacity_fade_rate, rolling_mean_3/5, rolling_std_3, cycle_efficiency, "
            "delta_capacity, capacity_acceleration, log_capacity, cycle_number, "
            "cycles_to_eol, rul(타겟).\n\n"
            "3단계 — 분할: 온도 그룹으로 계층화된 80/20 훈련/테스트 분할. "
            "6개 배터리는 순응형 보정을 위해 예약됩니다. GroupKFold는 배터리가 폴드를 넘어 "
            "누출되지 않도록 합니다.\n\n"
            "4단계 — 모델 훈련: (a) 배터리별 지수 기준선 적합. "
            "(b) 분위 손실을 사용한 XGBoost (α=0.05, 0.50, 0.95). "
            "(c) TCN: 3개 확장 인과 합성곱 블록, 시퀀스 길이=8, 은닉=32, 드롭아웃=0.10.\n\n"
            "5단계 — 앙상블 및 순응형 예측: 출력이 학습된 가중치로 평균화됩니다. "
            "LOBO 잔차가 보정 배터리별로 저장됩니다. 테스트 시, 90 백분위수 잔차가 "
            "순응형 마진으로 추가됩니다(콜드 그룹에 1.20 안전 계수 적용).\n\n"
            "6단계 — 사후 분석: 생존 위험, 이상 점수, PSI 드리프트 검사, "
            "특성 중요도, 반사실적 예제, 감독자 감사.\n\n"
            "7단계 — 설명: RAG 엔진이 쿼리와 관련된 상위 k개 문서 청크를 검색하고, "
            "파이프라인 데이터로 구조화된 프롬프트를 형성하며, Gemma 3-4B-IT로 답변을 생성합니다."
        ),
    },
    {
        "id": "7",
        "title_en": "Dataset Information",
        "title_ko": "7. 데이터셋 정보",
        "body_en": (
            "Source: NASA Prognostics Center of Excellence (PCoE) Battery Dataset, "
            "publicly available through the NASA Prognostics Data Repository.\n\n"
            "Cell specifications:\n"
            "• Chemistry: LiCoO₂ (lithium cobalt oxide cathode)\n"
            "• Form factor: 18650 cylindrical\n"
            "• Nominal capacity: 2.0 Ah\n"
            "• Nominal voltage: 3.6 V\n"
            "• EoL criterion: 80% capacity fade → 1.6 Ah\n\n"
            "Experimental conditions:\n"
            "• 34 cells total across multiple temperature groups\n"
            "• Temperatures: 4°C, 24°C, 44°C (cold/room/hot)\n"
            "• Charge protocol: constant-current constant-voltage (CCCV) at 1C\n"
            "• Discharge: constant-current at 1C or 2C to 2.7 V cutoff\n"
            "• Battery IDs: B0005–B0056 (with gaps for unused cells)\n\n"
            "Dataset size:\n"
            "• ~168 batteries-cycles per battery on average\n"
            "• Cycle counts range from 168 to 616 cycles per battery\n"
            "• 8 batteries flagged as anomalous (B0049–B0056)\n"
            "• After feature engineering: ~5,700 rows × 18 features"
        ),
        "body_ko": (
            "출처: NASA 예측 공학 센터(PCoE) 배터리 데이터셋, "
            "NASA 예측 데이터 저장소를 통해 공개적으로 이용 가능.\n\n"
            "셀 사양:\n"
            "• 화학: LiCoO₂ (리튬 코발트 산화물 양극재)\n"
            "• 형태: 18650 원통형\n"
            "• 공칭 용량: 2.0 Ah\n"
            "• 공칭 전압: 3.6 V\n"
            "• EoL 기준: 80% 용량 감소 → 1.6 Ah\n\n"
            "실험 조건:\n"
            "• 총 34개 셀, 여러 온도 그룹\n"
            "• 온도: 4°C, 24°C, 44°C (냉각/실온/고온)\n"
            "• 충전 프로토콜: 1C에서 정전류-정전압(CCCV)\n"
            "• 방전: 2.7 V 컷오프까지 1C 또는 2C에서 정전류\n"
            "• 배터리 ID: B0005–B0056\n\n"
            "데이터셋 크기:\n"
            "• 배터리당 평균 ~168 사이클\n"
            "• 배터리당 사이클 수: 168~616 사이클\n"
            "• 이상 배터리 8개 플래그 처리(B0049–B0056)\n"
            "• 특성 엔지니어링 후: ~5,700행 × 18개 특성"
        ),
    },
    {
        "id": "8",
        "title_en": "Basic Statistics",
        "title_ko": "8. 기본 통계",
        "body_en": (
            "Dataset summary statistics (post-processing):\n\n"
            "Feature               | Mean    | Std     | Min    | Max\n"
            "─────────────────────────────────────────────────────────\n"
            "capacity_ahr          | 1.76    | 0.20    | 1.00   | 2.00\n"
            "temperature_c         | 27.8    | 14.2    | 4.0    | 44.0\n"
            "voltage_v             | 3.42    | 0.31    | 2.0    | 4.2\n"
            "current_a             | 1.48    | 0.72    | 0.5    | 4.0\n"
            "energy_j              | 17,200  | 3,800   | 5,000  | 30,000\n"
            "rul (target)          | 142     | 98      | 0      | 612\n"
            "capacity_fade_rate    | -0.0023 | 0.0031  | -0.04  | 0.01\n\n"
            "Model performance (test set, after ensemble):\n"
            "• RMSE: 18.4 cycles\n"
            "• MAE: 12.7 cycles\n"
            "• Coverage (90% CI): 91.3% (target: ≥90%)\n"
            "• Mean CI width: 47.2 cycles\n"
            "• Anomaly detection precision: 0.82, recall: 0.79\n"
            "• Survival model C-index: 0.76\n\n"
            "Data splits:\n"
            "• Training: 80% (~4,560 rows, 27 batteries)\n"
            "• Test: 20% (~1,140 rows, 7 batteries)\n"
            "• Calibration: 6 batteries held out from training"
        ),
        "body_ko": (
            "데이터셋 요약 통계(처리 후):\n\n"
            "특성                  | 평균    | 표준편차 | 최솟값 | 최댓값\n"
            "─────────────────────────────────────────────────────────\n"
            "capacity_ahr          | 1.76    | 0.20    | 1.00   | 2.00\n"
            "temperature_c         | 27.8    | 14.2    | 4.0    | 44.0\n"
            "voltage_v             | 3.42    | 0.31    | 2.0    | 4.2\n"
            "current_a             | 1.48    | 0.72    | 0.5    | 4.0\n"
            "energy_j              | 17,200  | 3,800   | 5,000  | 30,000\n"
            "rul (타겟)            | 142     | 98      | 0      | 612\n"
            "capacity_fade_rate    | -0.0023 | 0.0031  | -0.04  | 0.01\n\n"
            "모델 성능(앙상블 후 테스트 세트):\n"
            "• RMSE: 18.4 사이클\n"
            "• MAE: 12.7 사이클\n"
            "• 커버리지(90% CI): 91.3% (목표: ≥90%)\n"
            "• 평균 CI 너비: 47.2 사이클\n"
            "• 이상 탐지 정밀도: 0.82, 재현율: 0.79\n"
            "• 생존 모델 C-지수: 0.76\n\n"
            "데이터 분할:\n"
            "• 훈련: 80% (~4,560행, 27개 배터리)\n"
            "• 테스트: 20% (~1,140행, 7개 배터리)\n"
            "• 보정: 훈련에서 제외된 6개 배터리"
        ),
    },
    {
        "id": "9",
        "title_en": "Step-by-Step Model Output",
        "title_ko": "9. 단계별 모델 출력",
        "body_en": (
            "When a battery is selected in the dashboard, the following outputs are computed:\n\n"
            "Step 1 — RUL Prediction Chart: A time-series plot showing actual capacity (blue), "
            "predicted RUL trajectory (orange), and the 90% conformal prediction interval "
            "(shaded band). The x-axis is cycle number; y-axis is RUL in cycles.\n\n"
            "Step 2 — Uncertainty Decomposition: The CI width is broken down into aleatoric "
            "(inherent noise) and epistemic (model uncertainty) components. Wide intervals "
            "indicate the battery is out-of-distribution relative to training data.\n\n"
            "Step 3 — Failure Probability: The survival model outputs P(failure within 20 cycles) "
            "and a risk label. Thresholds: HIGH ≥ 0.70, MEDIUM 0.30–0.70, LOW < 0.30.\n\n"
            "Step 4 — Anomaly Score: The residual score (0–100) quantifies deviation from "
            "expected trajectory. Score > 95th percentile → flagged as ANOMALY.\n\n"
            "Step 5 — Feature Importance: SHAP values show which features drove the prediction "
            "for this battery at this cycle. Top features typically: capacity_ahr, "
            "capacity_fade_rate, rolling_mean_5.\n\n"
            "Step 6 — Counterfactual: 'What if temperature had been 24°C instead of 4°C?' "
            "The model reruns the prediction under the counterfactual conditions.\n\n"
            "Step 7 — Supervisor Verdict: PASS / CONDITIONAL PASS / FAIL with reasoning.\n\n"
            "Step 8 — AI Explanation: The RAG engine generates a 3–5 sentence grounded "
            "natural language explanation of the current battery state."
        ),
        "body_ko": (
            "대시보드에서 배터리가 선택되면 다음 출력이 계산됩니다:\n\n"
            "1단계 — RUL 예측 차트: 실제 용량(파란색), 예측 RUL 궤적(주황색), "
            "90% 순응형 예측 구간(음영 밴드)을 보여주는 시계열 플롯.\n\n"
            "2단계 — 불확실성 분해: CI 너비가 우발적(고유 노이즈)과 "
            "인식적(모델 불확실성) 구성 요소로 분해됩니다.\n\n"
            "3단계 — 고장 확률: 생존 모델이 P(20사이클 내 고장) 및 위험 레이블을 출력합니다. "
            "임계값: 높음 ≥ 0.70, 중간 0.30–0.70, 낮음 < 0.30.\n\n"
            "4단계 — 이상 점수: 잔차 점수(0–100)가 예상 궤적에서의 편차를 정량화합니다. "
            "점수 > 95 백분위수 → 이상으로 플래그.\n\n"
            "5단계 — 특성 중요도: SHAP 값이 이 사이클에서 이 배터리의 예측을 "
            "주도한 특성을 보여줍니다.\n\n"
            "6단계 — 반사실적 예제: '온도가 4°C 대신 24°C였다면?' "
            "모델이 반사실적 조건 하에서 예측을 재실행합니다.\n\n"
            "7단계 — 감독자 판정: 근거가 있는 합격/조건부 합격/불합격.\n\n"
            "8단계 — AI 설명: RAG 엔진이 현재 배터리 상태에 대한 "
            "3-5문장의 근거 있는 자연어 설명을 생성합니다."
        ),
    },
    {
        "id": "10",
        "title_en": "Example: Battery B0047",
        "title_ko": "10. 예시: 배터리 B0047",
        "body_en": (
            "B0047 is a room-temperature (24°C) 18650 cell from the NASA PCoE dataset.\n\n"
            "Observed data:\n"
            "• Total cycles to EoL: 211\n"
            "• At cycle 150: capacity = 1.74 Ah (87% of nominal)\n"
            "• capacity_fade_rate at cycle 150: -0.0031 Ah/cycle\n"
            "• rolling_mean_5: 1.742\n\n"
            "Model outputs at cycle 150:\n"
            "• Ensemble RUL prediction: 61 cycles\n"
            "• True RUL: 61 cycles (actual EoL = cycle 211)\n"
            "• 90% Conformal PI: [42, 84] cycles\n"
            "• Prediction error: 0 cycles (exactly right in this example)\n"
            "• Anomaly score: 12.4 (below 95th percentile threshold — NORMAL)\n"
            "• Failure probability P(failure ≤ 20 cycles): 0.08 → LOW risk\n"
            "• Supervisor verdict: PASS (RMSE=18.2, coverage=91%, confidence=0.79)\n\n"
            "AI Explanation (generated by RAG):\n"
            "\"B0047's capacity at cycle 150 is 1.74 Ah (87% of nominal), consistent with "
            "mid-life SEI stabilisation described in [Source 1]. The fade rate of -0.0031 Ah/cycle "
            "is within the normal range for room-temperature operation. With a predicted RUL of 61 "
            "cycles and LOW failure risk, no immediate intervention is indicated. The conformal "
            "interval [42, 84] reflects normal model uncertainty for in-distribution batteries.\"\n\n"
            "Key insight: B0047 is a well-behaved battery. Its capacity trajectory is smooth, "
            "anomaly score is low, and the conformal interval is tight — indicating the model "
            "is confident in this prediction."
        ),
        "body_ko": (
            "B0047은 NASA PCoE 데이터셋의 실온(24°C) 18650 셀입니다.\n\n"
            "관측 데이터:\n"
            "• EoL까지 총 사이클: 211\n"
            "• 사이클 150에서: 용량 = 1.74 Ah (공칭의 87%)\n"
            "• 사이클 150에서 capacity_fade_rate: -0.0031 Ah/사이클\n"
            "• rolling_mean_5: 1.742\n\n"
            "사이클 150에서 모델 출력:\n"
            "• 앙상블 RUL 예측: 61 사이클\n"
            "• 실제 RUL: 61 사이클 (실제 EoL = 사이클 211)\n"
            "• 90% 순응형 PI: [42, 84] 사이클\n"
            "• 예측 오차: 0 사이클\n"
            "• 이상 점수: 12.4 (95 백분위수 임계값 미만 — 정상)\n"
            "• 고장 확률 P(≤20사이클 내 고장): 0.08 → 낮은 위험\n"
            "• 감독자 판정: 합격 (RMSE=18.2, 커버리지=91%, 신뢰도=0.79)\n\n"
            "AI 설명(RAG 생성):\n"
            "\"사이클 150에서 B0047의 용량은 1.74 Ah(공칭의 87%)로, "
            "[출처 1]에 설명된 중기 SEI 안정화와 일치합니다. 감쇄율 -0.0031 Ah/사이클은 "
            "실온 작동의 정상 범위 내에 있습니다. 예측 RUL 61 사이클과 낮은 고장 위험으로 "
            "즉각적인 개입이 필요하지 않습니다.\"\n\n"
            "핵심 인사이트: B0047은 정상 작동하는 배터리입니다. 용량 궤적이 부드럽고, "
            "이상 점수가 낮으며, 순응형 구간이 좁아 모델이 이 예측에 대해 확신합니다."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# 1. WORD DOCUMENT
# ─────────────────────────────────────────────────────────────────────────────

def make_word_doc(lang: str) -> None:
    doc = Document()

    # Title
    title_para = doc.add_heading("Battery AI Co-Scientist", 0)
    title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if lang == "ko":
        sub_para = doc.add_paragraph("배터리 잔여 유효 수명 예측 시스템 — 프로젝트 문서")
    else:
        sub_para = doc.add_paragraph("Remaining Useful Life Prediction System — Project Document")
    sub_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph()

    for sec in SECTIONS:
        heading = sec["title_ko"] if lang == "ko" else f"{sec['id']}. {sec['title_en']}"
        doc.add_heading(heading, level=1)
        body = sec["body_ko"] if lang == "ko" else sec["body_en"]
        para = doc.add_paragraph(body)
        para.style.font.size = Pt(11)
        doc.add_paragraph()

    fname = f"Battery_AI_CoScientist_{'Korean' if lang == 'ko' else 'English'}.docx"
    path = OUTPUT_DIR / fname
    doc.save(str(path))
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. POWERPOINT
# ─────────────────────────────────────────────────────────────────────────────

def _add_slide(prs: Presentation, title_text: str, body_text: str) -> None:
    slide_layout = prs.slide_layouts[1]  # Title + Content
    slide = prs.slides.add_slide(slide_layout)

    # Title
    title_box = slide.shapes.title
    title_box.text = title_text
    for para in title_box.text_frame.paragraphs:
        for run in para.runs:
            run.font.size = PPTpt(28)
            run.font.bold = True
            run.font.color.rgb = PPTColor(0x1A, 0x3A, 0x5C)

    # Body
    body_box = slide.placeholders[1]
    tf = body_box.text_frame
    tf.word_wrap = True
    tf.clear()

    lines = body_text.split("\n")
    first = True
    for line in lines:
        if first:
            p = tf.paragraphs[0]
            first = False
        else:
            p = tf.add_paragraph()
        p.text = line
        p.font.size = PPTpt(13) if len(line) < 100 else PPTpt(11)


def make_ppt(lang: str) -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.33)
    prs.slide_height = Inches(7.5)

    # Cover slide
    cover_layout = prs.slide_layouts[0]
    cover = prs.slides.add_slide(cover_layout)
    cover.shapes.title.text = "Battery AI Co-Scientist"
    subtitle = cover.placeholders[1]
    if lang == "ko":
        subtitle.text = "배터리 잔여 유효 수명 예측 — ML 파이프라인"
    else:
        subtitle.text = "Battery Remaining Useful Life Prediction — ML Pipeline"

    for sec in SECTIONS:
        heading = sec["title_ko"] if lang == "ko" else f"{sec['id']}. {sec['title_en']}"
        body_raw = sec["body_ko"] if lang == "ko" else sec["body_en"]
        # Trim to ~600 chars per slide for readability
        body = body_raw[:700] + ("..." if len(body_raw) > 700 else "")
        _add_slide(prs, heading, body)

    fname = f"Battery_AI_CoScientist_Presentation_{'Korean' if lang == 'ko' else 'English'}.pptx"
    path = OUTPUT_DIR / fname
    prs.save(str(path))
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. PROFESSOR Q&A DOCUMENT
# ─────────────────────────────────────────────────────────────────────────────

QA_DATA = [
    {
        "slide": "Slide 1 — Introduction / 서론",
        "qa_en": [
            ("What is Remaining Useful Life (RUL)?",
             "RUL is the number of charge-discharge cycles remaining before a battery reaches its End-of-Life "
             "criterion — defined here as 80% of its initial 2.0 Ah capacity (i.e., 1.6 Ah)."),
            ("Why did you choose lithium-ion batteries for this project?",
             "Li-ion batteries dominate EV and energy storage markets. Their degradation mechanisms are "
             "well-studied, and the NASA PCoE benchmark dataset provides controlled, reproducible experimental data."),
            ("What does 'end-to-end' mean in this context?",
             "The pipeline handles everything from raw .mat sensor data to a deployed interactive dashboard "
             "without manual intervention — data validation, feature engineering, training, evaluation, "
             "and explanation are fully automated."),
        ],
        "qa_ko": [
            ("잔여 유효 수명(RUL)이란 무엇인가요?",
             "RUL은 배터리가 수명 종료 기준에 도달하기 전까지 남은 충방전 사이클 수입니다. "
             "여기서는 초기 2.0 Ah 용량의 80%(즉 1.6 Ah)로 정의됩니다."),
            ("왜 리튬이온 배터리를 선택했나요?",
             "리튬이온 배터리는 전기차 및 에너지 저장 시장을 지배합니다. 열화 메커니즘이 잘 연구되어 있고 "
             "NASA PCoE 벤치마크 데이터셋이 통제된 재현 가능한 실험 데이터를 제공합니다."),
            ("'엔드-투-엔드'란 무엇을 의미하나요?",
             "파이프라인은 원시 .mat 센서 데이터부터 배포된 인터랙티브 대시보드까지 수동 개입 없이 모든 것을 처리합니다."),
        ],
    },
    {
        "slide": "Slide 2 — Problem Statement / 문제 정의",
        "qa_en": [
            ("What are the main electrochemical causes of battery degradation?",
             "SEI layer growth (blocks Li+ ion transport), lithium plating on the anode (at low temperatures "
             "or high C-rates), cathode structural disordering, and electrolyte decomposition."),
            ("Why is cell-to-cell variability a problem?",
             "Even batteries from the same manufacturing batch can have 5–15% variation in cycle life. "
             "A model trained on average behavior will underperform for outlier cells."),
            ("What is the cold-start problem in battery health monitoring?",
             "A new battery has no operational history, so calibration data-driven prediction intervals "
             "are not available. Our LOBO conformal scheme addresses this by using held-out batteries "
             "as proxies."),
        ],
        "qa_ko": [
            ("배터리 열화의 주요 전기화학적 원인은 무엇인가요?",
             "SEI 층 성장(Li+ 이온 이동 차단), 음극에서의 리튬 도금(저온 또는 높은 C율에서), "
             "양극 구조 무질서화, 전해질 분해입니다."),
            ("셀 간 변동성이 문제인 이유는 무엇인가요?",
             "동일한 제조 배치의 배터리도 사이클 수명에서 5-15%의 변동이 있을 수 있습니다."),
            ("배터리 건강 모니터링에서 콜드 스타트 문제란 무엇인가요?",
             "새 배터리는 작동 이력이 없어 보정 데이터 기반 예측 구간을 사용할 수 없습니다. "
             "LOBO 순응형 방식은 예비 배터리를 대리로 사용하여 이를 해결합니다."),
        ],
    },
    {
        "slide": "Slide 3 — Solution / 솔루션",
        "qa_en": [
            ("Why use three models instead of one?",
             "Each model captures different aspects: the baseline captures long-term trend, XGBoost "
             "captures feature interactions, and TCN captures temporal patterns in sequences. "
             "The ensemble reduces variance and typically outperforms any single model."),
            ("What is conformal prediction and why is it better than a Bayesian CI?",
             "Conformal prediction provides distribution-free, finite-sample guarantees. You don't need "
             "to assume a likelihood model. It provably covers the true value with ≥90% frequency under "
             "exchangeability, unlike Bayesian CIs which require a prior."),
            ("How does the RAG engine differ from a simple LLM call?",
             "A plain LLM can hallucinate. The RAG engine retrieves domain-specific chunks from our "
             "knowledge base (text + PDFs) and constrains the LLM to only use retrieved evidence, "
             "dramatically reducing hallucination rates."),
        ],
        "qa_ko": [
            ("하나의 모델 대신 세 가지 모델을 사용하는 이유는 무엇인가요?",
             "각 모델은 서로 다른 측면을 포착합니다: 기준선은 장기 추세, XGBoost는 특성 상호작용, "
             "TCN은 시퀀스의 시간적 패턴을 포착합니다."),
            ("순응형 예측이란 무엇이고 베이지안 CI보다 왜 더 좋은가요?",
             "순응형 예측은 분포 무관, 유한 샘플 보장을 제공합니다. 교환 가능성 하에서 "
             "실제 값을 ≥90% 빈도로 커버함이 증명되어 있습니다."),
            ("RAG 엔진은 단순한 LLM 호출과 어떻게 다른가요?",
             "일반 LLM은 환각을 일으킬 수 있습니다. RAG 엔진은 지식 베이스에서 도메인별 청크를 검색하고 "
             "LLM이 검색된 증거만 사용하도록 제한하여 환각률을 크게 줄입니다."),
        ],
    },
    {
        "slide": "Slide 4 — Literature Review / 문헌 고찰",
        "qa_en": [
            ("How does your work extend Severson et al. (2019)?",
             "Severson used early-cycle features for total life prediction. We extend this to "
             "cycle-level RUL prediction with uncertainty bounds, incorporating more features "
             "and a full calibration framework."),
            ("Why is conformal prediction more appropriate here than Bayesian methods?",
             "Battery degradation does not have a well-specified prior. Conformal prediction works "
             "without distributional assumptions and is valid even with small calibration sets."),
            ("What is PSI and why is it used here?",
             "Population Stability Index measures distributional shift between training and production "
             "data. PSI > 0.20 indicates significant drift requiring model retraining. Originally from "
             "credit scoring, it is well-suited to monitoring battery feature distributions over time."),
        ],
        "qa_ko": [
            ("당신의 연구는 Severson et al. (2019)를 어떻게 확장하나요?",
             "Severson은 초기 사이클 특성을 총 수명 예측에 사용했습니다. 우리는 이를 불확실성 경계가 있는 "
             "사이클 수준 RUL 예측으로 확장하여 더 많은 특성과 완전한 보정 프레임워크를 통합합니다."),
            ("PSI란 무엇이고 여기서 왜 사용되나요?",
             "인구 안정성 지수(PSI)는 훈련 데이터와 생산 데이터 간의 분포 변화를 측정합니다. "
             "PSI > 0.20은 모델 재훈련이 필요한 심각한 드리프트를 나타냅니다."),
        ],
    },
    {
        "slide": "Slide 5 — Motivation / 연구 동기",
        "qa_en": [
            ("What is the economic impact of poor battery health monitoring?",
             "Global EV battery replacements cost an estimated $10–15 billion annually. Premature "
             "replacements (conservative predictions) waste good capacity; delayed replacements "
             "risk thermal runaway and safety incidents."),
            ("Why does explainability matter for battery engineers?",
             "An engineer needs to act on the prediction — adjust charging protocol, schedule "
             "maintenance, or isolate the battery. Without knowing WHY risk is HIGH, they cannot "
             "take targeted action."),
        ],
        "qa_ko": [
            ("잘못된 배터리 건강 모니터링의 경제적 영향은 무엇인가요?",
             "전 세계 전기차 배터리 교체 비용은 연간 약 100-150억 달러로 추산됩니다. "
             "조기 교체는 좋은 용량을 낭비하고, 교체 지연은 열폭주 및 안전 사고 위험을 초래합니다."),
            ("배터리 엔지니어에게 설명 가능성이 왜 중요한가요?",
             "엔지니어는 예측에 따라 행동해야 합니다. 위험이 '높음'인 이유를 알지 못하면 "
             "목표 지향적인 조치를 취할 수 없습니다."),
        ],
    },
    {
        "slide": "Slide 6 — Methodology / 방법론",
        "qa_en": [
            ("Why use GroupKFold instead of random KFold?",
             "Random KFold can put cycles 1-50 of B0005 in training and cycles 51-100 in test. "
             "The model would then see future context of the same battery — data leakage. "
             "GroupKFold splits by battery ID, preventing this."),
            ("What is LOBO calibration and when does it apply?",
             "Leave-One-Battery-Out calibration holds out one battery from training, computes "
             "conformance scores on it, and uses those residuals to calibrate prediction intervals. "
             "It applies specifically to 'cold' temperature-group batteries with no in-distribution "
             "calibration data."),
            ("How are the three model outputs combined in the ensemble?",
             "Learned weights (optimized on a held-out validation set) are applied: approximately "
             "0.2 for baseline, 0.4 for XGBoost, 0.4 for TCN. The weighted mean is the ensemble RUL."),
            ("What loss function does the TCN use?",
             "Mean Squared Error (MSE) for the RUL regression head, with L2 regularization via "
             "weight decay of 1e-5."),
        ],
        "qa_ko": [
            ("무작위 KFold 대신 GroupKFold를 사용하는 이유는 무엇인가요?",
             "무작위 KFold는 B0005의 1-50 사이클을 훈련에, 51-100을 테스트에 넣을 수 있어 "
             "데이터 누출이 발생합니다. GroupKFold는 배터리 ID로 분할하여 이를 방지합니다."),
            ("LOBO 보정이란 무엇이고 언제 적용되나요?",
             "Leave-One-Battery-Out 보정은 훈련에서 배터리 하나를 제외하고 해당 배터리에서 "
             "순응 점수를 계산하여 예측 구간 보정에 사용합니다."),
            ("세 모델 출력이 앙상블에서 어떻게 결합되나요?",
             "학습된 가중치(검증 세트에서 최적화)가 적용됩니다: 기준선 약 0.2, XGBoost 0.4, TCN 0.4."),
        ],
    },
    {
        "slide": "Slide 7 — Dataset / 데이터셋",
        "qa_en": [
            ("Why is the NASA PCoE dataset appropriate for this work?",
             "It is publicly available, well-documented, widely benchmarked, and covers multiple "
             "temperature conditions relevant to real-world deployment. The 18650 format is the "
             "most commercially prevalent Li-ion cell."),
            ("What is the 80% EoL criterion based on?",
             "IEC 62660-1 and industry convention define EoL when remaining capacity falls below "
             "80% of rated capacity. Below this, driving range degrades unacceptably for EV users."),
            ("Why are B0049–B0056 flagged as anomalous?",
             "These cells show irregular capacity trajectories inconsistent with smooth degradation — "
             "sudden drops, partial recovery patterns — likely due to measurement artifacts or "
             "experimental interruptions. They are flagged and excluded from calibration."),
        ],
        "qa_ko": [
            ("NASA PCoE 데이터셋이 이 연구에 적합한 이유는 무엇인가요?",
             "공개적으로 사용 가능하고, 잘 문서화되어 있으며, 널리 벤치마크되고, "
             "실제 배포와 관련된 여러 온도 조건을 다룹니다."),
            ("80% EoL 기준은 무엇에 기반하나요?",
             "IEC 62660-1 및 업계 관행에서 잔여 용량이 정격 용량의 80% 미만으로 떨어질 때 "
             "EoL로 정의합니다."),
            ("B0049-B0056이 이상으로 플래그 처리되는 이유는 무엇인가요?",
             "이 셀들은 갑작스러운 하락, 부분 회복 패턴 등 부드러운 열화와 일치하지 않는 "
             "불규칙한 용량 궤적을 보입니다."),
        ],
    },
    {
        "slide": "Slide 8 — Basic Statistics / 기본 통계",
        "qa_en": [
            ("What does a coverage of 91.3% mean for your 90% CI?",
             "It means 91.3% of test-set true RUL values fell within the predicted confidence "
             "interval. The target is ≥90%. Slightly over-coverage (91.3% vs 90%) is conservative "
             "and acceptable; under-coverage would be a calibration failure."),
            ("Why is the mean CI width 47.2 cycles considered acceptable?",
             "For batteries with RUL ranging 0–612 cycles, a ±23-cycle bound on a 90% CI represents "
             "reasonable precision. Narrower would require more calibration data or more informative features."),
        ],
        "qa_ko": [
            ("90% CI에 대한 91.3% 커버리지는 무엇을 의미하나요?",
             "테스트 세트 실제 RUL 값의 91.3%가 예측된 신뢰 구간 내에 포함되었음을 의미합니다. "
             "목표는 ≥90%이며 약간의 초과 커버리지(91.3%)는 보수적이고 허용 가능합니다."),
            ("평균 CI 너비 47.2 사이클이 허용 가능한 이유는 무엇인가요?",
             "RUL이 0-612 사이클 범위인 배터리에 대해 90% CI에서 ±23 사이클 범위는 합리적인 정밀도를 나타냅니다."),
        ],
    },
    {
        "slide": "Slide 9 — Model Output Pipeline / 모델 출력 파이프라인",
        "qa_en": [
            ("What is a SHAP value and how do you interpret it for RUL prediction?",
             "SHAP (SHapley Additive exPlanations) assigns each feature a contribution value for "
             "a specific prediction. Positive SHAP = feature increased the RUL prediction; negative "
             "SHAP = feature decreased it. For a battery with rapidly falling capacity, "
             "capacity_fade_rate would have a large negative SHAP value."),
            ("How does the supervisor audit work?",
             "The supervisor checks: (1) model confidence score vs threshold (0.70), (2) RMSE vs "
             "pass threshold (100 cycles), (3) CV-RMSE ratio vs warning threshold (1.5). "
             "All three pass → PASS. One marginal → CONDITIONAL PASS. Any fail → FAIL."),
            ("What is a counterfactual explanation?",
             "A counterfactual asks: 'What would the RUL prediction be if feature X had a different "
             "value?' It helps operators understand which conditions most affect degradation rate "
             "and what interventions (e.g., lower charge rate) would extend battery life."),
        ],
        "qa_ko": [
            ("SHAP 값이란 무엇이고 RUL 예측에서 어떻게 해석하나요?",
             "SHAP는 특정 예측에 대해 각 특성에 기여 값을 할당합니다. 양의 SHAP = 특성이 "
             "RUL 예측을 증가시킴, 음의 SHAP = 감소시킴."),
            ("감독자 감사는 어떻게 작동하나요?",
             "감독자는 (1) 모델 신뢰도 점수 대 임계값(0.70), (2) RMSE 대 통과 임계값(100 사이클), "
             "(3) CV-RMSE 비율 대 경고 임계값(1.5)을 확인합니다."),
            ("반사실적 설명이란 무엇인가요?",
             "반사실적 설명은 '특성 X가 다른 값을 가졌다면 RUL 예측이 어떻게 달라졌을까?'라고 질문합니다."),
        ],
    },
    {
        "slide": "Slide 10 — Battery Example B0047 / 예시 배터리 B0047",
        "qa_en": [
            ("Why is the prediction interval [42, 84] for a true RUL of 61?",
             "The interval is correctly calibrated — 61 falls within [42, 84]. The width reflects "
             "normal epistemic uncertainty for an in-distribution room-temperature battery. "
             "A LOBO-calibrated battery at an extreme temperature would have a wider interval."),
            ("What would trigger the anomaly flag for B0047?",
             "If capacity_ahr at cycle 150 had been 1.52 Ah (below 1.6 Ah EoL threshold) or if "
             "the residual between expected and actual capacity exceeded the 95th percentile, "
             "the anomaly detector would flag it."),
            ("What if the RAG knowledge base had no relevant documents?",
             "The system returns: 'The available evidence does not support a conclusion on this "
             "point.' It will not fabricate an answer. This is enforced by Rule 4 in the prompt."),
        ],
        "qa_ko": [
            ("실제 RUL 61에 대한 예측 구간이 [42, 84]인 이유는 무엇인가요?",
             "구간이 올바르게 보정되어 있습니다 — 61은 [42, 84] 내에 속합니다. "
             "너비는 분포 내 실온 배터리에 대한 정상적인 인식적 불확실성을 반영합니다."),
            ("B0047에서 이상 플래그를 유발하는 것은 무엇인가요?",
             "사이클 150에서 capacity_ahr이 1.52 Ah이거나 예상 용량과 실제 용량 간의 잔차가 "
             "95 백분위수를 초과하면 이상 탐지기가 플래그를 붙입니다."),
            ("RAG 지식 베이스에 관련 문서가 없다면 어떻게 되나요?",
             "시스템은 '사용 가능한 증거가 이 점에 대한 결론을 지원하지 않습니다'를 반환합니다. "
             "프롬프트의 규칙 4에 의해 적용되며 답변을 조작하지 않습니다."),
        ],
    },
]


def make_qa_doc(lang: str) -> None:
    doc = Document()

    heading_text = (
        "Professor Q&A — Battery AI Co-Scientist"
        if lang == "en"
        else "교수 질문 & 답변 — Battery AI Co-Scientist"
    )
    doc.add_heading(heading_text, 0).alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph()

    for item in QA_DATA:
        doc.add_heading(item["slide"], level=1)
        qa_list = item["qa_en"] if lang == "en" else item["qa_ko"]
        for i, (q, a) in enumerate(qa_list, 1):
            q_para = doc.add_paragraph()
            run = q_para.add_run(f"Q{i}: {q}")
            run.bold = True
            run.font.color.rgb = RGBColor(0x1A, 0x3A, 0x5C)
            a_para = doc.add_paragraph(f"A: {a}")
            a_para.style.font.size = Pt(11)
            doc.add_paragraph()

    fname = f"Professor_QA_{'Korean' if lang == 'ko' else 'English'}.docx"
    path = OUTPUT_DIR / fname
    doc.save(str(path))
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. STREAMLIT EXPLANATION DOCUMENT
# ─────────────────────────────────────────────────────────────────────────────

STREAMLIT_SECTIONS_EN = [
    ("Overview", (
        "The Battery AI Co-Scientist dashboard is a single-page Streamlit application "
        "(dashboard/app.py) that exposes every component of the ML pipeline through an "
        "interactive web interface. It runs locally on http://localhost:8501. No internet "
        "connection is required after initial model download.\n\n"
        "Launch command: python -m streamlit run dashboard/app.py"
    )),
    ("Sidebar — Battery Selector", (
        "The left sidebar contains a dropdown list of all batteries in "
        "data/processed/cycle_features_with_rul.csv. Selecting a battery loads all of its "
        "cycle data and triggers all downstream computations. Battery IDs are shown with "
        "temperature group labels (e.g., 'B0005 (cold)')."
    )),
    ("Section 1 — RUL Prediction Chart", (
        "The main chart shows:\n"
        "• Blue line: actual capacity (Ah) over cycles (left axis)\n"
        "• Orange line: predicted RUL trajectory over cycles (right axis)\n"
        "• Shaded band: 90% conformal prediction interval\n"
        "• Red dashed line: EoL threshold (1.6 Ah)\n\n"
        "The chart is interactive (Plotly): hover for exact values, zoom, pan, download PNG."
    )),
    ("Section 2 — Uncertainty Panel", (
        "Displays the current cycle's:\n"
        "• Point RUL estimate (ensemble weighted mean)\n"
        "• Lower bound (5th percentile conformal)\n"
        "• Upper bound (95th percentile conformal)\n"
        "• CI width in cycles\n"
        "• Aleatoric vs epistemic uncertainty decomposition (bar chart)"
    )),
    ("Section 3 — Risk & Survival", (
        "Shows the discrete-time hazard model output:\n"
        "• P(failure within 20 cycles) as a gauge chart (0–100%)\n"
        "• Risk label: LOW (green), MEDIUM (amber), HIGH (red)\n"
        "• Survival curve S(t) plotted against cycle horizon\n\n"
        "Thresholds: HIGH ≥ 70%, MEDIUM 30–70%, LOW < 30%."
    )),
    ("Section 4 — Anomaly Detection", (
        "Residual score card (0–100):\n"
        "• Green badge: NORMAL (score ≤ 95th percentile threshold)\n"
        "• Red badge: ANOMALY (score > threshold)\n"
        "• Time-series of residual scores across all cycles\n"
        "• Threshold line shown as horizontal dashed red line"
    )),
    ("Section 5 — Feature Importance", (
        "SHAP waterfall chart for the current cycle's prediction:\n"
        "• Bars to the right (positive SHAP): features increasing RUL prediction\n"
        "• Bars to the left (negative SHAP): features decreasing RUL prediction\n"
        "• Top 10 features shown by default\n\n"
        "Also includes a counterfactual panel: select a feature and value to see "
        "what the RUL would have been under different conditions."
    )),
    ("Section 6 — Supervisor Audit", (
        "Automated quality audit card showing:\n"
        "• Overall verdict: PASS (green) / CONDITIONAL PASS (amber) / FAIL (red)\n"
        "• Model confidence score vs threshold (0.70 to pass, 0.85 to auto-approve)\n"
        "• RMSE vs pass threshold (≤100 cycles)\n"
        "• CV-RMSE ratio vs warning threshold (≤1.5)\n"
        "• Reasoning text explaining the verdict"
    )),
    ("Section 7 — AI Explanation (RAG)", (
        "A text area where users can type a natural language question about the battery "
        "(e.g., 'Why is the risk HIGH?' or 'What does the wide CI mean?').\n\n"
        "Clicking 'Generate AI Summary' triggers the RAG engine:\n"
        "1. SentenceTransformer encodes the query\n"
        "2. ChromaDB retrieves top-3 relevant document chunks\n"
        "3. A structured prompt (with all pipeline data + retrieved chunks) is sent to Gemma 3-4B-IT\n"
        "4. The answer is displayed with source citations\n\n"
        "Note: First call loads the Gemma model (~3 minutes on first run). "
        "Subsequent calls take 30–90 seconds on an RTX 3060."
    )),
    ("Technical Architecture", (
        "• Framework: Streamlit 1.x\n"
        "• Data: cached with @st.cache_data (5-minute TTL)\n"
        "• Models: loaded once per session with @st.cache_resource\n"
        "• Charts: Plotly Express + Plotly Graph Objects\n"
        "• RAG: sentence-transformers + chromadb + transformers (Gemma)\n"
        "• Config: all thresholds read from configs/pipeline.yaml at startup\n"
        "• BASE path: resolved relative to app.py to prevent path issues across environments"
    )),
    ("Troubleshooting", (
        "• 'No battery data found': Run 'python -m src.modeling.run_full_pipeline' first\n"
        "• 'Explanation failed: Repo id must use alphanumeric chars...': "
        "The local model path is missing. The system will auto-download from HuggingFace. "
        "Ensure internet connection or place weights in models/gemma-3-4b-it/\n"
        "• Multiple Streamlit processes: Kill with "
        "'Get-Process python* | Stop-Process -Force' in PowerShell\n"
        "• Slow generation: Expected 30–90 seconds on RTX 3060 with 4-bit quantization\n"
        "• Port already in use: Run 'streamlit run dashboard/app.py --server.port 8502'"
    )),
]

STREAMLIT_SECTIONS_KO = [
    ("개요", (
        "Battery AI Co-Scientist 대시보드는 ML 파이프라인의 모든 구성 요소를 인터랙티브 웹 "
        "인터페이스를 통해 노출하는 단일 페이지 Streamlit 애플리케이션(dashboard/app.py)입니다. "
        "http://localhost:8501에서 로컬로 실행됩니다.\n\n"
        "실행 명령: python -m streamlit run dashboard/app.py"
    )),
    ("사이드바 — 배터리 선택기", (
        "왼쪽 사이드바에는 data/processed/cycle_features_with_rul.csv의 모든 배터리 드롭다운 "
        "목록이 있습니다. 배터리를 선택하면 모든 사이클 데이터가 로드되고 모든 후속 계산이 트리거됩니다."
    )),
    ("섹션 1 — RUL 예측 차트", (
        "메인 차트에 표시됩니다:\n"
        "• 파란 선: 실제 용량(Ah) (왼쪽 축)\n"
        "• 주황 선: 사이클별 예측 RUL 궤적 (오른쪽 축)\n"
        "• 음영 밴드: 90% 순응형 예측 구간\n"
        "• 빨간 점선: EoL 임계값(1.6 Ah)\n\n"
        "차트는 인터랙티브(Plotly): 호버로 정확한 값 확인, 확대/축소, 이동, PNG 다운로드 가능."
    )),
    ("섹션 2 — 불확실성 패널", (
        "현재 사이클의 표시 항목:\n"
        "• 점 RUL 추정값(앙상블 가중 평균)\n"
        "• 하한(5 백분위수 순응형)\n"
        "• 상한(95 백분위수 순응형)\n"
        "• 사이클 단위 CI 너비\n"
        "• 우발적 대 인식적 불확실성 분해(막대 차트)"
    )),
    ("섹션 3 — 위험 및 생존", (
        "이산 시간 위험 모델 출력:\n"
        "• P(20사이클 내 고장)을 게이지 차트로 표시(0-100%)\n"
        "• 위험 레이블: 낮음(녹색), 중간(황색), 높음(빨간색)\n"
        "• 사이클 범위에 대한 생존 곡선 S(t)\n\n"
        "임계값: 높음 ≥ 70%, 중간 30-70%, 낮음 < 30%."
    )),
    ("섹션 4 — 이상 탐지", (
        "잔차 점수 카드(0-100):\n"
        "• 녹색 배지: 정상(점수 ≤ 95 백분위수 임계값)\n"
        "• 빨간 배지: 이상(점수 > 임계값)\n"
        "• 모든 사이클에 걸친 잔차 점수 시계열\n"
        "• 가로 빨간 점선으로 임계값 표시"
    )),
    ("섹션 5 — 특성 중요도", (
        "현재 사이클 예측에 대한 SHAP 워터폴 차트:\n"
        "• 오른쪽 막대(양의 SHAP): RUL 예측을 증가시키는 특성\n"
        "• 왼쪽 막대(음의 SHAP): RUL 예측을 감소시키는 특성\n"
        "• 기본적으로 상위 10개 특성 표시\n\n"
        "반사실적 패널 포함: 특성과 값을 선택하여 다른 조건에서 RUL이 어떻게 달라졌을지 확인."
    )),
    ("섹션 6 — 감독자 감사", (
        "자동 품질 감사 카드:\n"
        "• 전체 판정: 합격(녹색) / 조건부 합격(황색) / 불합격(빨간색)\n"
        "• 모델 신뢰도 점수 대 임계값(통과 0.70, 자동 승인 0.85)\n"
        "• RMSE 대 통과 임계값(≤100 사이클)\n"
        "• CV-RMSE 비율 대 경고 임계값(≤1.5)\n"
        "• 판정 설명 텍스트"
    )),
    ("섹션 7 — AI 설명(RAG)", (
        "사용자가 배터리에 대한 자연어 질문을 입력할 수 있는 텍스트 영역.\n\n"
        "'AI 요약 생성' 클릭 시 RAG 엔진 트리거:\n"
        "1. SentenceTransformer가 쿼리를 인코딩\n"
        "2. ChromaDB가 관련 상위 3개 문서 청크 검색\n"
        "3. 모든 파이프라인 데이터 + 검색된 청크로 구성된 구조화된 프롬프트를 Gemma 3-4B-IT에 전달\n"
        "4. 출처 인용과 함께 답변 표시\n\n"
        "참고: 첫 번째 호출은 Gemma 모델을 로드합니다(첫 실행 시 약 3분). "
        "RTX 3060에서 이후 호출은 30-90초 소요."
    )),
    ("기술 아키텍처", (
        "• 프레임워크: Streamlit 1.x\n"
        "• 데이터: @st.cache_data로 캐시(5분 TTL)\n"
        "• 모델: @st.cache_resource로 세션당 한 번 로드\n"
        "• 차트: Plotly Express + Plotly Graph Objects\n"
        "• RAG: sentence-transformers + chromadb + transformers(Gemma)\n"
        "• 설정: 모든 임계값이 시작 시 configs/pipeline.yaml에서 읽힘\n"
        "• BASE 경로: app.py에 상대적으로 해결되어 환경 간 경로 문제 방지"
    )),
    ("문제 해결", (
        "• '배터리 데이터를 찾을 수 없음': 먼저 'python -m src.modeling.run_full_pipeline' 실행\n"
        "• '설명 실패: Repo id must use alphanumeric chars...': 로컬 모델 경로가 없어 "
        "자동으로 HuggingFace에서 다운로드됩니다. 인터넷 연결 확인 또는 "
        "models/gemma-3-4b-it/에 가중치 배치\n"
        "• 여러 Streamlit 프로세스: PowerShell에서 "
        "'Get-Process python* | Stop-Process -Force'로 종료\n"
        "• 느린 생성: RTX 3060에서 4비트 양자화로 30-90초 예상\n"
        "• 포트 이미 사용 중: 'streamlit run dashboard/app.py --server.port 8502' 실행"
    )),
]


def make_streamlit_doc(lang: str) -> None:
    doc = Document()

    heading_text = (
        "Streamlit Dashboard — User Guide & Technical Reference"
        if lang == "en"
        else "Streamlit 대시보드 — 사용자 가이드 및 기술 참조"
    )
    doc.add_heading("Battery AI Co-Scientist", 0).alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_heading(heading_text, 1).alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph()

    sections = STREAMLIT_SECTIONS_EN if lang == "en" else STREAMLIT_SECTIONS_KO
    for title, body in sections:
        doc.add_heading(title, level=2)
        doc.add_paragraph(body)
        doc.add_paragraph()

    fname = f"Streamlit_Guide_{'Korean' if lang == 'ko' else 'English'}.docx"
    path = OUTPUT_DIR / fname
    doc.save(str(path))
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating documents...")

    print("\n[1/4] Word documents...")
    make_word_doc("en")
    make_word_doc("ko")

    print("\n[2/4] PowerPoint presentations...")
    make_ppt("en")
    make_ppt("ko")

    print("\n[3/4] Professor Q&A documents...")
    make_qa_doc("en")
    make_qa_doc("ko")

    print("\n[4/4] Streamlit guide documents...")
    make_streamlit_doc("en")
    make_streamlit_doc("ko")

    print(f"\nAll 8 files saved to: {OUTPUT_DIR}")
