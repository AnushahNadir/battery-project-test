"""Generate Professor Q&A document in English and Korean."""
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
GREEN     = RGBColor(0x1A, 0x6B, 0x3C)
ORANGE    = RGBColor(0xC0, 0x50, 0x00)

def add_slide_header(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(20)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    r.font.size = Pt(16)
    r.font.bold = True
    r.font.color.rgb = DARK_BLUE
    # bottom border
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement('w:pBdr')
    bot = OxmlElement('w:bottom')
    bot.set(qn('w:val'), 'single')
    bot.set(qn('w:sz'), '8')
    bot.set(qn('w:space'), '1')
    bot.set(qn('w:color'), '1A3A5C')
    pBdr.append(bot)
    pPr.append(pBdr)

def add_q(doc, num, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.left_indent = Inches(0.2)
    rl = p.add_run(f"Q{num}  ")
    rl.font.size = Pt(11)
    rl.font.bold = True
    rl.font.color.rgb = ORANGE
    rt = p.add_run(text)
    rt.font.size = Pt(11)
    rt.font.bold = True
    rt.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)

def add_a(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.left_indent = Inches(0.4)
    rl = p.add_run("A  ")
    rl.font.size = Pt(11)
    rl.font.bold = True
    rl.font.color.rgb = GREEN
    rt = p.add_run(text)
    rt.font.size = Pt(11)
    rt.font.color.rgb = RGBColor(0x22, 0x22, 0x22)

def add_tip(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.left_indent = Inches(0.4)
    r = p.add_run("Tip: " + text)
    r.font.size = Pt(10)
    r.font.italic = True
    r.font.color.rgb = RGBColor(0x55, 0x55, 0x88)


QA_EN = [
    {
        "slide": "Slide 1-2: Introduction — What the project does and why",
        "items": [
            ("What is the definition of Remaining Useful Life (RUL) in this project?",
             "RUL is the number of charge-discharge cycles remaining before the battery's discharge capacity drops below 80% of its initial rated capacity (1.6 Ah for a 2.0 Ah nominal cell). This follows the IEC 62660-1 standard and the NASA PCoE benchmark convention.",
             "Draw the capacity-vs-cycle curve and mark where it crosses 1.6 Ah — that is the EoL point, and the distance from current cycle to that crossing is RUL."),
            ("Why use the 80% capacity threshold as End-of-Life, and not some other value?",
             "80% is the industry-standard threshold established in IEC 62660-1 and widely adopted in EV applications. Below 80%, the battery's driving range degrades by more than 20%, which is considered unacceptable for most EV users. Some aerospace applications use 70%, but 80% is standard for automotive and grid storage.",
             "Mention that the threshold is configurable in pipeline.yaml (eol.capacity_fade_fraction = 0.80) and could be adjusted for different applications."),
            ("What does 'end-to-end' mean in the context of this system?",
             "End-to-end means the pipeline handles everything from raw .mat sensor files to a deployed interactive dashboard without manual intervention. Every stage — data parsing, schema validation, feature engineering, model training, conformal calibration, and explanation generation — is automated and triggered by a single pipeline run command.",
             "Contrast with a typical workflow where a data scientist prepares features manually, a modeler trains separately, and deployment is done by a different team."),
            ("Why did you choose the NASA PCoE dataset instead of other battery datasets?",
             "The NASA PCoE dataset is one of the most widely benchmarked publicly available Li-ion battery datasets, with controlled experimental conditions at multiple temperatures. It covers enough cells (34) and cycles (168-616 per cell) to train and evaluate ensemble models. The clean experimental setup (no real-world noise) makes it ideal for developing and validating a methodology before deployment on field data.",
             "Acknowledge the limitation: lab data under constant current may not represent real-world variable-load EV charging. Mention this is a direction for future work."),
        ]
    },
    {
        "slide": "Slide 3-4: Problem Statement — Degradation mechanisms and challenges",
        "items": [
            ("What is SEI layer growth and why is it the dominant degradation mechanism?",
             "The Solid Electrolyte Interphase forms on the graphite anode surface during the first charge as electrolyte reacts with lithium. It grows irreversibly each cycle, consuming cyclable lithium and increasing internal resistance. It is the dominant mechanism in room-temperature batteries at normal C-rates because it operates continuously throughout the battery's life from the first cycle.",
             "Contrast with lithium plating (which is dominant at low temperatures) and cathode cracking (which becomes dominant late in life when the cathode is structurally weakened)."),
            ("Why is cell-to-cell variability a problem for machine learning models?",
             "If a model is trained on average degradation behavior, it will systematically over-predict RUL for batteries that degrade faster than average and under-predict for batteries that degrade slower. This is particularly problematic for safety-critical decisions. Our ensemble + conformal approach mitigates this by providing prediction intervals that are wide enough to cover the true RUL even for outlier cells.",
             "Mention that the width of the conformal prediction interval is empirically calibrated to cover 90% of batteries, including outliers, rather than being derived from parametric assumptions."),
            ("What is the cold-start problem and how does your system solve it?",
             "The cold-start problem occurs when a new battery with no operational history needs a prediction. Standard calibration breaks down because you need historical residuals from similar batteries to compute conformal intervals. Our LOBO (Leave-One-Battery-Out) scheme solves this by training the calibration on one battery at a time, using the other batteries' residuals as a proxy. For cold-temperature batteries, a 1.20x safety factor additionally inflates the interval to account for greater out-of-distribution uncertainty.",
             "LOBO is essentially conformal prediction applied in the grouped time-series setting where the natural grouping is by battery identity."),
            ("Why do you need uncertainty quantification at all? A point prediction of RUL=50 seems sufficient.",
             "A point prediction of 50 cycles is ambiguous for decision-making. If the true interval is [48, 52], the operator can confidently plan maintenance at cycle 45. If the true interval is [20, 80], the operator must plan maintenance at cycle 15 to be safe. The second case requires replacing the battery 30 cycles earlier than the first, representing significant economic cost. Without uncertainty, these two scenarios are indistinguishable.",
             "Use an analogy: a weather forecast of '20 deg C' is less useful than '20 +/- 2 deg C'. The uncertainty informs how much to prepare."),
        ]
    },
    {
        "slide": "Slide 5-6: Solution — Model architecture and components",
        "items": [
            ("Why use three different model types instead of just training one very large model?",
             "No single architecture is universally optimal. The exponential baseline is physically interpretable but rigid. XGBoost captures feature interactions efficiently but cannot model temporal sequences. TCN captures temporal dynamics but requires more data. The ensemble inherits the strengths of all three: physical interpretability, feature interaction modeling, and temporal pattern recognition. Empirically, the ensemble RMSE (18.4) is lower than any individual model.",
             "Also mention bias-variance: different models make different types of errors. Averaging tends to cancel out individual model biases."),
            ("What is conformal prediction and why is it better than a Bayesian credible interval for this application?",
             "Conformal prediction provides a mathematical guarantee: the prediction interval will contain the true value at least 90% of the time, regardless of the true data distribution. Bayesian credible intervals require specifying a prior distribution over parameters. For battery degradation, there is no well-justified prior. If the prior is misspecified, the Bayesian interval can have much lower actual coverage than its stated confidence level. Conformal prediction avoids this by making no distributional assumptions.",
             "The tradeoff: conformal intervals may be wider than a correctly specified Bayesian interval. The guarantee comes at the cost of some efficiency."),
            ("What is the RAG engine and how does it prevent hallucination?",
             "RAG (Retrieval-Augmented Generation) combines a retrieval step (finding relevant document chunks using embedding similarity) with a generation step (Gemma 3-4B-IT generating an answer). The key anti-hallucination mechanism is the structured prompt with 9 rules that explicitly constrain the model to only use retrieved evidence, to say 'the available evidence does not support a conclusion' when evidence is insufficient, and to cite sources inline. The model is instructed not to claim specific causal mechanisms unless a retrieved source directly supports that diagnosis.",
             "Mention that this was evaluated on 40 queries with a 5-dimension rubric: groundedness, correctness, completeness, battery specificity, and hallucination flag."),
            ("What is the Population Stability Index (PSI) and why is it used?",
             "PSI measures the distributional shift between a baseline distribution (training data) and a current distribution (new predictions). It is computed by binning both distributions into 10 equal-width bins and computing a symmetric KL-divergence-like metric. PSI < 0.10: no significant drift (green). PSI 0.10-0.20: moderate drift (amber, monitor). PSI > 0.20: significant drift requiring model retraining (red). It is used here to detect when incoming battery data has drifted from the training distribution, which would invalidate the model's predictions.",
             "PSI originated in credit scoring (Lewis, 1998) where it was used to monitor whether a population of loan applicants had shifted since the model was built."),
        ]
    },
    {
        "slide": "Slide 7-8: Literature Review — Research foundations",
        "items": [
            ("How does your project extend Severson et al. (2019)?",
             "Severson et al. predicted total cycle life (a single number per battery) using features from the first 100 cycles. This project extends this in three ways: (1) we predict RUL at every cycle, not just at the start; (2) we provide calibrated uncertainty intervals, not just a point prediction; (3) we include a full post-hoc analysis pipeline with survival analysis, anomaly detection, and RAG explanation. Severson used ~124 batteries from a different dataset; we use the NASA PCoE 18650 dataset.",
             ""),
            ("Why is GroupKFold essential and what would go wrong with regular KFold?",
             "Regular KFold splits cycles randomly across training and test sets. This means cycles 1-50 of B0005 might go to training and cycles 51-100 to test. The model sees future context from the same battery it is being evaluated on — this is data leakage. The model learns the specific degradation trajectory of each battery, leading to optimistically biased RMSE estimates. GroupKFold splits by battery ID: all cycles of B0005 are either in training or in test, never both.",
             "This is a common mistake in battery ML papers. Always split by battery identity, not by cycle index."),
        ]
    },
    {
        "slide": "Slide 9-10: Methodology — Training pipeline in detail",
        "items": [
            ("What loss function does the TCN use and why MSE instead of quantile loss?",
             "The TCN uses Mean Squared Error (MSE) for the single-output RUL regression head. The quantile regression is handled by XGBoost (three separate models at alpha=0.05, 0.50, 0.95) rather than by the TCN. Using MSE for the TCN simplifies training and produces a well-calibrated point estimate. The conformal prediction step then adds properly calibrated uncertainty bands around the ensemble's point estimate, so the TCN does not need to produce its own intervals.",
             ""),
            ("How are the ensemble weights determined?",
             "The weights are learned by minimizing the mean squared error of the weighted combination on a held-out validation set. The optimization uses scipy.optimize.minimize with the constraint that weights sum to 1 and are all non-negative. The result is approximately 0.20 for the baseline (lower weight because it is less flexible), 0.40 for XGBoost, and 0.40 for TCN.",
             "Mention that equal weights (0.33 each) can work nearly as well in practice, and that weight optimization is sensitive to validation set size."),
            ("What is dilation in the TCN and why use it?",
             "Dilation in a causal convolution means inserting zeros between filter weights, effectively increasing the receptive field without increasing the number of parameters. With dilation factors of 1, 2, 4, a TCN with 3 blocks and kernel size 3 has a receptive field of 1 + (3-1)*1 + (3-1)*2 + (3-1)*4 = 15 time steps, covering the full 8-cycle input window with capacity to spare. Dilation allows the TCN to capture both short-term (recent cycle) and longer-term (multi-cycle trend) patterns.",
             "Draw the dilated convolution diagram: the gaps between taps show how far back in the sequence each output can 'see'."),
            ("What is a conformal calibration set and why must it be fully held out?",
             "The calibration set consists of 6 batteries that are never used during model training. After training, the models make predictions on each calibration battery, and the residuals (absolute prediction errors) are stored. At test time, the 90th percentile of these residuals is added as the conformance margin to produce a 90% coverage interval. If the calibration set batteries were used during training, the residuals would be smaller than true generalization residuals, leading to under-coverage — the intervals would be too narrow.",
             "Contaminating the calibration set is a common implementation mistake that makes coverage appear valid in evaluation but fails in deployment."),
        ]
    },
    {
        "slide": "Slide 11-12: Dataset and Statistics",
        "items": [
            ("Why are B0049-B0056 flagged as anomalous?",
             "These cells exhibit irregular capacity trajectories inconsistent with smooth monotonic degradation: sudden capacity drops of 10-20% followed by partial recovery, large cycle-to-cycle variance. These patterns are likely due to measurement artifacts, experimental interruptions, or actual cell defects. They are flagged in pipeline.yaml under anomaly.flagged_battery_prefixes and excluded from the conformal calibration set to prevent their irregular residuals from distorting the calibration.",
             "Mention that flagged batteries are still included in training — their anomalous patterns may actually help the anomaly detector learn what abnormal behavior looks like."),
            ("What does the 91.3% coverage mean and is it a good result?",
             "91.3% coverage means that out of all test-set cycles, 91.3% of the true RUL values fell within the model's predicted 90% confidence interval. The target was at least 90%. Slightly exceeding the target (91.3% vs 90%) is the preferred direction for safety-critical applications — conservative (wide) intervals are safer than over-confident narrow ones. Significantly under-coverage (e.g., 85%) would indicate a calibration failure requiring investigation.",
             ""),
            ("Why is RMSE 18.4 cycles a good result for RUL prediction?",
             "RMSE of 18.4 cycles should be evaluated relative to the RUL range. The target variable ranges from 0 to 612 cycles (mean 142, std 98). An RMSE of 18.4 is approximately 13% of the mean RUL — reasonable for a complex non-linear prediction task. For comparison, a naive baseline of always predicting the mean (RUL=142) would have RMSE equal to the standard deviation (98 cycles). Our model's RMSE of 18.4 represents an 81% improvement over this naive baseline.",
             ""),
        ]
    },
    {
        "slide": "Slide 13-14: Step-by-Step Output and Dashboard",
        "items": [
            ("What is a SHAP value and how do you interpret it for RUL prediction?",
             "SHAP (SHapley Additive exPlanations) assigns each feature a contribution value for a specific prediction, based on cooperative game theory. A positive SHAP value for capacity_ahr means that this battery's current capacity is higher than average, which pushed the RUL prediction higher. A negative SHAP for capacity_fade_rate means the fade rate is faster than average, pulling the RUL prediction down. The sum of all SHAP values plus the base value equals the final prediction.",
             "SHAP values are locally faithful: they explain one specific prediction, not the model's global behavior. This is important for battery engineers who care about why THIS battery was flagged, not about average behavior."),
            ("What is a counterfactual explanation in this context?",
             "A counterfactual asks: what would the RUL prediction be if a specific feature had a different value, holding all other features constant? For example: 'If B0047 had been operating at 4 deg C instead of 24 deg C, what would the predicted RUL be?' The model reruns the prediction with temperature_c changed to 4.0. This helps engineers understand the sensitivity of the prediction to each condition and plan interventions.",
             "Counterfactuals are not causal claims — they are sensitivity analyses. The model is not simulating the physical effect of changing temperature; it is only showing how its prediction changes under different inputs."),
            ("What is the supervisor audit and who is it designed for?",
             "The supervisor audit is an automated quality gate designed for production deployment. It checks three criteria before accepting a prediction batch: (1) model confidence score >= 0.70, (2) RMSE <= 100 cycles, (3) CV-RMSE ratio <= 1.5. If all pass, verdict is PASS. If one is marginal (between pass and auto-approve thresholds), verdict is CONDITIONAL PASS, triggering manual review. If any fail, verdict is FAIL and the batch is rejected. It is designed for a fleet operator who needs to know if a particular batch of predictions is trustworthy before acting on them.",
             ""),
        ]
    },
    {
        "slide": "Slide 15-16: Battery Example B0047",
        "items": [
            ("Why is the conformal interval [42, 84] for B0047 with a true RUL of 61?",
             "The interval [42, 84] is correctly calibrated — the true RUL of 61 falls inside the interval. The width of 42 cycles reflects the empirical 90th-percentile residual from the calibration set for room-temperature batteries. B0047 is an in-distribution battery (room temperature, smooth degradation) so the interval is relatively tight. A cold-temperature battery at the same cycle would have a wider interval due to the 1.20x LOBO safety factor.",
             ""),
            ("If the RAG knowledge base contained no relevant documents, what would the system return?",
             "The system would return: 'The available evidence does not support a conclusion on this point.' This is enforced by Rule 4 in the structured prompt: 'If the retrieved evidence is insufficient to answer the question, say explicitly: The available evidence does not support a conclusion on this point. Do not fabricate an answer.' The model is instructed not to fall back on its pre-training knowledge if retrieved evidence is insufficient.",
             "This is one of the key properties that distinguishes RAG from a plain LLM call — the grounding constraint is explicit and verifiable."),
            ("What would cause B0047 to receive a HIGH risk classification?",
             "B0047 would receive HIGH risk if: (1) the capacity at cycle 150 had been 1.61 Ah or lower (close to EoL threshold, suggesting imminent failure), (2) the capacity_fade_rate had accelerated significantly to -0.010 Ah/cycle or faster, or (3) the predicted RUL dropped below 20 cycles, which would push P(fail within 20 cycles) above 0.70. The hazard model uses a logistic function of 10 features, so no single feature triggers HIGH risk — it is a combination.",
             ""),
            ("How would the explanation differ for a battery with HIGH risk compared to B0047?",
             "For a HIGH-risk battery, the RAG engine would retrieve chunks discussing advanced degradation, late-life failure modes, and risk thresholds. The pipeline data in the prompt would include high failure_prob, low RUL, and potentially an ANOMALY flag. The generated explanation would reference these values and the retrieved mechanisms, noting the contradiction if, say, the anomaly score is low but failure probability is high — per Rule 7 in the prompt, the system is required to flag such contradictions explicitly.",
             ""),
        ]
    },
]

QA_KO = [
    {
        "slide": "슬라이드 1-2: 서론 - 프로젝트 개요",
        "items": [
            ("잔여 유효 수명(RUL)의 정의는 무엇인가요?",
             "RUL은 배터리의 방전 용량이 초기 정격 용량의 80% 미만(2.0 Ah 공칭 셀의 경우 1.6 Ah)으로 떨어지기 전까지 남은 충방전 사이클 수입니다. 이는 IEC 62660-1 표준과 NASA PCoE 벤치마크 관행을 따릅니다.",
             "용량-사이클 곡선을 그리고 1.6 Ah를 교차하는 지점을 표시하세요. 그것이 EoL 지점이며, 현재 사이클에서 그 교차점까지의 거리가 RUL입니다."),
            ("왜 80% 용량 임계값을 EoL로 사용하나요?",
             "80%는 IEC 62660-1에 확립된 업계 표준 임계값으로 전기차 응용에서 널리 채택되었습니다. 80% 미만에서 배터리의 주행 거리가 20% 이상 감소하며, 이는 대부분의 전기차 사용자에게 허용 불가능한 수준으로 간주됩니다.",
             "pipeline.yaml에서 이 임계값을 설정 가능하다고 언급하세요(eol.capacity_fade_fraction = 0.80)."),
            ("'엔드-투-엔드'는 이 시스템에서 무엇을 의미하나요?",
             "엔드-투-엔드는 파이프라인이 원시 .mat 센서 파일부터 배포된 인터랙티브 대시보드까지 수동 개입 없이 모든 것을 처리함을 의미합니다. 데이터 파싱, 스키마 검증, 특성 엔지니어링, 모델 훈련, 순응형 보정, 설명 생성의 모든 단계가 자동화됩니다.",
             ""),
            ("왜 다른 데이터셋 대신 NASA PCoE 데이터셋을 선택했나요?",
             "NASA PCoE 데이터셋은 여러 온도에서의 통제된 실험 조건을 갖춘 가장 널리 벤치마크된 공개 Li-ion 배터리 데이터셋 중 하나입니다. 34개 셀과 셀당 168-616 사이클을 포함하여 앙상블 모델을 훈련하고 평가하기에 충분합니다.",
             "한계를 인정하세요: 일정 전류 하에서의 실험실 데이터는 실제 가변 부하 전기차 충전을 대표하지 않을 수 있습니다."),
        ]
    },
    {
        "slide": "슬라이드 3-4: 문제 정의 - 열화 메커니즘",
        "items": [
            ("SEI 층 성장이란 무엇이고 왜 지배적인 열화 메커니즘인가요?",
             "고체 전해질 계면(SEI)은 첫 번째 충전 시 전해질이 리튬과 반응하면서 흑연 음극 표면에 형성됩니다. 매 사이클 비가역적으로 성장하여 순환 리튬을 소모하고 내부 저항을 증가시킵니다. 실온에서 정상 C율로 작동하는 배터리에서 첫 번째 사이클부터 평생 동안 지속적으로 작동하므로 지배적인 메커니즘입니다.",
             "저온에서는 리튬 도금이 지배적이고, 수명 후기에는 양극 균열이 지배적이 된다는 점과 대조하세요."),
            ("콜드 스타트 문제란 무엇이고 어떻게 해결하나요?",
             "콜드 스타트 문제는 운영 이력이 없는 새 배터리에 예측이 필요할 때 발생합니다. LOBO(Leave-One-Battery-Out) 방식은 한 번에 하나의 배터리를 보정에 사용하여 다른 배터리의 잔차를 대리로 사용함으로써 이를 해결합니다. 저온 배터리의 경우 1.20x 안전 계수가 추가로 구간을 확장합니다.",
             ""),
            ("왜 불확실성 정량화가 필요한가요? 점 예측으로 충분하지 않나요?",
             "점 예측 RUL=50은 의사결정에 모호합니다. 실제 구간이 [48, 52]라면 사이클 45에서 유지보수를 계획할 수 있습니다. 실제 구간이 [20, 80]이라면 안전을 위해 사이클 15에서 계획해야 합니다. 두 번째 경우는 첫 번째보다 30 사이클 일찍 교체를 의미하며 상당한 경제적 비용입니다.",
             "비유: '20도C' 날씨 예보는 '20 +/- 2도C'보다 유용하지 않습니다."),
        ]
    },
    {
        "slide": "슬라이드 5-6: 솔루션 - 모델 아키텍처",
        "items": [
            ("왜 하나의 큰 모델 대신 세 가지 모델 유형을 사용하나요?",
             "단일 아키텍처가 보편적으로 최적이 아닙니다. 지수 기준선은 물리적으로 해석 가능하지만 경직됩니다. XGBoost는 특성 상호작용을 효율적으로 포착하지만 시간적 시퀀스를 모델링할 수 없습니다. TCN은 시간적 역학을 포착하지만 더 많은 데이터가 필요합니다. 앙상블은 세 가지 모두의 장점을 활용합니다.",
             ""),
            ("순응형 예측이란 무엇이고 이 응용에서 베이지안 신용 구간보다 왜 더 좋은가요?",
             "순응형 예측은 분포 가정 없이 수학적 보장을 제공합니다: 예측 구간이 실제 값을 90% 이상의 시간 동안 포함합니다. 베이지안 신용 구간은 사전 분포 지정이 필요한데, 배터리 열화에는 잘 정당화된 사전 분포가 없습니다.",
             "트레이드오프: 순응형 구간이 올바르게 지정된 베이지안 구간보다 더 넓을 수 있습니다."),
            ("RAG 엔진이란 무엇이고 환각을 어떻게 방지하나요?",
             "RAG는 검색 단계(임베딩 유사성으로 관련 문서 청크 찾기)와 생성 단계(Gemma가 답변 생성)를 결합합니다. 핵심 환각 방지 메커니즘은 모델을 검색된 증거만 사용하도록 명시적으로 제한하는 9개 규칙이 포함된 구조화된 프롬프트입니다.",
             ""),
        ]
    },
    {
        "slide": "슬라이드 7-8: 문헌 고찰",
        "items": [
            ("프로젝트가 Severson et al. (2019)를 어떻게 확장하나요?",
             "Severson et al.은 처음 100 사이클의 특성으로 총 사이클 수명을 예측했습니다. 이 프로젝트는 (1) 모든 사이클에서 RUL 예측, (2) 보정된 불확실성 구간 제공, (3) 생존 분석, 이상 탐지, RAG 설명을 포함한 사후 분석 파이프라인으로 확장합니다.",
             ""),
            ("GroupKFold가 왜 필수적이고 일반 KFold를 사용하면 무슨 문제가 발생하나요?",
             "일반 KFold는 사이클을 무작위로 훈련과 테스트 세트에 분할합니다. B0005의 사이클 1-50은 훈련에, 51-100은 테스트에 들어갈 수 있어 데이터 누출이 발생합니다. 모델이 평가 중인 배터리의 미래 맥락을 봐서 낙관적으로 편향된 RMSE 추정치가 생성됩니다.",
             "이것은 배터리 ML 논문에서 흔한 실수입니다. 항상 사이클 인덱스가 아닌 배터리 ID로 분할하세요."),
        ]
    },
    {
        "slide": "슬라이드 9-10: 방법론 상세",
        "items": [
            ("TCN은 어떤 손실 함수를 사용하고 왜 분위 손실 대신 MSE를 사용하나요?",
             "TCN은 단일 출력 RUL 회귀 헤드에 평균 제곱 오차(MSE)를 사용합니다. 분위 회귀는 XGBoost(세 개 별도 모델, alpha=0.05/0.50/0.95)가 처리합니다. TCN에 MSE를 사용하면 훈련이 단순화되고 잘 보정된 점 추정값이 생성됩니다.",
             ""),
            ("TCN의 팽창(dilation)이란 무엇이고 왜 사용하나요?",
             "인과 합성곱의 팽창은 필터 가중치 사이에 0을 삽입하여 매개변수 수를 늘리지 않고 효과적으로 수용 필드를 확장합니다. 팽창 계수 1, 2, 4로 TCN은 8 사이클 입력 윈도우 전체를 커버하며 단기와 장기 패턴을 모두 포착합니다.",
             ""),
            ("앙상블 가중치는 어떻게 결정되나요?",
             "가중치는 보류된 검증 세트에서 가중 조합의 평균 제곱 오차를 최소화하여 학습됩니다. 제약 조건: 가중치 합=1, 모두 비음수. 결과: 기준선 약 0.20, XGBoost 0.40, TCN 0.40.",
             ""),
        ]
    },
    {
        "slide": "슬라이드 11-12: 데이터셋 및 통계",
        "items": [
            ("B0049-B0056이 이상으로 플래그 처리되는 이유는 무엇인가요?",
             "이 셀들은 10-20%의 갑작스러운 용량 하락 후 부분 회복, 사이클 간 큰 분산 등 부드러운 단조 열화와 일치하지 않는 불규칙한 용량 궤적을 보입니다. 측정 아티팩트, 실험 중단, 또는 실제 셀 결함 때문일 가능성이 있습니다.",
             ""),
            ("91.3% 커버리지가 좋은 결과인가요?",
             "91.3% 커버리지는 테스트 세트 사이클의 91.3%에서 실제 RUL 값이 모델의 예측 90% 신뢰 구간 내에 포함되었음을 의미합니다. 목표는 최소 90%였습니다. 안전-중요 응용에서는 목표를 약간 초과하는 것(91.3% vs 90%)이 과도하게 자신감 있는 좁은 구간보다 바람직합니다.",
             ""),
            ("RMSE 18.4 사이클이 왜 좋은 결과인가요?",
             "RMSE 18.4는 RUL 범위(0-612, 평균 142, 표준편차 98)에 비해 평가해야 합니다. 항상 평균(RUL=142)을 예측하는 나이브 기준선은 RMSE가 표준편차(98 사이클)와 같습니다. 우리 모델의 RMSE 18.4는 이 나이브 기준선 대비 81% 개선입니다.",
             ""),
        ]
    },
    {
        "slide": "슬라이드 13-14: 단계별 출력 및 대시보드",
        "items": [
            ("SHAP 값이란 무엇이고 RUL 예측에서 어떻게 해석하나요?",
             "SHAP(SHapley Additive exPlanations)는 협력 게임 이론을 기반으로 각 특성에 특정 예측에 대한 기여값을 할당합니다. capacity_ahr에 대한 양의 SHAP 값은 이 배터리의 현재 용량이 평균보다 높아 RUL 예측이 더 높아졌음을 의미합니다. capacity_fade_rate에 대한 음의 SHAP는 감쇄율이 평균보다 빠르다는 것을 의미합니다.",
             "SHAP 값은 지역적으로 충실합니다: 모델의 전역 동작이 아닌 하나의 특정 예측을 설명합니다."),
            ("감독자 감사란 무엇이고 누구를 위해 설계되었나요?",
             "감독자 감사는 생산 배포를 위한 자동화된 품질 게이트입니다. 예측 배치를 수락하기 전에 세 가지 기준을 확인합니다: (1) 모델 신뢰도 점수 >= 0.70, (2) RMSE <= 100 사이클, (3) CV-RMSE 비율 <= 1.5. 예측 배치를 신뢰할 수 있는지 여부를 알아야 하는 운영자를 위해 설계되었습니다.",
             ""),
        ]
    },
    {
        "slide": "슬라이드 15-16: 예시 배터리 B0047",
        "items": [
            ("실제 RUL 61에 대한 순응형 구간이 [42, 84]인 이유는 무엇인가요?",
             "구간 [42, 84]는 올바르게 보정되어 있습니다. 실제 RUL 61이 구간 내에 있습니다. 42 사이클의 너비는 실온 배터리 보정 세트의 90 백분위수 잔차를 반영합니다. B0047은 분포 내 배터리(실온, 부드러운 열화)이므로 구간이 상대적으로 좁습니다.",
             ""),
            ("RAG 지식 베이스에 관련 문서가 없다면 어떻게 되나요?",
             "시스템은 '사용 가능한 증거가 이 점에 대한 결론을 지원하지 않습니다'를 반환합니다. 이는 구조화된 프롬프트의 규칙 4에 의해 강제됩니다: 검색된 증거가 질문에 답하기에 불충분하다면, '사용 가능한 증거가 이 점에 대한 결론을 지원하지 않습니다'라고 명시적으로 말하세요.",
             ""),
            ("B0047이 높은 위험 분류를 받으려면 무엇이 필요한가요?",
             "B0047은 다음 경우 높은 위험을 받을 것입니다: (1) 사이클 150에서 용량이 1.61 Ah 이하 (EoL 임계값에 가까움), (2) capacity_fade_rate가 -0.010 Ah/사이클 이상으로 가속화, (3) 예측 RUL이 20 사이클 미만으로 떨어져 P(20사이클 내 고장) > 0.70.",
             ""),
        ]
    },
]


def build_qa(doc, data, lang):
    if lang == "en":
        title_text = "Battery AI Co-Scientist"
        sub_text = "Professor Q&A Guide — Potential Questions Per Slide With Full Answers"
        intro = ("This document contains likely exam and presentation questions organized by slide. "
                 "Each question is followed by a complete answer and, where appropriate, a presentation tip. "
                 "Questions range from conceptual ('What is RUL?') to technical ('Why GroupKFold?') "
                 "to critical ('Why not Bayesian CIs?'). Be prepared to answer any of these.")
    else:
        title_text = "Battery AI Co-Scientist"
        sub_text = "교수 질문 & 답변 가이드 - 슬라이드별 예상 질문 및 완전한 답변"
        intro = ("이 문서는 슬라이드별로 정리된 예상 시험 및 발표 질문을 포함합니다. "
                 "각 질문에는 완전한 답변이 포함되어 있으며, 적절한 경우 발표 팁도 제공됩니다.")

    # Title
    tp = doc.add_paragraph()
    tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    tr = tp.add_run(title_text)
    tr.font.size = Pt(28)
    tr.font.bold = True
    tr.font.color.rgb = DARK_BLUE

    sp = doc.add_paragraph()
    sp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sr = sp.add_run(sub_text)
    sr.font.size = Pt(14)
    sr.font.color.rgb = MID_BLUE

    doc.add_paragraph()
    ip = doc.add_paragraph(intro)
    ip.paragraph_format.left_indent = Inches(0.3)
    ip.paragraph_format.right_indent = Inches(0.3)
    for run in ip.runs:
        run.font.size = Pt(11)
        run.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
        run.font.italic = True

    doc.add_page_break()

    for section in data:
        add_slide_header(doc, section["slide"])
        for i, item in enumerate(section["items"], 1):
            q, a = item[0], item[1]
            tip = item[2] if len(item) > 2 else ""
            add_q(doc, i, q)
            add_a(doc, a)
            if tip:
                add_tip(doc, tip)


def make_qa_en():
    doc = Document()
    for sec in doc.sections:
        sec.top_margin = Cm(2.5)
        sec.bottom_margin = Cm(2.5)
        sec.left_margin = Cm(2.8)
        sec.right_margin = Cm(2.5)
    build_qa(doc, QA_EN, "en")
    path = OUT / "Professor_QA_English_Detailed.docx"
    doc.save(str(path))
    print(f"  Saved: {path}")


def make_qa_ko():
    doc = Document()
    for sec in doc.sections:
        sec.top_margin = Cm(2.5)
        sec.bottom_margin = Cm(2.5)
        sec.left_margin = Cm(2.8)
        sec.right_margin = Cm(2.5)
    build_qa(doc, QA_KO, "ko")
    path = OUT / "Professor_QA_Korean_Detailed.docx"
    doc.save(str(path))
    print(f"  Saved: {path}")


if __name__ == "__main__":
    print("Building Q&A (English)...")
    make_qa_en()
    print("Building Q&A (Korean)...")
    make_qa_ko()
    print("Done.")
