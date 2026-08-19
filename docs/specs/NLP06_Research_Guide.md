# NLP-06 Research Guide
## Unique Contributions, Base Paper Analysis, and Experimental Plan
### Everything needed to write the paper and run the experiments

---

## 1. The Research Question

> *In a federated, differentially private clinical anomaly detection setting trained on longitudinal ward vitals, at what privacy budget (ε) does model utility fall below a clinically meaningful threshold — and do LSTM/GRU sequence architectures tolerate DP-SGD gradient noise better than CNN baselines?*

This is the question the paper answers. Everything in the research track exists to answer it rigorously.

---

## 2. What Each Base Paper Does — Precise Technical Summary

### b1 — Shukla et al. (2025) — Scientific Reports
**"Federated learning with differential privacy for breast cancer diagnosis"**

| Aspect | Detail |
|---|---|
| Data | Breast Cancer Wisconsin Diagnostic — 569 records, 32 features, static tabular |
| Model | Random Forest (non-FL baseline) + Feed-Forward Neural Network (FL) |
| FL framework | TensorFlow Federated (TFF) |
| DP mechanism | Gaussian mechanism, DP-SGD via TensorFlow Privacy |
| Privacy accounting | Basic composition (NOT Rényi DP / Moments Accountant) |
| ε tested | 0.5, 1.0, 1.9, 3.0, 5.0, ∞ (no DP) |
| Best result | 96.1% accuracy at ε = 1.9, δ = 10⁻⁵ |
| Clients | 10 simulated clients, IID distribution |
| Evaluation metrics | Accuracy, precision, recall, F1 per class |
| What it lacks | No time-series data, no sequence model, no conversational interface, no non-IID heterogeneity experiment, no RAG, no clinical deployment scenario |

**What b1 gives NLP-06:**
- The ε-utility tradeoff framing — this is the template for C1
- The ε = 1.9 result as a comparison point (they claim this is the "optimal" privacy budget)
- Evidence that DP introduces only small accuracy drops on tabular data — your hypothesis is this changes significantly for time-series clinical vitals

---

### b2 — Tanveer et al. (2025) — DIGITAL HEALTH (SAGE)
**"Balancing privacy and performance in healthcare: A federated learning framework for sensitive data"**

| Aspect | Detail |
|---|---|
| Data | Stroke Prediction Dataset (Kaggle) — 5110 records, 12 features, static tabular |
| Model | 3-layer fully-connected DNN |
| FL framework | Flower (flwr) — same as NLP-06 |
| DP mechanism | DP-SGD, L2 norm clip = 1.0, noise multiplier = 1.0 |
| Privacy accounting | TensorFlow Privacy compute_dp_sgd_privacy — Rényi DP accountant |
| ε achieved | ε ≈ 0.69 at δ = 10⁻⁵ after 10 rounds |
| Clients | 5 simulated clients, IID + non-IID tested |
| Architecture | 5-layer pipeline: Edge → Privacy → Aggregation → App → Decision |
| Evaluation metrics | Accuracy 93%, F1 0.82, precision 0.78, recall 0.89 |
| What it lacks | Static dataset (no temporal sequences), no LLM or RAG, no alert system, no clinical anomaly detection, non-IID analysis is basic |

**What b2 gives NLP-06:**
- The 5-layer architectural template (Edge/Privacy/Aggregation/App layers) — directly adopted
- Flower + DP-SGD integration pattern — code-level reference
- ε ≈ 0.69 as a reference operating point for comparison
- Non-IID partitioning approach using work_type attribute — analogous to our ward condition skewing
- Statistical validation approach (95% CI across evaluation rounds)

---

### b3 — Mosaiyebzadeh et al. (2025) — MDPI Electronics
**"Privacy-Preserving Federated Learning-Based Intrusion Detection System for IoHT Devices"**

| Aspect | Detail |
|---|---|
| Data | wustl-ehms-2020 (36 features, IoHT network traffic + biometric) and ECU-IoHT (7 features, network traffic) |
| Models | Feedforward DNN + CNN — compared directly |
| FL framework | Custom FL with Opacus (PyTorch) — same DP library as NLP-06 |
| DP mechanism | DP-SGD via Opacus PrivacyEngine, gradient clip norm = 10⁻⁴ |
| Noise levels tested | 0.5 and 1.5 — two points only |
| ε results | Noise=1.5 → ε ≈ 0.43 (wustl) / 0.34 (ECU); Noise=0.5 → ε ≈ 6.69 / 5.20 |
| Clients | 100 FL rounds, single-machine simulation |
| Explainability | SHAP values — a strong addition |
| What it lacks | Only 2 noise levels (not a systematic ε-sweep), network traffic not clinical vitals, no temporal/sequence model (DNN and CNN only, no LSTM/GRU), no RAG or LLM, no clinical anomaly scenario |

**What b3 gives NLP-06:**
- Opacus implementation pattern — most directly applicable code reference
- CNN vs DNN comparison design — template for our CNN vs LSTM/GRU comparison (C2)
- Evidence that noise = 1.5 is a practical operating point worth testing
- SHAP explainability as an optional addition if time permits
- Demonstrates the pattern of using two noise levels — we improve on this with a full ε-sweep

---

### b4 — Wada et al. (2025) — npj Digital Medicine
**"Retrieval-augmented generation elevates local LLM quality in radiology contrast media consultation"**

| Aspect | Detail |
|---|---|
| LLM | Llama 3.2-11B (local, via GroqCloud for benchmarking; on-premise hardware for validation) |
| RAG knowledge base | ACR Manual on Contrast Media, ESUR guidelines, institutional protocols — 66 chunks, avg 337 chars, max 600 tokens |
| Embedding model | OpenAI text-embedding-3-large |
| Retrieval | Hybrid: semantic vector search + keyword (TopK = 4, cosine similarity) |
| Deployment | Dify platform (v0.10.0), temperature = 0.2 |
| Evaluation | 100 synthetic ICM consultation scenarios; 1 radiologist (blinded) + 3 LLM judges (GPT-4o, Gemini 2.0 Flash Thinking, Claude 3.5 Sonnet) |
| Key result | Hallucinations: 8% → 0% with RAG (χ²Yates = 6.38, p = 0.012); mean rank improved by 1.3 (Z = -4.82, p < 0.001) |
| Response time | RAG-enhanced: 2.6s vs cloud models 4.9-7.3s |
| Cloud comparison | GPT-4o mini, Gemini 2.0 Flash, Claude 3.5 Haiku |
| What it lacks | No FL or DP, no continuous monitoring, no federated training, centralized architecture, static knowledge base (no real-time vitals), domain-specific only (radiology) |

**What b4 gives NLP-06:**
- Proof that local LLM + RAG reaches near-zero hallucination — directly justifies our RAG design choice
- Hybrid retrieval design (semantic + keyword) — adopt this pattern
- Temperature = 0.2 for clinical LLMs — adopt
- Evaluation methodology: synthetic scenarios + blinded radiologist — template for evaluating C3
- The "0% hallucinations" claim as a benchmark — we target the same for our alert summaries

---

### b5 — Miao et al. (2025) — JMIR
**"Improving Large Language Model Applications in Medical and Nursing Domains With RAG: Scoping Review"**

| Aspect | Detail |
|---|---|
| Scope | 67 studies from Nov 2022 – May 2025; PubMed, Web of Science, IEEE Xplore, arXiv |
| RAG categories identified | Text-based RAG (54%), KG-enhanced (25%), Agentic (9%), Multimodal (3%), Plug-and-play (9%) |
| RAG workflow stages | Intent recognition → Knowledge retrieval → Knowledge integration → Generation |
| Key finding 1 | 94% of studies target medical (physician) workflows; only 6% target nursing |
| Key finding 2 | None of the 67 studies integrate real-time telemetry or continuous vital monitoring |
| Key finding 3 | Only 26/67 studies include explicit reasoning support; none use causal modeling |
| Key finding 4 | Only 9/67 explicitly address patient data privacy |
| Ethical gap | Only 1 study addresses patient safety flags; 2 address fairness |
| What it lacks | It is a review — no system built, no experimental results, no FL or DP integration |

**What b5 gives NLP-06:**
- Authoritative confirmation that NLP-06's gap is real — a peer-reviewed scoping review saying "no study integrates real-time telemetry with RAG" is the strongest possible related-work citation
- The 4-stage RAG workflow (intent → retrieval → integration → generation) as a framework to describe our RAG pipeline
- The nursing gap (6%) — NLP-06 targets ward nursing/clinical staff, directly addressing this
- Privacy gap (9/67 address privacy) — NLP-06 addresses it, positioning is clean

---

## 3. The Gap Table — What Every Paper Misses

| Capability | b1 | b2 | b3 | b4 | b5 | **NLP-06** |
|---|---|---|---|---|---|---|
| Federated learning | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ |
| Differential privacy (DP-SGD) | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ |
| Time-series / sequence data | ❌ | ❌ | Partial* | ❌ | ❌ | ✅ |
| LSTM / GRU model | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Systematic ε-sweep (5+ points) | ✅ | ❌ | ❌ | N/A | N/A | ✅ |
| Non-IID ward heterogeneity | ❌ | Partial | ❌ | N/A | N/A | ✅ |
| Local LLM + RAG | ❌ | ❌ | ❌ | ✅ | Review | ✅ |
| Patient-scoped retrieval | ❌ | ❌ | ❌ | Partial | ❌ | ✅ |
| Real-time vital monitoring | ❌ | ❌ | Partial* | ❌ | ❌ | ✅ |
| Alert → RAG clinician summary | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Privacy + RAG combined | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Clinical ward deployment scenario | ❌ | ❌ | ❌ | Partial | ❌ | ✅ |
| Rényi DP / Moments Accountant | ❌ | ✅ | ❌ | N/A | N/A | ✅ |

*b3 uses IoHT network traffic (not clinical vitals), making it adjacent but not equivalent to clinical time-series.

---

## 4. NLP-06's Unique Contributions — Precise Statements

### C1 — Empirical ε-utility floor for federated clinical time-series anomaly detection

**What we do:** Sweep ε across 6 levels (0.5, 1, 2, 4, 8, no-DP baseline), train LSTM, GRU, and CNN under FL+DP-SGD on Synthea-generated longitudinal ward vitals, and identify the ε value below which F1 and AUC-ROC drop below clinically actionable thresholds. Report false negative rate separately — missed anomalies are the safety-critical metric in clinical settings.

**Why it's novel:** b1 does this on static tabular data (569 records, 32 features). Our data is longitudinal sequences (12-timestep sliding windows of 6 vitals). The temporal structure of clinical vitals creates a fundamentally different learning signal — the question is whether the ε-utility curve has the same shape or a different profile. Nobody has published this for clinical vital-sign sequences under FL+DP.

**What the result tells clinicians:** "At ε < X, the federated anomaly model misses more than Y% of true anomalies — below this level the privacy guarantee costs more than the clinical benefit." This is an actionable deployment recommendation.

---

### C2 — LSTM/GRU vs CNN robustness to DP-SGD gradient noise on temporal clinical data

**What we do:** Compare the ε-utility curves of LSTM, GRU, and CNN on the same data, same FL setup, same noise levels. Test the hypothesis that temporal averaging in LSTM/GRU naturally smooths injected Gaussian gradient noise, making sequence models more noise-tolerant than CNNs at low epsilon.

**Why it's novel:** b3 compares DNN vs CNN under DP but on IoHT network traffic (not clinical vitals) and at only 2 noise levels (not a sweep). Our comparison uses time-series sequence models (LSTM, GRU) which have not been compared against CNNs under DP-SGD in a clinical context. The temporal architecture hypothesis has never been tested under DP noise conditions.

**What the result tells researchers:** If confirmed, this gives future FL-for-healthcare practitioners a principled reason to choose LSTM/GRU over CNN when strong DP guarantees are required. If disconfirmed, it tells practitioners that temporal architecture provides no advantage, simplifying model selection.

**Statistical test:** Mann-Whitney U test on F1 scores across 5 seeds at each ε level comparing LSTM/GRU vs CNN. Report effect size (Cohen's d or rank-biserial correlation).

---

### C3 — RAG-FL alert interface: anomaly alerts with patient-scoped clinical context summaries

**What we do:** Design, implement, and evaluate a novel interface where a federated anomaly detection alert automatically triggers a patient-scoped RAG query, retrieving clinically relevant history and generating a plain-language summary for the attending clinician. The FL model and RAG pipeline are architecturally decoupled but semantically connected at the alert layer.

**Why it's novel:** No paper in b1-b5, and to our knowledge no paper in the 67 studies reviewed by b5, implements a system where a privacy-preserving federated model's output directly triggers a patient-scoped RAG-LLM response. This is C3's core novelty — not that RAG and FL both exist, but that the alert output of a DP-protected FL model becomes the semantic query input to a grounded LLM. The two systems interact through the alert, not through shared training.

**Evaluation (three-layer, following b4's methodology):**

1. **Human rubric** — Structured rubric across 20-30 alert scenarios: factual accuracy (0-3), relevance to flagged vitals (0-3), clinical completeness (0-3), hallucination absence (binary), conciseness (0-3). Two raters, inter-rater agreement via Cohen's kappa.

2. **LLM-as-judge** — Following b4 (Wada et al.), have GPT-4o and Claude score the same summaries against the same rubric. This grounds the evaluation in a peer-reviewed npj Digital Medicine methodology rather than only student ratings.

3. **Programmatic fact-verification** — Because you control the source records (ground truth), programmatically check whether every fact in each summary appears in the source patient record. This gives an objective, automatable hallucination rate — stronger than subjective rating and verifiable (unlike b4's claim, which relied on expert review). Target: match b4's 0% hallucination benchmark.

```python
# integration/fact_verifier.py
def verify_summary_facts(summary, source_record):
    """Extract claims from summary, check each against source record."""
    claims = extract_claims(summary)  # LLM or rule-based extraction
    verified, hallucinated = [], []
    for claim in claims:
        if claim_supported_by_record(claim, source_record):
            verified.append(claim)
        else:
            hallucinated.append(claim)
    hallucination_rate = len(hallucinated) / max(len(claims), 1)
    return hallucination_rate, hallucinated
```

---

### C4 — Tight Rényi DP composition across 100 FL rounds with provable ε < 2

**What we do:** Use Opacus's Rényi DP accountant to track cumulative privacy loss tightly across 100 FL training rounds, demonstrating that with appropriate noise multiplier and clipping norm, the total ε remains within NIST SP 800-226's recommended bound (ε ≤ 1 for strong privacy) even with substantial local training.

**Honest assessment:** This is NOT a novel contribution — Rényi DP and Opacus's accountant are standard. This is methodological rigor. In the paper, frame it as: "unlike b1 which uses basic DP composition, we use Rényi DP accounting to provide tighter and more meaningful privacy bounds."

**What it adds:** It lets you claim a provable privacy envelope rather than an estimated one, which strengthens the paper's privacy claims and differentiates from b1.

---

## 5. Experimental Design — Complete Specification

### 5.1 Data

**Primary dataset — Synthea (synthetic):**

| Parameter | Value |
|---|---|
| Generator | Synthea v3.x |
| Total patients | 1000 |
| Ward nodes | 5 (non-IID by condition) |
| Patients per ward | ~200 |
| Vitals used | HR, BP systolic, SpO2, resp rate, temperature, glucose |
| Window size | 12 timesteps × 6 vitals |
| Stride | 1 (overlapping windows) |
| Train/test split | 80/20 per ward (local); 100 held-out patients (global test set) |
| Anomaly label | NEWS2 aggregate score + MEWS cross-check (binary: 0 = normal, 1 = anomaly) |
| Expected class imbalance | ~10-15% anomalous (Synthea produces mostly healthy trajectories) |
| Imbalance handling | Weighted loss function (BCEWithLogitsLoss with pos_weight) |

**Secondary dataset — PTB-XL (real physiological signals) — CRITICAL for defensibility:**

| Parameter | Value |
|---|---|
| Dataset | PTB-XL (PhysioNet) — 21,837 real 12-lead ECGs, open access, no credentialing |
| Alternative | MIT-BIH Arrhythmia Database (real ECG with beat-level annotations) |
| Purpose | Validate that the ε-utility trend holds on REAL physiological data, not just synthetic |
| Anomaly label | Real diagnostic labels (PTB-XL provides SCP-ECG statements: normal vs abnormal) |
| Use in paper | Run the ε-sweep on PTB-XL and confirm the same trend appears |

**Why PTB-XL matters:** This is the single most important addition to the paper. Running the ε-sweep on one real dataset converts the paper from "a synthetic-data study" (a Q3 objection) into "a study validated on real physiological signals" (a Q2 position). A reviewer cannot say "this is all synthetic" once PTB-XL results confirm the trend. Even if PTB-XL is narrower than full ward vitals, it breaks the data-realism objection completely.

**Noise-injected Synthea (robustness validation):**

To counter the "Synthea is too clean" criticism, create a noised variant of the Synthea data:

```python
# shared/noise_injection.py
def inject_realistic_noise(vitals_df, noise_std_pct=0.05,
                            missingness_rate=0.10, artifact_rate=0.02):
    """Make Synthea data realistically messy."""
    noisy = vitals_df.copy()
    for col in VITALS:
        # Gaussian measurement noise (calibrated to sensor error literature)
        noise = np.random.normal(0, noisy[col].std() * noise_std_pct, len(noisy))
        noisy[col] = noisy[col] + noise
        # Random missingness (simulate real dropped readings)
        mask = np.random.random(len(noisy)) < missingness_rate
        noisy.loc[mask, col] = np.nan
        # Occasional equipment artifacts (implausible spikes)
        artifact_mask = np.random.random(len(noisy)) < artifact_rate
        noisy.loc[artifact_mask, col] *= np.random.uniform(1.5, 3.0)
    return noisy
```

Report results on both clean and noised Synthea. This directly answers "Synthea is unrealistic" with "the finding is robust to injected measurement noise and missingness."

### 5.2 Models

All three models receive identical input: `[batch, 12, 6]` tensor.

**LSTM:**
```python
class LSTMClassifier(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return torch.sigmoid(self.fc(out[:, -1, :]))
```

**GRU:**
```python
class GRUClassifier(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return torch.sigmoid(self.fc(out[:, -1, :]))
```

**CNN (1D):**
```python
class CNNClassifier(nn.Module):
    def __init__(self, input_size=6, num_filters=64):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(num_filters, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch, features, timesteps]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return torch.sigmoid(self.fc(x))
```

All models: ~50K-100K parameters. Keep small — this is not a model architecture paper.

### 5.3 FL Setup

| Parameter | Value |
|---|---|
| Framework | Flower (flwr) v1.6+ |
| Aggregation | FedAvg (weighted by local dataset size) |
| Rounds | 100 |
| Clients per round | All 5 (full participation) + dropout variant |
| Local epochs per round | 5 |
| Local batch size | 32 |
| Local optimizer | SGD (lr = 0.01, momentum = 0.9) |
| Communication | Docker containers via Flower gRPC (genuine process-level federation) |

**Genuine distributed federation (defensibility upgrade):**
Instead of in-process simulation, deploy each ward node as an isolated Docker container communicating via Flower's gRPC protocol. This provides real process-level federation — not just a loop pretending to be clients. If team hardware allows, run containers across 2-3 physical machines. This lets the paper claim genuine distributed FL rather than simulation.

**Client dropout and straggler robustness (defensibility upgrade):**
Run an additional experiment where 20% of clients randomly drop out per round (1 of 5 unavailable), plus artificial communication latency. Show FedAvg still converges. This directly answers the "you ignored real FL network challenges" objection.

```python
# research_track/federated/dropout_strategy.py
class DropoutFedAvg(fl.server.strategy.FedAvg):
    def configure_fit(self, server_round, parameters, client_manager):
        clients = super().configure_fit(server_round, parameters, client_manager)
        # randomly drop 20% of clients this round
        n_keep = max(1, int(len(clients) * 0.8))
        return random.sample(clients, n_keep)
```

### 5.4 DP Setup

| Parameter | Value |
|---|---|
| Library | Opacus v1.4+ |
| Mechanism | DP-SGD (Gaussian) |
| Gradient clipping norm | 1.0 (L2) |
| Privacy accountant | Rényi DP (Opacus default) |
| δ | 10⁻⁵ |
| Noise multipliers tested | Derived from target ε values |

**Mapping target ε → noise multiplier:**
Use `opacus.accountant.analysis.rdp.get_noise_multiplier()` to compute the noise multiplier required to achieve each target ε after 100 rounds × 5 local epochs.

```python
from opacus.accountant.analysis import rdp as privacy_analysis

def get_noise_multiplier_for_epsilon(target_epsilon, delta=1e-5,
                                      sample_rate=0.2, steps=500):
    return privacy_analysis.get_noise_multiplier(
        target_epsilon=target_epsilon,
        target_delta=delta,
        sample_rate=sample_rate,
        steps=steps,
        accountant='rdp'
    )
```

### 5.5 Epsilon Sweep — Full Experimental Grid

| Run ID | ε target | Privacy level | Expected noise multiplier |
|---|---|---|---|
| E0 | ∞ | No DP baseline | 0 (no noise) |
| E1 | 8.0 | Very weak | ~0.4 |
| E2 | 4.0 | Weak | ~0.7 |
| E3 | 2.0 | Moderate (NIST threshold) | ~1.0 |
| E4 | 1.0 | Strong | ~1.5 |
| E5 | 0.5 | Very strong | ~2.5 |

**Core grid:** 6 ε levels × 3 models (LSTM, GRU, CNN) × 5 seeds = **90 training runs on clean Synthea.**

**Validation grid (defensibility additions):**

| Dataset variant | Purpose | Runs |
|---|---|---|
| Clean Synthea | Main results | 90 (6 ε × 3 models × 5 seeds) |
| Noised Synthea | Robustness to measurement noise/missingness | Best model only, 6 ε × 5 seeds = 30 |
| PTB-XL (real signals) | Confirm ε-trend on real data | Best model only, 6 ε × 5 seeds = 30 |

The noised-Synthea and PTB-XL runs use only the best-performing model from the core grid (likely LSTM or GRU) — you don't need to re-run all three architectures. This keeps the additional compute manageable while directly answering the two biggest reviewer objections.

Each run records:
- F1 score (primary metric)
- Precision
- Recall
- AUC-ROC
- False negative rate (FNR) — clinically critical (missed anomalies)
- Actual achieved ε (from Opacus accountant)
- Training time per round

### 5.6 Seeds and Statistical Reporting

```python
SEEDS = [42, 123, 456, 789, 1337]

for seed in SEEDS:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # run full FL+DP training
    results[seed] = run_experiment(epsilon, model_type, seed)

# Report: mean ± std across seeds
mean_f1 = np.mean([results[s]['f1'] for s in SEEDS])
std_f1  = np.std([results[s]['f1'] for s in SEEDS])
```

**Never report single-run numbers in the paper.** Always mean ± std across 5 seeds.

### 5.7 Statistical Tests

**For C1 (ε-utility floor):**
- Friedman test across ε levels for each model — non-parametric (DP noise causes non-normal F1 distributions)
- Post-hoc Nemenyi test for pairwise ε comparisons
- Identify the ε* where F1 drops below 0.75 (proposed clinical threshold — justify in paper)

**For C2 (LSTM/GRU vs CNN):**
- At each ε level: Mann-Whitney U test comparing LSTM F1 scores vs CNN F1 scores (across 5 seeds)
- Report: U statistic, p-value, rank-biserial correlation (effect size)
- Run separately for GRU vs CNN as well
- Bonferroni correction for multiple comparisons (6 ε levels × 2 comparisons = 12 tests)

---

## 6. Paper Structure — Section by Section

### Abstract (≤250 words)
State: problem (privacy vs utility in federated clinical AI), method (FL+DP-SGD on Synthea ward vitals, LSTM/GRU vs CNN, RAG-FL alert interface), key results (ε* for clinical utility floor, LSTM/GRU robustness comparison), conclusion (dual-track system enables on-premise privacy-preserving clinical decision support).

### 1. Introduction
- Healthcare AI privacy problem — HIPAA, GDPR, reconstruction attacks
- FL solves data centralization but not gradient leakage
- DP-SGD solves gradient leakage but at accuracy cost
- RAG-LLM solves conversational clinical support but requires centralized deployment
- Gap: no system combines FL+DP anomaly detection with local RAG in one ward-level system
- Cite b5 directly: "a 2025 scoping review of 67 clinical RAG studies found no study integrating real-time telemetry or FL/DP protection"
- State 4 contributions clearly

### 2. Related Work
Structure as three paragraphs:

**Paragraph 1 — FL+DP in healthcare (b1, b2, b3):**
Summarize each paper's DP approach and limitation. Key differentiation: "all three use static tabular datasets and none employ sequence architectures capable of temporal anomaly detection."

**Paragraph 2 — Clinical RAG and local LLMs (b4, b5):**
Summarize b4's hallucination results and b5's gap findings. Key differentiation: "neither incorporates privacy-preserving federated learning; b4 transmits patient queries to cloud APIs during benchmarking."

**Paragraph 3 — The NLP-06 gap:**
"To our knowledge, no prior work combines a differentially private federated anomaly detector with a patient-scoped local RAG assistant in a unified ward-level deployment. NLP-06 addresses this gap."

### 3. System Architecture
- Figure 1: dual-track architecture diagram (two-track system)
- 5-layer pipeline (adopted from b2): Edge → Privacy → Aggregation → App → Decision
- Data flow between tracks (alert bridge)
- Privacy guarantees at each layer

### 4. Data and Preprocessing
- Synthea generation (justify: FL methodology research does not require real data; cite b2 and b3 which use non-clinical datasets)
- **PTB-XL real-signal validation set** (justify: confirms trend on real physiological data)
- **Noise-injected Synthea variant** (justify: tests robustness to realistic measurement noise/missingness)
- Non-IID ward partitioning (Table: ward configuration)
- Anomaly labeling (NEWS2 aggregate score + MEWS cross-check — cite Royal College of Physicians 2017)
- Sliding window design
- Class imbalance handling

### 5. FL + DP Methodology
- Model architectures (LSTM, GRU, CNN) with parameter counts
- Flower FL setup (FedAvg, rounds, clients)
- Opacus DP-SGD (clipping norm, noise multiplier, Rényi accountant)
- Privacy budget tracking across rounds (Figure: ε vs round curve — same as b2 Figure 5)
- Reference: NIST SP 800-226 (ε ≤ 1 recommendation)

### 6. Experiments and Results

**6.1 — Epsilon sweep (C1):**
- Table: F1, precision, recall, AUC, FNR for each (model × ε) combination, mean ± std
- Figure: ε-utility tradeoff curves for LSTM, GRU, CNN on same axes (main paper figure)
- Identify ε* for each model
- Discussion: what the FNR means clinically

**6.2 — LSTM/GRU vs CNN robustness (C2):**
- Figure: bar chart or line chart showing F1 gap between LSTM/GRU and CNN across ε values
- Table: Mann-Whitney U test results at each ε level
- Discuss: does temporal architecture provide noise robustness?

**6.3 — Real-data validation on PTB-XL:**
- Run the ε-sweep on real physiological signals
- Figure: PTB-XL ε-utility curve overlaid on Synthea curve
- Key claim: "the ε-utility trend observed on Synthea is confirmed on real ECG data, demonstrating the finding is not a synthetic-data artifact"

**6.4 — Robustness to measurement noise:**
- Compare clean Synthea vs noise-injected Synthea across ε levels
- Table: F1 at each ε for clean vs noised
- Key claim: "the finding is robust to injected Gaussian noise, 10% missingness, and equipment artifacts"

**6.5 — FL network robustness:**
- FedAvg convergence under 20% per-round client dropout + latency
- Figure: convergence curve with vs without dropout
- Key claim: "the federated model converges despite realistic client unavailability"

**6.6 — Non-IID heterogeneity (if run):**
- Varied non-IID skew intensity → measured convergence and final F1
- Table or figure

### 7. RAG-FL Alert Interface (C3)
- System design description (alert → RAG query → LLM summary → clinician display)
- Evaluation rubric (Table: criteria and scoring)
- Results: mean rubric scores across 20-30 scenarios, inter-rater Cohen's kappa
- Example: one annotated alert + summary (anonymized synthetic patient)

### 8. Discussion
- Interpretation of ε* — what it means for real ward deployment
- Interpretation of LSTM/GRU vs CNN finding
- Limitations (be explicit — this is required and reviewers expect it)
- Future work: real MIMIC data, real clinician evaluation, wearable integration

### 9. Conclusion
- Restate 4 contributions in one sentence each
- Reiterate the core finding (ε*, architecture recommendation)
- One sentence on practical implication for hospital AI deployment

---

## 7. Limitations Section — Mitigated and Defended

This section must appear in the paper. The strategy below fixes what can be fixed, mitigates what can be reduced, and defends what is inherent to scope. Each limitation is categorized:

- 🟢 **Fixed** — eliminated before submission, do not list as a limitation
- 🟡 **Mitigated** — substantially reduced; state the remaining gap honestly
- 🔴 **Defended** — inherent to scope; argue why it is acceptable

### 7.1 Action plan summary

| Limitation | Category | What you did | What to write |
|---|---|---|---|
| Synthea data | 🟡 Mitigated | Added PTB-XL real-signal validation + noise injection | See 7.2 |
| Simulated FL | 🟡 Mitigated | Dockerized wards (gRPC) + dropout/latency experiment | See 7.3 |
| Rule-based labels | 🟢 Fixed | Full NEWS2 + MEWS cross-check + sample clinician review | See 7.4 |
| C3 evaluation | 🟡 Mitigated | LLM-judge (per b4) + programmatic fact-verification | See 7.5 |
| RAG encryption | 🟢 Fixed | Encrypted volume (LUKS/AES-256) — remove from limitations | Do not list |
| No deployment | 🔴 Defended | Cite precedent from b1/b2/b3 | See 7.6 |

### 7.2 Synthetic data (mitigated)
> "We use Synthea-generated data to enable controlled non-IID ward simulation with reproducible population skew — a property unavailable in single-institution datasets like MIMIC. To address the realism gap, we validate the central ε-utility finding on real physiological signals from PTB-XL and confirm the trend holds. We further demonstrate robustness under injected Gaussian measurement noise, 10% missingness, and equipment artifacts. Full validation on live hospital telemetry across multiple institutions remains future work."

### 7.3 Simulated federation (mitigated)
> "Ward nodes are deployed as isolated Docker containers communicating via Flower's gRPC protocol, providing genuine process-level federation rather than in-process simulation. We additionally evaluate robustness to 20% per-round client dropout and communication latency, confirming convergence. Multi-institution physical deployment across geographically distributed hospitals remains future work."

### 7.4 Anomaly labels (fixed)
> "Anomaly labels are derived from the NEWS2 aggregate early-warning score, a clinically validated protocol adopted across NHS hospitals, and cross-validated against the MEWS scoring scheme, with consistent results. A qualified clinician reviewed a random sample of 30 labeled windows (agreement: X%)."

*(If no clinician is available, drop the last sentence and rely on NEWS2 + MEWS agreement.)*

### 7.5 C3 evaluation (mitigated)
> "Following the evaluation methodology of Wada et al. (b4), we assess alert summaries using two human raters, three LLM judges (GPT-4o, Claude, Gemini), and a programmatic fact-verification protocol confirming every stated fact traces to the source patient record (hallucination rate: X%). A formal multi-clinician usability study is planned as future work."

### 7.6 No real-world deployment (defended)
> "Consistent with prior federated healthcare studies (b1, b2, b3), this work evaluates in a controlled environment. The contribution is the empirical characterization of the privacy-utility relationship and the novel RAG-FL interface design, both prerequisites to — not substitutes for — clinical deployment. Live deployment requires IRB approval, EHR/FHIR integration, and prospective clinical validation, which we identify as essential future work."

### 7.7 Highest-leverage move (do this first)
If you do only one thing from this list: **add PTB-XL real-signal validation.** It is the single change that moves the paper toward Q2 — it converts the work from "a synthetic-data study" into "a study validated on real physiological signals," directly answering the objection reviewers at Computers in Biology and Medicine and JMIR care about most. Every other mitigation is secondary to the data-realism concern.

---

## 8. Figures Required for the Paper

| Figure | What it shows | How to generate |
|---|---|---|
| Fig 1 | Dual-track system architecture | Draw in draw.io or matplotlib |
| Fig 2 | ε-utility tradeoff curves (LSTM, GRU, CNN) | matplotlib with error bands (±1 std) |
| Fig 3 | F1 gap between LSTM/GRU and CNN across ε | matplotlib bar chart or grouped line |
| Fig 4 | Privacy budget (ε) accumulation across FL rounds | matplotlib line plot (like b2 Fig 5) |
| Fig 5 | Confusion matrices at ε* for each model | seaborn heatmap |
| Fig 6 | Annotated alert example (screenshot or diagram) | Streamlit screenshot or mockup |
| Fig 7 | PTB-XL ε-curve overlaid on Synthea (real-data validation) | matplotlib, two curves same axes |
| Fig 8 | Clean vs noise-injected Synthea F1 across ε | matplotlib grouped bar |
| Fig 9 | FedAvg convergence with vs without client dropout | matplotlib line plot |

---

## 9. Key Citations to Secure

Beyond b1-b5, these must appear in the paper:

| Citation | For what |
|---|---|
| Lewis et al. (2020) — original RAG paper | RAG definition |
| McMahan et al. (2017) — FedAvg | FedAvg algorithm |
| Dwork & Roth (2014) — DP foundations | DP definition |
| Abadi et al. (2016) — DP-SGD / Moments Accountant | DP-SGD algorithm |
| Mironov (2017) — Rényi DP | Rényi DP accountant |
| Royal College of Physicians (2017) — NEWS2 | Anomaly label justification |
| Subbe et al. (2001) — MEWS | Cross-validation labeling scheme |
| Wagner et al. (2020) — PTB-XL | Real physiological signal validation dataset |
| NIST SP 800-226 (2025) — DP guidelines | ε ≤ 1 recommendation |
| Flower paper (Beutel et al. 2022) | FL framework |
| Opacus paper (Yousefpour et al. 2021) | DP library |
| Synthea paper (Walonoski et al. 2018) | Synthetic data generator |

---

## 10. Target Venues and Submission Checklist

### Primary target (Q2)
**Computers in Biology and Medicine** (Elsevier)
- Accepts FL+DP methodology papers with synthetic data
- Regularly publishes system + experiment hybrid papers
- Impact Factor ~7.0

**JMIR (Journal of Medical Internet Research)**
- More tolerant of system-level papers
- Accepts Synthea-based studies with proper justification
- Impact Factor ~7.1

### Fallback target (Q3)
**IEEE Access** — broad scope, rigorous peer review, fast turnaround
**Frontiers in Digital Health** — system papers with methodology welcome
**MDPI Electronics** — b3 published here; NLP-06 is clearly stronger

### Submission checklist
- [ ] All experiments run across 5 seeds
- [ ] **PTB-XL real-signal validation run and trend confirmed** (highest priority)
- [ ] **Noise-injected Synthea robustness run**
- [ ] **FL dropout/latency robustness run**
- [ ] **NEWS2 + MEWS label cross-check completed**
- [ ] **C3: LLM-judge + programmatic fact-verification completed**
- [ ] **Encrypted volume deployed (RAG encryption limitation eliminated)**
- [ ] All figures generated at 300 DPI (now 9 figures)
- [ ] Statistical tests completed and reported
- [ ] Limitations section written per the mitigated strategy (Section 7)
- [ ] All key citations secured and formatted (now includes PTB-XL, MEWS)
- [ ] Code repository made public (GitHub) with README
- [ ] Supplementary material: full experimental tables, Synthea configuration, model hyperparameters
- [ ] Ethics statement: "Primary data is synthetic (Synthea); PTB-XL is a de-identified public research dataset"
- [ ] Author contributions stated
- [ ] Conflict of interest statement
