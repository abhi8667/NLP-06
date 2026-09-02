# 🏥 AI-Assisted Clinical Deterioration Monitoring & Federated Early Warning System

An end-to-end, privacy-preserving clinical decision support system for ICU acute physiological deterioration monitoring, combining **Federated Learning with Differential Privacy (Track B)** and a **Real-Time Clinician Workstation with Local RAG & Fact Verification (Track A)**.

---

## 🧭 System Architecture & Dual Track Structure

```
                               ┌────────────────────────────────────────────────────────┐
                               │             PhysioNet 2019 Sepsis Dataset              │
                               │  Hospital Site A (20,336 ICU)   Hospital Site B (20,000) │
                               └──────────────────────────┬─────────────────────────────┘
                                                          │
                  ┌───────────────────────────────────────┴───────────────────────────────────────┐
                  │                                                                               │
                  ▼                                                                               ▼
   ┌───────────────────────────────┐                                               ┌───────────────────────────────┐
   │    PERSON B: RESEARCH TRACK   │                                               │    PERSON A: PRODUCT TRACK    │
   │  (Federated DP Sequence ML)   │                                               │   (Clinician AI Workstation)  │
   ├───────────────────────────────┤                                               ├───────────────────────────────┤
   │ • 12h Windows (4-6h Horizon)  │                                               │ • Real-time Streamlit UI      │
   │ • Opacus DP-SGD & RDP Account │                                               │ • Vitals Telemetry Replay     │
   │ • DPLSTM, DPGRU, CNN1D        │                                               │ • NEWS2 Standardized Scoring  │
   │ • Flower FedAvg Coordinator   │                                               │ • RAG Clinical Evidence Search│
   │ • 60-Epoch Campaign Sweeper   │                                               │ • Ollama Structured Insights  │
   │ • AUROC ~0.90, AUPRC ~0.725   │                                               │ • Stage 6 Fact Verification   │
   └──────────────┬────────────────┘                                               └───────────────▲───────────────┘
                  │                                                                                │
                  │                        Weights Checkpoint Handoff                              │
                  └─────────────────────────► detector_first_checkpoint.pt ────────────────────────┘
```

---

## 📂 Repository Structure

```
NLP_6/
├── README.md                      # Primary project overview and quickstart guide
├── PROJECT_GUIDE.md               # Complete architectural guidelines and contracts
├── PERSON_A_PRODUCT_TRACK.md      # Person A product specifications & exit criteria
├── PERSON_B_RESEARCH_TRACK.md     # Person B research track specifications
├── TRACK_B_RESEARCH_REPORT.md     # Detailed Track B research findings, figures & tables
│
├── product_track/                 # [Track A] Clinician Workstation & Decision Support
│   ├── bridge/                    # Vitals replay, NEWS2 calculation, RiskScorer & alerts
│   ├── rag/                       # Clinical guidelines ingestion & ChromaDB semantic search
│   ├── llm/                       # Ollama / Llama-3 client & prompt engineering
│   ├── verification/              # Stage 6 clinical fact-checker (hallucination mitigation)
│   └── interfaces/                # Interactive Streamlit Clinician Workstation App
│
├── research_track/                # [Track B] Privacy-Preserving Federated ML & DP Campaign
│   ├── data/                      # 12h windowing, 4-6h horizon labeling, freeze manifest
│   ├── models/                    # DPLSTM, DPGRU, CNN1D, BCE loss, clinical metrics
│   ├── federation/                # Flower FedAvg, Opacus DP-SGD, continuous RDP accountant
│   ├── campaign/                  # Automated grid sweeper across ε × architectures × seeds
│   ├── analysis/                  # Publication figure generator & LaTeX table exporter
│   ├── results/                   # Checkpoints (.pt), campaign JSON/CSV, figures & tables
│   └── tests/                     # Automated unit and integration test suite
│
├── shared/                        # Shared physiological constants, NEWS2 logic & preprocessing
└── physioNet/                     # Raw PhysioNet 2019 dataset partitions (Site A & Site B)
```

---

## ⚡ Quickstart Guide

### 1. Environment Setup
```bash
# Clone the repository
git clone <repo_url>
cd NLP_6

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required dependencies
pip install torch opacus flwr streamlit scikit-learn pandas numpy matplotlib tabulate chromadb requests
```

---

### 2. Launch Track A: Clinician Deterioration Workstation
Launch the interactive browser UI for live patient vital telemetry replay, NEWS2 alert generation, and AI clinical summaries:

```bash
streamlit run product_track/interfaces/clinician_app.py --server.port 8501
```
Navigate to `http://localhost:8501` in your browser.

---

### 3. Run Track B: Research Track Tests & Simulations

#### Run the Automated Unit Test Suite:
```bash
python3 -m unittest discover research_track/tests
```

#### Run Multi-Site Federated DP Simulation:
```bash
python3 -c "
from research_track.federation import run_federated_simulation
model, rep, hist = run_federated_simulation(
    rounds=10,
    target_epsilon=2.0,
    architecture='DPLSTM',
    save_checkpoint_path='research_track/results/checkpoints/detector_first_checkpoint.pt'
)
print('Achieved ε:', rep.achieved_epsilon, '| AUROC:', rep.auroc, '| AUPRC:', rep.auprc)
"
```

#### Run Full 60-Epoch Experiment Campaign Sweep & Export Figures:
```bash
python3 -c "
from research_track.campaign import ExperimentCampaignRunner
from research_track.analysis import generate_all_figures, export_results_tables

runner = ExperimentCampaignRunner(rounds=20, local_epochs=3)
runner.run_campaign()
generate_all_figures()
export_results_tables()
"
```

---

## 📊 Track B Empirical Results (60 Epochs)

| Architecture | Privacy Budget | AUPRC | AUROC | FNR (Missed Det) | F1-Score | Deterioration Detection |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **DPLSTM** | $\varepsilon = 2.0$ | **0.725** | **0.898** | 1.000 | 0.000 | (Strict $\tau=0.5$) |
| **DPLSTM** | $\varepsilon = 1.0$ | **0.719** | **0.885** | **0.136** | **0.808** | **86.4% Detection** |
| **DPLSTM** | $\varepsilon = 0.5$ | 0.620 | 0.878 | **0.136** | 0.613 | **86.4% Detection** |
| **DPLSTM** | $\varepsilon = 4.0$ | 0.659 | 0.884 | 0.545 | 0.571 | 45.5% Detection |
| **DPLSTM** | $\varepsilon = 8.0$ | 0.645 | 0.885 | 0.341 | 0.630 | 65.9% Detection |
| **DPLSTM** | $\infty$ (Non-DP) | 0.509 | 0.824 | 0.409 | 0.433 | 59.1% Detection |
| **DPGRU** | $\infty$ (Non-DP) | 0.550 | **0.880** | **0.227** | 0.544 | **77.3% Detection** |
| **DPGRU** | $\varepsilon = 8.0$ | **0.622** | 0.875 | **0.273** | **0.660** | **72.7% Detection** |
| **DPGRU** | $\varepsilon = 4.0$ | 0.614 | 0.873 | 0.341 | 0.617 | 65.9% Detection |
| **DPGRU** | $\varepsilon = 2.0$ | 0.605 | 0.870 | 0.523 | 0.553 | 47.7% Detection |
| **DPGRU** | $\varepsilon = 1.0$ | 0.563 | 0.859 | 0.705 | 0.426 | 29.5% Detection |
| **DPGRU** | $\varepsilon = 0.5$ | 0.560 | 0.858 | 0.477 | 0.535 | 52.3% Detection |
| **CNN1D** | $\varepsilon = 8.0$ | 0.578 | **0.874** | 0.955 | 0.087 | (Strict $\tau=0.5$) |
| **CNN1D** | $\varepsilon = 4.0$ | 0.436 | 0.818 | 1.000 | 0.000 | (Strict $\tau=0.5$) |
| **CNN1D** | $\infty$ (Non-DP) | 0.398 | 0.824 | 0.432 | 0.495 | 56.8% Detection |

For full details and analysis, see [`TRACK_B_RESEARCH_REPORT.md`](file:///Users/daivikmankame/NLP_6/TRACK_B_RESEARCH_REPORT.md).

---

## 🔒 Key Design Decisions & Privacy Guarantees

1. **4–6 Hour Prediction Horizon:** Slicing sliding windows at a 4–6h forward horizon prevents artificial AUROC inflation caused by autoregressive vital inertia at horizon 0.
2. **Opacus Differential Privacy Architecture:** Standard `nn.LSTM`/`nn.GRU` and `nn.BatchNorm1d` fail under DP-SGD. We use `opacus.layers.DPLSTM`, `opacus.layers.DPGRU`, and `nn.GroupNorm`.
3. **Continuous Rényi DP Accounting:** The Flower federated client carries forward `pe.accountant.state_dict()` across rounds, preventing the $2.43\times$ under-reporting trap of recreating accountants per round.
4. **Clinical Safety Focus:** Models are evaluated on False Negative Rate (FNR) alongside AUPRC and AUROC to ensure deteriorating patients are not missed due to privacy noise.
