# NLP-06 — Project Guide

**Shared reference for both team members.** Read this first, then your own track file.

| | |
|---|---|
| **Person A** | Product track — assistant, retrieval, interfaces, voice → [`PERSON_A_PRODUCT_TRACK.md`](PERSON_A_PRODUCT_TRACK.md) |
| **Person B** | Research track — detector, federation, privacy, paper → [`PERSON_B_RESEARCH_TRACK.md`](PERSON_B_RESEARCH_TRACK.md) |
| **Mentor** | Dr. M. S. Srividya |
| **Authority** | `docs/02_scope_lock.html` wins over the PRD, Execution Plan, Prototype Guide and Research Guide |

---

## 1. What we are building

A hospital ward system that does two things, **without any patient data leaving the hospital**:

1. **Warns nurses early** when a patient is deteriorating — a small model reads a rolling window of vitals and produces a risk score.
2. **Answers questions in plain English** about a patient, grounded only in that patient's own record.

The two connect at one point: **when the detector raises an alert, it automatically triggers a written clinical summary**. That junction is the project's most original contribution — no paper in our related work does it.

Alongside the system, we publish a paper answering one question:

> In a federated, differentially private clinical anomaly detector, **at what privacy budget (ε) does utility fall below clinical usefulness** — and do sequence models tolerate that noise better than convolutional ones?

---

## 2. The two halves — do not confuse them

There are **two separate AI systems**. Different models, trained differently, doing different jobs.

| | Detector (Person B) | Assistant (Person A) |
|---|---|---|
| What it is | Small sequence classifier (~100K params) | Local LLM + retrieval |
| Input | 12 hours of 6 vitals | A question + that patient's records |
| Output | Risk score | Plain-English text |
| Trained? | Yes — federated, with DP | No — frozen, used as-is |
| In the paper? | Yes, it *is* the paper | Only via the alert bridge |

Federating a whole LLM is not feasible at this scale and is explicitly out of scope. **Only the small detector is federated.**

---

## 3. Locked decisions

Everything here is settled. Do not relitigate; if something must change, change the Scope Lock.

### Data

| Item | Value |
|---|---|
| Dataset | **PhysioNet/CinC 2019** — ODbL, open, already downloaded to `physioNet/` |
| Patients | 40,336 (20,336 site A + 20,000 site B) |
| Task | **Deterioration monitoring** — not sepsis classification |
| Labels | **NEWS2 aggregate ≥ 5**, derived from vitals |
| Label cross-check | The dataset's own sepsis label (independent clinical adjudication) |
| Positive rate | **12.6%** of hours, 55% of patients |
| Vitals | HR, SBP, O2Sat, Resp, Temp, Glucose |
| Window | 12 timesteps × 6 vitals, stride 1 |
| **Prediction horizon** | **4–6 hours ahead** |
| **Train/test split** | **By patient, never by window** |
| Imputation | Forward-fill — **must be declared in the paper**, with per-vital rates |
| Federated clients | 2 real hospital systems |

### Models and stack

| Layer | Choice | Note |
|---|---|---|
| Detector | `DPLSTM`, `DPGRU`, `CNN` | **Not** `nn.LSTM`/`nn.GRU` — Opacus rejects them |
| Model output | **Raw logits** | Not sigmoid — the loss applies it internally |
| Loss | `BCEWithLogitsLoss` + `pos_weight` | Weighted for ~12.6% prevalence |
| Federation | Flower, **deployment path** (Docker + gRPC) | Not the Ray simulation backend |
| Privacy | Opacus DP-SGD, Rényi accountant | See §4 |
| LLM — alerts | Llama 3.1 8B Q4 | 21 tok/s, ~25% on CPU |
| LLM — chat | Llama 3.2 3B Q4 | 66 tok/s, fully on GPU |
| LLM temperature | **0.2** | Both models |
| Speech-to-text | `faster-whisper` small int8 | ~500 MB VRAM, demo scope |
| Text-to-speech | Piper | CPU only, demo scope |
| Vector store | ChromaDB, per-patient collections | Not pgvector |
| Relational | SQLite + AES-256 | Not PostgreSQL |
| Backend / UI | FastAPI + Streamlit | Not React |
| Hardware | RTX 4050, 6 GB | Detector needs 0.03 GB — coexists fine |

### Metrics

- **AUPRC** — headline
- **AUROC** — reported alongside (informative at 12.6%, would not have been at 1.5%)
- **False negative rate** — the clinical measure; missed deterioration is not symmetric with a false alarm
- **F1, precision, recall** — secondary
- Always **mean ± std across seeds**. Never a single run.

### Scope

**In:** patient chat · clinician dashboard · alert→summary bridge · federated detector · differential privacy · lab-report PDF ingest · voice (demo only)

**Out:** meal/exercise/medication logging · federated LLM fine-tuning · UI polish · EHR/FHIR integration · live deployment · PTB-XL validation (optional, see below) · voice as a research contribution

---

## 4. Corrections to the inherited documents

Four things in the original guides are **wrong** and were verified as wrong by running them:

1. **`nn.LSTM` / `nn.GRU` do not work under Opacus.** It raises `ShouldReplaceModuleError`. Use `DPLSTM`/`DPGRU` from `opacus.layers`. `ModuleValidator.fix()` will *not* substitute them for you.

2. **The models double-apply sigmoid.** They end with `torch.sigmoid(...)` while the loss is `BCEWithLogitsLoss`, which applies it internally. Measured gradient shrink: **4.2×**. Training looks mysteriously flat.

3. **The privacy accounting silently resets.** Building a `PrivacyEngine` inside each federated round restarts the accountant, so ε reads flat forever. Measured: naive reports **0.75** every round while correct accounting reaches **1.83** after six — a **2.43× under-report**, in the flattering direction.

4. **The import path `opacus.accountant.analysis` does not exist.** It is `opacus.accountants` (plural), and `get_noise_multiplier` lives in its `utils`.

### The binding privacy protocol

```
1. Size the noise multiplier ONCE for rounds × local_epochs × steps_per_epoch
2. Hold it fixed for the whole run
3. Persist each client's accountant across rounds:
       pe.accountant.state_dict() / load_state_dict()
4. Report the WORST-CASE site's ε, never the mean
5. State the privacy unit explicitly: this is WINDOW-level, not patient-level
```

Point 5 matters: one patient contributes many overlapping windows, so a per-window ε of 2 is **not** a per-patient guarantee. Health-privacy reviewers will ask.

---

## 5. Shared conventions

**The data freeze.** P2 ends by versioning and hashing every dataset variant. Every experiment logs that hash. After the freeze, changing the data invalidates every completed run — so it is locked.

**Patient-level everything.** Splits, cross-validation folds, held-out sets. Never split windows randomly.

**Nothing hand-typed into the paper.** Every figure and table regenerates from a script reading the results files.

**Update the Scope Lock, not a side note.** When an open item closes, edit `docs/02_scope_lock.html`. That is what stops it becoming a sixth document that disagrees with the others.

---

## 6. Phase sequence and handoffs

| Phase | Who | Depends on |
|---|---|---|
| **P0** Feasibility spikes | Both | — (complete) |
| **P1** Scope lock | Both | P0 decisions |
| **P2** Data pipeline → **freeze** | Both | P1 |
| **P3A** Product build | **A** | P2 freeze |
| **P3B** FL+DP harness | **B** | P2 freeze |
| **P4** Experiment campaign | **B** (A supports) | P3B |
| **P5** Alert→summary bridge | **A** (B supports) | P3A + *first checkpoint* from P3B |
| **P6** Summary evaluation | Both | P5 |
| **P7** Analysis & figures | **B** | P4 + P6 |
| **P8** Paper | **B** (A supports) | P7 — overlaps P4–P7 |
| **P9** Polish & defend | Both | P8 |

**Two handoffs matter:**

- **P2 → both tracks.** Neither track can start real work until the data is frozen. This is the bottleneck; get through it fast.
- **P3B → P5.** Person A needs *a* trained model, not the *final* one. Start the bridge against the first checkpoint — do **not** wait for P4 to finish.

**Person A has slack.** The product track is off the critical path. Spend that slack on the campaign and the figures, not on UI polish.

---

## 7. Compute discipline

Measured on the project GPU: one 100-round run takes **4.1 h** (DPLSTM), **4.4 h** (DPGRU), **43 min** (CNN). The full planned grid is about **22 days** of continuous compute.

**It must be trimmed.** Cut in this order — the obvious cuts are the wrong ones:

| Cut | Saves | Kind |
|---|---|---|
| **Halve the rounds** (verify convergence by 50 first) | **50%** | methodological |
| **Halve windows per client** | **50%** | methodological |
| Drop PTB-XL | ~5.5 days | scope — rationale already void |
| 4 ε levels instead of 6 | 33% | scope |
| Drop DPGRU | 17% | scope |

**Never cut the seed count.** It is a validity constraint, not a scope choice.

---

## 8. Open items

| # | Item | Owner |
|---|---|---|
| 1 | **Statistical plan** — the specified test cannot reach significance at any outcome. Blocking: it sets the seed count and therefore the compute budget. | B |
| 2 | **Base paper citations b2 and b3** — the PRD and Research Guide give different titles, and different years for b3. Resolve against the actual papers. | B |
| 3 | **Number of federated clients** — two real sites, or subdivide for stronger heterogeneity | B |
| 4 | **Mentor sign-off** on the dataset switch and the monitoring framing | Both |
| 5 | **Confirm the 4–6h horizon with seeds** before locking it | B |

---

## 9. Where everything lives

```
NLP_06/
├── PROJECT_GUIDE.md              this file
├── PERSON_A_PRODUCT_TRACK.md     Person A's work
├── PERSON_B_RESEARCH_TRACK.md    Person B's work
├── docs/
│   ├── index.html                open this
│   ├── 01_field_guide.html       plain-language explanation of everything
│   ├── 02_scope_lock.html        AUTHORITATIVE specification
│   ├── 03_progress.html          status
│   ├── 04_execution_phases.html  phases, gates, critical path
│   └── 05_p0_spike_guide.html    how P0 was run (historical)
├── p0/
│   ├── P0_MEMO.md                measurements + decisions
│   ├── .venv/                    Python 3.11 + torch cu126 + Opacus + Flower
│   ├── scripts/                  nine spike scripts
│   └── results/                  twelve JSON files — the evidence
├── physioNet/                    the dataset
└── synthea/                      abandoned; kept for the record
```

**Every measured claim in any document traces to a file in `p0/results/`.** If a number appears without a source, treat it as unverified.
