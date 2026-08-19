# Person B — Research Track

**You own:** the data pipeline, the detector models, federated training, differential privacy, the experiment campaign, the analysis, and the paper.

Read [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md) first for the locked decisions. This file is your work.

> **You are the critical path.** Every phase after P2 waits on this track. Person A has slack; you do not.

---

## Already done in P0

- [x] Environment: Python 3.11, torch 2.13+cu126 (CUDA on RTX 4050), Opacus 1.6, Flower 1.33
- [x] DP training verified on a recurrent model — achieved ε 1.99 against target 2.0
- [x] Federated averaging verified — parameters round-trip and converge
- [x] Privacy accounting trap reproduced and the fix established
- [x] Compute budget measured, cut sensitivity mapped
- [x] Dataset secured — PhysioNet/CinC 2019, tested and adopted
- [x] NEWS2 labels derived and validated against clinical adjudication
- [x] Patient-split and prediction-horizon effects measured

Evidence: `p0/results/*.json` — twelve files.

---

## Four corrections you must apply before writing any pipeline code

These were all verified by running them. The inherited guides are wrong on each.

### 1. The model layers do not work

```python
# WRONG — Opacus raises ShouldReplaceModuleError
self.lstm = nn.LSTM(6, 64, num_layers=2, batch_first=True)

# RIGHT
from opacus.layers import DPLSTM, DPGRU
self.lstm = DPLSTM(6, 64, num_layers=2, batch_first=True)
```

`ModuleValidator.fix()` will **not** substitute these for you.

### 2. The models double-apply sigmoid

```python
# WRONG — BCEWithLogitsLoss applies sigmoid internally. Gradients shrink 4.2×.
return torch.sigmoid(self.fc(out[:, -1, :]))

# RIGHT
return self.fc(out[:, -1, :])   # raw logits
```

### 3. The privacy accountant silently resets

Building a `PrivacyEngine` inside each round restarts accounting. Measured over six rounds: naive reports a flat **0.75**, correct accounting reaches **1.83** — a **2.43× under-report**, in the flattering direction, so it looks fine.

```python
from opacus.accountants.utils import get_noise_multiplier   # NOT opacus.accountant.analysis

TOTAL_STEPS = ROUNDS * LOCAL_EPOCHS * steps_per_epoch
sigma = get_noise_multiplier(
    target_epsilon=EPS, target_delta=1e-5,
    sample_rate=BATCH / len(local_dataset),
    steps=TOTAL_STEPS, accountant="rdp",
)
# then, each round:
pe = PrivacyEngine()
pe.accountant.load_state_dict(saved_state[client_id])   # carry it forward
model, opt, dl = pe.make_private(module=..., noise_multiplier=sigma, max_grad_norm=1.0)
# ... train ...
saved_state[client_id] = pe.accountant.state_dict()
```

Report the **worst-case site's** ε, never the mean.

### 4. State the privacy unit

Opacus gives **sample-level** DP, and your samples are overlapping windows. One patient contributes many, so a per-window ε of 2 is **not** a per-patient guarantee. Say so explicitly in the paper, and give the implied patient-level bound alongside it. Health-privacy reviewers will ask.

---

## P2 — Data pipeline → freeze

**This is the bottleneck. Both tracks wait on it.**

- [ ] Load PhysioNet `.psv` files — hourly rows, one file per patient
- [ ] Forward-fill the six vitals; **record the per-vital imputation rate** (temperature ~2 h in 3; glucose ~4 h in 5) — this goes in the paper
- [ ] Compute NEWS2 aggregate per hour; label = score ≥ 5
- [ ] Build windows: 12 timesteps, stride 1, **label taken 4–6 hours after the window ends**
- [ ] Partition clients by hospital system (site A / site B)
- [ ] **Split by patient** — hold out patients, never windows
- [ ] Reserve the held-out patients Person A needs for the C3 scenarios
- [ ] Subsample from the ~800,000 available windows to fit the compute budget; record the sampling rule
- [ ] Deploy on the encrypted volume

**Exit gate — the freeze:**
- [ ] Every dataset variant versioned and **content-hashed**; every experiment logs the hash
- [ ] Class balance reported per site (~12.6% expected)
- [ ] NEWS2 / sepsis-label agreement recorded
- [ ] **No phase past this may begin until the hash exists**

---

## P3B — Federated + DP training harness

Build in this order. Each layer is only debuggable once the one beneath it is known good.

- [ ] **Centralised baseline** — single node, no FL, no DP. Establishes the ceiling.
- [ ] **Flower FedAvg**, still without DP — confirm convergence
- [ ] **Opacus DP-SGD** per client, with the accounting protocol above
- [ ] **Dockerised sites over gRPC** — the deployment path, not the Ray simulation backend (Ray is not installed and building simulation first means building twice)
- [ ] **Dropout strategy** for the client-unavailability experiment
- [ ] **Experiment runner** — config-driven, checkpointed, resumable

Each run must write one complete row: AUPRC, AUROC, F1, precision, recall, **false negative rate**, achieved ε, wall-clock, **dataset hash**, seed.

**Exit gate:**
- [ ] One full (model × ε × seed) run completes unattended and writes a complete row
- [ ] Killing the runner mid-campaign and restarting resumes without duplicating or losing cells
- [ ] Achieved ε matches target within tolerance

> **Tell Person A the moment you have any trained checkpoint.** They need it to start P5 and must not wait for the campaign.

---

## P4 — The experiment campaign

Mostly waiting — which is exactly why the paper's Methods and Related Work get written now.

- [ ] Core grid: ε levels × architectures × seeds. **Set the seed count from the revised statistical plan, not convention.**
- [ ] Client-dropout and latency robustness run
- [ ] Non-IID skew sweep if compute remains
- [ ] **Nearly free bonus:** report per-site metrics at each ε. DP is known to degrade underrepresented groups disproportionately; you already have two sites, so it costs a `groupby` and could be a genuine additional finding.

**Compute reality:** the full grid is ~22 days. Cut **rounds and dataset size before scope** — each saves 50%, versus 17–24% for dropping model types. Verify convergence by round 50 before paying for 100.

**Exit gate:**
- [ ] Every grid cell has a result row — no silent failures
- [ ] Every row carries the frozen hash and its seed
- [ ] Spot-rerunning one cell reproduces its metrics

---

## P7 — Analysis and figures

- [ ] Run the revised statistical plan
- [ ] Identify **ε\*** — the budget below which utility falls under the pre-declared clinical threshold. **Declare that threshold before seeing results**, not after.
- [ ] Interpret false negative rate as the clinically dominant metric
- [ ] Generate every figure from a script reading the results files

**Exit gate:**
- [ ] Every figure and table regenerates from one command
- [ ] No number in the draft exists outside a generated artefact

---

## P8 — The paper

Overlaps P4–P7 deliberately.

**Write during the campaign:** Introduction · Related Work + gap table · System Architecture · Data and Preprocessing · FL+DP Methodology

**Write after P7:** Results · Discussion · Conclusion

**Must appear in the limitations, honestly:**
- Forward-fill imputation, with per-vital rates
- Privacy unit is the window, not the patient
- Two NEWS2 components (consciousness, supplemental oxygen) are not recorded, so the attainable maximum is 14 not 20
- Sites differ mainly in measurement practice, not patient population
- Federation is process-level, not multi-institution physical deployment

---

## Your open items

| # | Item | Why it matters |
|---|---|---|
| 1 | **Statistical plan** — as specified (pairwise tests, 5 seeds, correction for 12 comparisons) **no outcome can reach significance**: the smallest attainable p sits above the corrected threshold. Recommended: one omnibus model over architecture × ε, with per-ε effect sizes reported descriptively. | **Blocking** — sets the seed count and therefore the compute budget |
| 2 | **Citations b2 and b3** — the PRD and Research Guide give different titles, and different years for b3. Either two papers are cited under one label or one reference is wrong. | Wrong citations are a credibility problem |
| 3 | **Client count** — two real sites, or subdivide by unit/age band for stronger heterogeneity | Affects the federation claim |
| 4 | **Confirm the 4–6 h horizon with seeds** | Single-run evidence so far |

---

## The risk nobody else will catch

**If the task is too easy, the paper has no finding.**

Measured: at horizon 0 the model scores AUROC 0.90 — because NEWS2 is computed from vitals, vitals barely move hour to hour, and forward-fill means a carried value can be both model input and label basis. At 6 hours ahead it drops to 0.78.

An easy task leaves no room for DP noise to differentiate anything. Accuracy stays high and flat, the architectures do not separate, and **C1 and C2 both return null** — not because the hypothesis is wrong, but because the task could not discriminate.

Before committing 22 days of compute, run a **small pilot at two ε values** (say ∞ and 0.5) and confirm the gap between them is measurable. If DP noise barely moves the metric, the task needs to be harder — longer horizon, or a stricter NEWS2 threshold — before the full campaign is worth running.
