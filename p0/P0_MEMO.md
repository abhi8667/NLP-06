# NLP-06 · P0 Decision Memo

> **See also: the Scope Lock document**, which is now the canonical specification and
> supersedes the PRD, Technical Execution Plan, Prototype Guide and Research Guide
> wherever they disagree. This memo records the P0 *measurements*; the Scope Lock
> records the *decisions* taken from them.
>
> **Scope change since P0 began:** the voice pipeline has been restored as a
> demo-scope feature (local `faster-whisper` ASR + Piper TTS, ~500 MB VRAM,
> runs sequentially with the LLM so there is no contention). It is explicitly
> **not** a paper contribution and is first on the product-track de-scoping ladder.

> **P0 is complete when this file is filled in and both team members plus the mentor
> have read it.** Not when the spikes have run — when the decisions are recorded.
> Every answer here is an input to the P1 scope lock.

| | |
|---|---|
| Date completed | _______ |
| Person A | _______ |
| Person B | _______ |
| Mentor reviewed | ☐ |

---

## Environment of record

Captured from `results/00_env.json` — already measured on this machine.

| Item | Value |
|---|---|
| Python | 3.11.15 |
| torch (build) | 2.13.0+cu126 |
| opacus | 1.6.0 |
| flwr | 1.33.0 |
| GPU / VRAM | NVIDIA RTX 4050 Laptop, 6.00 GB, sm_89 |
| Java (Synthea) | 17.0.12 LTS |
| Ollama | not installed |

---

## S1 — Synthea telemetry cadence  →  **FAIL** *(superseded — see S6)*

> **Resolved.** Synthea was tested, failed on two independent counts, and has been
> replaced by the PhysioNet/CinC 2019 dataset, which passed. The S1 record below is
> kept because it is the evidence for the switch. **The decision is made — see S6.**

**Question:** can a 12-timestep window be formed from readings hours apart rather than months apart?

Run on 52 patients / 45,143 observations (`synthea -p 50 -s 12345`).

| Measurement | Value |
|---|---|
| Distinct times-of-day in `DATE` | 3,075 — timestamp *resolution* is fine |
| Overall median inter-observation gap | **3,696 h = 154 days** |
| Patients forming a 12-step window @ 1h | **0 / 52** |
| Patients forming a 12-step window @ 24h | **1 / 52 (1.9%)** |
| Patients forming a 12-step window @ 30d | 2 / 52 (3.8%) |
| Verdict | **FAIL** |

### Second, independent blocker: NEWS2 cannot be computed

The assumed SpO2 code `59408-5` returns **zero** rows. Synthea emits arterial oxygen
saturation as `2708-6`, but only **48 readings across 52 patients**. Temperature is
similarly sparse at **73 readings**.

| NEWS2 component | Readings | State |
|---|---|---|
| Respiratory rate | 773 | OK |
| Systolic BP | 782 | OK |
| Heart rate | 773 | OK |
| **SpO2** | **48** | **unusable** |
| **Temperature** | **73** | **unusable** |

NEWS2 aggregate scoring needs every component. This breaks the labelling scheme on its
own, independently of the cadence result — and it is *not* fixed by reframing.

**Decision — pick one and justify:**

*Keep-Synthea options — both weak:*

- ☐ **A · Reframe** as longitudinal encounter sequences. **Does not fix the NEWS2 blocker** — without SpO2 and temperature you still cannot label, whatever you call the time axis.
- ☐ **B · Synthesize** the missing vitals *and* the cadence. Keeps the ward framing, but you would be generating most of the signal yourself, at which point "grounded in Synthea's validated care pathways" is doing very little work.

*Replace-the-dataset options:*

- ☐ **D · PhysioNet/CinC 2019 Sepsis Challenge.** **Open access, no credentialing, available immediately.** Hourly vitals including HR, O2Sat, Temp, SBP, MAP, DBP, Resp — every NEWS2 component. ~40,000 patients across two real hospital systems, with hourly deterioration labels already provided. Changes the clinical task from generic deterioration to sepsis prediction; leaves every research contribution intact. **Recommended primary.**
- ☐ **E · eICU-CRD.** Credentialed. **208 hospitals** — the strongest federation story available, since the sites are genuinely different institutions rather than slices of one population. An open demo subset exists for immediate pipeline work.
- ☐ **C · MIMIC-IV** `chartevents`. Credentialed. Real ICU telemetry, real SpO2/temperature, care units as proxy sites. Open demo subset (~100 patients) available immediately.
- ☐ **F · HiRID** (2-minute resolution) or **G · AmsterdamUMCdb**. Credentialed / data agreement. Very high resolution, single centre.
- ☐ **H · Promote PTB-XL to primary.** Open, real, already validated by S3 — zero blockers, could start P2 tomorrow. Cost: the task becomes ECG classification, not ward monitoring.

*Reframing option:*

- ☐ **I · Keep the research question, change the clinical task.** The contribution is "how much privacy can a federated clinical time-series model absorb before it stops being useful." That question is indifferent to whether the task is deterioration, sepsis, or arrhythmia. Choosing the task the data supports dissolves the problem without touching any of C1–C4.

**Note on credentialed options:** MIMIC-IV and eICU both publish **open-access demo subsets**. Build the pipeline against the real schema now, swap in the full dataset when approval lands — so "waiting on credentialing" never blocks anything.

**Verify before committing:** confirm current licence and access terms on each dataset's page. The summaries above are from general knowledge, not from reading the download pages.

**Justification:**

> _______

**MIMIC credentialing application submitted?** ☐ yes ☐ no — _recommended yes regardless of the verdict; the review clock is long and starting it now is free optionality._

---

## S6 — PhysioNet/CinC 2019 replacement dataset  →  **PASS**

Downloaded and tested (40,336 patients; 800 sampled per site). Same three questions
S1 asked of Synthea, plus one Synthea could not answer.

### Coverage — the Synthea blockers are gone

| Vital | Column | set A | set B | Verdict |
|---|---|---|---|---|
| Heart rate | `HR` | 92.0% | 87.4% | OK |
| Systolic BP | `SBP` | 85.4% | 85.5% | OK |
| **SpO2** | `O2Sat` | **87.8%** | **85.4%** | OK — *Synthea had 48 readings total* |
| Respiratory rate | `Resp` | 89.5% | 77.8% | OK |
| **Temperature** | `Temp` | **33.2%** | **34.0%** | measured ~every 3h |
| Glucose | `Glucose` | 11.8% | 21.2% | sparse |

All five NEWS2 components are genuinely measured. 96% of patients have stays of ≥13 hours.

### Cadence — requires forward-fill, and this must be declared

| Window formation | set A | set B |
|---|---|---|
| All six vitals in the **same hour** | 0.5% | 0.1% |
| After **forward-fill** | **84.2%** | **78.9%** |

Different vitals are charted at different times, so strict simultaneity essentially never
occurs. Carrying the last observed value forward — standard clinical practice — yields
~20 usable windows per patient, roughly **800,000 windows** across the full dataset.

- ☐ **Methods declaration:** state that forward-fill imputation is used, and report the
  per-vital imputation rate (temperature carried ~2h in 3; glucose ~4h in 5).

### Labels — imbalance is ~10× worse than planned

| | set A | set B |
|---|---|---|
| Positive **hours** | 1.50% | 1.40% |
| Patients **ever** positive | 6.1% | 5.6% |

The guides assumed 10–15% anomalous. Actual prevalence is ~1.5% of hours.

- ☐ **Metric change — required:** at 1.5% prevalence **AUC-ROC is misleadingly
  optimistic** because true negatives dominate. Switch the headline metric to **AUPRC**
  (precision–recall), keeping false-negative rate as the clinical measure. The Research
  Guide currently specifies AUC-ROC throughout.
- ☐ **Recalibrate** `pos_weight` in the loss for ~1.5% prevalence, not 10–15%.

### Sites — genuine but modest heterogeneity

| Metric | set A | set B |
|---|---|---|
| Median stay (hours) | 39 | 37 |
| Patients ever positive | 6.1% | 5.6% |
| Median age | 65.5 | 64.0 |
| Male share | 58.5% | 57.8% |
| **Glucose coverage** | **11.8%** | **21.2%** |
| **Resp coverage** | **89.5%** | **77.8%** |

Populations are similar; the sites differ mainly in **what gets measured**. That is real
institutional variation — and arguably a more interesting federated problem than
population skew — but it is not the dramatic ward contrast the original design imagined.

- ☐ **Decide:** accept two-site federation as-is, or partition further within each site
  (by ICU unit or age band) to create additional clients with stronger heterogeneity.

### Dataset decision

- ☑ **PhysioNet/CinC 2019 adopted as the primary dataset.**
- ☐ PTB-XL retained as the real-signal cross-check (unchanged).
- ☐ eICU / MIMIC-IV application started as an upgrade path (optional).

**Note the framing change:** the clinical task is now **sepsis onset prediction**, not
generic deterioration. All four research contributions survive unchanged — the privacy
question is indifferent to which clinical label is being predicted.

---

## S2 — Opacus × Flower × sequence models

### Confirmed by `s2a` — already run on this machine

| Finding | Result |
|---|---|
| `nn.LSTM` incompatibilities under Opacus | **1** — `ShouldReplaceModuleError` |
| `DPLSTM` incompatibilities | **0** |
| Gradient shrink factor from double-sigmoid bug | **4.2×** |
| DP training reached target ε | ✅ achieved 1.9947 against target 2.0 |

**Action taken:** model spec updated to use `DPLSTM`/`DPGRU` and to return **raw logits**, not `torch.sigmoid(...)`. ☐ done

### Confirmed by `s2c` — the accounting trap, already reproduced

| | ε after round 1 | ε after final round |
|---|---|---|
| Naive (fresh engine per round) | 0.7524 | **0.7524 — flat** |
| Correct (accountant carried) | 0.7524 | **1.8255 — grows** |

**Understatement factor: 2.43× after only 6 rounds.** The gap widens with round count,
so at the planned 100 rounds it is far larger. C4's "provable ε < 2 across 100 rounds"
depends entirely on getting this right.

**Accounting decision for P3B:**

- ☐ Size σ **once** for `ROUNDS × LOCAL_EPOCHS × steps_per_epoch`, hold fixed.
- ☐ Persist each client's accountant via `pe.accountant.state_dict()` / `load_state_dict()`.
- ☐ Report the **worst-case ward's** ε, not the mean.

### Unit of privacy — must be stated in the paper

Opacus provides **sample-level** DP. Samples are overlapping sliding windows, and one
patient contributes many. The reported ε is therefore **not** a per-patient guarantee.

**Decision:**

- ☐ Report the unit as the window, and give the implied patient-level bound alongside it.
- ☐ Restructure sampling so the privacy unit is the patient.

**Wording to use in the paper:**

> _______

### Flower execution path (`s2d`)

- ☐ Deployment path (SuperLink/SuperNode over gRPC) — matches the Dockerised ward-node commitment in P3B.
- ☐ Simulation path (requires installing Ray).

**Chosen:** _______

---

## S3 — PTB-XL mapping

| Item | Value |
|---|---|
| Mapping option chosen | A (interval features) / B (different input width) |
| Window shape produced | `[batch, ___, ___]` |
| Features used | |
| Label scheme | |
| Class balance (NORM vs abnormal) | |

**Claim wording — what the PTB-XL experiment does and does not show:**

> _______

*(Guardrail: the defensible claim is that the ε-utility **trend** appears on real
physiological signals, so the finding is not a synthetic-data artifact. It is **not**
evidence that the anomaly detector works on real patients.)*

---

## S4 — Compute budget

Measured on the RTX 4050 at 4,000 windows/ward (an assumption — S1 sets the real figure).

| Model | Local train (s) | Full 100-round run | Peak VRAM |
|---|---|---|---|
| DPLSTM | 29.5 | 4.1 h | 0.03 GB |
| DPGRU | 31.5 | 4.4 h | 0.03 GB |
| CNN | 5.2 | 43 min | 0.03 GB |

DP-SGD costs roughly **5.7×** on a recurrent model versus the convolutional baseline.

| | |
|---|---|
| Total runs in the planned grid | **150** |
| Estimated total compute | **~530 h (22 days)** |
| Wall-clock actually available | ______ ← fill this in |
| **Fits?** | ☐ yes ☐ no |

**GPU contention:** peak training VRAM is only 0.03 GB against 6 GB total, so Ollama
and the campaign **can** share this GPU. This worry was unfounded — no sharing plan needed.

### Which cuts actually help

The de-scoping ladder is ordered by scientific cost, not compute cost, and against
compute it underperforms:

| Cut | Saves | Kind |
|---|---|---|
| Drop DPGRU | 17% | scope |
| Drop noise-injected Synthea | 24% | scope |
| 4 ε levels instead of 6 | 33% | scope |
| 3 seeds instead of 5 | 40% | **statistical validity — avoid** |
| 50 rounds instead of 100 | **50%** | methodological |
| Half the windows per ward | **50%** | methodological |
| Drop GRU + noised + 50 rounds | **71%** → ~6.5 days | combined |

**Decision — cuts applied:**

1. ☐ Verify FedAvg convergence by round 50; if converged, halve the rounds *(biggest single lever, no scope loss)*
2. ☐ Drop DPGRU
3. ☐ Drop noise-injected Synthea
4. ☐ Drop client-dropout experiment
5. ☐ Drop lab-report PDF ingestion
6. ☐ Drop MEWS cross-check

*Never cut: PTB-XL validation, the seed count, the C3 bridge.*

**Re-run `s4_timing.py` after S1**, since windows-per-ward scales this table linearly.

---

## S4b — Revised statistical plan for C2

The plan as written in the Research Guide **cannot produce a significant result**:
Mann-Whitney U at n=5 vs 5 has a minimum two-sided p of ≈0.0079, while Bonferroni
correction for 12 tests sets the threshold at 0.05/12 ≈ 0.0042.

**Chosen replacement:**

- ☐ Single omnibus model over architecture × ε (aligned rank transform / mixed model), with per-ε effect sizes and CIs reported descriptively. *(recommended)*
- ☐ Raise seeds to ___ and reduce comparisons to ___.

**Final seed count:** ___  → feeds back into the S4 grid arithmetic.

---

## Sign-off

P0 exits when every box above is filled and nothing reads "we'll figure it out later".

| | Name | Date |
|---|---|---|
| Person A | | |
| Person B | | |
| Mentor | | |
