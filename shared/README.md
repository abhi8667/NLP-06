# `shared/` — preprocessing used by BOTH tracks

**Import from here. Never copy-paste these functions.**

If Person A's NEWS2 disagrees with Person B's, the alert says one thing and the summary another — and that surfaces in a demo, not in development. This module exists so that cannot happen.

```bash
pytest shared/test_shared.py -q     # 95 tests
```

---

## Quick start

```python
from shared import load_patient, fill_vitals, add_news2, make_windows, patient_facts

df = load_patient("physioNet/training_setA/training_setA/p000001.psv")
df, imputation = fill_vitals(df)     # ffill + bfill, with a report
df = add_news2(df)                   # adds news2, news2_label, news2_band
```

### Person B — tensors for the detector

```python
from shared import make_windows, patient_split
import numpy as np

X, y, pid = make_windows(df, window=12, horizon=6)   # [n,12,6], [n], [n]
train, test = patient_split(pid, test_frac=0.2, seed=42)
```

### Person A — ground-truth facts for note generation

```python
facts = patient_facts(df)
# {'age': 83.1, 'sex': 'female', 'icu_hours': 54,
#  'vitals': {'HR': {'first':97, 'last':84, 'min':.., 'max':.., 'trend':'falling'}, ...},
#  'news2': {'peak': 9, 'max_attainable': 15, 'peak_hour': 8,
#            'ever_crossed_threshold': True, 'first_crossing_hour': 5,
#            'band_at_peak': 'high', 'response_at_peak': 'emergency response...'},
#  'labs': {'Creatinine': {'n_measurements': 2, 'first': .., ...}, ...}}
```

**This dict is the ground truth.** Generate notes only from these facts, and the P6 fact-verifier checks summaries against the same dict. A claim not derivable from it is a hallucination.

### Whole cohort

```python
from shared import load_cohort
frames, report = load_cohort("physioNet/training_setA/training_setA", limit=200)
print(report.summary())    # per-vital imputation rates — REQUIRED in the paper
```

---

## Decisions baked in

| Decision | Value | Why |
|---|---|---|
| Imputation | forward-fill **then** backward-fill | ffill alone leaves ~0.6 leading rows blank per patient |
| Label | NEWS2 aggregate **≥ 5** | The RCP escalation threshold. **Not** the dataset's `SepsisLabel` |
| Prediction horizon | **6 hours** | At horizon 0 the task is near-trivial (AUROC 0.90 vs 0.78) |
| Split | **by patient** | Stride-1 windows overlap 11/12 timesteps |
| Non-integer vitals | round **half up**, then chart | See below |
| Temperature | **not** rounded | Charted to 0.1 °C — rounding destroys signal |
| Missing vital | scores **0** | Never imputed as abnormal; biases scores down, state it |

### The rounding rule — read this one

NEWS2's chart is defined on **integer** observations ("21–24 → 2, ≥25 → 3"), but PhysioNet contains averaged values like a respiration rate of **24.5** that fall in the gap between bands.

Two reasonable implementations disagreed on **5% of inputs** before this was pinned down. The rule is now fixed: **round half up, then apply the chart.** Half *up* because it errs toward the more severe score, which is the safe direction for an early-warning system.

---

## What we can and cannot compute

NEWS2 has seven parameters. This dataset has five:

| Available | Not available |
|---|---|
| Respiration rate, SpO₂, Systolic BP, Pulse, Temperature | Consciousness (ACVPU, max 3), Air-or-oxygen (max 2) |

**Attainable maximum is 15, not the full 20.** `NEWS2_MAX` computes this from the component table rather than hardcoding it. Our scores are therefore systematically lower than a bedside nurse would record — **state this in the paper.**

---

## Columns that do not work

Measured over 400 patients / 15,500 patient-hours. Don't rediscover these:

| Column | Coverage | Verdict |
|---|---|---|
| `EtCO2` | **0%** | entirely empty |
| `Unit1`, `Unit2` | 53% | only half of patients — unusable for partitioning |
| `Temp` | 36% | usable, but ~64% forward-filled |
| `Glucose` | 12% | usable, but ~86% forward-filled |
| `HR`, `Resp`, `O2Sat`, `SBP` | 86–92% | good |

---

## If you change anything here

1. Run the tests — they encode agreed behaviour, not implementation detail
2. Tell the other person **before** pushing
3. If a label or scoring rule changes after the P2 data freeze, **every completed experiment is invalid**

The test file is the contract. If a test seems wrong, that is a conversation, not a fix.
