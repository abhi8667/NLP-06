# Cheat Sheet — Base Papers & Dataset

One page. Every citation, every number, every likely question. Print it or keep it open on a second screen.

---

## The five base papers — exact citations

| Tag | Full citation | Venue / date |
|---|---|---|
| **b1** | Shukla et al., "Federated learning with differential privacy for breast cancer diagnosis" | *Scientific Reports*, 2025 |
| **b2** | Tanveer, Iradat, Iqbal, Alsagri, Alhakbani, Ahmad, Khan, "Balancing privacy and performance in healthcare: A federated learning framework for sensitive data" | *Digital Health* (SAGE), Sept 2025. DOI: [10.1177/20552076251381769](https://journals.sagepub.com/doi/10.1177/20552076251381769) |
| **b3** | Mosaiyebzadeh, Pouriyeh, Han, Liu, Xie, Zhao, Batista, "Privacy-Preserving Federated Learning-Based Intrusion Detection System for IoHT Devices" | *Electronics* (MDPI) 14(1):67. Received Nov 2024, published Jan 2025. DOI: [10.3390/electronics14010067](https://doi.org/10.3390/electronics14010067) |
| **b4** | Wada et al., "Retrieval-augmented generation elevates local LLM quality in radiology contrast media consultation" | *npj Digital Medicine*, 2025 |
| **b5** | Miao et al., "Improving Large Language Model Applications in Medical and Nursing Domains With RAG: Scoping Review" | *JMIR*, 2025 |

**b2/b3 note, only if asked:** an earlier internal draft had the wrong title for b2 ("Privacy-Preserving Federated Reference Architecture") and a slightly different framing for b3 ("SECIoHT-FL"). Both are resolved — "SECIoHT-FL" is the *internal framework name inside* the b3 paper, not a separate work; the b2 title above is the verified real title, confirmed on SAGE and PubMed Central this week.

---

## Base paper facts, one line each

- **b1** — 569 records, 32 features, static tabular. Random Forest baseline + FFNN under FL. TensorFlow Federated. 96.1% accuracy at **ε=1.9**. Basic composition accounting (not Rényi). No sequence data, no RAG.
- **b2** — 5,110 records, static. 3-layer DNN. **Flower** (same framework as us). DP-SGD, **Rényi accounting** (same family as us). **ε≈0.69** after 10 rounds. 5-client, 5-layer pipeline (Edge/Privacy/Aggregation/App/Decision) — we borrowed this pattern. No temporal data, no LLM.
- **b3** — wustl-ehms-2020 (36 features) + ECU-IoHT (7 features), network + biometric traffic. DNN vs CNN. **Opacus** (same DP library as us). Only **2 noise levels tested** (0.5, 1.5) — not a sweep. ε≈0.43–6.69 depending on noise/dataset. No LSTM/GRU, not clinical vitals.
- **b4** — Llama 3.2-11B + RAG, radiology contrast-media guidelines. **Hallucinations: 8%→0%** with RAG (p=0.012). Temperature 0.2 — we adopted this. Hybrid semantic+keyword retrieval. No privacy, no monitoring, single-shot not continuous.
- **b5** — Scoping review, 67 studies, Nov 2022–May 2025. **94% target physician workflows, only 6% nursing.** "**None of the 67 studies integrate real-time telemetry with RAG.**" Only 9/67 address patient privacy. No system built — it's a review.

---

## The gap, in one sentence per paper

> b1/b2/b3 do privacy and federation well, on **static, non-clinical, or non-temporal** data.
> b4/b5 do medical language models well, with **no privacy protection and no continuous monitoring**.
> Nobody does both. We do both, on real vitals.

---

## Our four contributions

| | Claim | Built on the gap in |
|---|---|---|
| **C1** | Where does model utility break as DP noise increases, for clinical **time-series** specifically | b1 only tested static tabular data |
| **C2** | Do LSTM/GRU tolerate DP noise better than CNN, across a **full ε-sweep**, on real vitals | b3 tested only 2 noise points, on non-clinical traffic |
| **C3** ⭐ | A federated alert's **abnormal vitals become the retrieval query** into that patient's record — the model's output is the semantic input to the LLM | No paper in b1–b5, or the 67 b5 reviewed, connects a DP-protected model's output to patient-scoped RAG |
| **C4** | Tight **Rényi accounting** across the full 100-round run, provable ε<2 | Rigor, not novelty — b1 uses weaker basic composition |

**C3 is the one to lead with if asked "what's actually new here."**

---

## The dataset — PhysioNet/CinC 2019

- **40,336 patients**, two independent hospital systems (20,336 + 20,000). Open access, **no credentialing**.
- One row = one hour. Columns: HR, SBP, O2Sat, Resp, Temp, Glucose + 26 labs + demographics.
- **We tried Synthea first — it failed.** Median gap between readings: 154 days. Near-zero oxygen/temperature readings. Two independent test failures, documented in `p0/results/s1_cadence.json`.
- PhysioNet passed the **same tests**: coverage HR 87–92%, SpO₂ 85–88%, Resp 78–90%, SBP 85–86%, Temp 33–36%.

### Labels — NEWS2, not the built-in sepsis flag

- We derive **NEWS2** (National Early Warning Score, RCP 2017) hourly from the vitals — 5 of 7 official components available (no consciousness, no O₂-therapy flag), so our attainable max is **15, not 20**. State this if asked.
- Threshold ≥5 → **12.6% of hours flagged**, 55–64% of patients flagged at some point. Close to the 10–15% originally assumed.
- **Cross-checked against the dataset's own clinically-adjudicated sepsis label**: our NEWS2 flag caught **60–75% of patients** who the record confirms became septic (patient-level recall). This validates the derived label against real outcomes, not just against another rule.
- Why not just use the sepsis label directly: doing so would narrow the project to sepsis prediction specifically. NEWS2 keeps it a general deterioration monitor — the original framing — while still being checkable against real outcomes.

### Imputation — forward-fill, and we report the rate

- Missing values are **forward-filled then back-filled** (back-fill only fixes the very first row of a stay).
- Measured on a 200-patient sample: HR 7% imputed, Resp 9%, SpO₂ 12%, SBP 12%, **Temp 64% imputed**, **Glucose 86% imputed**.
- This goes directly in the paper's limitations section — it is not hidden.

### Prediction horizon — 6 hours, not 0

- At horizon 0 (predict the very next reading), the task is **near-trivial**: AUROC 0.90, because the label is computed from vitals nearly identical to the input.
- At **6 hours ahead**: AUROC drops to 0.78 — a real forecasting task.
- **Why this matters beyond honesty:** an easy task leaves no room for the DP-noise experiment to show anything — accuracy would stay high and flat regardless of ε, and C1/C2 would return a null result for the wrong reason. Horizon 0 would have silently wasted the whole campaign.

### Patient-level split

- Windows are split **by patient**, never by window — the deployed system meets patients it's never seen.
- Measured effect of getting this wrong: only −0.7% AUPRC (small, because NEWS2 is a smooth function of vitals, not patient-identity-specific). We still adopt patient-level splitting — it's correct on principle even though the penalty happened to be small here.

---

## Likely hard questions

**"Why not just use the sepsis label — isn't that more clinically important?"**
→ Because it would narrow the whole system to one condition. NEWS2 keeps it a general early-warning monitor — matches the original brief — and we validated it against the sepsis outcomes anyway, so we didn't lose rigor by not using it directly.

**"Only 6 vitals — why not use the 26 lab values too?"**
→ Labs are far sparser (most under 10% coverage) and ordered non-uniformly — including them would mean a much smaller usable dataset and a different missingness problem entirely. Vitals are the continuously-monitored signal; that's the deployment scenario we're modelling.

**"Isn't 6-hour horizon arbitrary?"**
→ It's measured, not arbitrary — 0h is near-trivial (0.90 AUROC), 6h is a real task (0.78). We picked the shortest horizon that still leaves a real forecasting problem, matching the "early warning" framing of NEWS2 itself.

**"Two hospitals isn't much of a federation, is it?"**
→ It's two *genuinely different* institutions, not synthetic wards — real non-IID heterogeneity in practice (e.g. one site logs glucose nearly twice as often as the other), which is closer to a real deployment than a hand-crafted split would be.

**"What's actually novel versus just combining existing techniques?"**
→ C3. No prior paper connects a privacy-protected federated model's *output* to a patient-scoped retrieval query as its *input*. That's the one to say if pressed for a single sentence.
