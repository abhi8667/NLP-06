# 🎬 WardSense Live Demonstration Script: Interactive Walkthrough Manual

> **System:** WardSense Clinician AI Workstation & Federated Deterioration Monitor  
> **Interface URL:** [http://localhost:8501](http://localhost:8501)  
> **Audience:** Examination Committee, Clinicians, Evaluators, Technical Jury  
> **Format:** Click-by-Click Guide with **Where to Click**, **What to Look At**, and **What to Say**  
> **Total Demonstration Duration:** 5 to 8 Minutes

---

## 🗺️ Demonstration Route at a Glance

```
  ┌──────────────────────┐         ┌──────────────────────┐         ┌──────────────────────┐         ┌──────────────────────┐
  │   1. OVERVIEW PAGE   │  ────►  │   2. LIVE CONSOLE    │  ────►  │  3. ASSURANCE PAGE   │  ────►  │  4. PATIENT PORTAL   │
  │                      │         │  (Chapters 1 to 5)   │         │                      │         │                      │
  │ • Architecture SVG   │         │ • Baseline (Hr 2)    │         │ • Stage 8 Provenance │         │ • Private Chat       │
  │ • Research Gap Recap │         │ • Decline (Hr 5)     │         │ • Epsilon Budget ε=1 │         │ • Patient-Friendly   │
  │ • Click 'Start Demo' │         │ • Alert & Summary(C3)│         │ • Inter-Rater Kappa  │         │ • Zero Data Egress   │
  └──────────────────────┘         └──────────────────────┘         └──────────────────────┘         └──────────────────────┘
```

---

## Pre-Flight Setup Checklist (Before Evaluators Arrive)

1. **Verify Web App is Live:**
   - Open browser at `http://localhost:8501`.
   - Ensure the sidebar navigation shows:
     - **Present:** `Overview`, `Live console`
     - **Verify:** `Assurance`
     - **Patient-facing:** `Patient portal`
2. **Set Initial State:**
   - Click **Overview** in the sidebar.
   - Maximize browser window (press `F11` or full-screen for clean presentation).

---

## Scene 1: The Overview Page — Framing the System

**Estimated Time:** 60 Seconds  
**Location:** Sidebar $\to$ `Overview` (`http://localhost:8501/overview`)

### 🖱️ Actions: Where & How to Click
1. Start on the **Overview** page.
2. Scroll slightly to bring the **Pipeline Flowchart** into view.
3. Hover your mouse over the dashed box labeled **`CONTRIBUTION C3 — ALERT-TO-SUMMARY BRIDGE`**.

---

### 👁️ Visual Cues to Point Out on Screen
- **The 7-Stage End-to-End Flow:** Point to the SVG pipeline:
  `Bedside replay` $\to$ `NEWS2 + detector` $\to$ `Alert raised (≥5)` $\to$ `Abnormalities (deterministic)` $\to$ `Patient-scoped retrieval` $\to$ `Local LLM narrative` $\to$ `Clinician review`.
- **Contribution C3 Box:** Highlight the cyan dashed outline connecting Stages 3 through 6.
- **Top Metrics Row:**
  - `40,336` ICU stays across two hospital sites (PhysioNet 2019).
  - `4–6h` prediction horizon (not trivial zero-hour autocorrelation).
  - `ε = 1.0` privacy budget under Opacus continuous Rényi DP.
  - `0%` cross-patient leakage via strict namespace isolation.

---

### 🗣️ What to Say (Verbatim Talking Points):
> *"Welcome. This is **WardSense**, an end-to-end clinical decision support system designed to monitor acute physiological deterioration in hospital wards without allowing patient data to leave hospital custody.*
> 
> *Before we jump into the live telemetry, look at this pipeline diagram:*
> *On the left, continuous vital signs are monitored by a lightweight sequence detector trained under Federated Learning and Differential Privacy. In the center is our primary innovation — **Contribution C3: The Alert-to-Summary Bridge**.*
> 
> *When an acute risk threshold is breached, the system doesn't merely sound an alarm. It deterministically compiles the physiological abnormalities, queries the patient's private admission history in ChromaDB, and uses a local language model to write an actionable clinical escalation brief.*
> 
> *Let's launch the guided presentation walkthrough."*

---

### 🖱️ Transition Action:
- Click the prominent button: **`Start guided demo`** (or select `Acute sepsis trajectory`).
- *The app instantly transitions to the **Live console** page with Demo Mode activated.*

---

## Scene 2: Live Console — Chapter 1: Baseline Telemetry

**Estimated Time:** 45 Seconds  
**Location:** Sidebar $\to$ `Live console` (`http://localhost:8501/console`)  
**Active Patient:** `Bed 101` (`p000001`, 83F — Acute sepsis trajectory)  
**Active Hour:** `Hour 2`

---

### 🖱️ Actions: Where & How to Click
1. Look at the **Demo Chapter Banner** at the very top of the console.
2. Notice the indicator: **`Chapter 1 of 5: Baseline`**.
3. Point to the **Vitals Strip** panel on the left (it is outlined in a highlighted focus ring; other panels recede into dim opacity).

---

### 👁️ Visual Cues to Point Out on Screen
- **The Focused Panel (`ws-focus`):** Notice the cyan glowing border around the vitals cards.
- **Vitals Strip Values at Hour 2:**
  - Heart Rate: `~82 bpm` (inside normal green band 51–90).
  - Systolic BP: `~122 mmHg` (inside normal band 111–219).
  - SpO₂: `98%` (normal band 96–100%).
  - Respiration: `16 /min` (normal band 12–20).
  - Temperature: `36.8 °C` (normal).
- **NEWS2 Score Gauge:** Shows an aggregate score of `0` or `1` (Green — Low Risk).
- **Detector Risk Bullet:** Sits comfortably below the decision threshold $\tau$.

---

### 🗣️ What to Say:
> *"We are now in Bed 101, replaying an 83-year-old female patient's recorded stay from PhysioNet.*
> 
> *At Chapter 1, we are at **Hour 2**. Notice that every vital sign sits comfortably inside its normal clinical reference band. The NEWS2 aggregate score is below the escalation threshold of 5, and the federated neural detector's risk score is resting at baseline.*
> 
> *Notice our UI design: the active focus panel is highlighted with a focused ring while non-critical panels recede, preventing cognitive overload for clinicians."*

---

### 🖱️ Transition Action:
- In the top demo banner, click: **`Next chapter →`** (or press the step button).

---

## Scene 3: Live Console — Chapter 2: Deterioration Begins

**Estimated Time:** 60 Seconds  
**Active Hour:** `Hour 5`  
**Focused Panel:** `Risk Scorer & Trend Forecast`

---

### 👁️ Visual Cues to Point Out on Screen
- **Demo Banner Updates:** **`Chapter 2 of 5: Deterioration begins`**.
- **The Focus Shifts:** The focus ring smoothly moves to the **Risk Scorer & NEWS2 Gauge** panel.
- **Vitals Subtle Drift:**
  - Heart Rate is creeping up: `94 bpm` (mild tachycardia).
  - Respiration is rising: `21 /min` (tachypneic).
- **The Neural Detector Bullet:** Notice that the **DPLSTM risk score is already climbing toward the alert threshold $\tau$**, even though the classical NEWS2 score is still hovering at 3 (sub-alert threshold).

---

### 🗣️ What to Say:
> *"Now we advance to **Hour 5**.*
> 
> *Watch what happens here: if you were only looking for single-vital red flags, you would miss this. The heart rate is only slightly elevated at 94, and respiration is 21. Standard hospital bedside monitors stay silent.*
> 
> *However, look at our federated **DPLSTM detector bullet** — because it was trained on 12-hour sliding temporal dynamics across 40,000 stays with differential privacy, it detects the subtle cross-channel covariance. It forecasts acute deterioration 4 to 6 hours into the future, and its risk probability is climbing toward our calibrated threshold $\tau$.*
> 
> *Let's see what happens as the physiological drift compounds."*

---

### 🖱️ Transition Action:
- Click: **`Next chapter →`**.

---

## Scene 4: Live Console — Chapter 3 & 4: Alert Fires & The Grounded Summary (Contribution C3)

**Estimated Time:** 90 Seconds  
**Active Hour:** `Hour 8`  
**Focused Panels:** `Alert Banner` then `Summary Card`

---

### 👁️ Visual Cues to Point Out on Screen
- **Demo Banner:** **`Chapter 3 of 5: Alert fires`** $\to$ then **`Chapter 4: Grounded summary (C3)`**.
- **Alert Banner:** Flashes an **Amber/Vermilion Alert Banner**:
  - `⚠️ ALERT: Physiological Deterioration Triggered (NEWS2 = 6, Risk P > τ)`.
- **Breached Vitals Table:** Point out the table listing:
  - `Heart Rate: 114 bpm (Score 2)`
  - `Systolic BP: 92 mmHg (Score 2)`
  - `Respiration: 24 /min (Score 2)`
- **The AI Clinical Escalation Card (Contribution C3):**
  - Point to the **Clinical Impression**:
    *"Patient exhibits decompensating tachycardia (114 bpm) with systemic hypotension (92 mmHg) and tachypnea, highly suggestive of severe sepsis / septic shock."*
  - Point to **Recommended Immediate Actions**:
    *"1. Initiate Sepsis Six protocol within 1 hour: draw blood cultures, measure serum lactate, administer IV broad-spectrum antibiotics, begin 30 mL/kg crystalloid fluid resuscitation."*
  - Point to **Telemetry Evidence Tags**:
    Click/hover over the tags showing exact timestamps and vital readings cited directly from the record.

---

### 🗣️ What to Say (Crucial Core Pitch):
> *"At **Hour 8**, the tipping point occurs. NEWS2 crosses the critical threshold of 5, reaching 6. The alert fires automatically — nobody clicked anything.*
> 
> *Now, examine what makes WardSense unique in the entire literature: **Contribution C3**.*
> 
> *In traditional systems, you get a generic buzzer. Here, the system executes our automated bridge:*
> 1. *It deterministically extracts the breached vitals: heart rate 114, blood pressure 92, respiration 24. These are computed in Python, meaning a vital breach can never be hallucinated or omitted.*
> 2. *These exact abnormalities form the semantic search query into the patient's isolated ChromaDB vector store.*
> 3. *Our local, offline Llama-3 model reads the retrieved admission notes — finding that this patient had a pre-existing history of urinary tract infection — and synthesizes this comprehensive clinical escalation brief.*
> 
> *Notice that every single clinical claim has a **provenance citation tag**. If the model mentions blood pressure 92, it points directly to the Hour 8 telemetry timestamp."*

---

## Scene 5: Live Console — Chapter 5: Ask the Record (Patient-Isolated Q&A)

**Estimated Time:** 45 Seconds  
**Location:** Bottom of Bedside Console  
**Focused Panel:** `Ask the Record / Clinical Q&A`

---

### 🖱️ Actions: Where & How to Click
1. Look at Chapter 5: **`Ask the record`**.
2. Notice the question pre-filled in the text box:
   `"What vital trajectory or abnormalities were recorded for this patient?"`
3. Click: **`Ask`** (or submit the pre-filled question).
4. Watch the grounded answer stream/display.

---

### 👁️ Visual Cues to Point Out on Screen
- **Hard Isolation Badge:** Point to the badge: `🛡️ Hard Namespace Filter: patient_id == 'p000001'`.
- **Response Content:** The response cites *only* Bed 101's history, noting the rise in heart rate and hypotension.
- **Explain the Cross-Patient Safety Boundary:** Mention that even if another patient on the ward has an identical condition, ChromaDB strictly segregates patient collections.

---

### 🗣️ What to Say:
> *"Finally, in Chapter 5, the clinician can interact with the record conversationally.*
> 
> *When we ask: 'What vital trajectory was recorded for this patient?', the retrieval engine enforces a **hard patient_id boundary** at the database layer. Cross-patient data leakage is not merely unlikely — it is mathematically impossible by construction.*
> 
> *The answer cites solely Patient p000001's admission and telemetry history, providing rapid bedside verification."*

---

### 🖱️ Transition Action:
- In the sidebar navigation under **Verify**, click on **`Assurance`**.

---

## Scene 6: The Assurance Page — Verification & Provenance

**Estimated Time:** 60 Seconds  
**Location:** Sidebar $\to$ `Assurance` (`http://localhost:8501/assurance`)

---

### 👁️ Visual Cues to Point Out on Screen
1. **Provenance Shield Banner (Top):**
   - Point to the green verified banner:
     `● Model-derived — figures below are benchmark results from 20 held-out scenarios`.
   - Emphasize that **nothing on this page is hardcoded HTML**; every metric is parsed dynamically from `stage8_evaluation_report.json`.
2. **The Three Architecture Cards:**
   - `Federated setup`: 2 hospital sites, Flower over gRPC, sample-weighted FedAvg, zero raw data movement.
   - `Differential privacy`: Target budget $\varepsilon = 1.0, \delta = 10^{-5}$, L2 gradient clipping $C=1.0$, stateful Rényi DP accountant.
   - `Inference boundary`: 100% on-premise local quantized LLM, zero network egress.
3. **Verification Table (Clinical Human-in-the-Loop):**
   - **`0.0%` Treatment Safety Violations** across all evaluated scenarios.
   - **`0.812` Cohen's Kappa ($\kappa$)** indicating high human inter-rater agreement.
   - **`1.0` (100%) agreement** on factual accuracy, clinical relevance, and completeness.

---

### 🗣️ What to Say:
> *"Now we move to the **Assurance** page. In clinical software, transparency is mandatory.*
> 
> *Notice the top banner: every figure here is read dynamically from our Stage 8 evaluation pipeline. If the test artifact were missing or fabricated, this page would show a warning banner rather than numbers.*
> 
> *Here are our three architectural pillars:*
> 1. *Decentralized training across 2 hospital sites via Flower.*
> 2. *Mathematical differential privacy at a strict epsilon of 1.0 using continuous Rényi DP bookkeeping.*
> 3. *Complete data sovereignty with 100% local, offline inference.*
> 
> *Most importantly, look at our human clinical validation: two independent raters evaluated 20 held-out deteriorating patient scenarios. We achieved an inter-rater Cohen's Kappa of **0.812**, with **zero treatment safety violations** across the entire cohort."*

---

### 🖱️ Transition Action:
- In the sidebar navigation under **Patient-facing**, click on **`Patient portal`**.

---

## Scene 7: The Patient Portal — Patient-Centered Transparency

**Estimated Time:** 45 Seconds  
**Location:** Sidebar $\to$ `Patient portal` (`http://localhost:8501/portal`)

---

### 👁️ Visual Cues to Point Out on Screen
- **The Sidebar Lock:** `🔒 RECORD ISOLATION ENFORCED` badge.
- **Quick Question Buttons:**
  - Click on: **`"What were my recorded heart rate and blood pressure averages?"`**
- **Empathetic, Accessible Tone:**
  - Point out how the AI translates frightening ICU telemetry into clear, reassuring, plain-English explanations without condescension or medical confusion.

---

### 🗣️ What to Say:
> *"Lastly, we recognize that patient deterioration impacts families deeply. We built a dedicated **Patient Portal**.*
> 
> *Patients and their families are often terrified by monitors beeping with terms like 'NEWS2 score of 6' or 'tachycardic crisis'.*
> 
> *Here, using the exact same private record isolation, the system answers patient questions in plain, empathetic English — explaining what their vital signs mean, how long their observation stay is, and reassuring them that the clinical team is monitoring them closely.*
> 
> *Zero data leaves the record; full reassurance is provided to the patient."*

---

## Scene 8: The Grand Finale & Q&A Handoff

**Estimated Time:** 30 Seconds

### 🗣️ Closing Pitch:
> *"To conclude:*
> - *We answered the research question: differential privacy on continuous clinical time-series is viable, achieving an **0.898 AUROC** and an **86.4% detection rate** at $\varepsilon = 1.0$.*
> - *We built the product: a real-time, patient-isolated, grounded workstation that bridges predictive ML directly with verified clinical language generation.*
> 
> *The entire system is running live on this machine right now. Thank you, and we welcome your questions!"*

---

## 🛠️ Quick Troubleshooting Guide During Demo

| Issue | Quick Fix |
|:---|:---|
| **App disconnects or reloads** | Refresh browser (`Ctrl + F5` or `Cmd + Shift + R`). The app persists state in Streamlit cache. |
| **Ollama LLM is slow to respond** | The system includes fallback deterministic summarization that outputs verified clinical templates if Ollama is busy. |
| **Want to replay from beginning** | In the Demo Banner at the top, click `Exit demo` or `Chapter 1`. Or reload `http://localhost:8501/overview`. |
| **Juror asks to see another patient** | In the Bedside Console, click the dropdown under `Select Patient` and choose `Bed 103 (p000008)` for Respiratory Decline or `Bed 105 (p000014)` for Stable Control. |
