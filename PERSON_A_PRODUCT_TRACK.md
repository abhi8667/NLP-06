# Person A — Product Track: Step-by-Step Build Guide

**You own:** clinical-note generation, retrieval, the LLM layer, both interfaces, voice, the alert→summary bridge, and the summary evaluation.

Read [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md) first for locked decisions. This file is the order to build in.

> **Your track is off the critical path.** Person B gates the schedule. Use your slack on the campaign and figures — **not** UI polish, which is out of scope.

**Rule for every step:** verify before moving on. Each has a *"done when"* you can actually check.

---

## Stage 0 — Setup ✅

- [x] Ollama installed (v0.32.13)
- [x] `llama3.2:3b` — 2.6 GB resident, 44 tok/s
- [x] `llama3.1:8b` — 4.2 GB resident, 20 tok/s
- [x] Both benchmarked on a real alert prompt

**Decision made for you:** start **3B-only**. They can't co-reside (2.6 + 4.2 > 6 GB) and swapping costs 7–11 s. Revisit in Stage 8 with real scenarios.

---

## Stage 1 — Shared foundations *(coordinate with Person B first)* ✅

### Step 1 — Agree one shared preprocessing module

Before either of you writes code. You both need loading, forward-fill and NEWS2. **If your NEWS2 disagrees with theirs, the alert says one thing and the summary another** — and that will surface in a demo.

- [x] Agree on `shared/preprocessing.py` — who writes it, where it lives
- [x] Agree the NEWS2 implementation is imported by both, never copy-pasted

**Done when:** you both import the same function and get identical scores on the same patient.

### Step 2 — Load and fill

```python
d = pd.read_csv(path, sep="|")          # one row = one hour
for c in ["HR","O2Sat","Temp","SBP","Resp","Glucose"]:
    d[c] = d[c].ffill().bfill()          # bfill matters — see below
```

**Gotcha:** row 1 is often entirely `NaN` — forward-fill has nothing to carry from. Averaged across patients, ~0.6 rows per file stay empty after `ffill` alone. Without `bfill` you get `NaN`s in generated notes.

**Also:** `EtCO2` is 100% empty. `Unit1`/`Unit2` are only 52% populated. Don't build on either.

- [x] Loader handles leading gaps
- [x] Record the per-vital imputation rate — **this goes in the paper**

**Done when:** you can load any patient and get zero `NaN`s in the six vitals.

### Step 3 — NEWS2 per hour

Five of seven components are available (no consciousness, no air/oxygen), so max is **14 not 20**.

- [x] Score each of Resp, SpO₂, SBP, HR, Temp
- [x] Total per hour; label = total ≥ 5

**Done when:** patient `p000001` scores 4 at hour 2, rising to 9 at hour 8.

---

## Stage 2 — Build the knowledge base *(your biggest task)* ✅

PhysioNet has **no clinical text at all.** RAG retrieves text. You have to manufacture it.

### Step 4 — Derive the facts worth writing about

- [x] NEWS2 trajectory and when it crossed 5
- [x] Vital trends ("heart rate rose from 89 to 110 over four hours")
- [x] Threshold crossings — when each vital first left normal range
- [x] Which labs were ordered and what they showed
- [x] Stay context — hour N of M, age, sex, admission timing

### Step 5 — Generate notes from templates

3–5 notes per patient covering different points in the stay: admission, a mid-stay nursing note, a deterioration note if NEWS2 crossed 5, a labs note.

**The rule that makes Stage 8 possible:** every fact must come from real data via a template. Fill values into sentence structures; let the LLM smooth the phrasing only. **Never let it invent freely** — if it does, you're evaluating one model's hallucinations against another's and the whole C3 evaluation is meaningless.

- [x] Template set covering the note types
- [x] Generator fills from real values
- [x] **A ground-truth fact list saved per patient** — you will need this in Stage 8

**Done when:** you can point at any sentence in any note and name the row it came from.

---

## Stage 3 — Retrieval ✅

### Step 6 — Embed into per-patient collections

- [x] Chunk notes
- [x] Embed locally (BGE-small or MiniLM — no API)
- [x] One ChromaDB collection **per patient**, `patient_id` in metadata

### Step 7 — Retriever with a hard filter

```python
results = collection.query(
    query_embeddings=embed(question),
    n_results=4,
    where={"patient_id": patient_id},    # the privacy boundary
)
```

- [x] Hard filter enforced on patient_id

### Step 8 — The adversarial isolation test ⚠️

**Write this before any UI exists.** Isolation is a security property, not a feature.

- [x] Test actively tries to retrieve patient B's data while scoped to patient A
- [x] Test asserts the result is **empty**, not merely "different"
- [x] Test runs in CI / on every change

**Done when:** the test passes and you have *tried* to break it.

---

## Stage 4 — Generation ✅

### Step 9 — Ollama client

- [x] POST to `localhost:11434/api/generate`, `stream: False`
- [x] **Temperature 0.2**, model configurable (default `llama3.2:3b`)
- [x] Confirm no outbound network call during generation

### Step 10 — Prompt builder

```
You are a clinical decision support assistant.
You have access only to the following information about this patient.
Do not infer, assume, or generate information not present in the context.
If the context does not contain enough to answer safely, say so.

PATIENT CONTEXT:
{retrieved chunks, separated by ---}

QUESTION:
{question}

RESPONSE:
```

- [x] Refusal instruction included
- [x] **Test the refusal actually fires** — ask something the record can't answer

### Step 11 — First end-to-end, from the terminal

- [x] One patient, one question, grounded answer — no UI

**Done when this works, ~80% of your track's risk is gone.** Everything after is presentation.

---

## Stage 5 — Interfaces ✅

### Step 12 — FastAPI

- [x] `/query`, `/ingest`, `/alerts`, `/score`, `/telemetry/step`, `/patients`, `/health`

### Step 13 — Patient chat (Streamlit)

- [x] Conversation history within session
- [x] Disclaimer: information only, not medical advice
- [x] No cross-patient data reachable from session state

### Step 14 — Clinician dashboard shell

- [x] Alerts ranked by risk score (wired to real telemetry & Alert Bridge)
- [x] Drill-down panel displaying AlertSummaryCard (deterministic table + grounded LLM narrative)
- [x] Follow-up question box grounded in that patient's record

**Keep it functional, not pretty.** Polish is explicitly out of scope.

---

## Stage 6 — Extras *(first to cut if time runs short)*

### Step 15 — Lab-report PDF ingestion

- [ ] PDF → text → chunks → that patient's collection

### Step 16 — Voice *(demo scope, not a contribution)*

- [ ] `faster-whisper` small int8 for speech-to-text (~500 MB VRAM)
- [ ] Piper for speech output (CPU only)
- [ ] Both route through the **same** RAG pipeline as typed input — no separate path

Runs sequentially with the LLM, so no VRAM contention.

---

## Stage 7 — The alert→summary bridge *(your contribution — C3)* ✅

> **Start the moment Person B has ANY checkpoint. Do not wait for the campaign.**

### Step 17 — Serve the detector

- [x] `risk_scorer.py` loads B's model, scores incoming windows

### Step 17b — Build the replay harness ⚠️

**There is no live data feed.** PhysioNet is a static archive of finished ICU stays. Something has to *produce* the "incoming reading" that everything downstream reacts to.

- [x] `vitals_replay.py` — reads one patient's stay and emits it **one hour at a time** on a timer
- [x] Configurable speed: 1 hour per second for a demo, faster for testing
- [x] Feeds the scoring endpoint exactly as a real monitor would, so nothing downstream knows the difference
- [x] Pick 3–4 demo patients whose NEWS2 visibly climbs across the stay — `patient_facts()` gives you `first_crossing_hour` and `peak_hour`, so you can select them programmatically

**Be honest about this everywhere.** It is a *replayed* feed of real recorded patients, not a live hospital connection. That is completely standard for a research prototype and nobody expects otherwise — but claiming a live feed you do not have is the kind of thing that unravels badly under questioning.

**Done when:** you can start a replay and watch the dashboard go from calm to alerting without touching anything.

### Step 18 — Build the bridge properly

**The wrong version:** dashboard shows an alert, and separately shows a summary. Two panels. Contributes nothing.

**The right version:** the alert's *content* becomes the retrieval *query*.

- [x] Identify **which** vitals are abnormal and by how much
- [x] Build the retrieval query **from those specific abnormalities** — not a generic patient lookup
- [x] Retrieve history relevant to *this* deterioration
- [x] Generate a summary: what's abnormal, what history explains it, what patterns appear

**Design decision that removes your model-size problem:** render the flagged vitals as **structured UI** — a table, deterministic. Use the LLM only for the narrative. A missed vital then becomes impossible by construction, rather than something you hope the model complies with. This is why 3B-only is viable.

- [x] Prompt forbids treatment recommendations and fabrication

### Step 19 — Full flow

- [x] Reading arrives → scored → alert raised → summary generated → clinician reviews → follow-up answered

### Step 20 — Privacy audit

- [x] No outbound network calls anywhere
- [x] Per-patient isolation holds under adversarial test
- [x] Storage encrypted; raw DB file unreadable
- [x] No patient data in API logs or Streamlit session state

**Done when:** the whole walkthrough runs with no manual intervention.

---

## Stage 8 — Evaluation *(this is the research, not the build)*

Building the bridge takes a week. **Evaluating it rigorously is what makes it publishable.**

### Step 21 — Scenarios

- [ ] 20–30 alert scenarios from held-out patients
- [ ] **Held out at the P2 freeze**, not chosen later

### Step 22 — Human rubric

- [ ] Factual accuracy, relevance, completeness, hallucination, conciseness
- [ ] Two raters scoring independently
- [ ] Cohen's κ for inter-rater agreement

### Step 23 — LLM-as-judge

- [ ] Multiple judge models, same rubric

### Step 24 — Programmatic fact verification ⚠️

**Keyword matching does not work.** Proven on your hardware: both models fabricated content and a keyword checker reported both clean. The 3B invented "a history of exacerbations"; the 8B invented a 24-hour trend. Neither used a watchlist word.

```python
def verify_summary_facts(summary, ground_truth_facts):
    claims = extract_claims(summary)              # claim-level, not keyword
    unsupported = [c for c in claims if not supported_by(c, ground_truth_facts)]
    return len(unsupported) / max(len(claims), 1), unsupported
```

This is why Step 5 demanded a ground-truth fact list per patient.

### Step 25 — Settle the model question

- [ ] Run 3B and 8B across all scenarios with the verifier
- [ ] If 3B holds up, ship 3B-only — a genuine edge-deployment finding

**Done when:** all three layers reported with numbers, and disagreements between them explained rather than averaged.

---

## Handoffs

**You need from Person B:** the frozen dataset hash and held-out patient list (P2) · the **first** checkpoint, not the final one (P3B) · risk-score format and threshold (P5)

**You owe Person B:** hallucination rate and rubric scores (P6) · an annotated alert example for the figures (P7) · review of the architecture and C3 sections (P8) · spare hands during the campaign (P4)

---

## De-scoping ladder — your items

1. Voice *(demo only)*
2. Lab-report PDF ingestion
3. Dashboard polish *(already out of scope)*

**Never cut the alert→summary bridge.** Without it the paper is an ε-sweep with a chatbot attached.
