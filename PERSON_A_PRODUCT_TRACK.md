# Person A — Product Track

**You own:** the patient assistant, the retrieval layer, both interfaces, the voice pipeline, and the alert→summary bridge.

Read [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md) first for the locked decisions. This file is your work.

> **Your track is off the critical path.** The research track gates the schedule, so you have slack. Spend it supporting the experiment campaign and the figures — **not** on UI polish, which is explicitly out of scope.

---

## Already done

- [x] Ollama installed and running (v0.32.13)
- [x] `llama3.1:8b` pulled — 4.9 GB, measured 21 tok/s
- [x] `llama3.2:3b` pulled — 2.0 GB, measured 66 tok/s
- [x] Both benchmarked on a real alert-summary prompt

---

## The rules you must not break

**Patient isolation is the privacy boundary.** Every retrieval filters on `patient_id`. A test must prove a cross-patient query returns nothing — verified by test, not by inspection.

**No outbound network calls during inference.** Ever. This is the project's entire premise. Audit it, don't assume it.

**Two models, split by job:**

| Job | Model | Why |
|---|---|---|
| Alert summaries | `llama3.1:8b` | The 3B **missed a low-oxygen reading** in testing — the most clinically important of three flagged vitals |
| Patient chat | `llama3.2:3b` | 3× faster, latency matters, stakes lower |

**Temperature 0.2** for both. Matches the paper we build on.

**Hallucination cannot be detected by keyword matching.** We proved this: both models fabricated content and a keyword checker reported both as clean. The 3B invented "a history of exacerbations"; the 8B invented a 24-hour trend. Neither used a watchlist word. Your verifier must work at the level of individual claims.

---

## P2 — Data pipeline *(shared with B)*

Your half is the retrieval side.

- [ ] Generate synthetic encounter notes per patient using the local LLM — these populate the knowledge base, since PhysioNet has vitals but no free text
- [ ] Chunk and embed notes into **per-patient ChromaDB collections**
- [ ] Tag every chunk with `patient_id` metadata
- [ ] Stand up SQLite + AES-256 on the encrypted volume
- [ ] **Gate:** a cross-patient retrieval probe returns nothing

---

## P3A — Build the assistant

### RAG layer

- [ ] `embedder.py` — chunk and embed patient records
- [ ] `retriever.py` — semantic search with a **hard `where` filter on `patient_id`**
- [ ] `prompt_builder.py` — assemble context + question, with an explicit instruction to refuse when context is insufficient

### LLM layer

- [ ] `ollama_client.py` — local HTTP calls, temperature 0.2, model selectable per job
- [ ] Confirm no network egress during generation

### Backend

- [ ] FastAPI: `/query`, `/ingest/lab_report`, `/alerts`, `/score`

### Interfaces

- [ ] Patient chat (Streamlit) — conversation history, clear "information only, not medical advice" disclaimer
- [ ] Clinician dashboard (Streamlit) — alerts ranked by risk, drill-down, follow-up question box
- [ ] Lab-report PDF ingestion → text → chunks → patient's collection

### Voice — demo scope only

Not a paper contribution. **First item on the de-scoping ladder** if the schedule tightens.

- [ ] `faster-whisper` small int8 for speech-to-text (~500 MB VRAM)
- [ ] Piper for speech output (CPU only)
- [ ] Wire both through the *same* RAG pipeline as typed input — no separate path

Both run **sequentially** with the LLM, not concurrently, so there is no VRAM contention on the 6 GB card.

**Exit gate:**
- [ ] A patient question returns an answer traceable to that patient's own record
- [ ] An adversarial probe for another patient's data returns nothing
- [ ] No outbound network call occurs during inference

---

## P5 — The alert→summary bridge *(your most important work)*

This is **C3**, the contribution with no analogue in any of the five base papers. Not that RAG and federated learning both exist — that the output of a privacy-protected federated model becomes the semantic query input to a grounded LLM.

> **Start this against the FIRST checkpoint Person B produces. Do not wait for the campaign to finish.** You need *a* trained model, not the final one.

- [ ] `risk_scorer.py` — serve the trained detector
- [ ] `alert_rag_bridge.py` — identify which vitals are abnormal, build an alert-specific retrieval query
- [ ] Alert summary prompt — explicit constraints: no treatment recommendations, no fabrication
- [ ] Live alerts in the dashboard, ranked by risk
- [ ] Full end-to-end flow: reading arrives → scored → alert raised → summary generated → clinician reviews → follow-up answered

**Exit gate:**
- [ ] The walkthrough runs start to finish with no manual intervention
- [ ] Privacy audit passes: no egress, per-patient isolation holds, storage encrypted, no patient data in logs or session state

---

## P6 — Summary evaluation *(shared with B)*

Three independent layers. The third is the strongest and the cheapest to defend, because we control ground truth.

- [ ] Generate 20–30 alert scenarios from held-out patients — **hold them out at the P2 freeze**, not later
- [ ] **Human rubric** — two raters scoring independently, then Cohen's κ
- [ ] **LLM-as-judge** — multiple judge models on the same rubric
- [ ] **Programmatic fact verification** — extract each claim, check it against the source record, report an objective hallucination rate

```python
# integration/fact_verifier.py — claim-level, NOT keyword matching
def verify_summary_facts(summary, source_record):
    claims = extract_claims(summary)
    unsupported = [c for c in claims if not supported_by(c, source_record)]
    return len(unsupported) / max(len(claims), 1), unsupported
```

**Exit gate:**
- [ ] All three layers reported with numbers, including κ and the objective hallucination rate
- [ ] Disagreements between layers explained, not averaged away

---

## What you owe Person B

| When | What |
|---|---|
| P6 | The hallucination rate and rubric scores — these go in the paper's C3 section |
| P7 | An annotated alert example for the figures (anonymised) |
| P8 | Review of the system-architecture and C3 sections |
| P4 | Spare hands — the campaign runs for days and needs babysitting |

## What you need from Person B

| When | What |
|---|---|
| P2 | The frozen dataset hash and the held-out patient list |
| P3B → P5 | The **first** trained checkpoint — not the final one |
| P5 | The risk-score output format and threshold |

---

## De-scoping ladder — your items

If the schedule tightens, drop in this order:

1. Voice pipeline *(demo only, not a contribution)*
2. Lab-report PDF ingestion *(the notes corpus already demonstrates RAG)*
3. Dashboard polish *(functional beats pretty — this is explicitly out of scope anyway)*

**Never cut:** the alert→summary bridge. Without it the paper is an ε-sweep with a chatbot attached.
