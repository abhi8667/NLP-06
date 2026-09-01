# Base Papers & Dataset — Talking Script

**Saturday, 22 August, 11:00. ~12 minutes + questions. Presenter: you, alone.**

Read this once tonight, once tomorrow, and again 30 minutes before. Don't memorise it word for word — know the shape of each beat so you can say it in your own words even if a question knocks you off the script.

---

## Opening (30 sec)

> "I'm going to cover two things quickly: the five base papers that ground our related work, and the dataset we actually built on. For each base paper I'll say what it does, and specifically what it *doesn't* do — because that gap is the whole reason our project exists."

---

## Slide 2 — The research question (30 sec)

> "Our question is: in a federated, differentially private clinical model trained on ward vitals, at what privacy budget does the model stop being clinically useful — and do sequence models like LSTM and GRU tolerate that privacy noise better than a plain CNN?"

**If asked "why does this matter":** privacy and usefulness trade off directly — more noise means stronger privacy but a worse model. Nobody has measured where that trade-off actually breaks for *clinical time-series* specifically.

---

## Slide 3 — Why five papers, two families (30 sec)

> "We grouped the five base papers into two families. Three of them — b1, b2, b3 — do privacy and federated learning well, but on static, non-clinical, or non-temporal data. The other two — b4, b5 — do medical language models well, but with no privacy protection at all. Our project sits exactly between them."

---

## Slide 4 — b1 (45 sec)

> "b1 is Shukla et al., 2025, Scientific Reports — federated learning with differential privacy for breast cancer diagnosis. Five hundred sixty-nine records, thirty-two features, completely static — one row per patient, no time dimension at all. They report ninety-six percent accuracy at epsilon 1.9.
>
> What we take from them is the *shape* of the experiment — sweep epsilon, measure how accuracy degrades. What they don't have is any time-series structure, so we don't know if that same curve holds for sequential vitals. That's our C1."

**Numbers if pressed:** ε tested at 0.5, 1.0, 1.9, 3.0, 5.0, and no-DP. Basic composition accounting, not Rényi — meaningfully weaker privacy bookkeeping than what we use.

---

## Slide 5 — b2 (45 sec)

> "b2 is Tanveer et al., 2025, Digital Health, published by SAGE — 'Balancing privacy and performance in healthcare.' Stroke prediction data, five thousand records, again static tabular. They use Flower for federation, same framework we use, and DP-SGD with Rényi accounting — same accountant family as us.
>
> They land at epsilon around 0.69. Their five-layer pipeline — edge, privacy, aggregation, app, decision — is a design pattern we borrowed. What's missing is any temporal data and any language model layer."

**Note if asked:** this paper's title in our PRD was wrong — an earlier draft called it "Privacy-Preserving Federated Reference Architecture." I checked the actual paper on SAGE and PubMed Central this week; the correct title and citation are what's on this slide.

---

## Slide 6 — b3 (45 sec)

> "b3 is Mosaiyebzadeh et al., Electronics, MDPI — 'Privacy-Preserving Federated Learning-Based Intrusion Detection for IoHT Devices.' Internally the paper calls its own framework SECIoHT-FL — that's not a separate paper, that's this paper's own name for itself.
>
> They compare a plain deep network against a CNN under DP-SGD using Opacus — same DP library we use — but only at two noise levels, not a full sweep, and on network intrusion traffic, not clinical vitals. That two-architecture comparison under DP is the direct template for our C2: LSTM and GRU against CNN, across a full six-point epsilon sweep, on real vitals instead of network traffic."

**Numbers if pressed:** ε ≈ 0.43 at noise 1.5 on one dataset, ε ≈ 6.69 at noise 0.5. Received Nov 2024, published in the January 2025 issue — so both "Dec 2024" and "2025" you may have seen in different documents are describing the same paper at different points in its publication timeline.

---

## Slide 7 — b4 (45 sec)

> "b4 flips to the other family. Wada et al., 2025, npj Digital Medicine — retrieval-augmented generation for radiology contrast-media consultation. A local Llama model, grounded in real clinical guidelines through RAG. Their headline result: hallucinations dropped from eight percent to zero once retrieval was added.
>
> That result is the direct justification for why our assistant uses RAG instead of a bare language model. What they don't have is any privacy mechanism, any federation, and no continuous vital monitoring — it's a one-shot consultation tool, not a ward system."

---

## Slide 8 — b5 (45 sec)

> "b5 is a scoping review, not an experimental paper — Miao et al., JMIR, sixty-seven studies on medical language models with retrieval. Two findings from it matter most to us. First: ninety-four percent of those studies target doctors, only six percent target nursing workflows — we're the six percent. Second, and this is the one I'd quote if asked to justify the whole project in one sentence: *none of the sixty-seven studies integrate real-time telemetry with retrieval-augmented generation.* That's a peer-reviewed review confirming our gap is real, not something we invented to sound novel."

---

## Slide 9 — The gap table (60 sec — this is the slide that matters most)

> "This is the summary. Down the left are the capabilities that matter for a system like ours. Across the top, the five papers and us. Full teal means the paper has it. Amber means partial. Grey means absent.
>
> The pattern is: b1 through b3 are solid teal down the privacy rows, and grey down the RAG and monitoring rows. b4 and b5 are the mirror image. Nobody has a column that's teal all the way down — except ours. That's not a coincidence, that's the actual definition of our contribution: we combine what the privacy papers do well with what the RAG papers do well, in one system, on real vitals."

**Pause here. Let them look at the grid. Don't rush past it.**

---

## Slide 10 — Our four contributions (60 sec)

> "Four contributions map directly onto that gap.
>
> C1: where does model usefulness break down as privacy noise increases, for clinical time-series specifically — nobody's measured that.
>
> C2: do LSTM and GRU tolerate that noise better than a CNN, on real vitals, across a full sweep — b3 only tested two noise points on non-clinical data.
>
> C3 is the one I'd call our actual novelty: when the federated detector raises an alert, the alert itself becomes the query into that patient's own record, and a language model writes the explanation. No paper in these five — and none of the sixty-seven b5 reviewed — connects a privacy-protected model's output to a retrieval system that way.
>
> C4 is rigor, not novelty: we use tight Rényi accounting across the full training run, which is a stronger privacy bookkeeping method than b1 uses."

---

## Slide 11 — The dataset, part 1 (45 sec)

> "For data, we use PhysioNet's 2019 sepsis challenge set — forty thousand, three hundred thirty-six real de-identified ICU patients across two independent hospital systems. Open access, no credentialing required. One row per patient per hour — heart rate, blood pressure, oxygen, respiration, temperature, glucose.
>
> We didn't start here. We first tried Synthea, a synthetic patient generator, and it failed two independent tests — readings months apart instead of hours, and almost no oxygen or temperature data at all. We tested this dataset the same way before committing to it, and it passed."

---

## Slide 12 — The dataset, part 2: what we measured (60 sec)

> "Three things we verified ourselves, on the real data, before building anything on top of it.
>
> First, coverage: heart rate and oxygen are recorded in roughly eighty-five to ninety percent of hours. Temperature is lower, around a third — so we forward-fill, and we report exactly how much of each vital is filled versus actually observed. That number goes straight into the paper's limitations section.
>
> Second, labelling: we don't use the dataset's built-in sepsis label as our target. We compute our own score every hour using NEWS2 — the early-warning chart used across the NHS — because that keeps the project a general deterioration monitor, not narrowly a sepsis detector. At the standard escalation threshold, that gives us about twelve point six percent of hours flagged, which is close to what our original plan assumed. And we cross-checked it: our NEWS2 flag caught roughly sixty to seventy-five percent of patients who the dataset's own clinical records confirmed became septic — so the label we derived tracks something real.
>
> Third, and this one changed our experiment design: if you ask the model to predict right now from the current vitals, it's almost trivially easy, because the label is computed from those same vitals. We measured it — accuracy is far higher at zero hours ahead than it is six hours ahead. So we forecast six hours out. That's both the clinically meaningful version of the problem, and the version that actually leaves room for the privacy experiment to show something."

---

## Closing (30 sec)

> "So: five base papers, three doing privacy without monitoring, two doing monitoring without privacy, and one review confirming nobody's combined them. One real dataset, tested the same way our failed one was tested, with the labelling and forecasting choices measured rather than assumed. That's the foundation the rest of the project builds on."

**Stop talking. Let the gap table or dataset slide stay up. Take questions.**

---

## If you only remember three things

1. **b5's finding, verbatim if you can:** "none of the sixty-seven studies integrate real-time telemetry with RAG." That's your one-sentence justification for the whole project.
2. **The gap table is the payoff slide.** Everything before it is setup; everything after it (C1–C4) is "here's what we do about that gap." Don't rush the table.
3. **The dataset section is yours to own** — nobody else measured these numbers, you did, on your own hardware. Say so plainly if asked: *"We ran this ourselves and the results are in `p0/results/`."*
