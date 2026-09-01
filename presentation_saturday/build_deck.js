/**
 * Base Papers & Dataset — Saturday review presentation.
 * Same visual family as the WardSense deck (clinical slate / trace teal),
 * different layouts: paper cards, gap matrix, dataset measurement panels.
 */
const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "NLP-06";
pres.title = "Base Papers and Dataset";

// ---------------------------------------------------------------- palette
const BG      = "0D1B24";
const CARD    = "162A38";
const CARD2   = "1D3546";
const INK     = "E8EEF2";
const MUTED   = "8CA3B3";
const FAINT   = "5C7387";
const TEAL    = "00C2A8";
const TEAL_DK = "00806F";
const AMBER   = "E8A33D";
const RED     = "E0544F";
const GREY    = "3A5062";

const H = "Cambria";
const B = "Calibri";
const W = 13.3, M = 0.75;

// ---------------------------------------------------------------- helpers
function base(s) { s.background = { color: BG }; }

function title(s, text, sub) {
  if (sub) {
    s.addText(sub, { x: M, y: 0.42, w: W - 2 * M, h: 0.3, margin: 0,
      fontFace: B, fontSize: 11.5, color: TEAL, charSpacing: 1.6 });
  }
  s.addText(text, { x: M, y: 0.74, w: W - 2 * M, h: 1.05, valign: "top", margin: 0,
    fontFace: H, fontSize: 30, bold: true, color: INK });
}

function card(s, x, y, w, h, fill) {
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.08,
    fill: { color: fill || CARD }, line: { color: fill || CARD, width: 0 } });
}

function dot(s, x, y, d, col) {
  s.addShape(pres.ShapeType.ellipse, { x, y, w: d, h: d,
    fill: { color: col }, line: { color: col, width: 0 } });
}

function trace(s, x0, y0, w, amp, seed, col, thick) {
  const pts = []; const n = 60; let r = seed;
  const rnd = () => { r = (r * 9301 + 49297) % 233280; return r / 233280; };
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    let v = Math.sin(t * 11) * 0.28 + Math.sin(t * 23 + 1) * 0.14 + (rnd() - 0.5) * 0.18;
    if (i % 12 === 3) v += 1.0;
    if (i % 12 === 4) v -= 0.55;
    pts.push([x0 + t * w, y0 - v * amp]);
  }
  for (let i = 0; i < pts.length - 1; i++) {
    const [ax, ay] = pts[i], [bx, by] = pts[i + 1];
    s.addShape(pres.ShapeType.line, {
      x: Math.min(ax, bx), y: Math.min(ay, by),
      w: Math.abs(bx - ax), h: Math.abs(by - ay),
      line: { color: col || TEAL_DK, width: thick || 1.25 },
      flipH: bx < ax, flipV: by < ay });
  }
}

// ================================================================ 1 TITLE
let s = pres.addSlide(); base(s);
trace(s, 0, 6.35, W, 0.42, 11, TEAL_DK, 1.3);

s.addText("Base Papers & Dataset", { x: M, y: 2.05, w: 11, h: 1.0, margin: 0,
  fontFace: H, fontSize: 46, bold: true, color: INK });
s.addText("What five papers establish, what none of them do, and the real\ndata NLP-06 is built and measured on.", {
  x: M, y: 3.15, w: 10.5, h: 0.9, margin: 0,
  fontFace: B, fontSize: 15, color: MUTED, lineSpacing: 22 });
s.addText("NLP-06   ·   Review presentation   ·   22 August 2026", {
  x: M, y: 5.6, w: 10, h: 0.35, margin: 0,
  fontFace: B, fontSize: 11.5, color: FAINT, charSpacing: 1.4 });

// ================================================================ 2 QUESTION
s = pres.addSlide(); base(s);
title(s, "The question every base paper is measured against", "THE RESEARCH QUESTION");

card(s, M, 2.2, W - 2 * M, 2.5, CARD2);
s.addText([
  { text: "In a federated, differentially private clinical model trained on ward vitals, ", options: {} },
  { text: "at what privacy budget (ε) does utility fall below clinical usefulness", options: { bold: true, color: TEAL } },
  { text: " — and do sequence models (LSTM/GRU) tolerate that noise better than a CNN?", options: {} },
], { x: M + 0.5, y: 2.55, w: W - 2 * M - 1, h: 1.8, margin: 0, valign: "middle",
     fontFace: H, fontSize: 22, color: INK, lineSpacing: 32 });

s.addText("Every paper reviewed next is scored against one thing: does it help answer this, and where does it fall short?",
  { x: M, y: 5.05, w: 11, h: 0.5, margin: 0, fontFace: B, fontSize: 13, color: MUTED, italic: true });

// ================================================================ 3 TWO FAMILIES
s = pres.addSlide(); base(s);
title(s, "Two families, and a gap between them", "WHY THESE FIVE PAPERS");

card(s, M, 2.1, 5.55, 3.9);
s.addText("Privacy & federation", { x: M + 0.4, y: 2.35, w: 4.8, h: 0.4, margin: 0,
  fontFace: B, fontSize: 17, bold: true, color: TEAL });
s.addText("b1 · b2 · b3", { x: M + 0.4, y: 2.78, w: 4.8, h: 0.35, margin: 0,
  fontFace: B, fontSize: 12, color: FAINT, charSpacing: 1 });
s.addText("Federated learning and differential privacy, done rigorously — on data that is static, tabular, or non-clinical.",
  { x: M + 0.4, y: 3.25, w: 4.8, h: 0.9, margin: 0, fontFace: B, fontSize: 13, color: MUTED, lineSpacing: 18 });
s.addText("No sequence modelling. No language model. No monitoring.",
  { x: M + 0.4, y: 5.3, w: 4.8, h: 0.5, margin: 0, fontFace: B, fontSize: 12, color: RED, italic: true });

card(s, 7.05, 2.1, 5.5, 3.9);
s.addText("Language models in medicine", { x: 7.45, y: 2.35, w: 4.7, h: 0.4, margin: 0,
  fontFace: B, fontSize: 17, bold: true, color: TEAL });
s.addText("b4 · b5", { x: 7.45, y: 2.78, w: 4.7, h: 0.35, margin: 0,
  fontFace: B, fontSize: 12, color: FAINT, charSpacing: 1 });
s.addText("Retrieval-grounded language models that work — one deployed system, one review of sixty-seven studies.",
  { x: 7.45, y: 3.25, w: 4.7, h: 0.9, margin: 0, fontFace: B, fontSize: 13, color: MUTED, lineSpacing: 18 });
s.addText("No privacy mechanism. No federation. No continuous monitoring.",
  { x: 7.45, y: 5.3, w: 4.7, h: 0.5, margin: 0, fontFace: B, fontSize: 12, color: RED, italic: true });

s.addShape(pres.ShapeType.rect, { x: 6.68, y: 2.4, w: 0.02, h: 3.4, fill: { color: GREY }, line: { width: 0 } });

// ================================================================ paper card template
function paperSlide(tag, cite, venue, factsL, factsR, gives) {
  const sl = pres.addSlide(); base(sl);
  title(sl, cite, `BASE PAPER ${tag}`);

  card(sl, M, 2.05, W - 2 * M, 0.55, CARD2);
  sl.addText(venue, { x: M + 0.35, y: 2.05, w: W - 2 * M - 0.7, h: 0.55, margin: 0, valign: "middle",
    fontFace: B, fontSize: 12.5, color: TEAL, italic: true });

  const colW = (W - 2 * M - 0.3) / 2;
  card(sl, M, 2.78, colW, 2.15);
  factsL.forEach((f, i) => {
    sl.addText(f[0], { x: M + 0.32, y: 2.98 + i * 0.52, w: 1.7, h: 0.42, margin: 0,
      fontFace: B, fontSize: 10.5, color: FAINT, charSpacing: 0.5 });
    sl.addText(f[1], { x: M + 2.0, y: 2.98 + i * 0.52, w: colW - 2.3, h: 0.42, margin: 0,
      fontFace: B, fontSize: 11.5, color: INK });
  });

  card(sl, M + colW + 0.3, 2.78, colW, 2.15);
  factsR.forEach((f, i) => {
    sl.addText(f[0], { x: M + colW + 0.3 + 0.32, y: 2.98 + i * 0.52, w: 1.7, h: 0.42, margin: 0,
      fontFace: B, fontSize: 10.5, color: FAINT, charSpacing: 0.5 });
    sl.addText(f[1], { x: M + colW + 0.3 + 2.0, y: 2.98 + i * 0.52, w: colW - 2.3, h: 0.42, margin: 0,
      fontFace: B, fontSize: 11.5, color: INK });
  });

  card(sl, M, 5.15, W - 2 * M, 1.55, CARD2);
  sl.addText("What NLP-06 takes from it — and what's missing", { x: M + 0.4, y: 5.3, w: 11, h: 0.3, margin: 0,
    fontFace: B, fontSize: 11.5, bold: true, color: AMBER });
  sl.addText(gives, { x: M + 0.4, y: 5.62, w: 11, h: 1.0, margin: 0,
    fontFace: B, fontSize: 12, color: INK, lineSpacing: 16 });
  return sl;
}

// ================================================================ 4 b1
paperSlide("b1",
  "Federated learning with differential privacy\nfor breast cancer diagnosis",
  "Shukla et al. · Scientific Reports · 2025",
  [["Data", "569 records, 32 features — static tabular"],
   ["Model", "Random Forest baseline + FL Feed-Forward NN"],
   ["Framework", "TensorFlow Federated"]],
  [["DP method", "Gaussian DP-SGD, basic composition"],
   ["Best result", "96.1% accuracy at ε = 1.9"],
   ["Clients", "10 simulated, IID"]],
  "Gives us the ε-sweep template — sweep privacy budget, measure accuracy degradation. Missing: no time-series structure at all, so it can't tell us whether the same curve holds for sequential clinical vitals. → basis for C1.");

// ================================================================ 5 b2
paperSlide("b2",
  "Balancing privacy and performance in healthcare:\nA federated learning framework for sensitive data",
  "Tanveer, Iradat, Iqbal, Alsagri, Alhakbani, Ahmad, Khan · Digital Health (SAGE) · Sept 2025",
  [["Data", "Stroke Prediction dataset — 5,110 records, static"],
   ["Model", "3-layer fully-connected DNN"],
   ["Framework", "Flower (flwr) — same as NLP-06"]],
  [["DP method", "DP-SGD, Rényi accounting — same family as us"],
   ["Best result", "ε ≈ 0.69 after 10 rounds"],
   ["Design", "5-layer pipeline: Edge/Privacy/Aggregation/App/Decision"]],
  "Gives us the Flower + Rényi accounting pattern and the layered architecture template — both directly adopted. Missing: no temporal data, no language model layer.");

// ================================================================ 6 b3
paperSlide("b3",
  "Privacy-Preserving Federated Learning-Based\nIntrusion Detection System for IoHT Devices",
  "Mosaiyebzadeh et al. · Electronics (MDPI) 14(1):67 · rec. Nov 2024, publ. Jan 2025",
  [["Data", "wustl-ehms-2020 + ECU-IoHT — network/biometric traffic"],
   ["Models", "DNN vs CNN — compared directly"],
   ["Framework", "Opacus (PyTorch) — same DP library as us"]],
  [["Noise tested", "Only 2 levels: 0.5 and 1.5 — not a sweep"],
   ["ε results", "≈0.43–6.69 depending on noise/dataset"],
   ["Internal name", "Calls itself “SECIoHT-FL” — not a separate paper"]],
  "Gives us the Opacus implementation pattern and the DNN-vs-CNN comparison design. Missing: only 2 noise points, non-clinical traffic, no LSTM/GRU. → direct template for C2, extended to a full sweep on real vitals.");

// ================================================================ 7 b4
paperSlide("b4",
  "Retrieval-augmented generation elevates local LLM\nquality in radiology contrast media consultation",
  "Wada et al. · npj Digital Medicine · 2025",
  [["Model", "Llama 3.2-11B, local deployment"],
   ["Knowledge base", "ACR/ESUR guidelines — 66 chunks"],
   ["Retrieval", "Hybrid semantic + keyword, top-4"]],
  [["Key result", "Hallucinations 8% → 0% with RAG (p = 0.012)"],
   ["Evaluation", "100 scenarios, radiologist + 3 LLM judges"],
   ["Setting", "Temperature 0.2 — adopted by us"]],
  "Directly justifies our RAG design and evaluation methodology. Missing: no privacy, no federation, no continuous monitoring — a single-shot consultation tool, not a ward system.");

// ================================================================ 8 b5
paperSlide("b5",
  "Improving Large Language Model Applications in\nMedical and Nursing Domains With RAG: Scoping Review",
  "Miao et al. · JMIR · 2025 · 67 studies, Nov 2022–May 2025",
  [["Coverage", "94% target physician workflows"],
   ["", "only 6% target nursing"],
   ["Privacy", "Only 9/67 studies address it"]],
  [["Key finding", "“None of the 67 studies integrate real-time"],
   ["", "telemetry with RAG.”"],
   ["Type", "Review — no system built, no experiments"]],
  "Peer-reviewed confirmation that NLP-06's gap is real, not asserted. We target the underserved nursing workflow and the exact telemetry gap this review names.");

// ================================================================ 9 GAP MATRIX
s = pres.addSlide(); base(s);
title(s, "Nobody has a column that's full teal — except ours", "THE GAP TABLE");

const cols = ["b1", "b2", "b3", "b4", "b5", "NLP-06"];
const rows = [
  ["Federated learning",              ["y","y","y","n","n","y"]],
  ["Differential privacy (DP-SGD)",   ["y","y","y","n","n","y"]],
  ["Time-series / sequence data",     ["n","n","p","n","n","y"]],
  ["Systematic ε-sweep (5+ points)",  ["y","n","n","x","x","y"]],
  ["Local LLM + RAG",                 ["n","n","n","y","p","y"]],
  ["Patient-scoped retrieval",        ["n","n","n","p","n","y"]],
  ["Continuous vital monitoring",     ["n","n","p","n","n","y"]],
  ["Alert → RAG clinician summary",   ["n","n","n","n","n","y"]],
  ["Rényi DP accounting",             ["n","y","n","x","x","y"]],
];

const labelW = 3.55, cellW = (W - 2 * M - labelW) / cols.length, rowH = 0.44, top = 2.15;
cols.forEach((c, i) => {
  s.addText(c, { x: M + labelW + i * cellW, y: top - 0.4, w: cellW, h: 0.35, align: "center", margin: 0,
    fontFace: B, fontSize: 11.5, bold: c === "NLP-06", color: c === "NLP-06" ? TEAL : MUTED });
});
rows.forEach((r, ri) => {
  const y = top + ri * rowH;
  if (ri % 2 === 0) card(s, M, y - 0.03, W - 2 * M, rowH - 0.02, CARD);
  s.addText(r[0], { x: M + 0.2, y, w: labelW - 0.3, h: rowH - 0.05, valign: "middle", margin: 0,
    fontFace: B, fontSize: 11, color: INK });
  r[1].forEach((v, ci) => {
    const cx = M + labelW + ci * cellW + cellW / 2 - 0.09;
    const cy = y + (rowH - 0.05) / 2 - 0.09;
    if (v === "y") dot(s, cx, cy, 0.18, TEAL);
    else if (v === "p") dot(s, cx, cy, 0.18, AMBER);
    else if (v === "n") dot(s, cx, cy, 0.14, GREY);
    // "x" = not applicable — no mark at all
  });
});

const legY = top + rows.length * rowH + 0.15;
[["Yes", TEAL], ["Partial", AMBER], ["No", GREY]].forEach((L, i) => {
  dot(s, M + i * 1.7, legY, 0.14, L[1]);
  s.addText(L[0], { x: M + i * 1.7 + 0.24, y: legY - 0.09, w: 1.3, h: 0.3, margin: 0,
    fontFace: B, fontSize: 10.5, color: MUTED });
});
s.addText("blank = not applicable (paper does not attempt FL/DP)", {
  x: M + 5.4, y: legY - 0.09, w: 6, h: 0.3, margin: 0, fontFace: B, fontSize: 10, italic: true, color: FAINT });

// ================================================================ 10 CONTRIBUTIONS
s = pres.addSlide(); base(s);
title(s, "Four contributions, mapped to the gap", "WHAT NLP-06 ADDS");

const contribs = [
  ["C1", "ε-utility floor for clinical time-series", "b1 only tested static tabular data — we measure the same curve on sequential vitals.", TEAL],
  ["C2", "LSTM/GRU vs CNN under a full DP sweep", "b3 tested 2 noise points on network traffic — we run a full sweep on real vitals.", TEAL],
  ["C3", "Alert → retrieval bridge", "No paper in b1–b5, or the 67 b5 reviewed, connects a DP-protected model's output to patient-scoped retrieval.", AMBER],
  ["C4", "Tight Rényi accounting, provable ε < 2", "Rigor, not novelty — stronger bookkeeping than b1's basic composition.", TEAL],
];
contribs.forEach((c, i) => {
  const y = 2.1 + i * 1.18;
  card(s, M, y, W - 2 * M, 1.0, i === 2 ? CARD2 : CARD);
  s.addText(c[0], { x: M + 0.32, y: y + 0.15, w: 0.85, h: 0.7, valign: "middle", margin: 0,
    fontFace: H, fontSize: 22, bold: true, color: c[3] });
  s.addText(c[1], { x: M + 1.3, y: y + 0.12, w: 4.6, h: 0.76, valign: "middle", margin: 0,
    fontFace: B, fontSize: 13.5, bold: true, color: INK, lineSpacing: 16 });
  s.addText(c[2], { x: M + 6.1, y: y + 0.12, w: 5.6, h: 0.76, valign: "middle", margin: 0,
    fontFace: B, fontSize: 11, color: MUTED, lineSpacing: 14 });
  // C3's own card fill (CARD2) plus the amber tag colour already flag it as
  // the standout row -- a star icon here only risked overlapping the text.
});

// ================================================================ 11 DATASET 1
s = pres.addSlide(); base(s);
title(s, "PhysioNet / CinC 2019 — not a simulation", "THE DATASET, PART 1");

const dstats = [["40,336", "real ICU patients"], ["2", "independent hospitals"], ["Open", "no credentialing"]];
dstats.forEach((d, i) => {
  const x = M + i * 3.95;
  card(s, x, 2.1, 3.7, 1.55);
  s.addText(d[0], { x: x + 0.3, y: 2.28, w: 3.1, h: 0.65, margin: 0,
    fontFace: H, fontSize: 30, bold: true, color: TEAL });
  s.addText(d[1], { x: x + 0.3, y: 2.9, w: 3.1, h: 0.6, margin: 0,
    fontFace: B, fontSize: 11.5, color: MUTED });
});

card(s, M, 3.95, W - 2 * M, 1.15, CARD2);
s.addText("We tried Synthea first. It failed two independent tests.", { x: M + 0.4, y: 4.12, w: 11, h: 0.35, margin: 0,
  fontFace: B, fontSize: 14, bold: true, color: RED });
s.addText("154-day median gap between readings (no telemetry cadence) · near-zero oxygen and temperature data. Both measured and recorded in p0/results/s1_cadence.json before switching.",
  { x: M + 0.4, y: 4.5, w: 11.3, h: 0.55, margin: 0, fontFace: B, fontSize: 11.5, color: MUTED, lineSpacing: 15 });

card(s, M, 5.3, W - 2 * M, 1.4);
s.addText("PhysioNet passed the same tests", { x: M + 0.4, y: 5.47, w: 11, h: 0.35, margin: 0,
  fontFace: B, fontSize: 14, bold: true, color: TEAL });
s.addText("HR 87–92%  ·  SpO₂ 85–88%  ·  Resp 78–90%  ·  SBP 85–86%  ·  Temp 33–36%   — hourly rows, real cadence.",
  { x: M + 0.4, y: 5.85, w: 11.3, h: 0.6, margin: 0, fontFace: B, fontSize: 12, color: INK, lineSpacing: 16 });

// ================================================================ 12 DATASET 2
s = pres.addSlide(); base(s);
title(s, "Three things we measured before building on it", "THE DATASET, PART 2");

const meas = [
  ["Labels: NEWS2, not the sepsis flag", "Derived hourly from vitals — keeps this a general deterioration monitor. 12.6% of hours flagged at threshold ≥ 5, close to the 10–15% originally planned. Cross-checked: caught 60–75% of patients later confirmed septic."],
  ["Imputation: forward-filled, and reported", "HR 7% · Resp 9% · SpO₂ 12% · SBP 12% · Temp 64% · Glucose 86% imputed. Goes directly into the paper's limitations — not hidden."],
  ["Horizon: 6 hours, not 0 — measured, not assumed", "At 0h the task is near-trivial (AUROC 0.90) because the label derives from the same vitals. At 6h: 0.78 — a real forecast, and one that leaves room for the DP experiment to show something."],
];
meas.forEach((m, i) => {
  const y = 2.05 + i * 1.62;
  card(s, M, y, W - 2 * M, 1.42);
  dot(s, M + 0.35, y + 0.28, 0.3, TEAL);
  s.addText(String(i + 1), { x: M + 0.35, y: y + 0.28, w: 0.3, h: 0.3, align: "center", valign: "middle", margin: 0,
    fontFace: B, fontSize: 12, bold: true, color: BG });
  s.addText(m[0], { x: M + 0.95, y: y + 0.15, w: 10.8, h: 0.38, margin: 0,
    fontFace: B, fontSize: 14.5, bold: true, color: INK });
  s.addText(m[1], { x: M + 0.95, y: y + 0.56, w: 10.8, h: 0.78, margin: 0,
    fontFace: B, fontSize: 11.5, color: MUTED, lineSpacing: 15 });
});

// ================================================================ 13 CLOSE
s = pres.addSlide(); base(s);
trace(s, 0, 6.3, W, 0.4, 23, TEAL_DK, 1.3);

s.addText("Three privacy papers without monitoring.\nTwo monitoring papers without privacy.\nOne review confirming nobody's combined them.", {
  x: M, y: 1.9, w: 11.3, h: 1.9, margin: 0,
  fontFace: H, fontSize: 26, bold: true, color: INK, lineSpacing: 36 });
s.addText("One real dataset, tested the same way our failed one was tested — with the labelling and forecasting choices measured, not assumed.",
  { x: M, y: 4.0, w: 10.3, h: 0.85, margin: 0, fontFace: B, fontSize: 14.5, color: TEAL, lineSpacing: 20 });
s.addText("Questions", { x: M, y: 5.35, w: 5, h: 0.5, margin: 0,
  fontFace: H, fontSize: 20, bold: true, color: MUTED });

pres.writeFile({ fileName: "Base_Papers_and_Dataset.pptx" }).then(f => console.log("wrote", f));
