/**
 * WardSense — electronica India idea-submission deck.
 * Dark "patient monitor" identity: deep clinical slate, trace teal, escalation amber.
 */
const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";           // 13.3 x 7.5 — MUST be set before adding slides
pres.author = "NLP-06";
pres.title = "WardSense";

// ---------------------------------------------------------------- palette
const BG      = "0D1B24";   // deep clinical slate — dominant
const CARD    = "162A38";   // raised surface
const CARD2   = "1D3546";   // higher surface
const INK     = "E8EEF2";
const MUTED   = "8CA3B3";
const FAINT   = "5C7387";
const TEAL    = "00C2A8";   // monitor trace
const TEAL_DK = "00806F";
const AMBER   = "E8A33D";   // escalation
const RED     = "E0544F";   // emergency

const H = "Cambria";        // safe-list serif, renders true-to-width
const B = "Calibri";        // safe-list sans

const W = 13.3, HT = 7.5, M = 0.75;

// ---------------------------------------------------------------- helpers
function base(s) {
  s.background = { color: BG };
}

/** Eyebrow label ABOVE the title, so a two-line title has room to breathe. */
function title(s, text, sub) {
  if (sub) {
    s.addText(sub, {
      x: M, y: 0.42, w: W - 2 * M, h: 0.3,
      fontFace: B, fontSize: 11.5, color: TEAL, margin: 0, charSpacing: 1.6,
    });
  }
  s.addText(text, {
    x: M, y: 0.74, w: W - 2 * M, h: 1.15, valign: "top",
    fontFace: H, fontSize: 32, bold: true, color: INK, margin: 0,
  });
}

function card(s, x, y, w, h, fill) {
  s.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.08,
    fill: { color: fill || CARD }, line: { color: fill || CARD, width: 0 },
  });
}

function marker(s, x, y, n, col) {
  s.addShape(pres.ShapeType.ellipse, {
    x, y, w: 0.42, h: 0.42,
    fill: { color: col || TEAL }, line: { color: col || TEAL, width: 0 },
  });
  s.addText(String(n), {
    x, y, w: 0.42, h: 0.42, align: "center", valign: "middle",
    fontFace: B, fontSize: 13, bold: true, color: BG, margin: 0,
  });
}

/** Vital-sign trace built from line segments — the deck's visual motif. */
function trace(s, x0, y0, w, amp, seed, col, thick) {
  const pts = [];
  const n = 60;
  let r = seed;
  const rnd = () => { r = (r * 9301 + 49297) % 233280; return r / 233280; };
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    let v = Math.sin(t * 11) * 0.28 + Math.sin(t * 23 + 1) * 0.14 + (rnd() - 0.5) * 0.18;
    if (i % 12 === 3) v += 1.0;                 // periodic beat spike
    if (i % 12 === 4) v -= 0.55;
    pts.push([x0 + t * w, y0 - v * amp]);
  }
  for (let i = 0; i < pts.length - 1; i++) {
    const [ax, ay] = pts[i], [bx, by] = pts[i + 1];
    s.addShape(pres.ShapeType.line, {
      x: Math.min(ax, bx), y: Math.min(ay, by),
      w: Math.abs(bx - ax), h: Math.abs(by - ay),
      line: { color: col || TEAL_DK, width: thick || 1.25 },
      flipH: bx < ax, flipV: by < ay,
    });
  }
}

function stat(s, x, y, w, value, label, col) {
  s.addText(value, {
    x, y, w, h: 0.75, align: "left", margin: 0,
    fontFace: H, fontSize: 40, bold: true, color: col || TEAL,
  });
  s.addText(label, {
    x, y: y + 0.72, w, h: 0.6, align: "left", margin: 0,
    fontFace: B, fontSize: 11.5, color: MUTED,
  });
}

// ================================================================ 1 TITLE
let s = pres.addSlide(); base(s);
trace(s, 0, 6.15, W, 0.5, 7, TEAL_DK, 1.4);
trace(s, 0, 1.55, W, 0.34, 31, "12303E", 1.1);

s.addText("WardSense", {
  x: M, y: 2.15, w: 9.5, h: 1.15, margin: 0,
  fontFace: H, fontSize: 60, bold: true, color: INK,
});
s.addText("Federated Edge AI for early deterioration monitoring", {
  x: M, y: 3.35, w: 10.2, h: 0.5, margin: 0,
  fontFace: B, fontSize: 20, color: TEAL,
});
s.addText("Warning nurses hours before deterioration becomes obvious —\nwithout a single byte of patient data leaving the hospital.", {
  x: M, y: 3.95, w: 9.2, h: 0.9, margin: 0,
  fontFace: B, fontSize: 14, color: MUTED, lineSpacing: 22,
});
s.addText("electronica India   ·   Edge AI & Healthcare   ·   Continuous Healthcare Monitoring", {
  x: M, y: 5.35, w: 11, h: 0.35, margin: 0,
  fontFace: B, fontSize: 11.5, color: FAINT, charSpacing: 1.5,
});
s.addNotes("WardSense: a two-part edge system. A small model watches vitals and forecasts deterioration 4-6 hours ahead; a local language model explains each alert in plain English. Everything runs on hardware the hospital owns.");

// ================================================================ 2 PROBLEM
s = pres.addSlide(); base(s);
title(s, "Deterioration is visible in the data hours before anyone sees it",
         "THE PROBLEM");

const probs = [
  ["Nobody is watching between rounds",
   "Wards check observations every 4–6 hours. Patients deteriorate continuously. The warning signs sit unread in a chart."],
  ["The obvious fix is illegal",
   "Streaming vitals to a cloud AI service is exactly what patient-privacy regulation forbids. Data cannot leave the hospital."],
  ["Alerts without context get ignored",
   "A red number tells a nurse something changed. It does not tell them why it matters for this patient."],
];
probs.forEach((p, i) => {
  const y = 1.95 + i * 1.55;
  card(s, M, y, 8.5, 1.28);
  marker(s, M + 0.35, y + 0.42, i + 1, TEAL);
  s.addText(p[0], { x: M + 1.0, y: y + 0.16, w: 7.2, h: 0.4, margin: 0,
    fontFace: B, fontSize: 16, bold: true, color: INK });
  s.addText(p[1], { x: M + 1.0, y: y + 0.58, w: 7.25, h: 0.6, margin: 0,
    fontFace: B, fontSize: 12, color: MUTED, lineSpacing: 16 });
});

card(s, 9.65, 1.95, 2.9, 4.15, CARD2);
s.addText("Between\nobservations", { x: 9.9, y: 2.25, w: 2.4, h: 0.7, margin: 0,
  fontFace: B, fontSize: 12, color: MUTED, lineSpacing: 16 });
s.addText("4–6", { x: 9.9, y: 2.95, w: 2.4, h: 0.9, margin: 0,
  fontFace: H, fontSize: 54, bold: true, color: AMBER });
s.addText("hours unmonitored", { x: 9.9, y: 3.85, w: 2.4, h: 0.4, margin: 0,
  fontFace: B, fontSize: 11.5, color: MUTED });
s.addText("WardSense forecasts the same window — continuously, on-device.",
  { x: 9.9, y: 4.45, w: 2.45, h: 1.2, margin: 0,
    fontFace: B, fontSize: 12, color: INK, lineSpacing: 16 });

// ================================================================ 3 SOLUTION
s = pres.addSlide(); base(s);
title(s, "Two systems, one junction", "WHAT WE BUILT");

card(s, M, 2.0, 5.6, 2.5);
s.addText("The detector", { x: M + 0.4, y: 2.25, w: 4.8, h: 0.4, margin: 0,
  fontFace: B, fontSize: 17, bold: true, color: TEAL });
s.addText([
  { text: "Reads 12 hours of six vitals, forecasts deterioration 4–6 hours ahead.", options: { breakLine: true } },
  { text: "Under 100K parameters. Trained across hospitals without sharing data.", options: {} },
], { x: M + 0.4, y: 2.68, w: 4.8, h: 1.5, margin: 0,
     fontFace: B, fontSize: 12.5, color: MUTED, lineSpacing: 18 });

card(s, 7.05, 2.0, 5.5, 2.5);
s.addText("The assistant", { x: 7.45, y: 2.25, w: 4.7, h: 0.4, margin: 0,
  fontFace: B, fontSize: 17, bold: true, color: TEAL });
s.addText([
  { text: "A local language model answering questions from that patient's own record.", options: { breakLine: true } },
  { text: "Runs offline. Strict per-patient isolation, adversarially tested.", options: {} },
], { x: 7.45, y: 2.68, w: 4.7, h: 1.5, margin: 0,
     fontFace: B, fontSize: 12.5, color: MUTED, lineSpacing: 18 });

card(s, M, 4.85, 11.8, 1.75, CARD2);
s.addText("They meet at exactly one point — and that is the contribution", {
  x: M + 0.45, y: 5.05, w: 11, h: 0.4, margin: 0,
  fontFace: B, fontSize: 16, bold: true, color: AMBER });
s.addText("When the detector raises an alert, the abnormal vitals themselves become the search query — pulling the history that explains this specific deterioration, then writing it in plain English. The output of a privacy-protected federated model becomes the semantic input to a grounded language model.",
  { x: M + 0.45, y: 5.5, w: 11, h: 0.95, margin: 0,
    fontFace: B, fontSize: 12.5, color: INK, lineSpacing: 17 });
s.addNotes("Two separate AI systems, not one. Different models, trained differently. The novelty is the junction: the alert's content becomes the retrieval query.");

// ================================================================ 4 FLOW
s = pres.addSlide(); base(s);
title(s, "How one alert becomes a clinical summary", "THE FLOW");

const steps = [
  ["Vitals arrive", "Heart rate, BP, oxygen,\nrespiration, temperature,\nglucose — every hour"],
  ["Scored on-device", "12-hour window through\nthe detector. No network\nround trip"],
  ["Risk crosses threshold", "NEWS2 ≥ 5 — the NHS\nescalation point. Alert\nraised"],
  ["Record retrieved", "The abnormal vitals\nbecome the search query\nfor this patient's history"],
  ["Nurse reads why", "Plain-English summary:\nwhat is abnormal, and\nwhat explains it"],
];
const cw = 2.24, gap = 0.19;
steps.forEach((st, i) => {
  const x = M + i * (cw + gap);
  const isLast = i === steps.length - 1;
  card(s, x, 2.05, cw, 3.05, isLast ? CARD2 : CARD);
  marker(s, x + 0.28, 2.32, i + 1, isLast ? AMBER : TEAL);
  s.addText(st[0], { x: x + 0.28, y: 2.92, w: cw - 0.40, h: 0.6, margin: 0,
    fontFace: B, fontSize: 13.5, bold: true, color: INK });
  s.addText(st[1], { x: x + 0.28, y: 3.55, w: cw - 0.5, h: 1.35, margin: 0,
    fontFace: B, fontSize: 11, color: MUTED, lineSpacing: 15 });
  if (i < steps.length - 1) {
    s.addShape(pres.ShapeType.triangle, {
      x: x + cw + 0.02, y: 3.47, w: 0.15, h: 0.17, rotate: 90,
      fill: { color: TEAL_DK }, line: { color: TEAL_DK, width: 0 },
    });
  }
});

card(s, M, 5.42, 11.8, 1.2, CARD);
s.addText("Every step happens on hardware the hospital owns. If the internet fails, monitoring continues.",
  { x: M + 0.45, y: 5.72, w: 11, h: 0.6, margin: 0,
    fontFace: B, fontSize: 13.5, color: TEAL, lineSpacing: 18 });

// ================================================================ 5 EDGE AI
s = pres.addSlide(); base(s);
title(s, "The whole stack fits in 6 GB", "EDGE AI — MEASURED, NOT ESTIMATED");

s.addText("We built to a hard constraint: a consumer laptop GPU. The resulting envelope is the evidence — a stack that fits commodity hardware runs on a ward mini-PC; one that needs a datacenter GPU is not deployable at all.",
  { x: M, y: 2.00, w: 11.8, h: 0.7, margin: 0,
    fontFace: B, fontSize: 13.5, color: MUTED, lineSpacing: 18 });

const stats = [
  ["6 GB", "total VRAM budget\nNVIDIA RTX 4050", TEAL],
  ["0.03 GB", "detector footprint\nleaves the GPU free", TEAL],
  ["66 tok/s", "local language model\nfully GPU-resident", TEAL],
  ["0", "external API calls\nat any point", AMBER],
];
stats.forEach((st, i) => {
  const x = M + i * 3.02;
  card(s, x, 2.88, 2.82, 1.85);
  s.addText(st[0], { x: x + 0.32, y: 3.11, w: 2.3, h: 0.72, margin: 0,
    fontFace: H, fontSize: 34, bold: true, color: st[2] });
  s.addText(st[1], { x: x + 0.32, y: 3.85, w: 2.3, h: 0.7, margin: 0,
    fontFace: B, fontSize: 11, color: MUTED, lineSpacing: 15 });
});

card(s, M, 5.02, 11.8, 1.65, CARD2);
s.addText("What building under a real budget taught us", {
  x: M + 0.45, y: 5.22, w: 11, h: 0.35, margin: 0,
  fontFace: B, fontSize: 14, bold: true, color: INK });
s.addText("Differential privacy costs roughly 5.7× more compute on recurrent models than convolutional ones — a trade-off that only surfaces when you measure on the target hardware, and one that directly decides which architecture is viable at the edge.",
  { x: M + 0.45, y: 5.62, w: 11, h: 0.85, margin: 0,
    fontFace: B, fontSize: 12.5, color: MUTED, lineSpacing: 17 });
s.addNotes("Every number on this slide was measured on the actual device, not estimated from spec sheets.");

// ================================================================ 6 PRIVACY
s = pres.addSlide(); base(s);
title(s, "The model travels. The data never does.", "PRIVACY BY CONSTRUCTION");

const priv = [
  ["Federated learning", "Each hospital trains on its own patients and sends back only what the model learned — never a patient record.", TEAL],
  ["Differential privacy", "Calibrated noise is added on-device before any update leaves, so individuals cannot be reconstructed from the gradients.", TEAL],
  ["Provable budget", "Privacy loss is accounted across every training round, not per round — a mistake that silently under-reports by 2.4×.", AMBER],
];
priv.forEach((p, i) => {
  const y = 1.95 + i * 1.42;
  card(s, M, y, 7.5, 1.2);
  s.addShape(pres.ShapeType.ellipse, { x: M + 0.35, y: y + 0.38, w: 0.44, h: 0.44,
    fill: { color: p[2] }, line: { color: p[2], width: 0 } });
  s.addText(p[0], { x: M + 1.05, y: y + 0.16, w: 6.2, h: 0.36, margin: 0,
    fontFace: B, fontSize: 15.5, bold: true, color: INK });
  s.addText(p[2] === AMBER ? p[1] : p[1], { x: M + 1.05, y: y + 0.55, w: 6.25, h: 0.6, margin: 0,
    fontFace: B, fontSize: 11.5, color: MUTED, lineSpacing: 15 });
});

card(s, 8.65, 1.95, 3.9, 4.25, CARD2);
s.addText("Measured privacy budget", { x: 8.95, y: 2.2, w: 3.3, h: 0.35, margin: 0,
  fontFace: B, fontSize: 12, color: MUTED });
s.addText("ε ≈ 2.0", { x: 8.95, y: 2.6, w: 3.3, h: 0.95, margin: 0,
  fontFace: H, fontSize: 48, bold: true, color: TEAL });
// superscript via run formatting — the literal ⁻⁵ glyphs are not in core Calibri
s.addText([
  { text: "at δ = 10", options: {} },
  { text: "-5", options: { superscript: true } },
  { text: ", tracked across the full training run using Rényi accounting", options: {} },
], { x: 8.95, y: 3.55, w: 3.35, h: 0.85, margin: 0,
     fontFace: B, fontSize: 11.5, color: MUTED, lineSpacing: 15 });
s.addText("Federated learning alone still leaks — gradients can be partly inverted. Differential privacy is the mathematical guarantee that closes it.",
  { x: 8.95, y: 4.55, w: 3.35, h: 1.4, margin: 0,
    fontFace: B, fontSize: 11.5, color: INK, lineSpacing: 16 });

// ================================================================ 7 THE ALERT
s = pres.addSlide(); base(s);
title(s, "An alert a nurse can act on", "THE CONTRIBUTION");

card(s, M, 1.95, 5.35, 4.4, CARD2);
s.addText("What most systems show", { x: M + 0.4, y: 2.18, w: 4.5, h: 0.35, margin: 0,
  fontFace: B, fontSize: 12, color: MUTED });
s.addText("Bed 7 — RISK 0.87", { x: M + 0.4, y: 2.62, w: 4.6, h: 0.55, margin: 0,
  fontFace: H, fontSize: 25, bold: true, color: RED });
s.addText("A number changed. The nurse must work out what it means, for which patient, and why — during a shift where twelve other things are also urgent.",
  { x: M + 0.4, y: 3.3, w: 4.55, h: 1.3, margin: 0,
    fontFace: B, fontSize: 12.5, color: MUTED, lineSpacing: 17 });
s.addText("Alert fatigue. Ignored.", { x: M + 0.4, y: 5.55, w: 4.5, h: 0.4, margin: 0,
  fontFace: B, fontSize: 13, bold: true, color: RED });

card(s, 6.85, 1.95, 5.7, 4.4);
s.addText("What WardSense shows", { x: 7.25, y: 2.18, w: 5, h: 0.35, margin: 0,
  fontFace: B, fontSize: 12, color: TEAL });
s.addText("Bed 7 — deteriorating, NEWS2 9", { x: 7.25, y: 2.6, w: 5.05, h: 0.4, margin: 0,
  fontFace: H, fontSize: 19, bold: true, color: AMBER });
s.addText([
  { text: "Respiratory rate 29 (normal 12–20), oxygen 90% (normal ≥ 96), blood pressure fallen to 93.", options: { breakLine: true } },
  { text: "", options: { breakLine: true } },
  { text: "Known COPD, managed with an inhaler; ex-smoker. Post-operative day 2, reported breathlessness on exertion yesterday.", options: { breakLine: true } },
  { text: "", options: { breakLine: true } },
  { text: "Escalation threshold crossed four hours ago and sustained.", options: {} },
], { x: 7.25, y: 3.1, w: 5.05, h: 2.4, margin: 0,
     fontFace: B, fontSize: 12, color: INK, lineSpacing: 17 });
s.addText("Context. Actionable.", { x: 7.25, y: 5.55, w: 5, h: 0.4, margin: 0,
  fontFace: B, fontSize: 13, bold: true, color: TEAL });
s.addNotes("The right-hand summary is generated automatically by retrieving this patient's own record, using the abnormal vitals themselves as the query.");

// ================================================================ 8 DATA
s = pres.addSlide(); base(s);
title(s, "Real patients. Real hospitals. Already tested.",
         "NOT A SIMULATION");

const dstats = [
  ["40,336", "de-identified ICU patients", TEAL],
  ["2", "independent hospital systems", TEAL],
  ["~800k", "hourly training windows", TEAL],
  ["4–6 h", "forecast horizon, ahead of onset", AMBER],
];
dstats.forEach((d, i) => {
  const x = M + i * 3.02;
  card(s, x, 1.95, 2.82, 1.75);
  s.addText(d[0], { x: x + 0.32, y: 2.15, w: 2.3, h: 0.72, margin: 0,
    fontFace: H, fontSize: 32, bold: true, color: d[2] });
  s.addText(d[1], { x: x + 0.32, y: 2.88, w: 2.3, h: 0.65, margin: 0,
    fontFace: B, fontSize: 11, color: MUTED, lineSpacing: 15 });
});

card(s, M, 3.98, 5.75, 2.45);
s.addText("Labels a clinician recognises", { x: M + 0.4, y: 4.2, w: 4.9, h: 0.35, margin: 0,
  fontFace: B, fontSize: 15, bold: true, color: INK });
s.addText("We score every hour with NEWS2 — the early-warning chart used across NHS hospitals — rather than inventing thresholds. Our labels were cross-checked against clinically adjudicated outcomes in the same records.",
  { x: M + 0.4, y: 4.62, w: 4.95, h: 1.6, margin: 0,
    fontFace: B, fontSize: 12, color: MUTED, lineSpacing: 17 });

card(s, 6.85, 3.98, 5.7, 2.45);
s.addText("Sites differ in what they measure", { x: 7.25, y: 4.2, w: 5, h: 0.35, margin: 0,
  fontFace: B, fontSize: 15, bold: true, color: INK });
s.addText("One hospital records blood glucose almost twice as often as the other — exactly the federation problem real deployments hit, and one synthetic data cannot reproduce.",
  { x: 7.25, y: 4.62, w: 5.05, h: 1.0, margin: 0,
    fontFace: B, fontSize: 12, color: MUTED, lineSpacing: 17 });
s.addText("Recorded stays are replayed hour by hour as a live feed. Connecting to hospital monitors is an integration step, not a research question.",
  { x: 7.25, y: 5.62, w: 5.05, h: 0.75, margin: 0,
    fontFace: B, fontSize: 11, color: TEAL, lineSpacing: 15 });

// ================================================================ 9 STACK
s = pres.addSlide(); base(s);
title(s, "Technology", "EVERY LAYER RUNS LOCALLY");

const stack = [
  ["Detector", "PyTorch sequence models\nDPLSTM / DPGRU, CNN baseline", "< 100K parameters"],
  ["Federation", "Flower, containers over gRPC", "genuine multi-site"],
  ["Privacy", "Opacus DP-SGD\nRényi accounting", "ε ≈ 2.0 measured"],
  ["Language model", "Llama 3.2 3B, 4-bit\nserved by Ollama", "66 tok/s, offline"],
  ["Retrieval", "ChromaDB, local embeddings\nper-patient isolation", "no external calls"],
  ["Platform", "FastAPI · Streamlit\nSQLite + AES-256", "encrypted at rest"],
];
stack.forEach((it, i) => {
  const col = i % 3, row = Math.floor(i / 3);
  const x = M + col * 4.03, y = 1.95 + row * 2.25;
  card(s, x, y, 3.75, 1.95);
  s.addText(it[0], { x: x + 0.32, y: y + 0.22, w: 3.1, h: 0.35, margin: 0,
    fontFace: B, fontSize: 15, bold: true, color: TEAL });
  s.addText(it[1], { x: x + 0.32, y: y + 0.65, w: 3.15, h: 0.75, margin: 0,
    fontFace: B, fontSize: 11.5, color: INK, lineSpacing: 15 });
  s.addText(it[2], { x: x + 0.32, y: y + 1.42, w: 3.15, h: 0.35, margin: 0,
    fontFace: B, fontSize: 11, color: MUTED, italic: true });
});

// ================================================================ 10 STATUS
s = pres.addSlide(); base(s);
title(s, "Where we are", "STATUS & ROADMAP");

const done = [
  "Dataset secured, tested and processed",
  "Privacy pipeline verified end to end at ε ≈ 2.0",
  "Federated averaging converging across sites",
  "Local language model benchmarked on-device",
  "Compute budget measured on target hardware",
];
const next = [
  "Full privacy-versus-accuracy experiment sweep",
  "Alert-to-summary bridge and clinician dashboard",
  "Three-layer evaluation of generated summaries",
  "Voice input for bedside accessibility",
  "Split across dedicated edge hardware",
];

card(s, M, 1.95, 5.75, 4.4);
s.addText("Built and measured", { x: M + 0.4, y: 2.18, w: 4.9, h: 0.4, margin: 0,
  fontFace: B, fontSize: 16, bold: true, color: TEAL });
done.forEach((t, i) => {
  // drawn, not a glyph: Calibri has no check mark and the fallback is unreliable
  s.addShape(pres.ShapeType.ellipse, {
    x: M + 0.44, y: 2.79 + i * 0.66, w: 0.17, h: 0.17,
    fill: { color: TEAL }, line: { color: TEAL, width: 0 },
  });
  s.addText(t, { x: M + 0.82, y: 2.7 + i * 0.66, w: 4.5, h: 0.6, margin: 0,
    fontFace: B, fontSize: 12, color: INK, lineSpacing: 15 });
});

card(s, 6.85, 1.95, 5.7, 4.4, CARD2);
s.addText("Next", { x: 7.25, y: 2.18, w: 5, h: 0.4, margin: 0,
  fontFace: B, fontSize: 16, bold: true, color: AMBER });
next.forEach((t, i) => {
  s.addShape(pres.ShapeType.ellipse, {
    x: 7.29, y: 2.79 + i * 0.66, w: 0.17, h: 0.17,
    fill: { color: AMBER }, line: { color: AMBER, width: 0 },
  });
  s.addText(t, { x: 7.67, y: 2.7 + i * 0.66, w: 4.5, h: 0.6, margin: 0,
    fontFace: B, fontSize: 12, color: INK, lineSpacing: 15 });
});
s.addNotes("Honest framing: the pipeline and hardware envelope are validated. The full experimental campaign and the clinician evaluation are the next phase.");

// ================================================================ 11 CLOSE
s = pres.addSlide(); base(s);
trace(s, 0, 6.4, W, 0.55, 19, TEAL_DK, 1.5);

s.addText("Privacy is the constraint.\nThe edge is the answer.", {
  x: M, y: 1.85, w: 10.5, h: 1.7, margin: 0,
  fontFace: H, fontSize: 40, bold: true, color: INK, lineSpacing: 50,
});
s.addText("A deterioration monitor that learns from every hospital, and reveals nothing about any patient — running inside 6 GB on hardware a ward can afford.",
  { x: M, y: 3.75, w: 9.6, h: 0.95, margin: 0,
    fontFace: B, fontSize: 15, color: TEAL, lineSpacing: 22 });

const closing = [["6 GB", "full stack"], ["ε ≈ 2.0", "privacy budget"],
                 ["40,336", "real patients"], ["4–6 h", "early warning"]];
closing.forEach((c, i) => {
  const x = M + i * 2.95;
  s.addText(c[0], { x, y: 4.95, w: 2.6, h: 0.6, margin: 0,
    fontFace: H, fontSize: 26, bold: true, color: INK });
  s.addText(c[1], { x, y: 5.5, w: 2.6, h: 0.35, margin: 0,
    fontFace: B, fontSize: 11, color: MUTED });
});

pres.writeFile({ fileName: "WardSense_electronica_India.pptx" })
  .then(f => console.log("wrote", f));
