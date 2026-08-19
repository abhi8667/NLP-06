"""
P0 · S7 — Can we derive NEWS2 deterioration labels from PhysioNet vitals?

The project is a DETERIORATION MONITORING system, not a sepsis classifier. The
PhysioNet dataset ships a sepsis label, but its real value here is the vitals:
S6 confirmed every NEWS2 component is adequately measured, which Synthea could
not offer. So the label can be derived rather than borrowed.

This script answers three questions:

  1. PREVALENCE   what fraction of hours score as deteriorating under NEWS2?
                  (This drives the loss weighting and the choice of metric.)
  2. AGREEMENT    do NEWS2-derived labels line up with the dataset's own
                  clinically-adjudicated sepsis labels? Agreement is evidence
                  that the derived labels track real clinical deterioration -
                  a validity check Synthea made impossible.
  3. USABILITY    how many labelled windows survive?

NEWS2 per Royal College of Physicians (2017). Two of the seven components -
level of consciousness and supplemental oxygen - are not recorded numerically
in this dataset, so the attainable maximum is 14 rather than 20. That is a
limitation to state, not a blocker: the standard escalation threshold of 5
still sits well inside the attainable range.

    python scripts/s7_news2_labels.py --root ../physioNet --sample 800
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

VITALS = ["HR", "SBP", "O2Sat", "Resp", "Temp", "Glucose"]
WINDOW = 12
NEEDED = WINDOW + 1


def score_resp(v):
    if pd.isna(v): return np.nan
    if v <= 8: return 3
    if v <= 11: return 1
    if v <= 20: return 0
    if v <= 24: return 2
    return 3


def score_spo2(v):
    if pd.isna(v): return np.nan
    if v <= 91: return 3
    if v <= 93: return 2
    if v <= 95: return 1
    return 0


def score_sbp(v):
    if pd.isna(v): return np.nan
    if v <= 90: return 3
    if v <= 100: return 2
    if v <= 110: return 1
    if v <= 219: return 0
    return 3


def score_hr(v):
    if pd.isna(v): return np.nan
    if v <= 40: return 3
    if v <= 50: return 1
    if v <= 90: return 0
    if v <= 110: return 1
    if v <= 130: return 2
    return 3


def score_temp(v):
    if pd.isna(v): return np.nan
    if v <= 35.0: return 3
    if v <= 36.0: return 1
    if v <= 38.0: return 0
    if v <= 39.0: return 1
    return 2


SCORERS = {"Resp": score_resp, "O2Sat": score_spo2, "SBP": score_sbp,
           "HR": score_hr, "Temp": score_temp}


def news2(df: pd.DataFrame) -> pd.Series:
    """Aggregate NEWS2 score per row, computed on forward-filled vitals."""
    filled = df[list(SCORERS)].ffill()
    total = pd.Series(0, index=df.index, dtype=float)
    for col, fn in SCORERS.items():
        total = total + filled[col].map(fn).fillna(0)
    return total


def load(folder: Path, sample: int, seed: int = 42):
    files = sorted(folder.glob("*.psv"))
    total = len(files)
    if sample and sample < total:
        random.Random(seed).shuffle(files)
        files = files[:sample]
    out = []
    for f in files:
        try:
            d = pd.read_csv(f, sep="|")
            d["_pid"] = f.stem
            out.append(d)
        except Exception:
            continue
    return out, total


def longest_run(mask: np.ndarray) -> int:
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("../physioNet"))
    ap.add_argument("--sample", type=int, default=800)
    ap.add_argument("--threshold", type=int, default=5)
    args = ap.parse_args()

    root = args.root.resolve()
    sets = {}
    for name in ("training_setA", "training_setB"):
        for cand in (root / name / name, root / name):
            if cand.is_dir() and any(cand.glob("*.psv")):
                sets[name] = cand
                break
    if not sets:
        raise SystemExit(f"no .psv files under {root}")

    print("=" * 70)
    print("S7 - NEWS2 DETERIORATION LABELS FROM PHYSIONET VITALS")
    print("=" * 70)
    print("  Framing: deterioration monitoring, NOT sepsis classification.")
    print("  The dataset supplies the vitals; the label is derived.\n")

    findings = {"threshold": args.threshold, "sets": {}}

    for name, folder in sets.items():
        frames, total = load(folder, args.sample)
        print("-" * 70)
        print(f"  {name}  ({len(frames):,} of {total:,} patients)")
        print("-" * 70)

        scores_all, labels_all, sepsis_all = [], [], []
        win_ok = 0
        for d in frames:
            s = news2(d)
            lab = (s >= args.threshold).astype(int)
            scores_all.append(s)
            labels_all.append(lab)
            if "SepsisLabel" in d.columns:
                sepsis_all.append(d["SepsisLabel"].astype(int))
            complete = d[VITALS].ffill().notna().all(axis=1)
            if longest_run(complete.values) >= NEEDED:
                win_ok += 1

        s_cat = pd.concat(scores_all, ignore_index=True)
        l_cat = pd.concat(labels_all, ignore_index=True)

        print(f"\n  NEWS2 score distribution (max attainable 14)")
        q = s_cat.describe(percentiles=[.5, .75, .9, .95, .99])
        for k in ("mean", "50%", "75%", "90%", "95%", "99%", "max"):
            print(f"    {k:<6} {q[k]:6.2f}")

        pos_hours = float(l_cat.mean())
        per_pat = float(np.mean([1.0 if l.max() > 0 else 0.0 for l in labels_all]))
        print(f"\n  --- prevalence at NEWS2 >= {args.threshold} ---")
        print(f"    positive hours          {pos_hours:6.2%}")
        print(f"    patients ever positive  {per_pat:6.1%}")

        print(f"\n  --- prevalence at other thresholds ---")
        thr_tab = {}
        for t in (3, 4, 5, 6, 7):
            r = float((s_cat >= t).mean())
            thr_tab[t] = round(r, 5)
            print(f"    NEWS2 >= {t}   {r:6.2%} of hours")

        entry = {
            "patients": len(frames),
            "news2_mean": round(float(s_cat.mean()), 3),
            "positive_hours": round(pos_hours, 5),
            "positive_patients": round(per_pat, 4),
            "prevalence_by_threshold": thr_tab,
            "window_formable_share": round(win_ok / max(len(frames), 1), 4),
        }

        # --- agreement with the dataset's own clinical label ---
        if sepsis_all:
            sep = pd.concat(sepsis_all, ignore_index=True)
            n = min(len(sep), len(l_cat))
            a, b = l_cat[:n], sep[:n]
            tp = int(((a == 1) & (b == 1)).sum())
            fp = int(((a == 1) & (b == 0)).sum())
            fn = int(((a == 0) & (b == 1)).sum())
            recall = tp / max(tp + fn, 1)
            prec = tp / max(tp + fp, 1)
            # patient-level: does NEWS2 ever fire for patients who became septic?
            pat_rec = []
            for lab, sp in zip(labels_all, sepsis_all):
                if sp.max() > 0:
                    pat_rec.append(1.0 if lab.max() > 0 else 0.0)
            print(f"\n  --- agreement with clinically-adjudicated sepsis label ---")
            print(f"    hours flagged by NEWS2 that were septic hours   {prec:6.1%}")
            print(f"    septic hours that NEWS2 also flagged            {recall:6.1%}")
            if pat_rec:
                print(f"    septic PATIENTS NEWS2 flagged at some point     {np.mean(pat_rec):6.1%}")
            print("    (high patient-level recall is the meaningful figure - NEWS2 is an")
            print("     escalation trigger, not a sepsis test, so it fires more broadly)")
            entry["agreement_precision"] = round(prec, 4)
            entry["agreement_recall"] = round(recall, 4)
            entry["patient_level_recall"] = round(float(np.mean(pat_rec)), 4) if pat_rec else None

        print(f"\n  windows formable: {win_ok:,}/{len(frames):,} "
              f"({win_ok / max(len(frames),1):.1%})")
        findings["sets"][name] = entry
        print()

    # -------- verdict --------
    print("=" * 70)
    prev = np.mean([s["positive_hours"] for s in findings["sets"].values()])
    print(f"  NEWS2 >= {args.threshold} gives {prev:.2%} positive hours, versus 1.5% for the")
    print("  dataset's sepsis label.")
    if prev >= 0.05:
        print("\n  This is a far healthier class balance than the sepsis label.")
        print("  AUPRC remains the right headline metric, but the imbalance is")
        print("  no longer severe enough to dominate every design choice.")
    else:
        print("\n  Still heavily imbalanced - keep AUPRC as the headline metric")
        print("  and weight the loss accordingly.")
    print("\n  Recommended: NEWS2-derived labels as PRIMARY (keeps the monitoring")
    print("  framing), with the sepsis label retained as an independent")
    print("  cross-check. That cross-check is a labelling-validity argument")
    print("  Synthea could never have supported.")
    print("=" * 70)
    findings["mean_positive_hours"] = round(float(prev), 5)

    out = RESULTS / "s7_news2_labels.json"
    out.write_text(json.dumps(findings, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
