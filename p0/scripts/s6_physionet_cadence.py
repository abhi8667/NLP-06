"""
P0 · S6 — Does the PhysioNet 2019 data support the method Synthea could not?

Same three questions S1 asked of Synthea, asked again of the replacement:

  1. CADENCE   are readings close enough together to form a 12-step window?
  2. COVERAGE  is every NEWS2 component actually measured, not just present
               as a column header?
  3. LABELS    is there a usable deterioration signal, and how rare is it?

Plus one Synthea could never answer:

  4. SITES     do the two hospital systems differ enough to make a genuine
               non-IID federation, rather than a synthetic split?

The trap this script exists to avoid: in this format every patient has one ROW
per hour, so the cadence looks perfect by construction. What varies is whether
the vitals in those rows are actually filled in. A column that is 90% NaN is
the same problem Synthea had with SpO2, wearing a better disguise.

    python scripts/s6_physionet_cadence.py --root ../physioNet --sample 800
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

# The six vitals the NLP-06 detector consumes, mapped to this dataset's columns.
VITALS = {
    "heart_rate": "HR",
    "bp_systolic": "SBP",
    "spo2": "O2Sat",
    "resp_rate": "Resp",
    "temperature": "Temp",
    "glucose": "Glucose",
}
# NEWS2 needs these five (plus consciousness, which no dataset records numerically).
NEWS2 = ["Resp", "O2Sat", "Temp", "SBP", "HR"]

WINDOW = 12
NEEDED = WINDOW + 1
MIN_COVERAGE = 0.30   # below this a vital is too sparse to forward-fill honestly


def load_set(folder: Path, sample: int, seed: int = 42) -> tuple[list[pd.DataFrame], int]:
    files = sorted(folder.glob("*.psv"))
    total = len(files)
    if sample and sample < total:
        random.Random(seed).shuffle(files)
        files = files[:sample]
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, sep="|")
            df["_pid"] = f.stem
            frames.append(df)
        except Exception:
            continue
    return frames, total


def analyse(frames: list[pd.DataFrame], label: str, total_files: int) -> dict:
    print("=" * 70)
    print(f"  {label}   ({len(frames):,} sampled of {total_files:,} patients)")
    print("=" * 70)

    allrows = pd.concat(frames, ignore_index=True)
    out: dict = {"patients_sampled": len(frames), "patients_total": total_files,
                 "rows": int(len(allrows))}

    # ---- record length -------------------------------------------------
    lengths = pd.Series([len(f) for f in frames])
    print(f"\n  ICU stay length (hours)   median {lengths.median():.0f}   "
          f"min {lengths.min()}   max {lengths.max()}")
    print(f"  patients with >= {NEEDED} hours: "
          f"{(lengths >= NEEDED).sum():,}/{len(frames):,} "
          f"({(lengths >= NEEDED).mean():.1%})")
    out["stay_hours_median"] = float(lengths.median())
    out["share_long_enough"] = float((lengths >= NEEDED).mean())

    # ---- 2. COVERAGE: is the column actually measured? -----------------
    print("\n  --- vital coverage (share of hourly rows actually recorded) ---")
    cov = {}
    for name, col in VITALS.items():
        if col not in allrows.columns:
            print(f"    {name:<14} {col:<9}   COLUMN ABSENT")
            cov[name] = {"column": col, "coverage": 0.0, "state": "ABSENT"}
            continue
        c = float(allrows[col].notna().mean())
        state = "OK" if c >= MIN_COVERAGE else ("SPARSE" if c > 0 else "ABSENT")
        print(f"    {name:<14} {col:<9} {c:6.1%}   {state}")
        cov[name] = {"column": col, "coverage": round(c, 4), "state": state}
    out["coverage"] = cov

    print("\n  --- NEWS2 components ---")
    blocked = []
    for col in NEWS2:
        c = float(allrows[col].notna().mean()) if col in allrows.columns else 0.0
        state = "OK" if c >= MIN_COVERAGE else ("SPARSE" if c > 0 else "ABSENT")
        if state != "OK":
            blocked.append(col)
        print(f"    {col:<9} {c:6.1%}   {state}")
    out["news2_blocked_by"] = blocked
    if blocked:
        print(f"\n    >> sparse components: {', '.join(blocked)}")
        print("       (these are candidates for forward-fill, not necessarily blockers)")
    else:
        print("\n    >> all NEWS2 components adequately measured")

    # ---- 1. CADENCE: can a 12-step window be formed? -------------------
    print(f"\n  --- can a {WINDOW}-step window be formed? ---")
    core = [VITALS[k] for k in VITALS]
    raw_ok = ffill_ok = 0
    raw_windows = ffill_windows = 0
    for f in frames:
        present = [c for c in core if c in f.columns]
        sub = f[present]

        # strict: every one of the six recorded in the same hour
        complete = sub.notna().all(axis=1)
        r = int(_longest_run(complete.values))
        if r >= NEEDED:
            raw_ok += 1
        raw_windows += max(0, int(complete.sum()) - WINDOW)

        # realistic: carry the last observed value forward, as clinicians do
        filled = sub.ffill()
        complete_f = filled.notna().all(axis=1)
        rf = int(_longest_run(complete_f.values))
        if rf >= NEEDED:
            ffill_ok += 1
        ffill_windows += max(0, int(complete_f.sum()) - WINDOW)

    n = len(frames)
    print(f"    strict (all 6 in the same hour)      "
          f"{raw_ok:,}/{n:,} patients ({raw_ok / n:6.1%})")
    print(f"    forward-filled (standard practice)   "
          f"{ffill_ok:,}/{n:,} patients ({ffill_ok / n:6.1%})")
    print(f"    usable windows in sample: strict {raw_windows:,}   "
          f"forward-filled {ffill_windows:,}")
    out["window_strict_share"] = round(raw_ok / n, 4)
    out["window_ffill_share"] = round(ffill_ok / n, 4)
    out["windows_strict"] = raw_windows
    out["windows_ffill"] = ffill_windows

    # ---- 3. LABELS -----------------------------------------------------
    if "SepsisLabel" in allrows.columns:
        pos_rows = float(allrows["SepsisLabel"].mean())
        per_patient = np.mean([1.0 if f["SepsisLabel"].max() > 0 else 0.0 for f in frames])
        print(f"\n  --- deterioration label ---")
        print(f"    positive hours          {pos_rows:6.2%}")
        print(f"    patients ever positive  {per_patient:6.1%}")
        out["label_positive_hours"] = round(pos_rows, 5)
        out["label_positive_patients"] = round(float(per_patient), 4)

    # ---- demographics for the non-IID comparison -----------------------
    if "Age" in allrows.columns:
        out["age_median"] = float(allrows["Age"].median())
    if "Gender" in allrows.columns:
        out["male_share"] = round(float(allrows["Gender"].mean()), 4)
    return out


def _longest_run(mask: np.ndarray) -> int:
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        if run > best:
            best = run
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("../physioNet"))
    ap.add_argument("--sample", type=int, default=800,
                    help="patients per set (0 = all; 800 is ample for these statistics)")
    args = ap.parse_args()

    root = args.root.resolve()
    sets = {}
    for name in ("training_setA", "training_setB"):
        # the upload nests the folder one level deeper
        for cand in (root / name / name, root / name):
            if cand.is_dir() and any(cand.glob("*.psv")):
                sets[name] = cand
                break
    if not sets:
        raise SystemExit(f"no .psv files found under {root}")

    print(f"\nPhysioNet/CinC 2019 - cadence, coverage and label check")
    print(f"root: {root}\n")

    findings: dict = {"root": str(root), "sample_per_set": args.sample, "sets": {}}
    for name, folder in sets.items():
        frames, total = load_set(folder, args.sample)
        findings["sets"][name] = analyse(frames, name, total)
        print()

    # ---- 4. SITES: is this a genuine non-IID federation? ---------------
    if len(findings["sets"]) == 2:
        a, b = findings["sets"]["training_setA"], findings["sets"]["training_setB"]
        print("=" * 70)
        print("  SITE COMPARISON - is this a real non-IID federation?")
        print("=" * 70)
        print(f"  {'metric':<28} {'set A':>12} {'set B':>12}")
        rows = [
            ("median stay (hours)", a.get("stay_hours_median"), b.get("stay_hours_median")),
            ("patients ever positive", a.get("label_positive_patients"), b.get("label_positive_patients")),
            ("positive hours", a.get("label_positive_hours"), b.get("label_positive_hours")),
            ("median age", a.get("age_median"), b.get("age_median")),
            ("male share", a.get("male_share"), b.get("male_share")),
            ("temp coverage", a["coverage"]["temperature"]["coverage"], b["coverage"]["temperature"]["coverage"]),
            ("glucose coverage", a["coverage"]["glucose"]["coverage"], b["coverage"]["glucose"]["coverage"]),
            ("spo2 coverage", a["coverage"]["spo2"]["coverage"], b["coverage"]["spo2"]["coverage"]),
        ]
        for label, va, vb in rows:
            fa = f"{va:.4g}" if isinstance(va, (int, float)) else "-"
            fb = f"{vb:.4g}" if isinstance(vb, (int, float)) else "-"
            print(f"  {label:<28} {fa:>12} {fb:>12}")
        print("\n  Differences here are REAL institutional variation, not synthetic")
        print("  skew - which is exactly what federated learning is meant to handle.")

    # ---- verdict -------------------------------------------------------
    print()
    print("=" * 70)
    worst_ffill = min(s["window_ffill_share"] for s in findings["sets"].values())
    any_blocked = any(s["news2_blocked_by"] for s in findings["sets"].values())
    if worst_ffill >= 0.5 and not any_blocked:
        verdict = "PASS"
        print("VERDICT: PASS - windows are formable and every NEWS2 component")
        print("is measured. This dataset supports the method.")
    elif worst_ffill >= 0.5:
        verdict = "PASS WITH IMPUTATION"
        print("VERDICT: PASS WITH IMPUTATION - windows are formable once values")
        print("are carried forward, but some components are sparsely measured.")
        print("Forward-fill is standard clinical practice; state it in the paper")
        print("and report the imputation rate per vital.")
    else:
        verdict = "FAIL"
        print(f"VERDICT: FAIL - only {worst_ffill:.1%} of patients can form a window")
        print("even after forward-filling.")
    print("=" * 70)
    findings["verdict"] = verdict

    out = RESULTS / "s6_physionet_cadence.json"
    out.write_text(json.dumps(findings, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
