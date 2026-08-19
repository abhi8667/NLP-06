"""
P0 · S1 — Does Synthea emit vitals at ward cadence?

The decisive question: can you form a 12-timestep window of vitals whose readings
sit hours apart rather than months apart? If not, the windows are life-history,
not telemetry, and the "real-time monitoring" gap claimed against b5 cannot be
claimed on this dataset.

Prerequisite — generate a cohort first (needs a JDK):

    git clone https://github.com/synthetichealth/synthea.git
    cd synthea
    .\\run_synthea.bat -p 50 --exporter.csv.export true --exporter.fhir.export false

Then:

    python scripts/s1_cadence.py --obs path/to/synthea/output/csv/observations.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

# The seven codes the NLP-06 guides assume. Step 2 verifies they are real.
VITAL_CODES = {
    "8867-4": "heart_rate",
    "8480-6": "bp_systolic",
    "8462-4": "bp_diastolic",
    "59408-5": "spo2",
    "9279-1": "resp_rate",
    "8310-5": "temperature",
    "2339-0": "glucose",
}

# Codes Synthea may use instead. 59408-5 in particular returns nothing;
# Synthea emits arterial oxygen saturation as 2708-6 (and rarely).
ALTERNATE_CODES = {
    "spo2": ["2708-6", "2710-2"],
    "temperature": ["8331-1"],
    "glucose": ["2345-7", "2344-0"],
}

# NEWS2 needs all of these. Missing any one makes the aggregate score
# uncomputable, which breaks the labelling scheme independently of cadence.
NEWS2_COMPONENTS = {
    "resp_rate": ["9279-1"],
    "spo2": ["59408-5", "2708-6", "2710-2"],
    "temperature": ["8310-5", "8331-1"],
    "bp_systolic": ["8480-6"],
    "heart_rate": ["8867-4"],
}
MIN_USABLE = 300  # readings below this are too sparse to build windows from

WINDOW = 12          # timesteps the model consumes
NEEDED = WINDOW + 1  # a window plus the label step
TOLERANCES_H = [1, 6, 24, 24 * 7, 24 * 30]


def longest_run(times: pd.Series, max_gap_h: float) -> int:
    """Longest run of readings each within max_gap_h of the previous one."""
    t = times.sort_values()
    if len(t) < 2:
        return len(t)
    within = t.diff().dt.total_seconds().div(3600).le(max_gap_h)
    best = run = 1
    for flag in within.iloc[1:]:
        run = run + 1 if flag else 1
        if run > best:
            best = run
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True, type=Path,
                    help="path to synthea output/csv/observations.csv")
    ap.add_argument("--vital", default="8867-4",
                    help="LOINC code used for the decisive window test")
    args = ap.parse_args()

    if not args.obs.exists():
        sys.exit(f"not found: {args.obs}\nGenerate a Synthea cohort first (see module docstring).")

    findings: dict = {"source": str(args.obs)}

    df = pd.read_csv(args.obs, low_memory=False)
    print(f"loaded {len(df):,} observations from {args.obs.name}\n")

    # ------------------------------------------------------------------
    # 1. Do timestamps carry a time of day at all?
    #    Fastest possible disqualifier: if Synthea writes date-only values,
    #    or pins everything to a handful of clock slots, sub-day cadence is
    #    not representable and the rest of the spike is moot.
    # ------------------------------------------------------------------
    print("-" * 62)
    print("1. TIMESTAMP RESOLUTION")
    print("-" * 62)
    raw = df["DATE"].astype(str)
    print("  sample raw values:")
    for val in raw.head(5):
        print(f"    {val}")

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce", utc=True)
    n_bad = int(df["DATE"].isna().sum())
    distinct_tod = int(df["DATE"].dt.time.nunique())
    has_t = float(raw.str.contains("T").mean())

    print(f"\n  unparseable dates      : {n_bad:,}")
    print(f"  values with 'T' marker : {has_t:.1%}")
    print(f"  distinct times-of-day  : {distinct_tod:,}")
    if distinct_tod <= 24:
        print("  >> timestamps are pinned to a few clock slots - sub-day cadence is synthetic")
    findings["timestamp"] = {
        "unparseable": n_bad,
        "share_with_time_marker": round(has_t, 4),
        "distinct_times_of_day": distinct_tod,
    }

    # ------------------------------------------------------------------
    # 2. What does Synthea actually emit? Do not trust the assumed code list.
    # ------------------------------------------------------------------
    print()
    print("-" * 62)
    print("2. WHAT CODES ARE ACTUALLY PRESENT")
    print("-" * 62)
    top = (df.groupby(["CODE", "DESCRIPTION"]).size()
             .sort_values(ascending=False).head(20))
    print("  20 most frequent observation codes:")
    for (code, desc), n in top.items():
        print(f"    {str(code):<12} {n:>7,}  {str(desc)[:44]}")

    counts = df["CODE"].value_counts()
    print("\n  the seven codes the guides assume:")
    missing = []
    for code, name in VITAL_CODES.items():
        n = int(counts.get(code, 0))
        flag = "" if n else "   <-- ABSENT, needs substituting before P2"
        print(f"    {code:<12} {name:<14} {n:>7,}{flag}")
        if not n:
            missing.append({"code": code, "name": name})
    findings["missing_codes"] = missing

    # look for the alternates Synthea actually uses
    alts_found = {}
    for name, codes in ALTERNATE_CODES.items():
        for c in codes:
            n = int(counts.get(c, 0))
            if n:
                alts_found[name] = {"code": c, "n": n}
                print(f"    -> '{name}' found instead under {c} ({n:,} readings)")
    findings["alternate_codes_found"] = alts_found

    # ------------------------------------------------------------------
    # 2b. Can NEWS2 even be computed? Independent of cadence.
    # ------------------------------------------------------------------
    print()
    print("-" * 62)
    print("2b. NEWS2 COMPONENT COVERAGE (labelling scheme viability)")
    print("-" * 62)
    news2_status, blocked = {}, []
    for name, codes in NEWS2_COMPONENTS.items():
        n = max(int(counts.get(c, 0)) for c in codes)
        best = max(codes, key=lambda c: int(counts.get(c, 0)))
        state = "OK" if n >= MIN_USABLE else ("SPARSE" if n else "ABSENT")
        if state != "OK":
            blocked.append(name)
        news2_status[name] = {"code": best, "n": n, "state": state}
        print(f"    {name:<14} {n:>6} readings via {best:<9} {state}")
    findings["news2_components"] = news2_status

    if blocked:
        print()
        print(f"  >> NEWS2 CANNOT BE COMPUTED: {', '.join(blocked)} unusable.")
        print("     The aggregate score needs every component. This blocks the")
        print("     labelling scheme on its own, regardless of the cadence result.")
    findings["news2_blocked_by"] = blocked

    present = [c for c in VITAL_CODES if counts.get(c, 0) > 0]
    if not present:
        sys.exit("\nNone of the assumed vital codes are present. Inspect the table above "
                 "and re-run with a corrected VITAL_CODES mapping.")

    # ------------------------------------------------------------------
    # 3. Gap distribution per vital.
    # ------------------------------------------------------------------
    print()
    print("-" * 62)
    print("3. INTER-OBSERVATION GAP DISTRIBUTION (hours)")
    print("-" * 62)
    v = df[df["CODE"].isin(present)].sort_values(["PATIENT", "CODE", "DATE"]).copy()
    v["gap_h"] = (v.groupby(["PATIENT", "CODE"])["DATE"]
                    .diff().dt.total_seconds() / 3600)

    desc = (v.groupby("CODE")["gap_h"]
              .describe(percentiles=[0.25, 0.5, 0.75])[["count", "25%", "50%", "75%", "max"]]
              .round(1))
    desc.index = [f"{c} ({VITAL_CODES[c]})" for c in desc.index]
    print(desc.to_string())

    med = v.groupby("CODE")["gap_h"].median()
    findings["median_gap_hours"] = {VITAL_CODES[c]: (None if pd.isna(x) else round(float(x), 2))
                                    for c, x in med.items()}
    overall_median = float(v["gap_h"].median())
    print(f"\n  overall median gap: {overall_median:,.1f} h  ({overall_median / 24:,.1f} days)")
    findings["overall_median_gap_hours"] = round(overall_median, 2)

    # ------------------------------------------------------------------
    # 4. THE DECISIVE TEST — can a 12-step window be formed at all?
    # ------------------------------------------------------------------
    print()
    print("-" * 62)
    print(f"4. DECISIVE TEST - runs of >={NEEDED} consecutive readings")
    print("-" * 62)
    code = args.vital if args.vital in present else present[0]
    print(f"  using {code} ({VITAL_CODES[code]})\n")

    sub = v[v["CODE"] == code]
    n_patients = sub["PATIENT"].nunique()
    table = []
    for tol in TOLERANCES_H:
        runs = sub.groupby("PATIENT")["DATE"].apply(longest_run, max_gap_h=tol)
        n_ok = int((runs >= NEEDED).sum())
        share = n_ok / max(n_patients, 1)
        table.append({"max_gap_h": tol, "patients_ok": n_ok,
                      "patients_total": n_patients, "share": round(share, 4)})
        label = f"{tol}h" if tol < 24 else f"{tol // 24}d"
        print(f"    max gap {label:>5}  ->  {n_ok:>4}/{n_patients} patients "
              f"({share:6.1%}) can form a {WINDOW}-step window")
    findings["window_formability"] = table

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    at_24h = next(r["share"] for r in table if r["max_gap_h"] == 24)
    print()
    print("=" * 62)
    # A cohort only counts if enough patients survive to train on. A few
    # percent is arithmetically non-zero and practically useless.
    if at_24h >= 0.5 and not blocked:
        verdict = "PASS"
        print("VERDICT: PASS - most patients form windows at <=24h tolerance")
        print("and every NEWS2 component is available.")
        print("Proceed as planned. Record the tolerance you will use in P2.")
    elif at_24h >= 0.25 and not blocked:
        verdict = "MARGINAL"
        print(f"VERDICT: MARGINAL - {at_24h:.1%} of patients qualify at 24h.")
        print("Decide whether the surviving cohort is large enough to train on.")
    else:
        verdict = "FAIL"
        print(f"VERDICT: FAIL - only {at_24h:.1%} of patients form a window at 24h"
              + (", and NEWS2 is uncomputable." if blocked else "."))
        print()
        print("Synthea has clinical structure but no ward telemetry.")
        if blocked:
            print(f"Independently, {', '.join(blocked)} is unusable, so the NEWS2")
            print("labelling scheme cannot be applied to this data at all.")
        print()
        print("Choose a path and record it in P0_MEMO.md:")
        print("  A. Reframe as longitudinal encounter sequences.")
        print("     Does NOT fix a missing NEWS2 component - you still cannot label.")
        print("  B. Synthesize the missing vitals and the cadence.")
        print("     The more you generate, the less the 'grounded in real care")
        print("     pathways' defense is doing for you.")
        print("  C. Move to MIMIC-IV chartevents.")
        print("     Real telemetry, real SpO2/temperature, and real per-care-unit")
        print("     non-IID partitions. Costs credentialing time.")
    print("=" * 62)
    findings["verdict"] = verdict
    findings["share_at_24h"] = at_24h

    out = RESULTS / "s1_cadence.json"
    out.write_text(json.dumps(findings, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
