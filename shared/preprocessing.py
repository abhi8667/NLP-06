"""
Shared preprocessing for NLP-06 — imported by BOTH tracks.

Person A (product) and Person B (research) must agree on loading, imputation and
NEWS2, or the alert says one thing and the summary another. Import from here;
never copy-paste these functions.

    Person B needs:  make_windows()      -> tensors for the detector
    Person A needs:  patient_facts()     -> ground-truth facts for note generation
    Both need:       load_patient(), fill_vitals(), add_news2()

DATASET SHAPE
-------------
PhysioNet/CinC 2019. One .psv per patient, pipe-separated, ONE ROW = ONE HOUR.
41 columns: 9 vitals, 26 labs, 5 demographic/admin, 1 label.

MEASURED COVERAGE (400-patient sample, 15,500 patient-hours)
    HR 92%   Resp 90%   MAP 89%   O2Sat 87%   SBP 86%
    Temp 36%   Glucose 12%   most labs < 10%
    EtCO2 0%        <- entirely empty, do not use
    Unit1/Unit2 53% <- only half of patients, not usable for partitioning

LOCKED DECISIONS (see docs/02_scope_lock.html)
    - Forward-fill then backward-fill. Leading rows are often all-NaN, so ffill
      alone leaves gaps (~0.6 rows/patient).
    - Labels are NEWS2 aggregate >= 5, derived. NOT the dataset's SepsisLabel,
      which is retained only as an independent cross-check.
    - Prediction horizon 4-6 hours. At horizon 0 the task is near-trivial
      (AUROC 0.90 vs 0.78 at 6h) because labels derive from the same vitals.
    - Splits are BY PATIENT, never by window.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .news2 import (COMPONENTS, DEFAULT_THRESHOLD, NEWS2_MAX, news2_total,
                    recommended_response, risk_band)

__all__ = [
    "VITALS", "WINDOW", "HORIZON", "UNUSABLE_COLUMNS",
    "load_patient", "fill_vitals", "add_news2", "make_windows",
    "patient_facts", "patient_split", "load_cohort", "ImputationReport",
    "read_norm_stats",
]

#: The six vitals the detector consumes, in fixed channel order.
VITALS = ["HR", "SBP", "O2Sat", "Resp", "Temp", "Glucose"]

WINDOW = 12    # timesteps fed to the model
HORIZON = 6    # hours ahead the label is taken

#: Columns measured as unusable. Documented so nobody rediscovers this.
UNUSABLE_COLUMNS = {
    "EtCO2": "0% coverage across the sampled cohort",
    "Unit1": "only ~53% of patients; unusable for partitioning",
    "Unit2": "only ~53% of patients; unusable for partitioning",
}


@dataclass
class ImputationReport:
    """Per-vital carry-forward rates. These MUST be reported in the paper."""
    observed: dict = field(default_factory=dict)
    imputed: dict = field(default_factory=dict)
    #: rows filled from `fallback` because the vital was NEVER observed for that
    #: patient — a strictly weaker form of imputation than carry-forward.
    never_observed: dict = field(default_factory=dict)
    rows: int = 0

    def rate(self, vital: str) -> float:
        return self.imputed.get(vital, 0) / max(self.rows, 1)

    def never_observed_rate(self, vital: str) -> float:
        return self.never_observed.get(vital, 0) / max(self.rows, 1)

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame({
            "observed": pd.Series(self.observed),
            "imputed": pd.Series(self.imputed),
            "never_observed": pd.Series(self.never_observed),
            "imputation_rate": pd.Series({v: self.rate(v) for v in self.observed}),
        }).fillna(0).sort_values("imputation_rate")

    def merge(self, other: "ImputationReport") -> "ImputationReport":
        out = ImputationReport(rows=self.rows + other.rows)
        for v in set(self.observed) | set(other.observed):
            out.observed[v] = self.observed.get(v, 0) + other.observed.get(v, 0)
            out.imputed[v] = self.imputed.get(v, 0) + other.imputed.get(v, 0)
            out.never_observed[v] = (self.never_observed.get(v, 0)
                                     + other.never_observed.get(v, 0))
        return out


def load_patient(path: str | Path) -> pd.DataFrame:
    """Read one .psv. Adds `patient_id` from the filename."""
    path = Path(path)
    df = pd.read_csv(path, sep="|")
    df["patient_id"] = path.stem
    return df


def fill_vitals(df: pd.DataFrame, vitals: list[str] | None = None,
                fallback: dict[str, float] | None = None
                ) -> tuple[pd.DataFrame, ImputationReport]:
    """
    Forward-fill then backward-fill the vitals, and report how much was imputed.

    Forward-fill mirrors what a clinician does reading a chart: the last observed
    value stands until a new one is taken. Backward-fill is needed only for the
    leading gap — the first rows of a stay are frequently blank, and ffill has
    nothing to carry from.

    THE CASE NEITHER FILL CAN HANDLE
    --------------------------------
    Roughly 4% of patients never have a given vital measured *at all* — most
    often glucose, occasionally SpO2 or temperature. Neither ffill nor bfill can
    fill a column with zero observations, so those columns stay NaN and later
    propagate silently into model inputs as NaN predictions.

    Pass `fallback` (normally the frozen training-set means) to fill those
    columns. Rows filled this way are counted separately in
    `report.never_observed`, because substituting a cohort mean is a much weaker
    claim than carrying a real observation forward, and the paper should report
    the two rates separately.

    Default is `None`, which preserves the original behaviour: never-observed
    columns are left as NaN for the caller to handle.
    """
    vitals = vitals or VITALS
    out = df.copy()
    rep = ImputationReport(rows=len(out))
    for c in vitals:
        if c not in out.columns:
            out[c] = np.nan
        observed = int(out[c].notna().sum())
        out[c] = out[c].ffill().bfill()
        rep.observed[c] = observed
        rep.imputed[c] = int(out[c].notna().sum()) - observed
        rep.never_observed[c] = 0

        if observed == 0 and fallback is not None and c in fallback:
            out[c] = float(fallback[c])
            rep.never_observed[c] = len(out)
    return out, rep


def add_news2(df: pd.DataFrame, threshold: int = DEFAULT_THRESHOLD) -> pd.DataFrame:
    """Add `news2`, `news2_label`, `news2_band` columns. Expects filled vitals."""
    out = df.copy()
    total = np.zeros(len(out), dtype=int)
    for name, (col, fn, _) in COMPONENTS.items():
        series = out[col] if col in out.columns else pd.Series(np.nan, index=out.index)
        total = total + series.map(fn).fillna(0).to_numpy(dtype=int)
    out["news2"] = total
    out["news2_label"] = (total >= threshold).astype(int)
    out["news2_band"] = [risk_band(t) for t in total]
    return out


def make_windows(df: pd.DataFrame, window: int = WINDOW, horizon: int = HORIZON
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sliding windows for the detector.

    Returns (X, y, patient_ids):
        X          [n, window, 6]  float32
        y          [n]             float32 — NEWS2 label `horizon` hours AFTER
                                   the window ends
        patient_ids[n]             so callers can split by patient

    Horizon 0 makes this near-trivial: the label derives from vitals almost
    identical to the last timestep, and forward-fill can make it literally the
    same value. Keep horizon >= 4.
    """
    need = window + horizon
    if len(df) <= need:
        return (np.empty((0, window, len(VITALS)), np.float32),
                np.empty(0, np.float32), np.empty(0, object))

    arr = df[VITALS].to_numpy(np.float32)
    lab = df["news2_label"].to_numpy(np.float32)
    pid = df["patient_id"].iloc[0] if "patient_id" in df.columns else "unknown"

    n = len(df) - need
    X = np.stack([arr[i:i + window] for i in range(n)])
    y = np.array([lab[i + window + horizon] for i in range(n)], dtype=np.float32)
    return X, y, np.array([pid] * n, dtype=object)


def patient_split(patient_ids: np.ndarray, test_frac: float = 0.2, seed: int = 42
                  ) -> tuple[np.ndarray, np.ndarray]:
    """
    Boolean masks for a PATIENT-level split. Never split windows randomly:
    stride-1 windows overlap by 11 of 12 timesteps, so a random split puts
    near-duplicates on both sides and the model is tested on what it memorised.
    """
    uniq = np.unique(patient_ids)
    shuffled = np.random.default_rng(seed).permutation(uniq)
    n_test = max(1, int(round(test_frac * len(shuffled))))
    test_set = set(shuffled[:n_test])
    is_test = np.fromiter((p in test_set for p in patient_ids), bool, len(patient_ids))
    return ~is_test, is_test


def _trend(series: pd.Series) -> str:
    s = series.dropna()
    if len(s) < 2:
        return "insufficient data"
    delta = float(s.iloc[-1] - s.iloc[0])
    rel = abs(delta) / max(abs(float(s.iloc[0])), 1e-6)
    if rel < 0.05:
        return "stable"
    return "rising" if delta > 0 else "falling"


def patient_facts(df: pd.DataFrame, threshold: int = DEFAULT_THRESHOLD) -> dict:
    """
    Every fact about a patient that is TRUE BY CONSTRUCTION.

    Person A: this is the source material for generated clinical notes, and the
    ground-truth fact list the P6 fact-verifier checks summaries against. If a
    claim in a generated summary is not derivable from this dict, it is a
    hallucination.

    Expects a frame that has been through fill_vitals() and add_news2().
    """
    first, last = df.iloc[0], df.iloc[-1]
    facts: dict = {
        "patient_id": df["patient_id"].iloc[0] if "patient_id" in df else None,
        "age": None if pd.isna(first.get("Age")) else round(float(first["Age"]), 1),
        "sex": {0: "female", 1: "male"}.get(int(first["Gender"]))
               if not pd.isna(first.get("Gender")) else None,
        "icu_hours": int(len(df)),
        "hours_in_hospital_before_icu": (
            None if pd.isna(first.get("HospAdmTime"))
            else round(abs(float(first["HospAdmTime"])), 2)),
        "vitals": {},
        "news2": {},
        "labs": {},
    }

    for v in VITALS:
        if v not in df.columns:
            continue
        s = df[v]
        if s.notna().sum() == 0:
            continue
        facts["vitals"][v] = {
            "first": round(float(s.iloc[0]), 1),
            "last": round(float(s.iloc[-1]), 1),
            "min": round(float(s.min()), 1),
            "max": round(float(s.max()), 1),
            "trend": _trend(s),
        }

    if "news2" in df.columns:
        peak_idx = int(df["news2"].idxmax())
        crossed = df.index[df["news2"] >= threshold]
        facts["news2"] = {
            "max_attainable": NEWS2_MAX,
            "threshold": threshold,
            "first": int(df["news2"].iloc[0]),
            "last": int(df["news2"].iloc[-1]),
            "peak": int(df["news2"].max()),
            "peak_hour": int(df.loc[peak_idx, "ICULOS"]) if "ICULOS" in df else peak_idx,
            "hours_at_or_above_threshold": int((df["news2"] >= threshold).sum()),
            "ever_crossed_threshold": bool(len(crossed) > 0),
            "first_crossing_hour": (int(df.loc[crossed[0], "ICULOS"])
                                    if len(crossed) and "ICULOS" in df else None),
            "band_at_peak": risk_band(int(df["news2"].max())),
            "response_at_peak": recommended_response(int(df["news2"].max())),
        }

    # labs that were actually ordered, with their observed range
    lab_cols = [c for c in df.columns
                if c not in VITALS + ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime",
                                      "ICULOS", "SepsisLabel", "patient_id", "MAP",
                                      "DBP", "EtCO2", "news2", "news2_label", "news2_band"]]
    for c in lab_cols:
        n = int(df[c].notna().sum())
        if n:
            facts["labs"][c] = {
                "n_measurements": n,
                "first": round(float(df[c].dropna().iloc[0]), 2),
                "last": round(float(df[c].dropna().iloc[-1]), 2),
                "min": round(float(df[c].min()), 2),
                "max": round(float(df[c].max()), 2),
            }

    if "SepsisLabel" in df.columns:
        # cross-check only — NOT the training label
        facts["sepsis_crosscheck"] = bool(df["SepsisLabel"].max() > 0)
    return facts


def read_norm_stats(path: str | Path, vitals: list[str] | None = None
                    ) -> tuple[np.ndarray, np.ndarray, dict] | None:
    """
    Read per-vital normalisation stats, tolerating either on-disk schema.

    Two freeze implementations have written this file with different shapes:

        nested :  {"stats": {"HR": {"mean": .., "std": ..}, ...}}
        flat   :  {"mean": {"HR": ..}, "std": {"HR": ..}}

    Standardisation MUST match whatever the model was trained with, so silently
    failing to read the file is worse than not having it — the model would
    receive raw vitals and produce meaningless probabilities. This accepts both
    and returns None only when neither shape is present.

    Returns (mu, sd, raw_document) with `sd` guarded against division by zero.
    """
    vitals = vitals or VITALS
    p = Path(path)
    if not p.exists():
        return None
    try:
        blob = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

    means: dict = {}
    stds: dict = {}
    if isinstance(blob.get("stats"), dict):                 # nested schema
        for v in vitals:
            entry = blob["stats"].get(v) or {}
            if "mean" in entry and "std" in entry:
                means[v], stds[v] = entry["mean"], entry["std"]
    elif isinstance(blob.get("mean"), dict) and isinstance(blob.get("std"), dict):
        for v in vitals:                                    # flat schema
            if v in blob["mean"] and v in blob["std"]:
                means[v], stds[v] = blob["mean"][v], blob["std"][v]

    if len(means) != len(vitals):
        return None

    mu = np.array([float(means[v]) for v in vitals], dtype=np.float32)
    sd = np.array([float(stds[v]) for v in vitals], dtype=np.float32)
    sd = np.where(np.abs(sd) < 1e-6, 1.0, sd)
    return mu, sd, blob


def load_cohort(folder: str | Path, limit: int | None = None,
                threshold: int = DEFAULT_THRESHOLD, seed: int = 42):
    """
    Load, fill and score a folder of .psv files.

    Returns (frames, imputation_report). `frames` are fully processed and ready
    for make_windows() or patient_facts().
    """
    files = sorted(Path(folder).glob("*.psv"))
    if limit and limit < len(files):
        rng = np.random.default_rng(seed)
        files = [files[i] for i in rng.permutation(len(files))[:limit]]

    frames, report = [], ImputationReport()
    for f in files:
        df = load_patient(f)
        df, rep = fill_vitals(df)
        df = add_news2(df, threshold=threshold)
        frames.append(df)
        report = report.merge(rep)
    return frames, report
