"""
Step 21 — Evaluation Scenarios from Held-Out Patient Cohort.

Extracts realistic acute deterioration alert scenarios from held-out PhysioNet patients
for rigorous, reproducible clinical summary evaluation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from product_track.bridge.risk_scorer import AbnormalityReport, identify_abnormal_vitals
from shared.news2 import news2_total, risk_band
from shared.preprocessing import (
    VITALS,
    WINDOW,
    add_news2,
    fill_vitals,
    load_patient,
    patient_facts,
    patient_split,
)


@dataclass
class AlertScenario:
    """A fully encapsulated alert scenario for clinical evaluation."""
    patient_id: str
    alert_hour: int
    news2_score: int
    risk_band: str
    current_vitals: dict[str, float]
    abnormal_vitals: list[dict[str, Any]]
    window_vitals: list[dict[str, float]]
    ground_truth_facts: dict[str, Any]
    source_file: str
    scenario_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AlertScenario:
        return cls(**data)


@dataclass
class ScenarioCohort:
    """A collection of curated evaluation scenarios with cohort-level statistics."""
    scenarios: list[AlertScenario]
    total_patients_screened: int
    held_out_count: int
    seed: int
    mean_news2: float
    alert_prevalence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_patients_screened": self.total_patients_screened,
            "held_out_count": self.held_out_count,
            "seed": self.seed,
            "mean_news2": self.mean_news2,
            "alert_prevalence": self.alert_prevalence,
            "scenarios": [s.to_dict() for s in self.scenarios],
        }

    def save_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> ScenarioCohort:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        scenarios = [AlertScenario.from_dict(s) for s in data["scenarios"]]
        return cls(
            scenarios=scenarios,
            total_patients_screened=data["total_patients_screened"],
            held_out_count=data["held_out_count"],
            seed=data["seed"],
            mean_news2=data["mean_news2"],
            alert_prevalence=data["alert_prevalence"],
        )


def build_scenario_from_patient(
    file_path: str | Path,
    scenario_idx: int = 0,
    preferred_crossing: bool = True,
) -> AlertScenario | None:
    """
    Build an evaluation scenario from a single patient's data file.
    Picks the hour where NEWS2 first crosses >= 5 or reaches maximum severity.
    """
    path = Path(file_path)
    df = load_patient(path)
    if len(df) < 2:
        return None

    df_filled, _ = fill_vitals(df)
    df_scored = add_news2(df_filled)
    facts = patient_facts(df_scored)

    news2_series = df_scored["news2"].values
    crossing_hour = facts.get("news2", {}).get("first_crossing_hour")
    peak_hour = facts.get("news2", {}).get("peak_hour")

    # Choose alert hour
    if preferred_crossing and crossing_hour is not None and crossing_hour < len(df_scored):
        alert_hr = int(crossing_hour)
    elif peak_hour is not None and peak_hour < len(df_scored):
        alert_hr = int(peak_hour)
    else:
        alert_hr = int(np.argmax(news2_series))

    # Extract current vitals at alert hour
    alert_row = df_scored.iloc[alert_hr]
    current_vitals = {v: float(alert_row[v]) for v in VITALS if v in alert_row and pd.notna(alert_row[v])}
    current_news2 = int(alert_row["news2"])
    current_risk_band = risk_band(current_news2)

    # Extract 12-hour sliding window (or padded if earlier in stay)
    start_hr = max(0, alert_hr - WINDOW + 1)
    window_df = df_scored.iloc[start_hr : alert_hr + 1]
    window_list: list[dict[str, float]] = []
    for _, row in window_df.iterrows():
        window_list.append({v: float(row[v]) for v in VITALS if v in row and pd.notna(row[v])})

    # Identify abnormal vitals
    abnormalities = identify_abnormal_vitals(current_vitals)

    return AlertScenario(
        patient_id=df["patient_id"].iloc[0],
        alert_hour=alert_hr,
        news2_score=current_news2,
        risk_band=current_risk_band,
        current_vitals=current_vitals,
        abnormal_vitals=abnormalities,
        window_vitals=window_list,
        ground_truth_facts=facts,
        source_file=str(path),
        scenario_index=scenario_idx,
        metadata={
            "age": facts.get("age"),
            "sex": facts.get("sex"),
            "icu_hours": facts.get("icu_hours"),
            "first_crossing_hour": crossing_hour,
            "peak_hour": peak_hour,
            "peak_news2": facts.get("news2", {}).get("peak"),
        },
    )


def select_alert_scenarios(
    data_dir: str | Path | None = None,
    n_scenarios: int = 25,
    test_fraction: float = 0.2,
    seed: int = 42,
    output_json: str | Path | None = None,
) -> ScenarioCohort:
    """
    Select 20-30 alert scenarios strictly from held-out patients partitioned
    using shared.patient_split().
    """
    if data_dir is None:
        base_dir = Path("physioNet")
        files = sorted(base_dir.rglob("*.psv"))
    else:
        files = sorted(Path(data_dir).rglob("*.psv"))

    if not files:
        # Fallback to demo synthetic generation if no psv files present
        scenarios: list[AlertScenario] = []
        return ScenarioCohort(
            scenarios=scenarios,
            total_patients_screened=0,
            held_out_count=0,
            seed=seed,
            mean_news2=0.0,
            alert_prevalence=0.0,
        )

    # Perform strict patient-level train/test split
    files_arr = np.array(files)
    _, test_mask = patient_split(files_arr, test_frac=test_fraction, seed=seed)
    held_out_files = list(files_arr[test_mask])

    scenarios: list[AlertScenario] = []
    total_screened = 0
    alert_count = 0
    news2_scores: list[int] = []

    # Prioritize patients who experienced deterioration (NEWS2 >= 5)
    deteriorating_candidates: list[Path] = []
    stable_candidates: list[Path] = []

    for f in held_out_files:
        total_screened += 1
        scenario = build_scenario_from_patient(f, scenario_idx=len(scenarios))
        if scenario is None:
            continue

        if scenario.news2_score >= 5:
            deteriorating_candidates.append(f)
        else:
            stable_candidates.append(f)

        # Stop screening if we have ample candidates
        if len(deteriorating_candidates) >= n_scenarios * 2 and total_screened >= 100:
            break

    # Select target mix: 80% deteriorating alerts (>=5) and 20% borderline/stable (<5)
    n_deteriorating = min(int(n_scenarios * 0.8), len(deteriorating_candidates))
    n_stable = min(n_scenarios - n_deteriorating, len(stable_candidates))

    selected_files = deteriorating_candidates[:n_deteriorating] + stable_candidates[:n_stable]

    for idx, f in enumerate(selected_files):
        scen = build_scenario_from_patient(f, scenario_idx=idx)
        if scen is not None:
            scenarios.append(scen)
            news2_scores.append(scen.news2_score)
            if scen.news2_score >= 5:
                alert_count += 1

    mean_news2 = float(np.mean(news2_scores)) if news2_scores else 0.0
    prevalence = (alert_count / len(scenarios)) if scenarios else 0.0

    cohort = ScenarioCohort(
        scenarios=scenarios,
        total_patients_screened=total_screened,
        held_out_count=len(held_out_files),
        seed=seed,
        mean_news2=round(mean_news2, 2),
        alert_prevalence=round(prevalence, 3),
    )

    if output_json:
        cohort.save_json(output_json)

    return cohort
