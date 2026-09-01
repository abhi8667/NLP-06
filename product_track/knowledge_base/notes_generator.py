"""
Deterministic, template-grounded clinical note generator.

Generates structured clinical notes for RAG retrieval directly from PhysioNet
patient records and shared.patient_facts().

Rule: Every sentence is constructed from verified data fields. No free-form
hallucinations are permitted.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from shared import add_news2, fill_vitals, load_patient, patient_facts


@dataclass
class ClinicalNote:
    """Represents a single deterministic clinical note for a patient."""
    patient_id: str
    note_type: str  # 'admission', 'nursing_vitals', 'deterioration', 'labs'
    title: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        return f"# {self.title}\n**Patient ID:** {self.patient_id} | **Type:** {self.note_type}\n\n{self.content}\n"


def _format_value(val: Any, default: str = "Not recorded") -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    return str(val)


def generate_admission_note(facts: dict[str, Any]) -> ClinicalNote:
    """
    Generate initial clinical baseline and admission assessment note.
    """
    pid = _format_value(facts.get("patient_id"), "Unknown")
    age = _format_value(facts.get("age"))
    sex = _format_value(facts.get("sex"))
    icu_hours = _format_value(facts.get("icu_hours"))
    pre_icu_hosp = facts.get("hours_in_hospital_before_icu")
    pre_icu_str = f"{pre_icu_hosp:.2f} hours" if pre_icu_hosp is not None else "Direct ICU admission / Not recorded"

    vitals = facts.get("vitals", {})
    v_lines = []
    for v_key, v_name, unit in [
        ("HR", "Heart Rate", "bpm"),
        ("SBP", "Systolic Blood Pressure", "mmHg"),
        ("Resp", "Respiration Rate", "breaths/min"),
        ("O2Sat", "Oxygen Saturation (SpO2)", "%"),
        ("Temp", "Body Temperature", "deg C"),
        ("Glucose", "Serum Glucose", "mg/dL"),
    ]:
        if v_key in vitals:
            val = vitals[v_key].get("first")
            v_lines.append(f"- **{v_name}:** {_format_value(val)} {unit}")
        else:
            v_lines.append(f"- **{v_name}:** Not recorded on admission")

    news2_info = facts.get("news2", {})
    init_news2 = _format_value(news2_info.get("first"))
    max_news2 = _format_value(news2_info.get("max_attainable", 15))

    lines = [
        f"Patient {pid} is a {age}-year-old {sex} admitted to the Intensive Care Unit (ICU).",
        f"Prior to ICU admission, the patient spent {pre_icu_str} in the hospital.",
        f"Total monitored ICU stay duration is {icu_hours} hours.",
        "",
        "### Initial Baseline Vital Signs (ICU Hour 0):",
        *v_lines,
        "",
        f"Initial calculated NEWS2 physiological score on arrival was {init_news2}/{max_news2}.",
    ]

    return ClinicalNote(
        patient_id=pid,
        note_type="admission",
        title=f"Admission Clinical Assessment - Patient {pid}",
        content="\n".join(lines),
        metadata={
            "patient_id": pid,
            "note_type": "admission",
            "age": facts.get("age"),
            "sex": facts.get("sex"),
            "initial_news2": news2_info.get("first"),
        },
    )


def generate_nursing_vitals_note(facts: dict[str, Any]) -> ClinicalNote:
    """
    Generate longitudinal vital signs trajectory and nursing shift summary.
    """
    pid = _format_value(facts.get("patient_id"), "Unknown")
    vitals = facts.get("vitals", {})
    stay_hours = _format_value(facts.get("icu_hours"))

    table_rows = [
        "| Vital Sign | Initial (Hr 0) | Final | Minimum | Maximum | Observed Trend |",
        "|---|---|---|---|---|---|",
    ]

    vital_meta = [
        ("HR", "Heart Rate (bpm)"),
        ("SBP", "Systolic BP (mmHg)"),
        ("Resp", "Respiration Rate (breaths/min)"),
        ("O2Sat", "SpO2 Oxygen Saturation (%)"),
        ("Temp", "Temperature (deg C)"),
        ("Glucose", "Glucose (mg/dL)"),
    ]

    for key, label in vital_meta:
        if key in vitals:
            v = vitals[key]
            first_v = _format_value(v.get("first"))
            last_v = _format_value(v.get("last"))
            min_v = _format_value(v.get("min"))
            max_v = _format_value(v.get("max"))
            trend_v = _format_value(v.get("trend")).capitalize()
            table_rows.append(f"| {label} | {first_v} | {last_v} | {min_v} | {max_v} | {trend_v} |")
        else:
            table_rows.append(f"| {label} | N/A | N/A | N/A | N/A | Not monitored |")

    lines = [
        f"Longitudinal vital signs summary for patient {pid} across the {stay_hours}-hour ICU stay.",
        "",
        *table_rows,
        "",
        "### Key Trajectory Notes:",
    ]

    # Deterministic narrative highlights
    highlights = []
    if "HR" in vitals:
        hr = vitals["HR"]
        highlights.append(f"- Heart rate shifted from {hr['first']} to {hr['last']} bpm (range: {hr['min']}-{hr['max']} bpm, trend: {hr['trend']}).")
    if "SBP" in vitals:
        sbp = vitals["SBP"]
        highlights.append(f"- Systolic blood pressure spanned {sbp['min']} to {sbp['max']} mmHg, ending at {sbp['last']} mmHg ({sbp['trend']}).")
    if "O2Sat" in vitals:
        o2 = vitals["O2Sat"]
        highlights.append(f"- Oxygen saturation reached a low of {o2['min']}% and high of {o2['max']}% (final: {o2['last']}%, {o2['trend']}).")
    if "Resp" in vitals:
        resp = vitals["Resp"]
        highlights.append(f"- Respiratory rate exhibited a range of {resp['min']}-{resp['max']} breaths/min ({resp['trend']}).")

    if not highlights:
        highlights.append("- No continuous vital parameters were recorded during this stay.")

    lines.extend(highlights)

    return ClinicalNote(
        patient_id=pid,
        note_type="nursing_vitals",
        title=f"Longitudinal Nursing Vitals Summary - Patient {pid}",
        content="\n".join(lines),
        metadata={
            "patient_id": pid,
            "note_type": "nursing_vitals",
            "icu_hours": facts.get("icu_hours"),
            "vitals_summary": {k: vitals[k].get("trend") for k in vitals},
        },
    )


def generate_deterioration_note(facts: dict[str, Any]) -> ClinicalNote:
    """
    Generate clinical deterioration and NEWS2 escalation record.
    """
    pid = _format_value(facts.get("patient_id"), "Unknown")
    news = facts.get("news2", {})
    threshold = news.get("threshold", 5)
    max_att = news.get("max_attainable", 15)
    crossed = news.get("ever_crossed_threshold", False)
    peak = news.get("peak", 0)
    peak_hr = news.get("peak_hour", "N/A")
    band = _format_value(news.get("band_at_peak"), "unknown").upper()
    response = _format_value(news.get("response_at_peak"), "Standard clinical observation")
    hrs_above = news.get("hours_at_or_above_threshold", 0)
    first_cross = news.get("first_crossing_hour")

    if crossed:
        lines = [
            "**CLINICAL ESCALATION RECORD - ACUTE DETERIORATION DETECTED**",
            "",
            f"Patient {pid} crossed the standardized clinical deterioration threshold (NEWS2 aggregate >= {threshold}).",
            f"- **First Threshold Crossing:** ICU Hour {first_cross}",
            f"- **Peak NEWS2 Score:** {peak}/{max_att} reached at ICU Hour {peak_hr}",
            f"- **Clinical Risk Band at Peak:** {band}",
            f"- **Recommended Clinical Escalation:** {response}",
            f"- **Total Monitored Duration >= Threshold:** {hrs_above} hours",
            "",
            "### Clinical Significance:",
            "The patient demonstrated acute physiological instability during this stay requiring continuous telemetry monitoring and clinical review.",
        ]
    else:
        lines = [
            "**PHYSIOLOGICAL STABILITY RECORD - NO ACUTE DETERIORATION**",
            "",
            f"Patient {pid} maintained physiological stability below the clinical escalation threshold (NEWS2 < {threshold}) throughout the entire stay.",
            f"- **Peak NEWS2 Score:** {peak}/{max_att} observed at ICU Hour {peak_hr}",
            f"- **Overall Risk Band:** {band}",
            f"- **Recommended Clinical Response:** {response}",
            f"- **Hours Above Threshold:** 0 hours",
            "",
            "### Clinical Significance:",
            "No emergency early-warning protocol escalations were triggered for this patient.",
        ]

    return ClinicalNote(
        patient_id=pid,
        note_type="deterioration",
        title=f"Deterioration & NEWS2 Escalation Record - Patient {pid}",
        content="\n".join(lines),
        metadata={
            "patient_id": pid,
            "note_type": "deterioration",
            "ever_crossed_threshold": crossed,
            "peak_news2": peak,
            "peak_hour": peak_hr,
            "band_at_peak": band,
        },
    )


def generate_labs_note(facts: dict[str, Any]) -> ClinicalNote:
    """
    Generate diagnostic and laboratory findings summary.
    """
    pid = _format_value(facts.get("patient_id"), "Unknown")
    labs = facts.get("labs", {})

    if not labs:
        content = (
            f"No routine laboratory panels (biochemical, hematological, or blood gas tests) "
            f"were ordered or recorded for patient {pid} during this ICU stay."
        )
    else:
        rows = [
            "| Laboratory Test | Observations (Count) | Initial Value | Final Value | Min Observed | Max Observed |",
            "|---|---|---|---|---|---|",
        ]
        for lab_name in sorted(labs.keys()):
            l_data = labs[lab_name]
            count = _format_value(l_data.get("n_measurements"))
            first_v = _format_value(l_data.get("first"))
            last_v = _format_value(l_data.get("last"))
            min_v = _format_value(l_data.get("min"))
            max_v = _format_value(l_data.get("max"))
            rows.append(f"| {lab_name} | {count} | {first_v} | {last_v} | {min_v} | {max_v} |")

        content = "\n".join([
            f"Diagnostic laboratory panel results ordered during ICU hospitalization for patient {pid}:",
            f"Total unique laboratory markers recorded: {len(labs)}.",
            "",
            *rows,
        ])

    return ClinicalNote(
        patient_id=pid,
        note_type="labs",
        title=f"Diagnostic Laboratory Findings - Patient {pid}",
        content=content,
        metadata={
            "patient_id": pid,
            "note_type": "labs",
            "n_lab_tests": len(labs),
            "lab_tests": list(labs.keys()),
        },
    )


def generate_patient_notes(
    source: pd.DataFrame | str | Path | dict[str, Any],
) -> tuple[list[ClinicalNote], dict[str, Any]]:
    """
    Generate the complete set of clinical notes and ground-truth facts for a patient.

    Accepts:
        - raw DataFrame (must be filled & scored or raw)
        - Path / str to a .psv file
        - pre-extracted `facts` dict

    Returns:
        (list of ClinicalNote objects, ground-truth facts dict)
    """
    if isinstance(source, dict):
        facts = source
    else:
        if isinstance(source, (str, Path)):
            df = load_patient(source)
        elif isinstance(source, pd.DataFrame):
            df = source.copy()
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        if "news2" not in df.columns or "HR" not in df.columns:
            df, _ = fill_vitals(df)
            df = add_news2(df)
        facts = patient_facts(df)

    notes = [
        generate_admission_note(facts),
        generate_nursing_vitals_note(facts),
        generate_deterioration_note(facts),
        generate_labs_note(facts),
    ]

    return notes, facts


def batch_generate_notes(
    cohort_dir: str | Path,
    output_dir: str | Path,
    limit: int | None = None,
) -> dict[str, int]:
    """
    Batch generate and save clinical notes and ground truth facts for a cohort.

    Saves for each patient:
        <output_dir>/<patient_id>/
            - admission_note.md
            - nursing_vitals_note.md
            - deterioration_note.md
            - labs_note.md
            - ground_truth_facts.json
            - all_notes.json
    """
    cohort_path = Path(cohort_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    files = sorted(cohort_path.glob("*.psv"))
    if limit is not None:
        files = files[:limit]

    count = 0
    for f in files:
        notes, facts = generate_patient_notes(f)
        pid = facts.get("patient_id") or f.stem
        p_dir = out_path / pid
        p_dir.mkdir(parents=True, exist_ok=True)

        # Save individual markdown files
        for note in notes:
            (p_dir / f"{note.note_type}_note.md").write_text(note.to_markdown(), encoding="utf-8")

        # Save ground truth facts JSON
        (p_dir / "ground_truth_facts.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")

        # Save all notes structured JSON
        all_notes_data = [n.to_dict() for n in notes]
        (p_dir / "all_notes.json").write_text(json.dumps(all_notes_data, indent=2), encoding="utf-8")

        count += 1

    return {"processed_patients": count, "output_directory": str(out_path)}
