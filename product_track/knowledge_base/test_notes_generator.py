"""
Tests for deterministic clinical note generation (Stage 2 - Person A).
"""

import json
from pathlib import Path
import pytest
import pandas as pd

from shared import add_news2, fill_vitals, load_patient, patient_facts
from product_track.knowledge_base import (
    ClinicalNote,
    generate_patient_notes,
    generate_admission_note,
    generate_nursing_vitals_note,
    generate_deterioration_note,
    generate_labs_note,
    batch_generate_notes,
)

SAMPLE_PATIENT = Path("physioNet/training_setA/training_setA/p000001.psv")


@pytest.fixture
def sample_facts():
    df = load_patient(SAMPLE_PATIENT)
    df, _ = fill_vitals(df)
    df = add_news2(df)
    return patient_facts(df)


def test_generate_patient_notes_from_file():
    notes, facts = generate_patient_notes(SAMPLE_PATIENT)
    assert len(notes) == 4
    assert facts["patient_id"] == "p000001"
    assert facts["age"] == 83.1
    assert facts["sex"] == "female"

    types = {n.note_type for n in notes}
    assert types == {"admission", "nursing_vitals", "deterioration", "labs"}


def test_admission_note_content(sample_facts):
    note = generate_admission_note(sample_facts)
    assert isinstance(note, ClinicalNote)
    assert note.note_type == "admission"
    assert "83.1-year-old female" in note.content
    assert "Patient p000001" in note.content
    assert "Heart Rate:" in note.content
    assert "NEWS2" in note.content


def test_nursing_vitals_note_table(sample_facts):
    note = generate_nursing_vitals_note(sample_facts)
    assert note.note_type == "nursing_vitals"
    assert "| Vital Sign |" in note.content
    assert "Heart Rate (bpm)" in note.content
    assert "Systolic BP (mmHg)" in note.content
    # Check that min/max from facts are in the note text
    hr_min = str(sample_facts["vitals"]["HR"]["min"])
    hr_max = str(sample_facts["vitals"]["HR"]["max"])
    assert hr_min in note.content
    assert hr_max in note.content


def test_deterioration_note_with_crossing(sample_facts):
    note = generate_deterioration_note(sample_facts)
    assert note.note_type == "deterioration"
    assert "ACUTE DETERIORATION DETECTED" in note.content
    assert "First Threshold Crossing" in note.content
    assert "ICU Hour 5" in note.content
    assert "Peak NEWS2 Score" in note.content
    assert "9/15" in note.content
    assert "HIGH" in note.content


def test_deterioration_note_without_crossing():
    facts = {
        "patient_id": "p_stable",
        "icu_hours": 24,
        "news2": {
            "threshold": 5,
            "max_attainable": 15,
            "first": 1,
            "last": 1,
            "peak": 2,
            "peak_hour": 10,
            "hours_at_or_above_threshold": 0,
            "ever_crossed_threshold": False,
            "first_crossing_hour": None,
            "band_at_peak": "low",
            "response_at_peak": "routine observation",
        },
    }
    note = generate_deterioration_note(facts)
    assert "PHYSIOLOGICAL STABILITY RECORD" in note.content
    assert "0 hours" in note.content
    assert "2/15" in note.content


def test_labs_note_with_and_without_labs(sample_facts):
    # With labs
    note_with_labs = generate_labs_note(sample_facts)
    assert note_with_labs.note_type == "labs"
    assert "WBC" in note_with_labs.content
    assert "Creatinine" in note_with_labs.content

    # Without labs
    empty_facts = {"patient_id": "p_no_labs", "labs": {}}
    note_empty = generate_labs_note(empty_facts)
    assert "No routine laboratory panels" in note_empty.content


def test_batch_generate_notes(tmp_path):
    cohort_dir = SAMPLE_PATIENT.parent
    output_dir = tmp_path / "cohort_notes"
    
    result = batch_generate_notes(cohort_dir, output_dir, limit=3)
    assert result["processed_patients"] == 3

    # Verify directory structure for p000001
    p1_dir = output_dir / "p000001"
    assert p1_dir.exists()
    assert (p1_dir / "admission_note.md").exists()
    assert (p1_dir / "nursing_vitals_note.md").exists()
    assert (p1_dir / "deterioration_note.md").exists()
    assert (p1_dir / "labs_note.md").exists()
    assert (p1_dir / "ground_truth_facts.json").exists()
    assert (p1_dir / "all_notes.json").exists()

    # Verify JSON content
    facts_json = json.loads((p1_dir / "ground_truth_facts.json").read_text(encoding="utf-8"))
    assert facts_json["patient_id"] == "p000001"
    all_notes = json.loads((p1_dir / "all_notes.json").read_text(encoding="utf-8"))
    assert len(all_notes) == 4
