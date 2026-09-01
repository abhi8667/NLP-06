"""
Knowledge base generation module for Person A (Product Track).
Deterministic, template-grounded clinical note generation from PhysioNet vitals.
"""

from .notes_generator import (
    ClinicalNote,
    generate_patient_notes,
    generate_admission_note,
    generate_nursing_vitals_note,
    generate_deterioration_note,
    generate_labs_note,
    batch_generate_notes,
)

__all__ = [
    "ClinicalNote",
    "generate_patient_notes",
    "generate_admission_note",
    "generate_nursing_vitals_note",
    "generate_deterioration_note",
    "generate_labs_note",
    "batch_generate_notes",
]
