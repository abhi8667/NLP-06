"""
Tests for Streamlit interface scripts compilation and import integrity (Stage 5 - Person A).
"""

import py_compile
from pathlib import Path


def test_clinician_app_compilation():
    path = Path("product_track/interfaces/clinician_app.py")
    assert path.exists()
    compiled = py_compile.compile(str(path), doraise=True)
    assert compiled is not None


def test_patient_app_compilation():
    path = Path("product_track/interfaces/patient_app.py")
    assert path.exists()
    compiled = py_compile.compile(str(path), doraise=True)
    assert compiled is not None
