"""
Tests for the Streamlit interface layer (Stage 5 - Person A).

Compilation integrity for every page module, plus the two invariants that matter
for a demo: evaluation numbers are never hardcoded in the UI, and the demo script
only points at panels the console actually renders.
"""

import py_compile
import re
from pathlib import Path

import pytest

INTERFACE_DIR = Path("product_track/interfaces")

MODULES = [
    "app.py",
    "assurance_page.py",
    "charts.py",
    "clinician_app.py",
    "demo_script.py",
    "metrics_source.py",
    "overview_page.py",
    "patient_app.py",
    "theme.py",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_compiles(module):
    path = INTERFACE_DIR / module
    assert path.exists(), f"{module} is missing"
    assert py_compile.compile(str(path), doraise=True) is not None


def test_no_hardcoded_evaluation_numbers():
    """
    Evaluation figures must come from metrics_source at render time. A literal
    kappa, AUROC or 'zero hallucinations' claim in the UI is exactly the thing
    that survives a broken evaluation run and gets shown to an audience.
    """
    banned = [
        re.compile(r"zero\s+hallucination", re.I),
        re.compile(r"100%\s+ground[- ]truth", re.I),
        re.compile(r"0\.812"),
        re.compile(r"0\.898"),
        re.compile(r"0\.725"),
    ]
    offenders = []
    for path in INTERFACE_DIR.glob("*.py"):
        if path.name == "test_interfaces.py":
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in banned:
            if pattern.search(text):
                offenders.append(f"{path.name}: {pattern.pattern}")
    assert not offenders, "hardcoded evaluation claims found: " + "; ".join(offenders)


def test_demo_chapters_target_real_panels():
    from product_track.interfaces.demo_script import FOCUS_PANELS, SCENARIOS

    for scenario in SCENARIOS.values():
        assert scenario.chapters, f"{scenario.key} has no chapters"
        for chapter in scenario.chapters:
            assert chapter.focus in FOCUS_PANELS, (
                f"{scenario.key}/{chapter.title} focuses unknown panel {chapter.focus!r}"
            )
            assert chapter.hour >= 0
            assert chapter.caption and chapter.watch_for


def test_metrics_source_reports_provenance():
    """
    An artifact whose models returned identical narratives was produced by the
    offline template path and must not be presented as a benchmark.
    """
    from product_track.interfaces.metrics_source import _detect_provenance

    identical = {
        "model_benchmarks": {
            "a": {"items": [{"narrative_summary": "same", "latency_s": 0.85}]},
            "b": {"items": [{"narrative_summary": "same", "latency_s": 2.65}]},
        }
    }
    source, _, trustworthy = _detect_provenance(identical)
    assert source == "template"
    assert trustworthy is False

    declared = {
        "model_benchmarks": {
            "a": {"items": [{"narrative_summary": "x", "narrative_source": "ollama",
                             "latency_s": 1.1}]},
        }
    }
    source, _, trustworthy = _detect_provenance(declared)
    assert source == "model"
    assert trustworthy is True


def test_missing_artifact_is_not_an_error():
    from product_track.interfaces.metrics_source import load_evaluation_metrics

    metrics = load_evaluation_metrics(Path("does/not/exist.json"))
    assert metrics.available is False
    assert metrics.trustworthy is False
    assert metrics.fmt_pct(None) == "— not yet measured"
