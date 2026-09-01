"""
Tests for Alert-to-Summary Bridge and Vitals Replay Harness (Stage 7 - Person A).
"""

from pathlib import Path
import pytest
import numpy as np

from product_track.bridge import (
    RiskScorer,
    VitalsReplayHarness,
    AlertBridge,
    AlertSummaryCard,
    select_demo_patients,
)
from product_track.llm import OllamaClient
from product_track.rag import PatientVectorStore

P1_PATH = Path("physioNet/training_setA/training_setA/p000001.psv")


@pytest.fixture
def ollama():
    client = OllamaClient()
    if not client.is_available():
        pytest.skip("Local Ollama server is not running on localhost:11434")
    return client


def test_abnormality_identification():
    scorer = RiskScorer()
    abnormal_vitals = {
        "HR": 135.0,     # severe tachycardia (+3)
        "SBP": 85.0,     # severe hypotension (+3)
        "Resp": 26.0,    # severe tachypnea (+3)
        "O2Sat": 90.0,   # severe hypoxia (+3)
        "Temp": 39.2,    # severe fever (+2)
        "Glucose": 220.0 # hyperglycemia
    }
    abns = scorer.identify_abnormalities(abnormal_vitals)
    assert len(abns) >= 5

    v_map = {a.vital: a for a in abns}
    assert v_map["HR"].subscore == 3
    assert v_map["HR"].severity == "severe"
    assert v_map["SBP"].subscore == 3
    assert v_map["Resp"].subscore == 3
    assert v_map["O2Sat"].subscore == 3


def test_vitals_replay_streaming():
    harness = VitalsReplayHarness(default_interval_s=0.0)
    readings = list(harness.stream_patient(P1_PATH, interval_s=0.0, max_hours=6))

    assert len(readings) == 6
    assert readings[0].hour == 0
    assert readings[0].patient_id == "p000001"
    assert readings[0].window_buffer.shape == (1, 6)

    assert readings[5].hour == 5
    assert readings[5].window_buffer.shape == (6, 6)
    assert readings[5].is_last_hour is True


def test_select_demo_patients():
    cohort_dir = P1_PATH.parent
    candidates = select_demo_patients(cohort_dir, min_peak=7, max_candidates=3)
    assert len(candidates) >= 1
    assert candidates[0]["patient_id"] == "p000001"
    assert candidates[0]["initial_news2"] <= 4
    assert candidates[0]["peak_news2"] >= 7
    assert candidates[0]["first_crossing_hour"] is not None


def test_end_to_end_alert_bridge_flow(ollama, tmp_path):
    # Setup vector store with patient's baseline notes
    store = PatientVectorStore(persist_dir=tmp_path / "chroma_bridge_test")
    store.index_patient_from_psv(P1_PATH)

    bridge = AlertBridge(vector_store=store, llm_client=ollama)
    harness = VitalsReplayHarness(default_interval_s=0.0)

    alerts_triggered = []
    summary_cards = []

    # Stream the first 10 hours of p000001
    for telemetry in harness.stream_patient(P1_PATH, interval_s=0.0, max_hours=10):
        report, card = bridge.process_telemetry(
            patient_id=telemetry.patient_id,
            hour=telemetry.hour,
            window=telemetry.window_buffer,
            current_vitals=telemetry.vitals,
        )

        if report.is_alert:
            alerts_triggered.append(telemetry.hour)
            if card:
                summary_cards.append(card)

    # Patient p000001 crosses threshold at hour 5 and peaks at hour 8
    assert len(alerts_triggered) > 0
    assert 5 in alerts_triggered
    assert len(summary_cards) > 0

    first_card = summary_cards[0]
    assert isinstance(first_card, AlertSummaryCard)
    assert first_card.patient_id == "p000001"
    assert len(first_card.abnormal_vitals) > 0
    assert len(first_card.narrative_summary) > 20
    assert len(first_card.retrieved_chunks) > 0

    # Markdown export check
    md = first_card.to_markdown()
    assert "Clinical Alert Summary Card" in md
    assert "Abnormal Vital Parameters" in md
