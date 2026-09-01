"""
Tests for FastAPI service endpoints (Stage 5 - Person A).
"""

from fastapi.testclient import TestClient
import pytest

from product_track.api.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "WardSense" in data["system"]


def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "vector_store_ready" in data


def test_list_patients_endpoint():
    response = client.get("/patients?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "demo_candidates" in data
    assert "sample_patient_ids" in data
    assert len(data["sample_patient_ids"]) > 0


def test_score_vitals_endpoint():
    payload = {
        "patient_id": "p000001",
        "hour": 5,
        "vitals": {
            "HR": 130.0,
            "SBP": 85.0,
            "Resp": 26.0,
            "O2Sat": 89.0,
            "Temp": 39.1,
            "Glucose": 210.0,
        },
    }
    response = client.post("/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["patient_id"] == "p000001"
    assert data["is_alert"] is True
    assert data["news2_score"] >= 10
    assert len(data["abnormalities"]) >= 4


def test_ingest_patient_endpoint():
    response = client.post("/ingest", json={"patient_id": "p000001"})
    assert response.status_code == 200
    data = response.json()
    assert data["patient_id"] == "p000001"
    assert data["indexed_chunks"] > 0


def test_ingest_nonexistent_patient_returns_404():
    response = client.post("/ingest", json={"patient_id": "p_nonexistent_99999"})
    assert response.status_code == 404


def test_telemetry_step_endpoint():
    payload = {
        "patient_id": "p000001",
        "hour": 8,
        "force_summary": True,
    }
    response = client.post("/telemetry/step", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "telemetry" in data
    assert "abnormality_report" in data
    assert data["telemetry"]["hour"] == 8
    assert data["abnormality_report"]["news2_score"] >= 5


def test_query_endpoint():
    payload = {
        "patient_id": "p000001",
        "question": "What is the patient's age and initial NEWS2 score on arrival?",
        "n_chunks": 2,
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["patient_id"] == "p000001"
    assert len(data["answer"]) > 10
    assert "83" in data["answer"] or "female" in data["answer"] or "4" in data["answer"]
