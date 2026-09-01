"""
FastAPI REST API Service for WardSense Local AI Clinical System.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from product_track.bridge import AlertBridge, AlertSummaryCard, VitalsReplayHarness, select_demo_patients
from product_track.knowledge_base import ClinicalNote, generate_patient_notes
from product_track.llm import ClinicalRAGPipeline, OllamaClient
from product_track.rag import PatientVectorStore

# Shared singleton instances for API
app = FastAPI(
    title="WardSense Local Clinical AI API",
    description="Privacy-preserving local clinical decision support system with Alert-to-Summary Bridge (Contribution C3)",
    version="1.0.0",
)

DATASET_DIR = Path("physioNet/training_setA/training_setA")
vector_store = PatientVectorStore(persist_dir="data/chroma_db")
llm_client = OllamaClient()
rag_pipeline = ClinicalRAGPipeline(vector_store=vector_store, llm_client=llm_client)
alert_bridge = AlertBridge(vector_store=vector_store, llm_client=llm_client)
replay_harness = VitalsReplayHarness()


# --- Request & Response Schemas ---

class HealthResponse(BaseModel):
    status: str
    ollama_connected: bool
    models_available: list[str]
    vector_store_ready: bool


class QueryRequest(BaseModel):
    patient_id: str = Field(..., description="Target patient identifier (e.g. p000001)")
    question: str = Field(..., description="Plain-English clinical question")
    n_chunks: int = Field(default=4, ge=1, le=10)


class QueryResponse(BaseModel):
    patient_id: str
    question: str
    answer: str
    model: str
    latency_s: float
    tokens_per_s: float
    retrieved_chunks_count: int


class ScoreRequest(BaseModel):
    patient_id: str
    hour: int = 0
    vitals: dict[str, float] = Field(..., description="Dictionary of vitals: HR, SBP, Resp, O2Sat, Temp, Glucose")


class TelemetryStepRequest(BaseModel):
    patient_id: str
    hour: int
    force_summary: bool = False


class IngestRequest(BaseModel):
    patient_id: str


# --- Endpoints ---

@app.get("/", tags=["General"])
def root() -> dict[str, str]:
    return {
        "system": "WardSense Local Clinical AI System",
        "description": "On-premise clinical monitoring with Alert-to-Summary Bridge",
        "version": "1.0.0",
        "documentation": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check() -> HealthResponse:
    ollama_ok = llm_client.is_available()
    models = llm_client.list_models() if ollama_ok else []
    return HealthResponse(
        status="healthy",
        ollama_connected=ollama_ok,
        models_available=models,
        vector_store_ready=True,
    )


@app.get("/patients", tags=["Cohort"])
def list_patients(limit: int = 20) -> dict[str, Any]:
    """List available demo patients with prominence flags."""
    if not DATASET_DIR.exists():
        return {"patients": [], "total_count": 0}

    demo_candidates = select_demo_patients(DATASET_DIR, min_peak=7, max_candidates=5)
    all_files = sorted(DATASET_DIR.glob("*.psv"))[:limit]
    patient_ids = [f.stem for f in all_files]

    return {
        "total_available": len(list(DATASET_DIR.glob("*.psv"))),
        "demo_candidates": demo_candidates,
        "sample_patient_ids": patient_ids,
    }


@app.post("/ingest", tags=["Knowledge Base"])
def ingest_patient(req: IngestRequest) -> dict[str, Any]:
    """Ingest and index notes for a patient into the vector store."""
    psv_file = DATASET_DIR / f"{req.patient_id}.psv"
    if not psv_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient record {req.patient_id}.psv not found in dataset directory.",
        )

    pid, count = vector_store.index_patient_from_psv(psv_file)
    return {
        "status": "success",
        "patient_id": pid,
        "indexed_chunks": count,
    }


@app.post("/score", tags=["Detection"])
def score_vitals(req: ScoreRequest) -> dict[str, Any]:
    """Score incoming vitals observation and detect abnormalities."""
    report = alert_bridge.risk_scorer.score_window(
        patient_id=req.patient_id,
        hour=req.hour,
        window=None,
        current_vitals=req.vitals,
    )
    return report.to_dict()


@app.post("/query", response_model=QueryResponse, tags=["Assistant"])
def query_patient(req: QueryRequest) -> QueryResponse:
    """Ask a grounded question scoped strictly to a patient's record."""
    # Ensure patient is indexed if not yet loaded
    if vector_store.count_for_patient(req.patient_id) == 0:
        psv_file = DATASET_DIR / f"{req.patient_id}.psv"
        if psv_file.exists():
            vector_store.index_patient_from_psv(psv_file)

    res = rag_pipeline.answer_question(
        patient_id=req.patient_id,
        question=req.question,
        n_chunks=req.n_chunks,
    )

    return QueryResponse(
        patient_id=res["patient_id"],
        question=res["question"],
        answer=res["answer"],
        model=res["model"],
        latency_s=res["total_duration_s"],
        tokens_per_s=res["eval_rate_tok_s"],
        retrieved_chunks_count=len(res["retrieved_chunks"]),
    )


@app.post("/telemetry/step", tags=["Bridge"])
def step_telemetry(req: TelemetryStepRequest) -> dict[str, Any]:
    """
    Step a patient's recorded telemetry by 1 hour.
    If deterioration occurs, automatically returns the AlertSummaryCard (Contribution C3).
    """
    psv_file = DATASET_DIR / f"{req.patient_id}.psv"
    if not psv_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient record {req.patient_id}.psv not found.",
        )

    # Ensure patient vector store is initialized
    if vector_store.count_for_patient(req.patient_id) == 0:
        vector_store.index_patient_from_psv(psv_file)

    # Stream up to target hour
    readings = list(replay_harness.stream_patient(psv_file, interval_s=0.0, max_hours=req.hour + 1))
    if not readings or req.hour >= len(readings):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Hour {req.hour} exceeds monitored stay length ({len(readings)} hours).",
        )

    target_telemetry = readings[req.hour]
    report, card = alert_bridge.process_telemetry(
        patient_id=req.patient_id,
        hour=target_telemetry.hour,
        window=target_telemetry.window_buffer,
        current_vitals=target_telemetry.vitals,
        force_generate_summary=req.force_summary,
    )

    return {
        "telemetry": target_telemetry.to_dict(),
        "abnormality_report": report.to_dict(),
        "alert_card": card.to_dict() if card else None,
    }
