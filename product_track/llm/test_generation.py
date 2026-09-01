"""
Tests for LLM generation, refusal mechanism, and end-to-end RAG pipeline (Stage 4 - Person A).
"""

from pathlib import Path
import pytest

from product_track.llm import OllamaClient, build_clinical_rag_prompt, ClinicalRAGPipeline
from product_track.rag import PatientVectorStore, RetrievedChunk

P1_PATH = Path("physioNet/training_setA/training_setA/p000001.psv")


@pytest.fixture
def ollama():
    client = OllamaClient()
    if not client.is_available():
        pytest.skip("Local Ollama is not running on localhost:11434")
    return client


@pytest.fixture
def populated_pipeline(tmp_path):
    store = PatientVectorStore(persist_dir=tmp_path / "chroma_llm_test")
    store.index_patient_from_psv(P1_PATH)
    pipeline = ClinicalRAGPipeline(vector_store=store)
    return pipeline


def test_prompt_builder_structure():
    chunks = [
        RetrievedChunk(
            content="Patient heart rate reached 110 bpm.",
            patient_id="p1",
            note_type="nursing",
            title="Nursing Note",
            chunk_id="c1",
        ),
        RetrievedChunk(
            content="NEWS2 score climbed to 9 at hour 8.",
            patient_id="p1",
            note_type="deterioration",
            title="Deterioration Record",
            chunk_id="c2",
        ),
    ]
    prompt = build_clinical_rag_prompt(chunks, "What was the heart rate and NEWS2 score?")
    assert "PATIENT CONTEXT:" in prompt
    assert "QUESTION:" in prompt
    assert "RESPONSE:" in prompt
    assert "Do not infer, assume, or generate information not present in the context" in prompt
    assert "Patient heart rate reached 110 bpm" in prompt
    assert "NEWS2 score climbed to 9 at hour 8" in prompt
    assert "---" in prompt


def test_ollama_client_list_models(ollama):
    models = ollama.list_models()
    assert len(models) > 0
    # Check that at least one supported llama model is present
    assert any("llama" in m for m in models)


def test_refusal_fires_on_out_of_scope_question(ollama, populated_pipeline):
    """
    Test that asking about information not in the record causes the refusal clause to trigger.
    """
    out_of_scope_q = "What is the patient's home address, social security number, and pet's name?"
    result = populated_pipeline.answer_question(
        patient_id="p000001",
        question=out_of_scope_q,
    )
    answer = result["answer"].lower()
    
    # Assert that the model acknowledges the absence of information rather than hallucinating
    refusal_keywords = ["not", "does not contain", "no information", "unavailable", "not present", "not mentioned", "not recorded"]
    assert any(kw in answer for kw in refusal_keywords), f"Model failed to refuse out-of-scope query: {result['answer']}"


def test_end_to_end_grounded_answer(ollama, populated_pipeline):
    """
    Test grounded factual answer on patient p000001 (known peak NEWS2 9, 83.1-year-old female).
    """
    question = "What is the patient's age and sex, and did they experience acute clinical deterioration?"
    result = populated_pipeline.answer_question(
        patient_id="p000001",
        question=question,
    )

    assert result["patient_id"] == "p000001"
    assert len(result["retrieved_chunks"]) > 0
    assert result["eval_rate_tok_s"] > 0
    
    ans = result["answer"].lower()
    # Factual grounding checks
    assert "83" in ans
    assert "female" in ans
    assert "deteriorat" in ans or "news2" in ans or "9" in ans or "threshold" in ans
