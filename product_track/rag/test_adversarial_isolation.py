"""
Adversarial Patient Isolation Security Tests (Stage 3 - Person A).

Verifies that patient records in ChromaDB are strictly isolated and that cross-patient
data leakage is impossible, even under adversarial queries.
"""

from pathlib import Path
import pytest

from product_track.knowledge_base import ClinicalNote, generate_patient_notes
from product_track.rag import PatientVectorStore, RetrievedChunk, chunk_text

P1_PATH = Path("physioNet/training_setA/training_setA/p000001.psv")
P2_PATH = Path("physioNet/training_setA/training_setA/p000002.psv")


@pytest.fixture
def isolated_store(tmp_path):
    """Creates a temporary isolated persistent vector store."""
    return PatientVectorStore(persist_dir=tmp_path / "chroma_test")


def test_chunking_utility():
    short_text = "Patient arrived with elevated pulse."
    assert chunk_text(short_text, chunk_size=100) == [short_text]

    long_text = "\n\n".join([f"Paragraph {i}: Detailed medical observation text." for i in range(10)])
    chunks = chunk_text(long_text, chunk_size=150)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 250


def test_adversarial_cross_patient_token_leakage(isolated_store):
    """
    Direct adversarial security test:
    Inject distinct high-entropy canary tokens into Patient A and Patient B.
    Verify that querying Patient A for Patient B's canary token returns an EMPTY result
    under distance threshold and NEVER returns Patient B's content.
    """
    token_a = "CANARY_TOKEN_PATIENT_A_SECRET_HYPERTENSION_109283"
    token_b = "CANARY_TOKEN_PATIENT_B_SECRET_SEPSIS_771829"

    note_a = ClinicalNote(
        patient_id="patient_A",
        note_type="admission",
        title="Admission Note Patient A",
        content=f"Confidential clinical assessment: {token_a}.",
    )
    note_b = ClinicalNote(
        patient_id="patient_B",
        note_type="admission",
        title="Admission Note Patient B",
        content=f"Confidential clinical assessment: {token_b}.",
    )

    isolated_store.index_patient_notes("patient_A", [note_a])
    isolated_store.index_patient_notes("patient_B", [note_b])

    # 1. Adversarial query: Query Patient A for Patient B's canary token with distance cutoff
    res_a_for_b = isolated_store.query(
        patient_id="patient_A",
        query_text=token_b,
        n_results=4,
        distance_threshold=0.5,
    )
    # MUST BE EMPTY (not merely different)
    assert len(res_a_for_b) == 0, f"Leakage detected! Patient A retrieved data: {res_a_for_b}"

    # 2. Adversarial query without threshold: verify Patient B's token NEVER leaks into Patient A results
    res_a_raw = isolated_store.query(
        patient_id="patient_A",
        query_text=token_b,
        n_results=4,
    )
    for chunk in res_a_raw:
        assert token_b not in chunk.content
        assert chunk.patient_id == "patient_A"

    # 3. Adversarial query: Query Patient B for Patient A's canary token with distance cutoff
    res_b_for_a = isolated_store.query(
        patient_id="patient_B",
        query_text=token_a,
        n_results=4,
        distance_threshold=0.5,
    )
    # MUST BE EMPTY
    assert len(res_b_for_a) == 0, f"Leakage detected! Patient B retrieved data: {res_b_for_a}"

    # 4. Legitimate queries: Each patient retrieves their own token
    res_a_own = isolated_store.query("patient_A", token_a, n_results=1, distance_threshold=0.5)
    assert len(res_a_own) == 1
    assert token_a in res_a_own[0].content
    assert res_a_own[0].patient_id == "patient_A"

    res_b_own = isolated_store.query("patient_B", token_b, n_results=1, distance_threshold=0.5)
    assert len(res_b_own) == 1
    assert token_b in res_b_own[0].content
    assert res_b_own[0].patient_id == "patient_B"


def test_real_physionet_patients_isolation(isolated_store):
    """
    Index real PhysioNet patients p000001 and p000002.
    Assert that all retrieved chunks match the queried patient_id.
    """
    pid1, count1 = isolated_store.index_patient_from_psv(P1_PATH)
    pid2, count2 = isolated_store.index_patient_from_psv(P2_PATH)

    assert pid1 == "p000001" and count1 > 0
    assert pid2 == "p000002" and count2 > 0

    # Query for vital signs on Patient 1
    res1 = isolated_store.query("p000001", "What was the peak NEWS2 score and heart rate?", n_results=5)
    assert len(res1) > 0
    for chunk in res1:
        assert chunk.patient_id == "p000001"
        assert "p000002" not in chunk.content

    # Query for laboratory tests on Patient 2
    res2 = isolated_store.query("p000002", "What diagnostic labs were ordered?", n_results=5)
    for chunk in res2:
        assert chunk.patient_id == "p000002"
        assert "p000001" not in chunk.content


def test_nonexistent_patient_returns_empty(isolated_store):
    """Querying an unknown patient returns an empty list without crashing."""
    res = isolated_store.query("patient_nonexistent_99999", "Any clinical findings?", n_results=4)
    assert res == []


def test_invalid_patient_id_raises_error(isolated_store):
    """Invalid or empty patient IDs must raise an explicit exception."""
    with pytest.raises(ValueError):
        isolated_store.query("", "query text")

    with pytest.raises(ValueError):
        isolated_store.index_patient_notes("", [])


def test_clear_patient_preserves_other_patients(isolated_store):
    """Clearing one patient does not affect other patients in the store."""
    token_a = "UNIQUE_ALPHA_RECORD"
    token_b = "UNIQUE_BETA_RECORD"

    note_a = ClinicalNote(patient_id="patient_A", note_type="admission", title="A", content=token_a)
    note_b = ClinicalNote(patient_id="patient_B", note_type="admission", title="B", content=token_b)

    isolated_store.index_patient_notes("patient_A", [note_a])
    isolated_store.index_patient_notes("patient_B", [note_b])

    assert isolated_store.count_for_patient("patient_A") >= 1
    assert isolated_store.count_for_patient("patient_B") >= 1

    # Clear Patient A
    isolated_store.clear_patient("patient_A")

    assert isolated_store.count_for_patient("patient_A") == 0
    assert isolated_store.count_for_patient("patient_B") >= 1
    assert len(isolated_store.query("patient_B", token_b)) >= 1
