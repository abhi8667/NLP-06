import sys
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from product_track.llm import ClinicalRAGPipeline, OllamaClient
from product_track.rag import PatientVectorStore

# Page Configuration
st.set_page_config(
    page_title="WardSense | Patient Care Portal",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded",
)

DATASET_DIR = Path("physioNet/training_setA/training_setA")


@st.cache_resource
def get_patient_backend():
    store = PatientVectorStore(persist_dir="data/chroma_db")
    client = OllamaClient()
    pipeline = ClinicalRAGPipeline(vector_store=store, llm_client=client)
    return store, client, pipeline


vector_store, llm_client, rag_pipeline = get_patient_backend()

# --- Sidebar: Patient Selection ---
st.sidebar.title("🩺 Patient Portal")
all_files = sorted(DATASET_DIR.glob("*.psv"))[:30] if DATASET_DIR.exists() else []
all_pids = [f.stem for f in all_files]

selected_patient = st.sidebar.selectbox(
    "Select Your Patient ID:",
    options=all_pids,
    index=0 if all_pids else 0,
)

# Initialize Session Chat History
if "patient_messages" not in st.session_state or st.session_state.get("current_patient") != selected_patient:
    st.session_state.current_patient = selected_patient
    st.session_state.patient_messages = [
        {
            "role": "assistant",
            "content": f"Hello! I am your local care assistant for patient record `{selected_patient}`. I can help explain your stay duration, baseline observations, and recorded vital signs. What would you like to know?",
        }
    ]

    # Ensure patient is indexed
    psv_path = DATASET_DIR / f"{selected_patient}.psv"
    if psv_path.exists() and vector_store.count_for_patient(selected_patient) == 0:
        vector_store.index_patient_from_psv(psv_path)

# --- Header & Medical Disclaimer ---
st.title("Patient Care Assistant")
st.warning(
    "⚠️ **Important Medical Disclaimer:** This assistant provides information derived strictly from your hospital records for educational and personal review. "
    "It is NOT a medical diagnosis and cannot prescribe treatments or medications. If you feel unwell or have medical concerns, please consult your doctor or nurse immediately."
)

st.caption(f"Currently viewing records for: **Patient {selected_patient}** (100% Private & Locally Grounded)")

# --- Chat Display ---
for msg in st.session_state.patient_messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- Chat Input ---
if prompt := st.chat_input("Ask a question about your hospital records..."):
    # Add user message
    st.session_state.patient_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Reviewing your local health records..."):
            ans_res = rag_pipeline.answer_question(
                patient_id=selected_patient,
                question=prompt,
                n_chunks=3,
            )
            response_text = ans_res["answer"]
            st.write(response_text)
            st.session_state.patient_messages.append({"role": "assistant", "content": response_text})
