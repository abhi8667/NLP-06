"""
WardSense | Patient Care Portal.
Grounded, privacy-first patient assistant interface for personal health review.
"""

from __future__ import annotations

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

# Clinical Dark Theme CSS
PATIENT_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Outfit:wght@500;600;700&display=swap');

    .stApp {
        background-color: #080E14 !important;
        color: #E2E8F0 !important;
        font-family: 'Inter', sans-serif !important;
    }

    [data-testid="stSidebar"] {
        background-color: #0B131B !important;
        border-right: 1px solid rgba(0, 229, 255, 0.12) !important;
    }

    h1, h2, h3 {
        font-family: 'Outfit', sans-serif !important;
        letter-spacing: -0.02em;
    }

    .disclaimer-card {
        background: rgba(255, 179, 0, 0.1);
        border-left: 4px solid #FFB300;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 20px;
        font-size: 0.88rem;
        color: #CBD5E1;
        line-height: 1.5;
    }

    .privacy-badge {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        background: rgba(0, 230, 118, 0.12);
        color: #00E676;
        border: 1px solid rgba(0, 230, 118, 0.3);
        padding: 4px 10px;
        border-radius: 4px;
        margin-bottom: 12px;
    }

    .stChatMessage {
        background: #111C26 !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 8px !important;
        margin-bottom: 10px !important;
    }
</style>
"""
st.markdown(PATIENT_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_patient_backend():
    store = PatientVectorStore(persist_dir="data/chroma_db")
    client = OllamaClient()
    pipeline = ClinicalRAGPipeline(vector_store=store, llm_client=client)
    return store, client, pipeline


vector_store, llm_client, rag_pipeline = get_patient_backend()

# --- Sidebar: Patient Selection ---
st.sidebar.markdown("""
<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
    <span style="font-size: 1.6rem;">🩺</span>
    <div>
        <h3 style="margin: 0; font-size: 1.15rem; color: #00E5FF;">Patient Care Portal</h3>
        <span style="font-size: 0.7rem; color: #64748B;">Private Health Record Review</span>
    </div>
</div>
""", unsafe_allow_html=True)

all_files = sorted(DATASET_DIR.glob("*.psv"))[:30] if DATASET_DIR.exists() else []
all_pids = [f.stem for f in all_files]

selected_patient = st.sidebar.selectbox(
    "Select Your Medical Record ID:",
    options=all_pids if all_pids else ["p000001"],
    index=0 if all_pids else 0,
)

st.sidebar.markdown("""
<div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #94A3B8; background: #0D1620; padding: 10px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.06); margin-top: 15px;">
    🔒 <b>Strict Record Isolation:</b><br>
    Queries are answered solely using this record. No data leaves the local hospital perimeter.
</div>
""", unsafe_allow_html=True)

# Initialize Session Chat History
if "patient_messages" not in st.session_state or st.session_state.get("current_patient") != selected_patient:
    st.session_state.current_patient = selected_patient
    st.session_state.patient_messages = [
        {
            "role": "assistant",
            "content": f"Hello! I am your personal health care assistant for record `{selected_patient}`. I can help explain your stay duration, baseline observations, and recorded vital signs. What would you like to know?",
        }
    ]

    # Ensure patient is indexed
    psv_path = DATASET_DIR / f"{selected_patient}.psv"
    if psv_path.exists() and vector_store.count_for_patient(selected_patient) == 0:
        vector_store.index_patient_from_psv(psv_path)

# --- Header & Medical Disclaimer ---
st.markdown(f"""
<div style="margin-bottom: 6px;">
    <span class="privacy-badge">🔒 ON-PREMISES PRIVATE HEALTH ASSISTANT · RECORD {selected_patient}</span>
    <h1 style="margin: 0; font-size: 1.8rem; color: #F8FAFC;">Personal Inpatient Care Assistant</h1>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-card">
    <strong>⚠️ Medical Information Disclaimer:</strong> This assistant provides information derived strictly from your local hospital records for educational review. It does NOT provide medical diagnoses and cannot prescribe treatments or medications. If you feel unwell or have medical concerns, notify your nurse or attending physician immediately.
</div>
""", unsafe_allow_html=True)

# Quick Query Chips
q_c1, q_c2, q_c3 = st.columns(3)
quick_q = None
with q_c1:
    if st.button("📊 My Average Vitals", use_container_width=True):
        quick_q = "What were my recorded heart rate and blood pressure averages?"
with q_c2:
    if st.button("⏱️ Length of Stay", use_container_width=True):
        quick_q = "How many hours of observations are recorded in my stay?"
with q_c3:
    if st.button("🩺 Admission Findings", use_container_width=True):
        quick_q = "What clinical baseline observations were recorded on admission?"

# --- Chat Display ---
for msg in st.session_state.patient_messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- Chat Input ---
chat_input = st.chat_input("Ask a question about your hospital records...")
prompt = quick_q or chat_input

if prompt:
    st.session_state.patient_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

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
