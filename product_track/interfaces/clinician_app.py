import sys
from pathlib import Path
import time

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import streamlit as st

from product_track.bridge import AlertBridge, AlertSummaryCard, VitalsReplayHarness, select_demo_patients
from product_track.llm import ClinicalRAGPipeline, OllamaClient
from product_track.rag import PatientVectorStore

# Page Configuration
st.set_page_config(
    page_title="WardSense | Clinician Deterioration Workstation",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATASET_DIR = Path("physioNet/training_setA/training_setA")


@st.cache_resource
def get_backend_services():
    store = PatientVectorStore(persist_dir="data/chroma_db")
    client = OllamaClient()
    bridge = AlertBridge(vector_store=store, llm_client=client)
    pipeline = ClinicalRAGPipeline(vector_store=store, llm_client=client)
    harness = VitalsReplayHarness()
    return store, client, bridge, pipeline, harness


vector_store, llm_client, alert_bridge, rag_pipeline, replay_harness = get_backend_services()

# --- Custom Styling ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #0d6efd;
    }
    .alert-box-high {
        background-color: #f8d7da;
        color: #842029;
        border: 1px solid #f5c2c7;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
    }
    .alert-box-med {
        background-color: #fff3cd;
        color: #664d03;
        border: 1px solid #ffecb5;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar Controls ---
st.sidebar.title("🏥 WardSense Workstation")
st.sidebar.caption("On-Premise Clinical AI & Alert-to-Summary Bridge (C3)")

# Patient Selection
demo_patients = select_demo_patients(DATASET_DIR, min_peak=7, max_candidates=5) if DATASET_DIR.exists() else []
demo_ids = [d["patient_id"] for d in demo_patients]
all_files = sorted(DATASET_DIR.glob("*.psv"))[:30] if DATASET_DIR.exists() else []
all_pids = [f.stem for f in all_files]

selected_patient = st.sidebar.selectbox(
    "Select Monitored Patient:",
    options=all_pids,
    index=0 if all_pids else 0,
    help="Patients marked with high deterioration are listed in demo candidates.",
)

if selected_patient in demo_ids:
    st.sidebar.success("⭐ Prominent Demo Candidate (Calm → Deterioration)")

# Initialize Session State for Patient
if "current_hour" not in st.session_state or st.session_state.get("active_patient") != selected_patient:
    st.session_state.active_patient = selected_patient
    st.session_state.current_hour = 0
    st.session_state.alerts_history = []
    st.session_state.telemetry_history = []
    st.session_state.chat_history = []

    # Ensure patient is indexed in ChromaDB
    psv_path = DATASET_DIR / f"{selected_patient}.psv"
    if psv_path.exists() and vector_store.count_for_patient(selected_patient) == 0:
        vector_store.index_patient_from_psv(psv_path)

psv_file = DATASET_DIR / f"{selected_patient}.psv"
all_readings = list(replay_harness.stream_patient(psv_file, interval_s=0.0)) if psv_file.exists() else []
max_available_hours = len(all_readings)

# Replay Controls
st.sidebar.subheader("⏱️ Bedside Telemetry Replay")
st.sidebar.caption("Simulates real-time monitor feeds from recorded ICU stays.")

col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    if st.button("▶ Step 1 Hour", use_container_width=True):
        if st.session_state.current_hour < max_available_hours - 1:
            st.session_state.current_hour += 1

with col_btn2:
    if st.button("🔄 Reset Stay", use_container_width=True):
        st.session_state.current_hour = 0
        st.session_state.alerts_history = []

target_hour = st.sidebar.slider(
    "Jump to ICU Hour:",
    min_value=0,
    max_value=max(0, max_available_hours - 1),
    value=st.session_state.current_hour,
)
if target_hour != st.session_state.current_hour:
    st.session_state.current_hour = target_hour

# --- Main Dashboard ---
st.title(f"Bedside Telemetry Monitor — Patient `{selected_patient}`")

if not all_readings:
    st.error(f"No records found for patient {selected_patient}")
    st.stop()

# Current Telemetry
cur_tel = all_readings[st.session_state.current_hour]
vitals = cur_tel.vitals

# Score Telemetry
report, card = alert_bridge.process_telemetry(
    patient_id=selected_patient,
    hour=cur_tel.hour,
    window=cur_tel.window_buffer,
    current_vitals=vitals,
)

# Metric Row
col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
with col1:
    st.metric("ICU Hour", f"{cur_tel.hour:02d} / {max_available_hours:02d}")
with col2:
    delta_color = "inverse" if report.news2_score >= 5 else "normal"
    st.metric("NEWS2 Score", f"{report.news2_score}/15", delta=f"{report.risk_band} Risk", delta_color=delta_color)
with col3:
    st.metric("Heart Rate", f"{vitals.get('HR', 0):.0f} bpm", delta="Normal: 51-90", delta_color="off")
with col4:
    st.metric("Systolic BP", f"{vitals.get('SBP', 0):.0f} mmHg", delta="Normal: 111-219", delta_color="off")
with col5:
    st.metric("SpO2", f"{vitals.get('O2Sat', 0):.0f} %", delta="Normal: ≥96%", delta_color="off")
with col6:
    st.metric("Respiration", f"{vitals.get('Resp', 0):.0f} /min", delta="Normal: 12-20", delta_color="off")
with col7:
    st.metric("Temp", f"{vitals.get('Temp', 0):.1f} °C", delta="Normal: 36.1-38", delta_color="off")

st.divider()

# --- Active Alert & Bridge Section (Contribution C3) ---
st.subheader("🚨 Clinical Anomaly Detection & Alert-to-Summary Bridge")

if report.is_alert:
    alert_class = "alert-box-high" if report.news2_score >= 7 else "alert-box-med"
    st.markdown(f"""
    <div class="{alert_class}">
        <h4>⚠️ ACUTE DETERIORATION ALERT DETECTED (ICU Hour {cur_tel.hour})</h4>
        <p><strong>Standardized Escalation:</strong> {report.recommended_response}</p>
        <p><strong>Neural Anomaly Probability:</strong> {report.risk_score:.1%}</p>
    </div>
    """, unsafe_allow_html=True)

    if card:
        tab1, tab2 = st.tabs(["📋 Clinical Summary Card", "🔍 Retrieved Record Evidence"])
        with tab1:
            st.markdown(card.to_markdown())
        with tab2:
            st.write("Targeted Semantic Query synthesized from abnormalities:")
            st.code(report.summary_query(), language="text")
            st.write("Retrieved Patient-Scoped Chunks:")
            for chunk in card.retrieved_chunks:
                st.info(f"**[{chunk['title']}]** (Doc ID: `{chunk['chunk_id']}`)\n\n{chunk['content']}")
else:
    st.success(f"✅ Physiological Stability: NEWS2 score is {report.news2_score} (below deterioration threshold 5). Routine monitoring in effect.")

st.divider()

# --- Follow-Up Interactive Assistant ---
st.subheader("💬 Interactive Clinical Q&A Assistant")
st.caption("Ask questions about this patient's medical history, lab panels, or deterioration trajectory. Answers are grounded 100% in this patient's record.")

user_q = st.text_input(
    "Clinical Inquiry:",
    placeholder="e.g. What baseline conditions or laboratory findings were noted on admission?",
    key="clinician_q_input",
)

if user_q:
    with st.spinner("Retrieving patient records and synthesizing answer..."):
        ans_res = rag_pipeline.answer_question(
            patient_id=selected_patient,
            question=user_q,
        )
        st.markdown(f"**Response:**\n\n{ans_res['answer']}")
        st.caption(f"Generated via {ans_res['model']} in {ans_res['total_duration_s']}s ({ans_res['eval_rate_tok_s']} tok/s) | 100% Patient Isolated")
