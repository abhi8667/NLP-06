"""
WardSense | Patient portal.

A patient-facing view over one record. Same retrieval boundary as the clinician
console — answers come from this record and nothing else.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from product_track.interfaces import theme as T
from product_track.llm import ClinicalRAGPipeline, OllamaClient
from product_track.rag import PatientVectorStore

DATASET_DIR = REPO_ROOT / "physioNet" / "training_setA" / "training_setA"

QUICK_QUESTIONS = [
    ("Recorded vitals", "What were my recorded heart rate and blood pressure averages?"),
    ("Length of stay", "How many hours of observations are recorded in my stay?"),
    ("On admission", "What clinical baseline observations were recorded on admission?"),
]


@st.cache_resource
def get_patient_backend():
    store = PatientVectorStore(persist_dir="data/chroma_db")
    client = OllamaClient()
    pipeline = ClinicalRAGPipeline(vector_store=store, llm_client=client)
    return store, client, pipeline


def render() -> None:
    vector_store, _, rag_pipeline = get_patient_backend()

    st.sidebar.markdown(
        f'<div style="display:flex;align-items:center;gap:9px;margin-bottom:14px;">'
        f'{T.icon("users", 19, T.ACCENT)}'
        f'<div><div style="font-size:1rem;font-weight:600;color:{T.INK};">Patient portal</div>'
        f'<div style="font-family:{T.FONT_MONO};font-size:.66rem;color:{T.INK_MUTED};'
        f'letter-spacing:.06em;">PRIVATE RECORD REVIEW</div></div></div>',
        unsafe_allow_html=True,
    )

    pids = sorted(f.stem for f in DATASET_DIR.glob("*.psv"))[:30] if DATASET_DIR.exists() else []
    selected = st.sidebar.selectbox(
        "Your record ID", options=pids or ["p000001"], index=0
    )

    st.sidebar.markdown(
        f'<div class="ws-prov" style="width:100%;box-sizing:border-box;margin-top:12px;">'
        f'{T.icon("lock", 12, T.GOOD)} RECORD ISOLATION ENFORCED</div>',
        unsafe_allow_html=True,
    )

    if (
        "patient_messages" not in st.session_state
        or st.session_state.get("portal_patient") != selected
    ):
        st.session_state.portal_patient = selected
        st.session_state.patient_messages = [
            {
                "role": "assistant",
                "content": (
                    f"Hello. I can answer questions about record `{selected}` — your stay "
                    f"duration, baseline observations, and recorded vital signs. What would "
                    f"you like to know?"
                ),
            }
        ]
        path = DATASET_DIR / f"{selected}.psv"
        if path.exists() and vector_store.count_for_patient(selected) == 0:
            vector_store.index_patient_from_psv(path)

    st.markdown(
        f'<div style="margin-bottom:10px;">'
        f'<h1 style="margin:0 0 6px 0;">Your care assistant</h1>'
        f'<div style="font-size:.92rem;color:{T.INK_2};">Answers come from record '
        f'<span class="ws-num" style="color:{T.ACCENT};">{selected}</span> and nothing else.'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        T.banner(
            "Information only — not medical advice",
            "This assistant explains what is recorded in your notes. It does not diagnose and "
            "cannot recommend treatments or medication. If you feel unwell, tell your nurse or "
            "doctor straight away.",
            T.WARNING, "triangle-alert",
        ),
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    quick = None
    cols = st.columns(len(QUICK_QUESTIONS))
    for col, (label, question) in zip(cols, QUICK_QUESTIONS):
        with col:
            if st.button(label, use_container_width=True, key=f"portal_q_{label}"):
                quick = question

    for msg in st.session_state.patient_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    prompt = quick or st.chat_input("Ask a question about your records…")

    if prompt:
        st.session_state.patient_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Reviewing your records…"):
                result = rag_pipeline.answer_question(
                    patient_id=selected, question=prompt, n_chunks=3
                )
            st.write(result["answer"])
            st.session_state.patient_messages.append(
                {"role": "assistant", "content": result["answer"]}
            )


if __name__ == "__main__":
    st.set_page_config(page_title="WardSense | Patient portal", layout="centered")
    T.inject()
    render()
