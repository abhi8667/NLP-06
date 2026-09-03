"""
WardSense — application entry point.

    streamlit run product_track/interfaces/app.py

Owns the one page config and the one theme injection; every page below is a
plain render() function so state is shared without file-path indirection.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

st.set_page_config(
    page_title="WardSense",
    layout="wide",
    initial_sidebar_state="expanded",
)

from product_track.interfaces import (  # noqa: E402  (must follow set_page_config)
    assurance_page,
    clinician_app,
    overview_page,
    patient_app,
    theme as T,
)

T.inject()

overview = st.Page(overview_page.render, title="Overview", url_path="overview", default=True)
console = st.Page(clinician_app.render, title="Live console", url_path="console")
assurance = st.Page(assurance_page.render, title="Assurance", url_path="assurance")
portal = st.Page(patient_app.render, title="Patient portal", url_path="portal")

# The overview's "Start guided demo" button needs a handle on the console page.
st.session_state["_ws_console_page"] = console

st.navigation(
    {
        "Present": [overview, console],
        "Verify": [assurance],
        "Patient-facing": [portal],
    }
).run()
