"""
WardSense | Clinician console.

Two views over the same replayed stay:

  * **Ward** — a triage list ordered by detector risk, not a grid of equal tiles.
    A ward board's job is to answer "who is deteriorating"; ordering is itself
    information, and six identical cards throw that away.
  * **Bedside** — traces on the left, current state and the alert on the right,
    so the alert sits beside the data that caused it rather than a screen below.

Presentation only — scoring, abnormality extraction, retrieval and generation all
belong to product_track.bridge / rag / llm.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from product_track.bridge import AlertBridge, AlertSummaryCard, VitalsReplayHarness
from product_track.interfaces import charts, demo_script, theme as T
from product_track.interfaces.metrics_source import (
    load_evaluation_metrics,
    scenario_hallucination_rate,
)
from product_track.llm import ClinicalRAGPipeline, OllamaClient
from product_track.rag import PatientVectorStore

DATASET_DIR = REPO_ROOT / "physioNet" / "training_setA" / "training_setA"
ALERT_THRESHOLD = 5

WARD_BEDS = [
    {"bed": "101", "pid": "p000001", "age": 83, "sex": "F", "condition": "Acute sepsis trajectory"},
    {"bed": "102", "pid": "p000003", "age": 67, "sex": "M", "condition": "Post-cardiac ICU"},
    {"bed": "103", "pid": "p000008", "age": 74, "sex": "M", "condition": "Respiratory failure risk"},
    {"bed": "104", "pid": "p000009", "age": 59, "sex": "F", "condition": "Hemodynamic deterioration"},
    {"bed": "105", "pid": "p000014", "age": 62, "sex": "M", "condition": "Post-op recovery (control)"},
    {"bed": "106", "pid": "p000018", "age": 81, "sex": "F", "condition": "Early shock indicator"},
]

VITAL_STRIP = [
    ("HR", "Heart rate", "bpm", 51, 90, 0),
    ("SBP", "Systolic BP", "mmHg", 111, 219, 0),
    ("O2Sat", "SpO₂", "%", 96, 100, 0),
    ("Resp", "Respiration", "/min", 12, 20, 0),
    ("Temp", "Temperature", "°C", 36.1, 38.0, 1),
]


# -----------------------------------------------------------------------------
# Backend services and cached reads
# -----------------------------------------------------------------------------
@st.cache_resource
def get_services():
    store = PatientVectorStore(persist_dir="data/chroma_db")
    client = OllamaClient()
    bridge = AlertBridge(vector_store=store, llm_client=client)
    pipeline = ClinicalRAGPipeline(vector_store=store, llm_client=client)
    harness = VitalsReplayHarness()
    return store, client, bridge, pipeline, harness


@st.cache_data(show_spinner=False)
def load_stay(patient_id: str, max_hours: int | None = None):
    _, _, _, _, harness = get_services()
    path = DATASET_DIR / f"{patient_id}.psv"
    if not path.exists():
        return []
    return list(harness.stream_patient(path, interval_s=0.0, max_hours=max_hours))


@st.cache_data(show_spinner=False)
def score_hour(patient_id: str, hour: int):
    """Deterministic score for one hour, no summary. Cached per (patient, hour)."""
    _, _, bridge, _, _ = get_services()
    readings = load_stay(patient_id)
    if not readings or hour >= len(readings):
        return None
    reading = readings[hour]
    report, _ = bridge.process_telemetry(
        patient_id=patient_id,
        hour=reading.hour,
        window=reading.window_buffer,
        current_vitals=reading.vitals,
        force_generate_summary=False,
    )
    return report


@st.cache_data(show_spinner="Retrieving history and generating summary…")
def summary_card(patient_id: str, hour: int) -> AlertSummaryCard | None:
    _, _, bridge, _, _ = get_services()
    report = score_hour(patient_id, hour)
    if report is None:
        return None
    return bridge.generate_summary_card(report)


def risk_threshold() -> float:
    """The detector's calibrated alert threshold (tau)."""
    _, _, bridge, _, _ = get_services()
    tau = getattr(bridge.risk_scorer, "alert_threshold_prob", 0.5)
    # Without a checkpoint the scorer parks tau above 1.0 to disable neural alerting.
    return tau if 0.0 < tau <= 1.0 else 0.5


def risk_colour(risk: float, tau: float) -> str:
    """Colour a risk value against its own threshold — never against the NEWS2 band."""
    if risk >= tau:
        return T.VERMILION
    if risk >= tau * 0.6:
        return T.AMBER
    return T.TRACE


def ensure_indexed(patient_id: str) -> None:
    store, *_ = get_services()
    path = DATASET_DIR / f"{patient_id}.psv"
    if path.exists() and store.count_for_patient(patient_id) == 0:
        store.index_patient_from_psv(path)


# -----------------------------------------------------------------------------
# State
# -----------------------------------------------------------------------------
def init_state() -> None:
    st.session_state.setdefault("active_patient", "p000001")
    st.session_state.setdefault("current_hour", 0)
    st.session_state.setdefault("is_playing", False)
    st.session_state.setdefault("play_speed", 1.0)
    st.session_state.setdefault("show_table", False)
    demo_script.init_state()


def available_patients() -> list[str]:
    if not DATASET_DIR.exists():
        return [b["pid"] for b in WARD_BEDS]
    pids = sorted(f.stem for f in DATASET_DIR.glob("*.psv"))
    return pids[:40] if pids else [b["pid"] for b in WARD_BEDS]


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
def render_sidebar(max_hours: int) -> None:
    st.sidebar.markdown(
        f'<div style="margin-bottom:16px;">'
        f'<div class="ws-display" style="font-size:{T.TYPE["xl"]};">WardSense</div>'
        f'<div style="font-family:{T.FONT_MONO};font-size:{T.TYPE["xs"]};'
        f'color:{T.INK_MUTED};letter-spacing:.14em;margin-top:2px;">ICU CENTRAL STATION</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        f'<div class="ws-prov" style="width:100%;box-sizing:border-box;">'
        f'{T.icon("lock", 12, T.INK_2)} LOCAL INFERENCE · NO EGRESS</div>',
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
    st.sidebar.markdown('<div class="ws-kpi-l">Guided demo</div>', unsafe_allow_html=True)

    labels = {k: s.label for k, s in demo_script.SCENARIOS.items()}
    keys = list(labels)
    chosen = st.sidebar.selectbox(
        "Scenario", options=keys, format_func=lambda k: labels[k],
        index=keys.index(st.session_state.get("demo_scenario", demo_script.DEFAULT_SCENARIO)),
        label_visibility="collapsed",
    )
    st.sidebar.markdown(
        f'<div style="font-family:{T.FONT_DISPLAY};font-style:italic;font-size:{T.TYPE["sm"]};'
        f'color:{T.INK_MUTED};line-height:1.5;margin:6px 0 10px 0;">'
        f'{demo_script.SCENARIOS[chosen].premise}</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.demo_active:
        if st.sidebar.button("Exit demo mode", use_container_width=True, key="sb_exit_demo"):
            demo_script.stop()
            st.rerun()
    else:
        if st.sidebar.button("Start guided demo", type="primary",
                             use_container_width=True, key="sb_start_demo"):
            demo_script.start(chosen)
            st.rerun()

    st.sidebar.divider()

    pids = available_patients()
    current = st.session_state.active_patient
    selected = st.sidebar.selectbox(
        "Monitored patient", options=pids,
        index=pids.index(current) if current in pids else 0,
        disabled=st.session_state.demo_active,
        help="Locked while a guided demo is running." if st.session_state.demo_active else None,
    )
    if selected != st.session_state.active_patient and not st.session_state.demo_active:
        st.session_state.active_patient = selected
        st.session_state.current_hour = 0
        st.session_state.is_playing = False
        st.rerun()

    st.sidebar.divider()
    st.sidebar.markdown(
        f'<div class="ws-kpi-l">Telemetry replay</div>'
        f'<div style="font-family:{T.FONT_DISPLAY};font-style:italic;'
        f'font-size:{T.TYPE["sm"]};color:{T.INK_MUTED};margin-bottom:10px;">'
        f'A recorded stay, played back one hour at a time.</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.sidebar.columns(2)
    with c1:
        if st.session_state.is_playing:
            if st.button("Pause", use_container_width=True, key="sb_pause"):
                st.session_state.is_playing = False
                st.rerun()
        else:
            if st.button("Play", type="primary", use_container_width=True, key="sb_play"):
                st.session_state.is_playing = True
                st.rerun()
    with c2:
        if st.button("Reset", use_container_width=True, key="sb_reset"):
            st.session_state.current_hour = 0
            st.session_state.is_playing = False
            st.rerun()

    st.session_state.play_speed = st.sidebar.select_slider(
        "Speed", options=[0.25, 0.5, 1.0, 2.0], value=st.session_state.play_speed,
        format_func=lambda s: f"{s}s per ICU hour",
    )

    s1, s2 = st.sidebar.columns(2)
    with s1:
        if st.button("−1 hour", use_container_width=True, key="sb_back"):
            st.session_state.current_hour = max(0, st.session_state.current_hour - 1)
            st.rerun()
    with s2:
        if st.button("+1 hour", use_container_width=True, key="sb_fwd"):
            st.session_state.current_hour = min(max_hours - 1, st.session_state.current_hour + 1)
            st.rerun()

    target = st.sidebar.slider(
        "ICU hour", min_value=0, max_value=max(0, max_hours - 1),
        value=min(st.session_state.current_hour, max(0, max_hours - 1)),
    )
    if target != st.session_state.current_hour:
        st.session_state.current_hour = target
        st.session_state.is_playing = False

    st.sidebar.divider()
    st.session_state.show_table = st.sidebar.toggle(
        "Show data table", value=st.session_state.show_table,
        help="Non-visual reading of the window shown in the charts.",
    )


# -----------------------------------------------------------------------------
# Ward — a triage list, ordered by risk
# -----------------------------------------------------------------------------
def render_ward() -> None:
    hour = min(st.session_state.current_hour, 15)
    tau = risk_threshold()

    T.section("Ward 4B · medical intensive care", "",
              f"{len(WARD_BEDS)} beds · ICU hour {hour:02d} · ordered by detector risk")

    rows = []
    for bed in WARD_BEDS:
        report = score_hour(bed["pid"], hour)
        readings = load_stay(bed["pid"])
        vitals = readings[min(hour, len(readings) - 1)].vitals if readings else {}
        rows.append((bed, report, vitals))

    # Ordering is the information a grid of equal tiles throws away.
    rows.sort(key=lambda r: (r[1].risk_score if r[1] else -1.0), reverse=True)

    html = [
        '<div class="ws-row ws-row--head">'
        "<div>Bed / patient</div><div>Status</div>"
        '<div style="text-align:right;">HR</div>'
        '<div style="text-align:right;">SBP</div>'
        '<div style="text-align:right;">SpO₂</div>'
        "<div>Detector risk</div></div>"
    ]

    for bed, report, vitals in rows:
        active = bed["pid"] == st.session_state.active_patient
        cls = "ws-row ws-row--active" if active else "ws-row"

        if report is None:
            html.append(
                f'<div class="{cls}"><div><b>Bed {bed["bed"]}</b> '
                f'<span style="font-family:{T.FONT_MONO};font-size:{T.TYPE["xs"]};'
                f'color:{T.INK_MUTED};">{bed["pid"]}</span></div>'
                f'<div style="color:{T.INK_MUTED};font-size:{T.TYPE["xs"]};">no recording</div>'
                f"<div></div><div></div><div></div><div></div></div>"
            )
            continue

        colour, icon_name, label = T.band_style(report.risk_band, report.news2_score)
        rcolour = risk_colour(report.risk_score, tau)

        def cell(key: str, digits: int = 0) -> str:
            value = vitals.get(key, 0.0)
            return (
                f'<div class="ws-num" style="text-align:right;font-size:{T.TYPE["base"]};'
                f'color:{T.INK};">{value:.{digits}f}</div>'
            )

        html.append(
            f'<div class="{cls}">'
            f'<div><b style="font-size:{T.TYPE["base"]};">Bed {bed["bed"]}</b>'
            f'<span style="font-family:{T.FONT_MONO};font-size:{T.TYPE["xs"]};'
            f'color:{T.INK_MUTED};margin-left:8px;">{bed["pid"]}</span>'
            f'<div style="font-size:{T.TYPE["xs"]};color:{T.INK_MUTED};margin-top:2px;">'
            f'{bed["age"]}y {bed["sex"]} · {bed["condition"]}</div></div>'
            f'<div>{T.status_chip(colour, icon_name, label, f"NEWS2 {report.news2_score}")}</div>'
            f"{cell('HR')}{cell('SBP')}{cell('O2Sat')}"
            f'<div><div style="display:flex;justify-content:space-between;'
            f'font-size:{T.TYPE["xs"]};color:{T.INK_MUTED};margin-bottom:4px;">'
            f'<span>risk</span><span class="ws-num" style="color:{rcolour};">'
            f"{report.risk_score:.0%}</span></div>"
            f"{T.rail(report.risk_score, rcolour)}</div>"
            f"</div>"
        )

    st.markdown("".join(html), unsafe_allow_html=True)
    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    if not st.session_state.demo_active:
        options = [f"Bed {b['bed']}" for b, _, _ in rows]
        picker = getattr(st, "pills", None) or getattr(st, "segmented_control", None)
        if picker is not None:
            choice = picker("Open a bed", options, key="ward_pick",
                            label_visibility="collapsed")
        else:  # pragma: no cover - older Streamlit
            choice = st.selectbox("Open a bed", [""] + options, key="ward_pick")
        if choice:
            target = next(b for b, _, _ in rows if f"Bed {b['bed']}" == choice)
            if target["pid"] != st.session_state.active_patient:
                st.session_state.active_patient = target["pid"]
                st.session_state.current_hour = hour
                st.session_state.is_playing = False
                st.rerun()


# -----------------------------------------------------------------------------
# Bedside
# -----------------------------------------------------------------------------
def _vitals_rows(vitals: dict[str, float]) -> str:
    """Compact ruled list for the state rail. In-range values carry no colour."""
    out = []
    for key, label, unit, lo, hi, digits in VITAL_STRIP:
        value = vitals.get(key, 0.0)
        breach = value < lo or value > hi
        colour = T.VERMILION if breach else T.INK
        mark = T.icon("triangle-alert" if breach else "minus", 12,
                      T.VERMILION if breach else T.INK_MUTED)
        out.append(
            f'<div style="display:grid;grid-template-columns:16px 1fr auto;gap:9px;'
            f'align-items:baseline;padding:9px 0;border-bottom:1px solid {T.RULE};">'
            f'<div style="align-self:center;">{mark}</div>'
            f'<div><div style="font-size:{T.TYPE["sm"]};color:{T.INK};">{label}</div>'
            f'<div style="font-size:{T.TYPE["xs"]};color:{T.INK_MUTED};font-family:{T.FONT_MONO};">'
            f'{lo:g}–{hi:g} {unit}</div></div>'
            f'<div class="ws-num" style="font-size:{T.TYPE["lg"]};color:{colour};">'
            f'{value:.{digits}f}</div></div>'
        )
    return "".join(out)


def _breach_table(report) -> str:
    if not report.abnormalities:
        return (
            f'<div style="font-family:{T.FONT_DISPLAY};font-style:italic;'
            f'font-size:{T.TYPE["sm"]};color:{T.INK_2};padding:8px 2px;">'
            f"The aggregate score crossed the threshold without any single vital "
            f"reaching an extreme value.</div>"
        )
    rows = "".join(
        f"<tr><td>{a.label}</td>"
        f'<td class="num" style="color:{T.VERMILION};">{a.value:.1f} {a.unit}</td>'
        f'<td class="num">{a.normal_range}</td>'
        f'<td class="num">+{a.subscore}</td>'
        f'<td style="font-size:{T.TYPE["xs"]};letter-spacing:.08em;">{a.severity.upper()}</td></tr>'
        for a in report.abnormalities
    )
    return (
        f'<table class="ws-table"><thead><tr><th>Parameter</th><th>Observed</th>'
        f"<th>Normal</th><th>NEWS2</th><th>Severity</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def render_bedside_body() -> None:
    """Everything that depends on the current hour. Runs inside the replay fragment."""
    patient = st.session_state.active_patient
    readings = load_stay(patient)
    if not readings:
        st.error(f"No recording found for patient `{patient}`.")
        return

    total = len(readings)
    hour = min(st.session_state.current_hour, total - 1)
    reading = readings[hour]
    report = score_hour(patient, hour)
    if report is None:
        st.error("Scoring unavailable for this hour.")
        return

    colour, icon_name, label = T.band_style(report.risk_band, report.news2_score)
    tau = risk_threshold()

    # ---- Header -------------------------------------------------------------
    h1, h2 = st.columns([3, 2])
    with h1:
        st.markdown(
            f'<div class="ws-display" style="font-size:{T.TYPE["xl"]};">'
            f'Bed {patient} <span style="font-family:{T.FONT_MONO};'
            f'font-size:{T.TYPE["sm"]};color:{T.INK_MUTED};letter-spacing:0;">'
            f'hour {hour:02d} of {total - 1:02d}</span></div>',
            unsafe_allow_html=True,
        )
    with h2:
        st.markdown(
            f'<div style="display:flex;gap:10px;justify-content:flex-end;'
            f'align-items:center;flex-wrap:wrap;padding-top:6px;">'
            f'{T.status_chip(colour, icon_name, label, f"NEWS2 {report.news2_score}/15")}'
            f'<span class="ws-prov">{T.icon("circle-dot", 11, T.INK_MUTED)}'
            f"REPLAY h{hour:02d} · {patient}.psv</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    window = readings[: hour + 1]
    hours = [r.hour for r in window]
    series = {key: [r.vitals.get(key, 0.0) for r in window] for key, *_ in VITAL_STRIP}

    # ---- Traces beside state ------------------------------------------------
    left, right = st.columns([2.1, 1], gap="medium")

    with left:
        demo_script.open_panel("vitals")
        T.section("Vital trajectories", "", "shaded band = normal range")
        st.plotly_chart(
            charts.vitals_small_multiples(hours, series),
            use_container_width=True, config={"displayModeBar": False},
            key=f"vitals_{patient}_{hour}",
        )
        demo_script.close_panel()

        demo_script.open_panel("risk")
        T.section("NEWS2 trajectory", "", f"alert threshold {ALERT_THRESHOLD} · maximum 15")
        st.plotly_chart(
            charts.news2_trajectory(hours, [r.news2 for r in window], ALERT_THRESHOLD),
            use_container_width=True, config={"displayModeBar": False},
            key=f"news2_{patient}_{hour}",
        )
        demo_script.close_panel()

        if st.session_state.show_table:
            st.dataframe(
                {
                    "ICU hour": hours,
                    "NEWS2": [r.news2 for r in window],
                    **{key: [round(v, 1) for v in series[key]] for key, *_ in VITAL_STRIP},
                },
                use_container_width=True, hide_index=True,
            )

    with right:
        T.section("Current reading", "")
        k1, k2 = st.columns(2)
        with k1:
            st.markdown(T.kpi("NEWS2", f"{report.news2_score}",
                              f"alert at ≥ {ALERT_THRESHOLD}", colour),
                        unsafe_allow_html=True)
        with k2:
            st.markdown(T.kpi("Risk", f"{report.risk_score:.0%}", "detector",
                              risk_colour(report.risk_score, tau)),
                        unsafe_allow_html=True)

        st.plotly_chart(charts.risk_bullet(report.risk_score, tau),
                        use_container_width=True, config={"displayModeBar": False},
                        key=f"bullet_{patient}_{hour}")

        st.markdown(_vitals_rows(reading.vitals), unsafe_allow_html=True)
        st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

        demo_script.open_panel("alert")
        if report.is_alert:
            severity = T.VERMILION if report.news2_score >= 7 else T.AMBER
            st.markdown(
                T.banner(
                    f"Deterioration alert · hour {hour:02d}",
                    f"{report.recommended_response.rstrip('.')}.",
                    severity, "triangle-alert", attention=True,
                ),
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
            T.section("Breached thresholds", "", "computed, not generated")
            st.markdown(_breach_table(report), unsafe_allow_html=True)
        else:
            st.markdown(
                T.banner(
                    f"Stable · hour {hour:02d}",
                    f"NEWS2 {report.news2_score}/15, below the alert threshold of "
                    f"{ALERT_THRESHOLD}. No summary is generated while the patient is stable.",
                    T.INK_MUTED, "circle-check",
                ),
                unsafe_allow_html=True,
            )
        demo_script.close_panel()

    # ---- The C3 bridge, full width -----------------------------------------
    card = summary_card(patient, hour) if report.is_alert else None

    if card:
        st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)
        T.section("Alert-to-summary bridge", "", "contribution C3")
        tab_note, tab_evidence = st.tabs(["Clinical handover note", "Retrieval evidence"])

        with tab_note:
            demo_script.open_panel("summary")
            metrics = load_evaluation_metrics()
            rate = scenario_hallucination_rate(metrics, patient)
            if rate is None:
                verification = (
                    f'<span class="ws-prov">{T.icon("circle-alert", 11, T.INK_MUTED)}'
                    f"Fact verification: not run for this scenario</span>"
                )
            else:
                ok = rate == 0.0
                verification = (
                    f'<span class="ws-prov" style="color:{T.INK_2 if ok else T.AMBER};">'
                    f'{T.icon("circle-check" if ok else "triangle-alert", 11, T.INK_2 if ok else T.AMBER)}'
                    f"Unsupported claims {rate:.1%}</span>"
                )

            st.markdown(
                f'<div class="ws-card">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'gap:10px;flex-wrap:wrap;padding-bottom:11px;margin-bottom:14px;'
                f'border-bottom:1px solid {T.RULE};">'
                f'<span class="ws-prov">{T.icon("file-text", 11, T.INK_2)}'
                f"Generated narrative · patient {patient} only</span>{verification}</div>"
                f'<div style="font-family:{T.FONT_DISPLAY};font-size:{T.TYPE["base"]};'
                f'color:{T.INK};line-height:1.75;white-space:pre-wrap;max-width:78ch;">'
                f"{card.narrative_summary}</div>"
                f'<div style="margin-top:16px;padding-top:11px;border-top:1px solid {T.RULE};'
                f'font-family:{T.FONT_MONO};font-size:{T.TYPE["xs"]};color:{T.INK_MUTED};">'
                f"model {card.model} · {card.latency_s:.2f}s · {card.tokens_per_s:.1f} tok/s "
                f"· measured at request time</div></div>",
                unsafe_allow_html=True,
            )
            demo_script.close_panel()

        with tab_evidence:
            demo_script.open_panel("evidence")
            st.markdown(
                '<div class="ws-kpi-l" style="margin-bottom:7px;">Retrieval query, '
                "synthesised from the abnormalities</div>",
                unsafe_allow_html=True,
            )
            st.code(report.summary_query(), language="text")
            st.markdown(
                f'<div class="ws-kpi-l" style="margin:16px 0 7px 0;">Retrieved chunks · '
                f"scoped to {patient}</div>",
                unsafe_allow_html=True,
            )
            for chunk in card.retrieved_chunks:
                st.markdown(
                    f'<div class="ws-card" style="margin-bottom:8px;">'
                    f'<div style="display:flex;justify-content:space-between;gap:10px;'
                    f'margin-bottom:7px;">'
                    f'<span style="font-weight:600;font-size:{T.TYPE["sm"]};color:{T.INK};">'
                    f'{chunk["title"]}</span>'
                    f'<span style="font-family:{T.FONT_MONO};font-size:{T.TYPE["xs"]};'
                    f'color:{T.INK_MUTED};">{chunk["chunk_id"]}</span></div>'
                    f'<div style="font-size:{T.TYPE["sm"]};color:{T.INK_2};line-height:1.65;">'
                    f'{chunk["content"]}</div></div>',
                    unsafe_allow_html=True,
                )
            demo_script.close_panel()

    # ---- Q&A ---------------------------------------------------------------
    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
    demo_script.open_panel("qa")
    T.section("Ask this patient's record", "",
              "hard patient_id filter — cross-patient retrieval is blocked, not merely unlikely")

    prefill = st.session_state.pop("demo_prefill", "")
    question = st.text_input(
        "Clinical question", value=prefill,
        placeholder="e.g. What baseline conditions explain the tachycardia observed?",
        key=f"qa_{patient}_{hour}",
    )

    if question:
        _, _, _, pipeline, _ = get_services()
        with st.spinner("Retrieving and generating…"):
            result = pipeline.answer_question(patient_id=patient, question=question)
        st.markdown(
            f'<div class="ws-card" style="border-left:2px solid {T.AMBER};">'
            f'<div class="ws-kpi-l" style="margin-bottom:7px;">Answer</div>'
            f'<div style="font-family:{T.FONT_DISPLAY};font-size:{T.TYPE["base"]};'
            f'color:{T.INK};line-height:1.75;max-width:78ch;">{result["answer"]}</div>'
            f'<div style="margin-top:14px;padding-top:10px;border-top:1px solid {T.RULE};'
            f'font-family:{T.FONT_MONO};font-size:{T.TYPE["xs"]};color:{T.INK_MUTED};">'
            f'{result["model"]} · {result["total_duration_s"]:.2f}s · '
            f'{result["eval_rate_tok_s"]:.1f} tok/s · scope {patient}</div></div>',
            unsafe_allow_html=True,
        )
    demo_script.close_panel()

    # ---- Replay advance -----------------------------------------------------
    if st.session_state.is_playing:
        if hour < total - 1:
            st.session_state.current_hour = hour + 1
        else:
            st.session_state.is_playing = False


# -----------------------------------------------------------------------------
# Page
# -----------------------------------------------------------------------------
def render() -> None:
    init_state()

    patient = st.session_state.active_patient
    ensure_indexed(patient)
    readings = load_stay(patient)
    total = len(readings) if readings else 1

    render_sidebar(total)

    if st.session_state.demo_active:
        demo_script.render_director_bar()
        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    tab_ward, tab_bed = st.tabs(["Ward", f"Bedside · {patient}"])

    with tab_ward:
        render_ward()

    with tab_bed:
        # Auto-advance repaints only this fragment, so the page does not flicker.
        interval = st.session_state.play_speed if st.session_state.is_playing else None
        st.fragment(run_every=interval)(render_bedside_body)()


if __name__ == "__main__":
    st.set_page_config(page_title="WardSense | Clinician console", layout="wide",
                       initial_sidebar_state="expanded")
    T.inject()
    render()
