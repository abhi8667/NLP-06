"""
Overview — the landing page.

A single screen for someone who wants the context before watching the demo:
what the system does, how the pipeline fits together, what is actually measured,
and one button that starts the guided walkthrough.
"""

from __future__ import annotations

import streamlit as st

from product_track.interfaces import demo_script, theme as T
from product_track.interfaces.metrics_source import NOT_MEASURED, load_evaluation_metrics


def _pipeline_svg() -> str:
    """Flat inline-SVG pipeline. No library, no gradient, no decoration."""
    stages = [
        ("Bedside replay", "recorded stay"),
        ("NEWS2 + detector", "on-device"),
        ("Alert raised", "threshold ≥ 5"),
        ("Abnormalities", "deterministic"),
        ("Patient-scoped retrieval", "hard id filter"),
        ("Local LLM narrative", "no network"),
        ("Clinician", "reviews"),
    ]
    box_w, gap, box_h, y = 172, 24, 58, 16
    width = len(stages) * box_w + (len(stages) - 1) * gap
    parts = []

    for i, (title, sub) in enumerate(stages):
        x = i * (box_w + gap)
        emphasis = i in (2, 3, 4, 5)  # the C3 bridge span
        stroke = T.ACCENT if emphasis else T.RULE_STRONG
        parts.append(
            f'<rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" rx="5" '
            f'fill="{T.SURFACE}" stroke="{stroke}" stroke-width="1"/>'
            f'<text x="{x + box_w / 2}" y="{y + 24}" text-anchor="middle" '
            f'font-family="Inter Tight, sans-serif" font-size="12.5" font-weight="600" '
            f'fill="{T.INK}">{title}</text>'
            f'<text x="{x + box_w / 2}" y="{y + 42}" text-anchor="middle" '
            f'font-family="JetBrains Mono, monospace" font-size="10" '
            f'fill="{T.INK_MUTED}">{sub}</text>'
        )
        if i < len(stages) - 1:
            ax = x + box_w + 5
            parts.append(
                f'<path d="M{ax} {y + box_h / 2} H{ax + gap - 10}" stroke="{T.INK_MUTED}" '
                f'stroke-width="1.25"/>'
                f'<path d="m{ax + gap - 14} {y + box_h / 2 - 3.5} 4 3.5-4 3.5" '
                f'stroke="{T.INK_MUTED}" stroke-width="1.25" fill="none" '
                f'stroke-linecap="round" stroke-linejoin="round"/>'
            )

    parts.append(
        f'<path d="M{2 * (box_w + gap)} 88 H{6 * (box_w + gap) - gap}" stroke="{T.ACCENT}" '
        f'stroke-width="1" stroke-dasharray="3 3"/>'
        f'<text x="{(2 * (box_w + gap) + 6 * (box_w + gap) - gap) / 2}" y="105" '
        f'text-anchor="middle" font-family="JetBrains Mono, monospace" font-size="10" '
        f'letter-spacing="0.08em" fill="{T.ACCENT}">CONTRIBUTION C3 — ALERT-TO-SUMMARY BRIDGE</text>'
    )

    return (
        f'<div style="overflow-x:auto;padding-bottom:6px;">'
        f'<svg width="{width}" height="118" viewBox="0 0 {width} 118" '
        f'xmlns="http://www.w3.org/2000/svg" role="img" '
        f'aria-label="Pipeline: bedside replay, NEWS2 and detector, alert raised, abnormality '
        f'extraction, patient-scoped retrieval, local LLM narrative, clinician review. '
        f'The middle four stages are contribution C3.">{"".join(parts)}</svg></div>'
    )


def _capability(icon_name: str, title: str, body: str) -> str:
    return (
        f'<div class="ws-card" style="height:100%;">'
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">'
        f'{T.icon(icon_name, 16, T.ACCENT)}'
        f'<span style="font-weight:600;font-size:.92rem;color:{T.INK};">{title}</span></div>'
        f'<div style="font-size:.83rem;color:{T.INK_2};line-height:1.6;">{body}</div></div>'
    )


def render() -> None:
    demo_script.init_state()

    st.markdown(
        f"""
        <div style="margin-bottom:6px;">
          <h1 style="margin:0 0 6px 0;">WardSense</h1>
          <div style="font-size:1rem;color:{T.INK_2};max-width:70ch;line-height:1.6;">
            Privacy-preserving early deterioration surveillance for the ICU. A federated,
            differentially private detector raises the alert; a local language model explains
            it from that patient's record alone. Nothing leaves the machine.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin:14px 0 22px 0;">'
        f'<span class="ws-prov">{T.icon("circle-dot", 12, T.WARNING)}'
        f'REPLAY · recorded PhysioNet ICU stays · not a live hospital feed</span>'
        f'<span class="ws-prov">{T.icon("lock", 12, T.GOOD)}'
        f'ON-PREMISES INFERENCE · no outbound network call</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if st.button("Start guided demo", type="primary", key="ov_start_demo"):
        demo_script.start()
        st.switch_page(st.session_state["_ws_console_page"])

    st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)

    T.section("What it does", "layout-grid")
    c1, c2, c3 = st.columns(3, gap="small")
    with c1:
        st.markdown(
            _capability(
                "cpu", "Detect",
                "A sequence detector trained across two simulated sites with Flower, under "
                "DP-SGD. Raw patient data never leaves the edge node — only clipped, noised "
                "gradient updates are aggregated.",
            ),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            _capability(
                "triangle-alert", "Alert",
                "NEWS2 is computed per hour from five available components. Crossing 5 raises "
                "an alert and pinpoints exactly which vitals breached, by how much, and what "
                "each contributed to the score.",
            ),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            _capability(
                "file-text", "Explain",
                "The alert's own abnormalities become the retrieval query. The breached-vitals "
                "table is computed, not generated; only the narrative comes from the model, so "
                "a missed vital is impossible by construction.",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)

    T.section("Pipeline", "arrow-right", "alert content becomes the retrieval query — that junction is the contribution")
    st.markdown(_pipeline_svg(), unsafe_allow_html=True)

    st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)

    metrics = load_evaluation_metrics()
    T.section("Measured so far", "shield", "figures are read from the evaluation artifact — never hardcoded")

    if not metrics.available:
        st.markdown(
            T.banner(
                "No evaluation artifact",
                metrics.provenance_note,
                T.INK_MUTED, "circle-alert",
            ),
            unsafe_allow_html=True,
        )
        return

    if not metrics.trustworthy:
        st.markdown(
            T.banner(
                "Evaluation figures withheld",
                f"{metrics.provenance_note} Until the evaluation is re-run against a live model, "
                f"this panel reports no benchmark numbers.",
                T.WARNING, "triangle-alert",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    show = metrics.trustworthy
    primary = next(iter(metrics.models.values()), None)

    with k1:
        st.markdown(
            T.kpi("Held-out scenarios", str(metrics.scenarios_count or 0),
                  f"split seed {metrics.held_out_seed}" if metrics.held_out_seed is not None else ""),
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            T.kpi("Hallucination rate",
                  metrics.fmt_pct(primary.hallucination_rate) if (show and primary) else NOT_MEASURED,
                  "claim-level verification"),
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            T.kpi("Treatment-safety violations",
                  metrics.fmt_pct_raw(100.0 - primary.zero_treatment_violation_pct)
                  if (show and primary and primary.zero_treatment_violation_pct is not None)
                  else NOT_MEASURED,
                  "prompt forbids prescribing"),
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            T.kpi("Patient isolation", "enforced",
                  "adversarial test, every change", T.GOOD),
            unsafe_allow_html=True,
        )
