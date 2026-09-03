"""
Assurance — model architecture, privacy posture, and evaluation provenance.

Every number on this page is read from the Stage 8 artifact through
metrics_source. Nothing is hardcoded, and a figure whose provenance we cannot
vouch for is shown as withheld rather than printed as a result.
"""

from __future__ import annotations

import streamlit as st

from product_track.interfaces import theme as T
from product_track.interfaces.metrics_source import NOT_MEASURED, load_evaluation_metrics


def _spec_card(icon_name: str, title: str, rows: list[tuple[str, str]]) -> str:
    body = "".join(
        f'<div style="display:flex;justify-content:space-between;gap:12px;padding:6px 0;'
        f'border-bottom:1px solid {T.RULE};">'
        f'<span style="font-size:.8rem;color:{T.INK_2};">{k}</span>'
        f'<span style="font-family:{T.FONT_MONO};font-size:.78rem;color:{T.INK};'
        f'text-align:right;">{v}</span></div>'
        for k, v in rows
    )
    return (
        f'<div class="ws-card" style="height:100%;">'
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
        f'{T.icon(icon_name, 16, T.ACCENT)}'
        f'<span style="font-weight:600;font-size:.92rem;color:{T.INK};">{title}</span></div>'
        f'{body}</div>'
    )


def render() -> None:
    metrics = load_evaluation_metrics()

    st.markdown("## Assurance")
    st.markdown(
        f'<div style="font-size:.9rem;color:{T.INK_2};max-width:78ch;line-height:1.6;'
        f'margin-bottom:20px;">The privacy posture this system claims, and the current state of '
        f'its evaluation evidence. Figures here are read from the evaluation artifact at render '
        f'time; when the artifact cannot support a claim, this page says so instead of showing a '
        f'number.</div>',
        unsafe_allow_html=True,
    )

    # ---- Provenance first. It governs how everything below should be read. ----
    T.section("Evaluation provenance", "shield")

    generated = metrics.generated_at.strftime("%Y-%m-%d %H:%M") if metrics.generated_at else "unknown"

    if not metrics.available:
        st.markdown(
            T.banner("No evaluation artifact found", metrics.provenance_note,
                     T.INK_MUTED, "circle-alert"),
            unsafe_allow_html=True,
        )
    elif metrics.trustworthy:
        st.markdown(
            T.banner(
                "Model-derived — figures below are benchmark results",
                f"{metrics.provenance_note} Artifact generated {generated} from "
                f"{metrics.scenarios_count} held-out scenarios (split seed {metrics.held_out_seed}).",
                T.GOOD, "circle-check",
            ),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            T.banner(
                "Template-derived — not a model benchmark",
                f"{metrics.provenance_note} Artifact generated {generated}. "
                f"The scores below are reported for transparency and are struck through: they "
                f"describe the runner's fallback template, not either model's output. Re-run the "
                f"evaluation against a live model to populate this page.",
                T.WARNING, "triangle-alert",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

    # ---- Architecture & privacy: design facts, not measurements ----
    T.section("Architecture and privacy posture", "cpu",
              "design parameters — configuration, not measured outcomes")
    c1, c2, c3 = st.columns(3, gap="small")
    with c1:
        st.markdown(
            _spec_card("database", "Federated setup", [
                ("Sites", "2 (simulated)"),
                ("Framework", "Flower over gRPC"),
                ("Aggregation", "sample-weighted FedAvg"),
                ("Raw data movement", "none"),
            ]),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            _spec_card("lock", "Differential privacy", [
                ("Target budget", "ε = 1.0, δ = 1e-5"),
                ("Gradient clipping", "L2 norm C = 1.0"),
                ("Accountant", "stateful Rényi DP"),
                ("Privacy unit", "per patient"),
            ]),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            _spec_card("shield", "Inference boundary", [
                ("LLM", "local Ollama, no network"),
                ("Retrieval scope", "hard patient_id filter"),
                ("Isolation test", "adversarial, on every change"),
                ("Vitals in summary", "deterministic table"),
            ]),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

    # ---- Measured results ----
    T.section("Stage 8 evaluation results", "file-text",
              "contribution C3 — alert-to-summary quality")

    if not metrics.available:
        st.markdown(
            f'<div style="font-size:.86rem;color:{T.INK_2};">{NOT_MEASURED} — run '
            f'<code>python -m product_track.evaluation.runner</code> to produce the artifact.</div>',
            unsafe_allow_html=True,
        )
        return

    strike = "" if metrics.trustworthy else "text-decoration:line-through;opacity:.55;"
    note = "" if metrics.trustworthy else (
        f'<div style="font-size:.78rem;color:{T.WARNING};margin-top:10px;">'
        f'Struck through: template-derived, not a model result.</div>'
    )

    rows = []
    for name, m in metrics.models.items():
        rows.append(
            f"<tr>"
            f'<td style="{strike}">{name}</td>'
            f'<td class="num" style="{strike}">{metrics.fmt_num(m.mean_latency_s, 2, " s")}</td>'
            f'<td class="num" style="{strike}">{metrics.fmt_num(m.mean_tokens_per_s, 1, " tok/s")}</td>'
            f'<td class="num" style="{strike}">{metrics.fmt_pct(m.hallucination_rate)}</td>'
            f'<td class="num" style="{strike}">'
            f'{metrics.fmt_pct_raw(100.0 - m.zero_treatment_violation_pct) if m.zero_treatment_violation_pct is not None else NOT_MEASURED}</td>'
            f'<td class="num" style="{strike}">{metrics.fmt_pct_raw(m.rubric_pct, 1)}</td>'
            f"</tr>"
        )

    st.markdown(
        f'<div class="ws-card"><table class="ws-table">'
        f"<thead><tr><th>Model</th><th>Mean latency</th><th>Throughput</th>"
        f"<th>Hallucination rate</th><th>Safety violations</th><th>Rubric</th></tr></thead>"
        f'<tbody>{"".join(rows)}</tbody></table>{note}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    a1, a2 = st.columns([1, 2])
    with a1:
        kappa_value = (
            f"{metrics.kappa:.3f}" if (metrics.kappa is not None and metrics.trustworthy)
            else NOT_MEASURED
        )
        st.markdown(
            f'<div class="ws-card">'
            + T.kpi("Inter-rater agreement (κ)", kappa_value,
                    metrics.kappa_interpretation or "" if metrics.trustworthy else
                    "withheld pending a model-derived run")
            + "</div>",
            unsafe_allow_html=True,
        )
    with a2:
        rec = metrics.recommendation if metrics.trustworthy else (
            "No model recommendation can be made from a template-derived artifact."
        )
        st.markdown(
            f'<div class="ws-card"><div class="ws-kpi-l">Recommendation</div>'
            f'<div style="font-size:.86rem;color:{T.INK_2};line-height:1.6;margin-top:4px;">'
            f"{rec}</div></div>",
            unsafe_allow_html=True,
        )
