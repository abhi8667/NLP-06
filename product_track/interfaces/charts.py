"""
WardSense chart builders — warm editorial dark.

Rules that hold across every figure here:

  * **One trace ink for all vitals.** Identity comes from the panel title, so no
    legend is needed and no signal hue is spent on an ordinary series.
  * **Signal colour marks deviation only.** Amber for elevated, vermilion for
    critical. A reading inside its range is drawn in plain ink.
  * Deviation is never colour alone — a breach gets a distinct marker shape and,
    for the most recent one, a text annotation.
  * NEWS2 is a different measure on a different scale, so it owns its own panel.
    Never a second y-axis.
  * Thin marks, hairline rules, recessive axes. The data is the only thing with
    weight.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from product_track.interfaces import theme as T

FONT_UI = "Inter Tight, sans-serif"
FONT_MONO = "JetBrains Mono, monospace"

VITAL_PANELS: list[dict[str, Any]] = [
    {"key": "HR", "title": "Heart rate", "unit": "bpm", "lo": 51, "hi": 90},
    {"key": "SBP", "title": "Systolic BP", "unit": "mmHg", "lo": 111, "hi": 219},
    {"key": "O2Sat", "title": "SpO₂", "unit": "%", "lo": 96, "hi": 100},
    {"key": "Resp", "title": "Respiration", "unit": "/min", "lo": 12, "hi": 20},
]


def _base_layout(fig: go.Figure, height: int, transparent: bool = False) -> go.Figure:
    ground = "rgba(0,0,0,0)" if transparent else T.SURFACE
    fig.update_layout(
        height=height,
        margin=dict(l=46, r=18, t=32, b=28),
        paper_bgcolor=ground,
        plot_bgcolor=ground,
        showlegend=False,
        hovermode="x unified",
        font=dict(family=FONT_UI, size=11, color=T.INK_2),
        hoverlabel=dict(
            bgcolor=T.INSET,
            bordercolor=T.RULE_STRONG,
            font=dict(family=FONT_MONO, size=11, color=T.INK),
        ),
    )
    axis = dict(
        showgrid=True, gridcolor=T.RULE, gridwidth=1,
        zeroline=False, linecolor=T.RULE_STRONG, tickcolor=T.RULE_STRONG,
        tickfont=dict(family=FONT_MONO, size=9, color=T.INK_MUTED),
    )
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def vitals_small_multiples(hours: list[int], series: dict[str, list[float]],
                           height: int = 250) -> go.Figure:
    """Four vitals as small multiples. One ink for every trace; signal marks breaches."""
    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=[f"{p['title']} · {p['unit']}" for p in VITAL_PANELS],
        horizontal_spacing=0.052,
    )

    for i, panel in enumerate(VITAL_PANELS, start=1):
        values = series.get(panel["key"], [])
        if not values:
            continue

        # The normal range is a recessed well, not a coloured wash — signal hues
        # would imply "good", and green is not part of this language.
        fig.add_hrect(
            y0=panel["lo"], y1=panel["hi"],
            fillcolor=T.INSET, opacity=1.0, line_width=0, layer="below",
            row=1, col=i,
        )

        fig.add_trace(
            go.Scatter(
                x=hours, y=values, mode="lines",
                line=dict(color=T.TRACE, width=1.75),
                hovertemplate=f"h%{{x}} · %{{y:.1f}} {panel['unit']}<extra></extra>",
            ),
            row=1, col=i,
        )

        breaches = [(h, v) for h, v in zip(hours, values)
                    if v < panel["lo"] or v > panel["hi"]]
        if breaches:
            bx = [h for h, _ in breaches]
            by = [v for _, v in breaches]
            fig.add_trace(
                go.Scatter(
                    x=bx, y=by, mode="markers",
                    marker=dict(color=T.VERMILION, size=8, symbol="diamond",
                                line=dict(color=T.SURFACE, width=1.5)),
                    hovertemplate=f"OUT OF RANGE · h%{{x}} · %{{y:.1f}} {panel['unit']}<extra></extra>",
                ),
                row=1, col=i,
            )
            fig.add_annotation(
                x=bx[-1], y=by[-1], text="out of range",
                showarrow=True, arrowhead=0, arrowcolor=T.VERMILION, arrowwidth=1,
                ax=0, ay=-20, row=1, col=i,
                font=dict(family=FONT_MONO, size=8.5, color=T.VERMILION),
            )

        latest_out = values[-1] < panel["lo"] or values[-1] > panel["hi"]
        fig.add_trace(
            go.Scatter(
                x=[hours[-1]], y=[values[-1]], mode="markers+text",
                marker=dict(color=T.VERMILION if latest_out else T.INK, size=6,
                            line=dict(color=T.SURFACE, width=1.5)),
                text=[f" {values[-1]:.0f}"], textposition="middle right",
                textfont=dict(family=FONT_MONO, size=11,
                              color=T.VERMILION if latest_out else T.INK),
                hoverinfo="skip",
            ),
            row=1, col=i,
        )

    fig = _base_layout(fig, height)
    for ann in fig.layout.annotations[:len(VITAL_PANELS)]:
        ann.font = dict(family=FONT_UI, size=10.5, color=T.INK_MUTED)
        ann.xanchor = "center"
    return fig


def news2_trajectory(hours: list[int], news2: list[int], threshold: int = 5,
                     height: int = 210) -> go.Figure:
    """NEWS2 on its own axis. Zones use the signal ramp; the threshold is labelled."""
    fig = go.Figure()

    fig.add_hrect(y0=threshold, y1=7, fillcolor=T.AMBER, opacity=0.09,
                  line_width=0, layer="below")
    fig.add_hrect(y0=7, y1=15, fillcolor=T.VERMILION, opacity=0.11,
                  line_width=0, layer="below")

    fig.add_hline(
        y=threshold, line=dict(color=T.AMBER, width=1, dash="dot"),
        annotation_text=f"  alert threshold {threshold}",
        annotation_position="top left",
        annotation_font=dict(family=FONT_MONO, size=9.5, color=T.AMBER),
    )

    fig.add_trace(go.Scatter(
        x=hours, y=news2, mode="lines",
        line=dict(color=T.TRACE, width=1.75, shape="hv"),
        hovertemplate="ICU hour %{x} · NEWS2 %{y}<extra></extra>",
    ))

    over = [(h, v) for h, v in zip(hours, news2) if v >= threshold]
    if over:
        fig.add_trace(go.Scatter(
            x=[h for h, _ in over], y=[v for _, v in over], mode="markers",
            marker=dict(color=T.VERMILION, size=8, symbol="diamond",
                        line=dict(color=T.SURFACE, width=1.5)),
            hovertemplate="ALERT · hour %{x} · NEWS2 %{y}<extra></extra>",
        ))

    if hours:
        latest_over = news2[-1] >= threshold
        fig.add_trace(go.Scatter(
            x=[hours[-1]], y=[news2[-1]], mode="markers+text",
            marker=dict(color=T.VERMILION if latest_over else T.INK, size=7,
                        line=dict(color=T.SURFACE, width=1.5)),
            text=[f" {news2[-1]}"], textposition="middle right",
            textfont=dict(family=FONT_MONO, size=13,
                          color=T.VERMILION if latest_over else T.INK),
            hoverinfo="skip",
        ))

    fig = _base_layout(fig, height)
    fig.update_yaxes(range=[0, 15], dtick=5)
    fig.update_xaxes(title=dict(text="ICU hour",
                                font=dict(family=FONT_MONO, size=9, color=T.INK_MUTED)))
    return fig


def risk_bullet(risk: float, threshold: float = 0.5, height: int = 62) -> go.Figure:
    """
    Detector risk as a bullet against its own calibrated threshold.

    The fill is coloured against `threshold` — the quantity the bar encodes — and
    never against the NEWS2 band, which measures something else.
    """
    if risk >= threshold:
        colour = T.VERMILION
    elif risk >= threshold * 0.6:
        colour = T.AMBER
    else:
        colour = T.TRACE

    fig = go.Figure()
    fig.add_trace(go.Bar(x=[1.0], y=[""], orientation="h",
                         marker=dict(color=T.INSET), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Bar(x=[risk], y=[""], orientation="h", marker=dict(color=colour),
                         hovertemplate=f"risk {risk:.1%}<extra></extra>", showlegend=False))
    fig.add_shape(type="line", x0=threshold, x1=threshold, y0=-0.5, y1=0.5,
                  line=dict(color=T.INK, width=1.5))
    fig.add_annotation(x=threshold, y=0.72, text=f"alerts at {threshold:.0%}",
                       showarrow=False,
                       font=dict(family=FONT_MONO, size=8.5, color=T.INK_MUTED))

    fig.update_layout(
        barmode="overlay", height=height,
        margin=dict(l=0, r=0, t=16, b=4),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(range=[0, 1], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        bargap=0.6,
    )
    return fig
