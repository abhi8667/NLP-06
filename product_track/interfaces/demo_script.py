"""
Guided demo mode — a scripted walkthrough for presenting WardSense.

The console can be driven freehand, but a judge watching a timeline advance has
no way to know which panel matters. Demo mode runs a named scenario as numbered
chapters: each sets the patient and hour, names one panel to focus, and says in a
sentence what to watch for. The focused panel is ringed; everything else recedes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import streamlit as st

from product_track.interfaces import theme as T

# Panels a chapter can point at. Must match the keys used in the console.
FOCUS_PANELS = ("vitals", "risk", "alert", "summary", "evidence", "qa")


@dataclass(frozen=True)
class Chapter:
    """One step of the walkthrough."""
    title: str
    hour: int
    focus: str
    caption: str
    watch_for: str
    prefill_question: str = ""


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    patient_id: str
    bed: str
    premise: str
    chapters: list[Chapter] = field(default_factory=list)


SCENARIOS: dict[str, Scenario] = {
    "sepsis": Scenario(
        key="sepsis",
        label="Acute sepsis trajectory",
        patient_id="p000001",
        bed="Bed 101",
        premise="83-year-old female. NEWS2 climbs from 4 to 9 across the recorded stay.",
        chapters=[
            Chapter(
                title="Baseline",
                hour=2,
                focus="vitals",
                caption="Two hours into the stay. Vitals sit inside their normal bands and nothing is firing.",
                watch_for="NEWS2 is below the alert threshold of 5.",
            ),
            Chapter(
                title="Deterioration begins",
                hour=5,
                focus="risk",
                caption="Respiration and heart rate start leaving their ranges. The detector's risk score rises before any single vital looks alarming.",
                watch_for="The risk bullet crosses toward its threshold while NEWS2 is still sub-alert.",
            ),
            Chapter(
                title="Alert fires",
                hour=8,
                focus="alert",
                caption="NEWS2 crosses 5. The alert is raised automatically — nobody clicked anything.",
                watch_for="The breached vitals table. It is computed, not generated, so a missed vital is impossible by construction.",
            ),
            Chapter(
                title="Grounded summary (C3)",
                hour=8,
                focus="summary",
                caption="The abnormal vitals become the retrieval query. History relevant to this specific deterioration is retrieved, and only the narrative is model-generated.",
                watch_for="The retrieval query is built from the abnormalities themselves, not from a generic patient lookup.",
            ),
            Chapter(
                title="Ask the record",
                hour=8,
                focus="qa",
                caption="A follow-up question, answered from this patient's record alone. The retrieval filter is a hard patient_id boundary.",
                watch_for="The answer cites only this patient. Cross-patient retrieval is blocked, not merely unlikely.",
                prefill_question="What vital trajectory or abnormalities were recorded for this patient?",
            ),
        ],
    ),
    "respiratory": Scenario(
        key="respiratory",
        label="Rapid respiratory decline",
        patient_id="p000008",
        bed="Bed 103",
        premise="74-year-old male. Oxygenation falls faster than the other vitals move.",
        chapters=[
            Chapter(
                title="Baseline",
                hour=2,
                focus="vitals",
                caption="Stable opening hours on the recorded stay.",
                watch_for="SpO₂ still inside its band.",
            ),
            Chapter(
                title="Oxygenation falls",
                hour=5,
                focus="vitals",
                caption="SpO₂ drops below 96% while heart rate and blood pressure stay unremarkable.",
                watch_for="Only one panel breaches. A single-vital rule would still be quiet here.",
            ),
            Chapter(
                title="Alert and summary",
                hour=8,
                focus="alert",
                caption="The aggregate score crosses the threshold and the bridge runs.",
                watch_for="The summary names respiration specifically rather than describing the patient generically.",
            ),
        ],
    ),
    "stable": Scenario(
        key="stable",
        label="Stable post-op control",
        patient_id="p000014",
        bed="Bed 105",
        premise="62-year-old male, post-operative. The negative control for the demo.",
        chapters=[
            Chapter(
                title="Stable throughout",
                hour=6,
                focus="vitals",
                caption="A patient who does not deteriorate. Six hours in, nothing fires.",
                watch_for="No alert, no summary generated. The system stays quiet when it should.",
            ),
        ],
    ),
}

DEFAULT_SCENARIO = "sepsis"


# -----------------------------------------------------------------------------
# State
# -----------------------------------------------------------------------------
def init_state() -> None:
    st.session_state.setdefault("demo_active", False)
    st.session_state.setdefault("demo_scenario", DEFAULT_SCENARIO)
    st.session_state.setdefault("demo_chapter", 0)


def active_scenario() -> Scenario:
    return SCENARIOS.get(st.session_state.get("demo_scenario", DEFAULT_SCENARIO),
                         SCENARIOS[DEFAULT_SCENARIO])


def current_chapter() -> Chapter | None:
    """The chapter now playing, or None when demo mode is off."""
    if not st.session_state.get("demo_active"):
        return None
    scenario = active_scenario()
    idx = min(st.session_state.get("demo_chapter", 0), len(scenario.chapters) - 1)
    return scenario.chapters[idx]


def start(scenario_key: str = DEFAULT_SCENARIO) -> None:
    """Enter demo mode at chapter 1 and move the console to that patient/hour."""
    st.session_state.demo_active = True
    st.session_state.demo_scenario = scenario_key
    st.session_state.demo_chapter = 0
    _apply_chapter()


def stop() -> None:
    st.session_state.demo_active = False
    st.session_state.is_playing = False


def _apply_chapter() -> None:
    """Push the current chapter's patient and hour into console state."""
    scenario = active_scenario()
    chapter = scenario.chapters[st.session_state.demo_chapter]
    st.session_state.active_patient = scenario.patient_id
    st.session_state.current_hour = chapter.hour
    if chapter.prefill_question:
        st.session_state.demo_prefill = chapter.prefill_question
    else:
        st.session_state.pop("demo_prefill", None)


def goto(index: int) -> None:
    scenario = active_scenario()
    st.session_state.demo_chapter = max(0, min(index, len(scenario.chapters) - 1))
    _apply_chapter()


def focus_class(panel: str) -> str:
    """
    CSS class for a panel given the active chapter.

    Returns "ws-focus" for the panel this chapter is about, "ws-dim" for the rest,
    and "" whenever demo mode is off so the console looks normal outside a demo.
    """
    chapter = current_chapter()
    if chapter is None:
        return ""
    return "ws-focus" if chapter.focus == panel else "ws-dim"


def open_panel(panel: str) -> None:
    """Open a focus-aware wrapper div. Pair with close_panel()."""
    cls = focus_class(panel)
    st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)


def close_panel() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Director bar
# -----------------------------------------------------------------------------
def render_director_bar() -> None:
    """
    The strip that makes the demo legible without narration: which chapter we are
    in, what to look at, and the controls to move.
    """
    scenario = active_scenario()
    idx = st.session_state.demo_chapter
    total = len(scenario.chapters)
    chapter = scenario.chapters[idx]

    rail = "".join(
        f'<span style="flex:1;height:3px;border-radius:2px;background:'
        f'{T.ACCENT if i <= idx else T.INSET};"></span>'
        for i in range(total)
    )

    st.markdown(
        f"""
        <div class="ws-card" style="border-left:2px solid {T.ACCENT};margin-bottom:10px;">
          <div style="display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;">
            <span style="font-family:{T.FONT_MONO};font-size:.72rem;color:{T.ACCENT};
                         letter-spacing:.08em;">CHAPTER {idx + 1} / {total}</span>
            <span style="font-size:1rem;font-weight:600;color:{T.INK};">{chapter.title}</span>
            <span style="font-family:{T.FONT_MONO};font-size:.72rem;color:{T.INK_MUTED};
                         margin-left:auto;">{scenario.label} · {scenario.bed} · {scenario.patient_id}</span>
          </div>
          <div style="display:flex;gap:4px;margin:10px 0 10px 0;">{rail}</div>
          <div style="font-size:.86rem;color:{T.INK_2};line-height:1.55;">{chapter.caption}</div>
          <div style="display:flex;gap:7px;align-items:flex-start;margin-top:9px;
                      padding-top:9px;border-top:1px solid {T.RULE};">
            <div style="padding-top:1px;">{T.icon("search", 14, T.ACCENT)}</div>
            <div style="font-size:.82rem;color:{T.INK};"><b style="color:{T.ACCENT};
                 font-weight:600;">Watch for:</b> {chapter.watch_for}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1.4])
    with c1:
        if st.button("Back", use_container_width=True, disabled=idx == 0, key="demo_back"):
            goto(idx - 1)
            st.rerun()
    with c2:
        if st.button("Next", type="primary", use_container_width=True,
                     disabled=idx >= total - 1, key="demo_next"):
            goto(idx + 1)
            st.rerun()
    with c3:
        if st.button("Restart", use_container_width=True, key="demo_restart"):
            goto(0)
            st.rerun()
    with c4:
        if st.button("Exit demo mode", use_container_width=True, key="demo_exit"):
            stop()
            st.rerun()
