"""
WardSense design system — warm editorial dark.

The reference is a medical journal and a calibrated instrument, not a SaaS
dashboard: warm charcoal ground, bone ink, a serif for display, a grotesque for
interface text, and mono for every numeral.

Discipline that shapes everything below:

  * **Signal colour is reserved for deviation.** Amber means elevated, vermilion
    means critical, and nothing else in the interface is allowed to use them. A
    stable patient carries no colour at all — absence of signal *is* the stable
    state. This is why the ward scans in one pass.
  * **Hierarchy comes from type, rules and space** — never from shadow, gradient
    or glass. Containers are rules and ground shifts, not floating cards.
  * **Status is never colour alone.** Every status pairs an icon with a word.

Contrast measured against the raised surface #1E1A15:
  ink 14.83:1 · ink-2 6.55:1 · muted 5.08:1 · amber 8.02:1 · vermilion 4.74:1
All clear WCAG AA for small text.
"""

from __future__ import annotations

import streamlit as st

# -----------------------------------------------------------------------------
# Tokens
# -----------------------------------------------------------------------------
GROUND = "#16130F"        # page plane, warm charcoal
SURFACE = "#1E1A15"       # raised surface / chart ground
INSET = "#12100C"         # recessed wells (numeric groups)
RULE = "#332C23"          # hairline rules — the primary container device
RULE_STRONG = "#4A4034"

INK = "#F2EDE3"           # bone, primary
INK_2 = "#A89E8E"         # secondary
INK_MUTED = "#948A79"     # meta / axis (5.08:1 — small text safe)

TRACE = "#C9BFAE"         # the single data-series ink. Deliberately near-neutral:
                          # signal hues stay reserved for deviation.

# Signal ramp — deviation only. Never decorative, never a series colour.
AMBER = "#E8A33D"         # elevated
VERMILION = "#E05C36"     # critical
SIGNAL_DIM = "#6E4A22"    # amber at rule strength, for zone washes

# Kept as named roles so callers never reach for a raw hex.
ACCENT = AMBER
GOOD = INK_2              # "stable" is the absence of signal, not a green
WARNING = AMBER
CRITICAL = VERMILION

SPACE = {"xs": "4px", "sm": "8px", "md": "12px", "lg": "16px", "xl": "24px", "2xl": "32px"}

FONT_DISPLAY = "'Newsreader', Georgia, 'Times New Roman', serif"
FONT_UI = "'Inter Tight', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
FONT_MONO = "'JetBrains Mono', 'SFMono-Regular', Consolas, monospace"

# Type scale, 1.25x from a 16px base.
TYPE = {
    "xs": "0.64rem", "sm": "0.8rem", "base": "1rem", "lg": "1.25rem",
    "xl": "1.563rem", "2xl": "1.953rem", "3xl": "2.441rem",
}

BAND_STYLES = {
    "critical": (VERMILION, "triangle-alert", "CRITICAL"),
    "high": (VERMILION, "triangle-alert", "HIGH"),
    "medium": (AMBER, "circle-alert", "ELEVATED"),
    "elevated": (AMBER, "circle-alert", "ELEVATED"),
    "low": (INK_2, "circle-check", "STABLE"),
    "normal": (INK_2, "circle-check", "STABLE"),
    "none": (INK_2, "circle-check", "STABLE"),
}


def band_style(band: str, news2: int = 0) -> tuple[str, str, str]:
    """Resolve a risk band to (colour, icon, label), falling back to NEWS2."""
    key = (band or "").strip().lower()
    if key in BAND_STYLES:
        return BAND_STYLES[key]
    if news2 >= 7:
        return BAND_STYLES["critical"]
    if news2 >= 5:
        return BAND_STYLES["medium"]
    return BAND_STYLES["normal"]


# -----------------------------------------------------------------------------
# Icons (Lucide, inline SVG — never emoji)
# -----------------------------------------------------------------------------
_ICON_PATHS: dict[str, str] = {
    "activity": '<path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2"/>',
    "triangle-alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "circle-alert": '<circle cx="12" cy="12" r="10"/><path d="M12 8v4"/><path d="M12 16h.01"/>',
    "circle-check": '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
    "circle-dot": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="1"/>',
    "shield": '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/>',
    "lock": '<rect width="18" height="11" x="3" y="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>',
    "play": '<polygon points="6 3 20 12 6 21 6 3"/>',
    "pause": '<rect x="14" y="4" width="4" height="16" rx="1"/><rect x="6" y="4" width="4" height="16" rx="1"/>',
    "chevron-left": '<path d="m15 18-6-6 6-6"/>',
    "chevron-right": '<path d="m9 18 6-6-6-6"/>',
    "rotate-ccw": '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/>',
    "bed": '<path d="M2 4v16"/><path d="M2 8h18a2 2 0 0 1 2 2v10"/><path d="M2 17h20"/><path d="M6 8v9"/>',
    "layout-grid": '<rect width="7" height="7" x="3" y="3" rx="1"/><rect width="7" height="7" x="14" y="3" rx="1"/><rect width="7" height="7" x="14" y="14" rx="1"/><rect width="7" height="7" x="3" y="14" rx="1"/>',
    "file-text": '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M16 13H8"/><path d="M16 17H8"/><path d="M10 9H8"/>',
    "search": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    "message-square": '<path d="M22 17a2 2 0 0 1-2 2H6l-4 4V5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2z"/>',
    "arrow-right": '<path d="M5 12h14"/><path d="m12 5 7 7-7 7"/>',
    "cpu": '<rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14a9 3 0 0 0 18 0V5"/><path d="M3 12a9 3 0 0 0 18 0"/>',
    "trending-up": '<path d="M16 7h6v6"/><path d="m22 7-8.5 8.5-5-5L2 17"/>',
    "trending-down": '<path d="M16 17h6v-6"/><path d="m22 17-8.5-8.5-5 5L2 7"/>',
    "minus": '<path d="M5 12h14"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>',
    "users": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
}


def icon(name: str, size: int = 16, color: str = "currentColor", stroke: float = 1.6) -> str:
    """Inline Lucide SVG. Decorative by default (aria-hidden)."""
    body = _ICON_PATHS.get(name, _ICON_PATHS["circle-dot"])
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" '
        f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" '
        f'style="vertical-align:-0.15em;flex:none;">{body}</svg>'
    )


def status_chip(color: str, icon_name: str, label: str, detail: str = "") -> str:
    """Status is icon + word. A stable patient reads in ink, carrying no signal."""
    tail = (
        f'<span style="font-family:{FONT_MONO};color:{INK_2};margin-left:7px;">{detail}</span>'
        if detail else ""
    )
    return (
        f'<span class="ws-chip" style="color:{color};">'
        f'{icon(icon_name, 13, color)}'
        f'<span style="margin-left:6px;letter-spacing:.09em;">{label}</span>{tail}</span>'
    )


# -----------------------------------------------------------------------------
# CSS
# -----------------------------------------------------------------------------
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&family=Inter+Tight:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --ws-ground: __GROUND__;
  --ws-surface: __SURFACE__;
  --ws-inset: __INSET__;
  --ws-rule: __RULE__;
  --ws-rule-strong: __RULE_STRONG__;
  --ws-ink: __INK__;
  --ws-ink-2: __INK2__;
  --ws-muted: __MUTED__;
  --ws-trace: __TRACE__;
  --ws-amber: __AMBER__;
  --ws-vermilion: __VERMILION__;
}

.stApp { background: var(--ws-ground); }

/* Streamlit sets a font on the markdown container, which blocks inheritance from
   body. These three rules have to outrank it, so they are the only !important
   font declarations in the stylesheet — applied widest-first, then overridden by
   role. Order matters: display and mono come after, and win the specificity tie. */
html body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] *,
.stMarkdown, .stMarkdown *, button, input, select, textarea, label {
  font-family: __FONT_UI__ !important;
}
html body { color: var(--ws-ink); }

/* Display type is the serif. Interface text never is. */
h1, h2, .ws-display, .ws-sec-s,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] .ws-display,
[data-testid="stMarkdownContainer"] .ws-sec-s {
  font-family: __FONT_DISPLAY__ !important;
  font-weight: 500;
  letter-spacing: -0.012em;
  color: var(--ws-ink);
  line-height: 1.18;
}
h1 { font-size: __T3XL__; }
h2 { font-size: __T2XL__; }
h3, h4, h5 {
  font-family: __FONT_UI__;
  font-weight: 600;
  letter-spacing: -0.008em;
  color: var(--ws-ink);
}
h3 { font-size: __TLG__; }

code, pre, .ws-num, .ws-kpi-v, .ws-prov, [data-testid="stMetricValue"],
[data-testid="stMarkdownContainer"] code,
[data-testid="stMarkdownContainer"] .ws-num,
[data-testid="stMarkdownContainer"] .ws-kpi-v,
[data-testid="stMarkdownContainer"] .ws-prov {
  font-family: __FONT_MONO__ !important;
  font-variant-numeric: tabular-nums;
}

/* --- Section header: a rule with a label sitting on it ------------------- */
.ws-sec {
  display: flex; align-items: baseline; gap: 10px;
  margin: 6px 0 12px 0; padding-bottom: 7px;
  border-bottom: 1px solid var(--ws-rule-strong);
}
.ws-sec .ws-sec-t {
  font-size: __TXS__; font-weight: 600; letter-spacing: .13em;
  text-transform: uppercase; color: var(--ws-ink-2);
}
.ws-sec .ws-sec-s {
  font-size: __TSM__; color: var(--ws-muted); margin-left: auto;
  font-style: italic; font-family: __FONT_DISPLAY__;
}

/* --- Containers are rules and ground shifts, never floating cards -------- */
.ws-card {
  background: var(--ws-surface);
  border: 1px solid var(--ws-rule);
  border-radius: 3px;
  padding: 15px 17px;
}
.ws-well {
  background: var(--ws-inset);
  border: 1px solid var(--ws-rule);
  border-radius: 3px;
  padding: 10px 12px;
}

/* --- Triage row: the ward reads as a ruled list, sorted by risk ---------- */
.ws-row {
  display: grid;
  grid-template-columns: minmax(150px,1.5fr) minmax(120px,1fr) repeat(3, minmax(58px,.6fr)) minmax(120px,1.1fr);
  align-items: center; gap: 14px;
  padding: 11px 12px;
  border-bottom: 1px solid var(--ws-rule);
  transition: background-color .18s ease;
}
.ws-row:hover { background: var(--ws-surface); }
.ws-row--active { background: var(--ws-surface); box-shadow: inset 2px 0 0 var(--ws-amber); }
.ws-row--head {
  padding: 7px 12px; border-bottom: 1px solid var(--ws-rule-strong);
  font-size: __TXS__; letter-spacing: .12em; text-transform: uppercase;
  color: var(--ws-muted); font-weight: 600;
}
.ws-row--head:hover { background: none; }

/* --- Status chip: no box, just the mark and the word -------------------- */
.ws-chip {
  display: inline-flex; align-items: center;
  font-size: __TXS__; font-weight: 600; white-space: nowrap;
}

/* --- Figures ------------------------------------------------------------ */
.ws-kpi-l {
  font-size: __TXS__; text-transform: uppercase; letter-spacing: .12em;
  color: var(--ws-muted); font-weight: 600; margin-bottom: 5px;
}
.ws-kpi-v {
  font-family: __FONT_MONO__; font-variant-numeric: tabular-nums;
  font-size: __T2XL__; font-weight: 500; color: var(--ws-ink); line-height: 1.05;
}
.ws-kpi-v--sm { font-size: __TLG__; }
.ws-kpi-m { font-size: __TSM__; color: var(--ws-ink-2); margin-top: 5px; line-height: 1.45; }
.ws-slot { display: inline-block; min-width: 3.6ch; text-align: right; }

/* --- Banner: a left rule in the signal colour, nothing more ------------- */
.ws-banner {
  display: flex; gap: 13px; align-items: flex-start;
  background: var(--ws-surface);
  border: 1px solid var(--ws-rule);
  border-left: 2px solid var(--ws-muted);
  border-radius: 3px; padding: 13px 16px;
}
.ws-banner-t { font-weight: 600; font-size: __TBASE__; color: var(--ws-ink); letter-spacing: -0.005em; }
.ws-banner-b { font-size: __TSM__; color: var(--ws-ink-2); margin-top: 4px; line-height: 1.6; }

/* --- Evidence table ----------------------------------------------------- */
.ws-table { width: 100%; border-collapse: collapse; font-size: __TSM__; }
.ws-table th {
  text-align: left; font-size: __TXS__; text-transform: uppercase; letter-spacing: .12em;
  color: var(--ws-muted); font-weight: 600; padding: 7px 10px;
  border-bottom: 1px solid var(--ws-rule-strong);
}
.ws-table td {
  padding: 9px 10px; border-bottom: 1px solid var(--ws-rule); color: var(--ws-ink);
}
.ws-table td.num { font-family: __FONT_MONO__; font-variant-numeric: tabular-nums; }
.ws-table tr:last-child td { border-bottom: none; }

/* --- Provenance mark ---------------------------------------------------- */
.ws-prov {
  display: inline-flex; align-items: center; gap: 7px;
  font-family: __FONT_MONO__; font-size: __TXS__; letter-spacing: .02em;
  color: var(--ws-ink-2);
  border: 1px solid var(--ws-rule);
  border-radius: 3px; padding: 5px 9px;
}

/* --- Pull quote / premise, set in the display serif --------------------- */
.ws-lede {
  font-family: __FONT_DISPLAY__; font-size: __TLG__; line-height: 1.55;
  color: var(--ws-ink-2); max-width: 62ch;
}

/* --- Demo focus --------------------------------------------------------- */
.ws-focus {
  border-left: 2px solid var(--ws-amber);
  padding: 8px 0 8px 14px; margin-left: -16px;
}
.ws-dim { opacity: .45; transition: opacity .3s ease; }

@keyframes ws-attn {
  0%   { border-left-color: __VERMILION__; }
  50%  { border-left-color: __AMBER__; }
  100% { border-left-color: __VERMILION__; }
}
.ws-attn { animation: ws-attn 1.6s ease-in-out 2; }
@media (prefers-reduced-motion: reduce) {
  .ws-attn { animation: none; }
  .ws-dim, .ws-row { transition: none; }
}

/* --- Streamlit surface alignment ---------------------------------------- */
[data-testid="stSidebar"] {
  background: var(--ws-surface);
  border-right: 1px solid var(--ws-rule);
}
[data-testid="stMetric"] {
  background: none; border: none; border-top: 1px solid var(--ws-rule);
  border-radius: 0; padding: 10px 0 0 0;
}
.stTabs [data-baseweb="tab-list"] { gap: 22px; border-bottom: 1px solid var(--ws-rule-strong); }
.stTabs [data-baseweb="tab"] {
  font-size: __TXS__; font-weight: 600; letter-spacing: .11em; text-transform: uppercase;
  padding-left: 0; padding-right: 0;
}
.stButton button {
  border-radius: 3px; font-weight: 500; font-size: __TSM__;
  min-height: 44px; letter-spacing: .01em;
  border: 1px solid var(--ws-rule-strong); background: transparent; color: var(--ws-ink);
  transition: background-color .2s ease, border-color .2s ease, color .2s ease;
}
.stButton button:hover { border-color: var(--ws-amber); color: var(--ws-amber); background: transparent; }
.stButton button[kind="primary"] {
  background: var(--ws-amber); border-color: var(--ws-amber); color: __GROUND__; font-weight: 600;
}
.stButton button[kind="primary"]:hover { background: #F2B357; border-color: #F2B357; color: __GROUND__; }
:focus-visible { outline: 2px solid var(--ws-amber) !important; outline-offset: 2px; }
hr { border-color: var(--ws-rule); }
.block-container { padding-top: 2.6rem; max-width: 1560px; }
</style>
"""

_CSS = (
    _CSS.replace("__GROUND__", GROUND)
    .replace("__SURFACE__", SURFACE)
    .replace("__INSET__", INSET)
    .replace("__RULE_STRONG__", RULE_STRONG)
    .replace("__RULE__", RULE)
    .replace("__INK2__", INK_2)
    .replace("__INK__", INK)
    .replace("__MUTED__", INK_MUTED)
    .replace("__TRACE__", TRACE)
    .replace("__AMBER__", AMBER)
    .replace("__VERMILION__", VERMILION)
    .replace("__FONT_DISPLAY__", FONT_DISPLAY)
    .replace("__FONT_UI__", FONT_UI)
    .replace("__FONT_MONO__", FONT_MONO)
    .replace("__T3XL__", TYPE["3xl"])
    .replace("__T2XL__", TYPE["2xl"])
    .replace("__TXL__", TYPE["xl"])
    .replace("__TLG__", TYPE["lg"])
    .replace("__TBASE__", TYPE["base"])
    .replace("__TSM__", TYPE["sm"])
    .replace("__TXS__", TYPE["xs"])
)


def inject() -> None:
    """Install the design system. Call once per page, before rendering."""
    st.markdown(_CSS, unsafe_allow_html=True)


def section(title: str, icon_name: str = "", meta: str = "") -> None:
    """A ruled section label. The rule is the container; there is no box."""
    ic = icon(icon_name, 13, INK_MUTED) if icon_name else ""
    mt = f'<span class="ws-sec-s">{meta}</span>' if meta else ""
    st.markdown(
        f'<div class="ws-sec">{ic}<span class="ws-sec-t">{title}</span>{mt}</div>',
        unsafe_allow_html=True,
    )


def kpi(label: str, value: str, meta: str = "", color: str = INK, small: bool = False) -> str:
    size_cls = " ws-kpi-v--sm" if small else ""
    mt = f'<div class="ws-kpi-m">{meta}</div>' if meta else ""
    return (
        f'<div class="ws-kpi-l">{label}</div>'
        f'<div class="ws-kpi-v{size_cls}" style="color:{color};">{value}</div>{mt}'
    )


def banner(title: str, body: str, color: str = INK_MUTED, icon_name: str = "circle-dot",
           attention: bool = False) -> str:
    cls = "ws-banner ws-attn" if attention else "ws-banner"
    return (
        f'<div class="{cls}" style="border-left-color:{color};">'
        f'<div style="padding-top:2px;">{icon(icon_name, 17, color)}</div>'
        f'<div><div class="ws-banner-t">{title}</div>'
        f'<div class="ws-banner-b">{body}</div></div></div>'
    )


def rail(value: float, color: str, height: int = 3) -> str:
    """A thin proportion rail. Colour is passed in by the caller's own logic."""
    return (
        f'<div style="background:{RULE};height:{height}px;border-radius:1px;overflow:hidden;">'
        f'<div style="width:{max(0.0, min(value, 1.0)) * 100:.1f}%;height:100%;'
        f'background:{color};"></div></div>'
    )
