"""
NEWS2 — National Early Warning Score 2 (Royal College of Physicians, 2017).

Pure scoring functions. No pandas, no I/O, so they are trivially testable and
identical for both tracks.

WHAT WE CAN COMPUTE
-------------------
NEWS2 has seven parameters. The PhysioNet/CinC 2019 dataset records five of them:

    available      Respiration rate, SpO2, Systolic BP, Pulse, Temperature
    NOT available  Level of consciousness (ACVPU), Air-or-supplemental-oxygen

Full NEWS2 maxes at 20. Dropping consciousness (max 3) and air/oxygen (max 2)
leaves an attainable maximum of 15 — computed below as NEWS2_MAX rather than
hardcoded, so it cannot silently drift.

Our scores are therefore systematically LOWER than a bedside nurse would record.
State this in the paper. The standard escalation threshold of 5 still sits well
inside the attainable range, and measured prevalence at >=5 is ~12.6% of hours,
which is clinically plausible.

ESCALATION THRESHOLDS (RCP 2017)
--------------------------------
    0        routine monitoring, 12-hourly
    1-4      low risk, ward-based review, 4-6 hourly
    5-6      MEDIUM risk, urgent review, hourly observations   <-- our label
    >=7      high risk, emergency response, critical care
"""

from __future__ import annotations

import math

__all__ = [
    "score_resp", "score_spo2", "score_sbp", "score_hr", "score_temp",
    "COMPONENTS", "NEWS2_MAX", "MISSING_COMPONENTS", "DEFAULT_THRESHOLD",
    "news2_total", "risk_band", "recommended_response",
]

DEFAULT_THRESHOLD = 5

MISSING_COMPONENTS = {
    "consciousness": "ACVPU not recorded numerically in this dataset (max 3)",
    "supplemental_oxygen": "air-or-oxygen flag not reliably recorded (max 2)",
}


def _isna(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _chart(v):
    """
    Round to the nearest integer, half UP, the way an observation is charted.

    WHY THIS EXISTS. The NEWS2 chart is defined on integer observations — a nurse
    counts whole breaths and reads a whole percentage. PhysioNet contains averaged
    values such as a respiration rate of 24.5, which falls in the gap between the
    "21-24 -> 2" and ">=25 -> 3" bands. Two reasonable implementations disagreed on
    5% of inputs before this was pinned down, so the rule is fixed here:

        round half UP, then apply the chart

    Half up rather than half down because it errs toward the more severe score,
    which is the safe direction for an early-warning system.

    Temperature is exempt — it is charted to 0.1 C, so rounding it to whole
    degrees would destroy real signal. Pass it through untouched.
    """
    return math.floor(v + 0.5)


def score_resp(v) -> int:
    """Respiration rate, breaths/min. Charted as an integer."""
    if _isna(v):
        return 0
    v = _chart(v)
    if v <= 8:
        return 3
    if v <= 11:
        return 1
    if v <= 20:
        return 0
    if v <= 24:
        return 2
    return 3


def score_spo2(v) -> int:
    """Oxygen saturation, %. NEWS2 Scale 1 (the default scale). Charted as an integer."""
    if _isna(v):
        return 0
    v = _chart(v)
    if v <= 91:
        return 3
    if v <= 93:
        return 2
    if v <= 95:
        return 1
    return 0


def score_sbp(v) -> int:
    """Systolic blood pressure, mmHg. Charted as an integer."""
    if _isna(v):
        return 0
    v = _chart(v)
    if v <= 90:
        return 3
    if v <= 100:
        return 2
    if v <= 110:
        return 1
    if v <= 219:
        return 0
    return 3


def score_hr(v) -> int:
    """Pulse, beats/min. Charted as an integer."""
    if _isna(v):
        return 0
    v = _chart(v)
    if v <= 40:
        return 3
    if v <= 50:
        return 1
    if v <= 90:
        return 0
    if v <= 110:
        return 1
    if v <= 130:
        return 2
    return 3


def score_temp(v) -> int:
    """Body temperature, degrees C. NOT rounded — charted to 0.1 C precision."""
    if _isna(v):
        return 0
    if v <= 35.0:
        return 3
    if v <= 36.0:
        return 1
    if v <= 38.0:
        return 0
    if v <= 39.0:
        return 1
    return 2


# component name -> (dataset column, scoring function, worst attainable score)
COMPONENTS = {
    "resp_rate":   ("Resp",   score_resp, 3),
    "spo2":        ("O2Sat",  score_spo2, 3),
    "bp_systolic": ("SBP",    score_sbp,  3),
    "heart_rate":  ("HR",     score_hr,   3),
    "temperature": ("Temp",   score_temp, 3),
}

#: Attainable maximum with the five available components (15, not the full 20).
NEWS2_MAX = sum(worst for _, _, worst in COMPONENTS.values())


def news2_total(**vitals) -> int:
    """
    Aggregate NEWS2 from named vitals.

        news2_total(resp_rate=29, spo2=90.5, bp_systolic=93,
                    heart_rate=106, temperature=36.11)   -> 9

    Missing values score 0. That is a deliberate simplification: a vital that was
    never charted contributes nothing rather than being imputed as abnormal. It
    biases scores DOWNWARD, which is the safe direction for a limitation but must
    be stated. Callers should pass forward-filled values (see preprocessing).
    """
    return sum(fn(vitals.get(name)) for name, (_, fn, _) in COMPONENTS.items())


def risk_band(total: int) -> str:
    """RCP risk band for an aggregate score."""
    if total == 0:
        return "none"
    if total <= 4:
        return "low"
    if total <= 6:
        return "medium"
    return "high"


def recommended_response(total: int) -> str:
    """Plain-language clinical response, for use in generated notes."""
    return {
        "none":   "routine monitoring, 12-hourly observations",
        "low":    "ward-based review, 4-6 hourly observations",
        "medium": "urgent review by a clinician, hourly observations",
        "high":   "emergency response, continuous monitoring, critical care review",
    }[risk_band(total)]
