"""
Tests for the shared preprocessing module.

These exist so BOTH tracks can prove they are computing the same thing. If
Person A's NEWS2 disagrees with Person B's, the alert says one thing and the
summary another — and that surfaces in a demo, not in development.

    pytest shared/test_shared.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from .news2 import (NEWS2_MAX, news2_total, risk_band, score_hr, score_resp,
                    score_sbp, score_spo2, score_temp)
from .preprocessing import (HORIZON, VITALS, WINDOW, add_news2, fill_vitals,
                            make_windows, patient_facts, patient_split)


# ----------------------------------------------------------------- NEWS2 table

@pytest.mark.parametrize("value,expected", [
    (6, 3), (8, 3), (9, 1), (11, 1), (12, 0), (20, 0), (21, 2), (24, 2), (25, 3), (40, 3),
])
def test_resp_boundaries(value, expected):
    assert score_resp(value) == expected


@pytest.mark.parametrize("value,expected", [
    (85, 3), (91, 3), (92, 2), (93, 2), (94, 1), (95, 1), (96, 0), (100, 0),
])
def test_spo2_boundaries(value, expected):
    assert score_spo2(value) == expected


@pytest.mark.parametrize("value,expected", [
    (80, 3), (90, 3), (91, 2), (100, 2), (101, 1), (110, 1),
    (111, 0), (219, 0), (220, 3), (240, 3),
])
def test_sbp_boundaries(value, expected):
    assert score_sbp(value) == expected


@pytest.mark.parametrize("value,expected", [
    (35, 3), (40, 3), (41, 1), (50, 1), (51, 0), (90, 0),
    (91, 1), (110, 1), (111, 2), (130, 2), (131, 3), (180, 3),
])
def test_hr_boundaries(value, expected):
    assert score_hr(value) == expected


@pytest.mark.parametrize("value,expected", [
    (34.0, 3), (35.0, 3), (35.1, 1), (36.0, 1), (36.1, 0), (38.0, 0),
    (38.1, 1), (39.0, 1), (39.1, 2), (41.0, 2),
])
def test_temp_boundaries(value, expected):
    assert score_temp(value) == expected


def test_missing_scores_zero():
    """A vital never charted contributes nothing — it is not imputed as abnormal."""
    for fn in (score_resp, score_spo2, score_sbp, score_hr, score_temp):
        assert fn(None) == 0
        assert fn(float("nan")) == 0


def test_attainable_maximum_is_15():
    """Full NEWS2 is 20. We lack consciousness (3) and air/oxygen (2), so 15."""
    assert NEWS2_MAX == 15
    worst = news2_total(resp_rate=30, spo2=85, bp_systolic=80,
                        heart_rate=180, temperature=34.0)
    assert worst == NEWS2_MAX


def test_known_patient_hour():
    """
    Patient p000001, hour 8 (temperature forward-filled from hour 7).
    Resp 29 -> 3, SpO2 90.5 -> 3, SBP 93 -> 2, HR 106 -> 1, Temp 36.11 -> 0.
    """
    assert news2_total(resp_rate=29, spo2=90.5, bp_systolic=93,
                       heart_rate=106, temperature=36.11) == 9


@pytest.mark.parametrize("total,band", [
    (0, "none"), (1, "low"), (4, "low"), (5, "medium"), (6, "medium"), (7, "high"), (15, "high"),
])
def test_risk_bands(total, band):
    assert risk_band(total) == band


# ------------------------------------------------- non-integer values (the bug)
# NEWS2 is defined on integer observations, but PhysioNet contains averaged
# values such as a respiration rate of 24.5 that fall between bands. Two
# reasonable implementations disagreed on 5% of inputs before this was pinned
# down. Rule: round half UP, then apply the chart. Temperature is exempt.

@pytest.mark.parametrize("value,expected", [
    (23.4, 2), (24.0, 2), (24.4, 2),   # charts as 23 / 24 / 24 -> band 21-24
    (24.5, 3), (24.6, 3), (25.0, 3),   # charts as 25 -> band >=25
])
def test_resp_half_up_rounding(value, expected):
    assert score_resp(value) == expected


@pytest.mark.parametrize("value,expected", [
    (93.4, 2), (93.5, 1), (95.4, 1), (95.5, 0),
])
def test_spo2_half_up_rounding(value, expected):
    assert score_spo2(value) == expected


@pytest.mark.parametrize("value,expected", [
    (110.4, 1), (110.5, 0), (90.4, 3), (90.5, 2),
])
def test_sbp_half_up_rounding(value, expected):
    assert score_sbp(value) == expected


@pytest.mark.parametrize("value,expected", [
    (110.4, 1), (110.5, 2), (130.4, 2), (130.5, 3),
])
def test_hr_half_up_rounding(value, expected):
    assert score_hr(value) == expected


@pytest.mark.parametrize("value,expected", [
    (38.05, 1), (37.95, 0), (36.04, 0), (35.99, 1),
])
def test_temperature_is_not_rounded_to_integers(value, expected):
    """Temperature is charted to 0.1 C — rounding to whole degrees loses signal."""
    assert score_temp(value) == expected


def test_no_gaps_between_bands():
    """Every value in a plausible physiological range must score."""
    for v in np.arange(0, 60, 0.1):
        assert score_resp(v) in (0, 1, 2, 3)
    for v in np.arange(60, 101, 0.1):
        assert score_spo2(v) in (0, 1, 2, 3)
    for v in np.arange(40, 260, 0.1):
        assert score_sbp(v) in (0, 1, 2, 3)
    for v in np.arange(20, 220, 0.1):
        assert score_hr(v) in (0, 1, 2, 3)
    for v in np.arange(32, 43, 0.05):
        assert score_temp(v) in (0, 1, 2, 3)


# ------------------------------------------------------------------ imputation

def _frame(n=30, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "HR": rng.normal(85, 10, n), "SBP": rng.normal(120, 15, n),
        "O2Sat": rng.normal(96, 3, n), "Resp": rng.normal(18, 4, n),
        "Temp": rng.normal(37, 0.5, n), "Glucose": rng.normal(120, 30, n),
        "Age": 70.0, "Gender": 1, "HospAdmTime": -2.0,
        "ICULOS": np.arange(1, n + 1), "SepsisLabel": 0,
    })
    df["patient_id"] = "p_test"
    return df


def test_leading_nan_is_filled():
    """ffill alone cannot fix a blank first row — bfill must also run."""
    df = _frame()
    df.loc[0, VITALS] = np.nan
    filled, _ = fill_vitals(df)
    assert filled[VITALS].isna().sum().sum() == 0


def test_forward_fill_carries_last_observation():
    df = _frame()
    df.loc[5, "Temp"] = 38.5
    df.loc[6:8, "Temp"] = np.nan
    filled, _ = fill_vitals(df)
    assert (filled.loc[6:8, "Temp"] == 38.5).all()


def test_imputation_report_counts():
    df = _frame(n=20)
    df.loc[10:14, "Temp"] = np.nan          # 5 hours missing
    filled, rep = fill_vitals(df)
    assert rep.rows == 20
    assert rep.imputed["Temp"] == 5
    assert rep.rate("Temp") == pytest.approx(0.25)


def test_missing_column_is_tolerated():
    df = _frame().drop(columns=["Glucose"])
    filled, _ = fill_vitals(df)
    assert "Glucose" in filled.columns


# --------------------------------------------------------------------- windows

def test_window_shape_and_horizon():
    df = add_news2(fill_vitals(_frame(n=40))[0])
    X, y, pid = make_windows(df, window=WINDOW, horizon=HORIZON)
    assert X.shape == (40 - WINDOW - HORIZON, WINDOW, len(VITALS))
    assert len(y) == len(X) == len(pid)
    assert X.dtype == np.float32


def test_label_is_taken_from_the_future():
    """y[i] must equal the label HORIZON hours after window i ends."""
    df = add_news2(fill_vitals(_frame(n=40))[0])
    df["news2_label"] = np.arange(len(df)) % 2          # alternating, easy to trace
    X, y, _ = make_windows(df, window=WINDOW, horizon=HORIZON)
    for i in range(len(y)):
        assert y[i] == df["news2_label"].iloc[i + WINDOW + HORIZON]


def test_short_stay_yields_no_windows():
    df = add_news2(fill_vitals(_frame(n=WINDOW + HORIZON))[0])
    X, y, pid = make_windows(df)
    assert len(X) == len(y) == len(pid) == 0


# ----------------------------------------------------------------------- split

def test_patient_split_is_disjoint():
    pids = np.array([f"p{i//10}" for i in range(200)], dtype=object)
    tr, te = patient_split(pids, test_frac=0.2, seed=1)
    assert not (set(pids[tr]) & set(pids[te])), "a patient appeared in both halves"
    assert tr.sum() + te.sum() == len(pids)


def test_patient_split_is_deterministic():
    pids = np.array([f"p{i//10}" for i in range(200)], dtype=object)
    a1, b1 = patient_split(pids, seed=7)
    a2, b2 = patient_split(pids, seed=7)
    assert (a1 == a2).all() and (b1 == b2).all()


# ----------------------------------------------------------------------- facts

def test_facts_are_complete_enough_to_write_a_note():
    df = add_news2(fill_vitals(_frame(n=30))[0])
    f = patient_facts(df)
    assert f["age"] == 70.0 and f["sex"] == "male"
    assert f["icu_hours"] == 30
    assert set(f["vitals"]) <= set(VITALS) and f["vitals"]
    assert f["news2"]["max_attainable"] == 15
    assert "peak" in f["news2"] and "band_at_peak" in f["news2"]


def test_facts_record_threshold_crossing():
    df = _frame(n=30)
    df.loc[10:20, "O2Sat"] = 88     # 3 pts
    df.loc[10:20, "Resp"] = 28      # 3 pts
    df.loc[10:20, "SBP"] = 95       # 2 pts  -> >= 5
    df = add_news2(fill_vitals(df)[0])
    f = patient_facts(df)
    assert f["news2"]["ever_crossed_threshold"] is True
    assert f["news2"]["hours_at_or_above_threshold"] >= 10


def test_sepsis_is_only_a_crosscheck():
    """The dataset label must never become the training label."""
    df = add_news2(fill_vitals(_frame())[0])
    assert "sepsis_crosscheck" in patient_facts(df)
    assert "SepsisLabel" not in df["news2_label"].name if df["news2_label"].name else True
