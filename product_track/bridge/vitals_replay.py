"""
PhysioNet ICU Vitals Replay Harness.

Streams finished ICU stays hour-by-hour to simulate real-time bedside telemetry.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Generator

import numpy as np
import pandas as pd

from shared import add_news2, fill_vitals, load_patient, patient_facts
from shared.preprocessing import VITALS


@dataclass
class HourlyTelemetry:
    """Represents a single hour's incoming vitals packet from bedside telemetry."""
    patient_id: str
    hour: int
    vitals: dict[str, float]
    news2: int
    window_buffer: np.ndarray  # Shape: (min(hour+1, 12), 6)
    is_last_hour: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "hour": self.hour,
            "vitals": {k: round(v, 2) for k, v in self.vitals.items()},
            "news2": self.news2,
            "is_last_hour": self.is_last_hour,
        }


class VitalsReplayHarness:
    """
    Simulates a live bedside vital signs monitor by replaying recorded PhysioNet stays.
    """

    def __init__(self, default_interval_s: float = 1.0):
        self.default_interval_s = default_interval_s

    def stream_patient(
        self,
        source: str | Path | pd.DataFrame,
        interval_s: float | None = None,
        max_hours: int | None = None,
    ) -> Generator[HourlyTelemetry, None, None]:
        """
        Stream a patient's ICU stay hour-by-hour.

        Parameters:
            source: Path to .psv file or preloaded DataFrame
            interval_s: Delay in seconds between hours (0 for instant processing)
            max_hours: Maximum hours to stream before stopping
        """
        sleep_dur = self.default_interval_s if interval_s is None else interval_s

        if isinstance(source, (str, Path)):
            df = load_patient(source)
        else:
            df = source.copy()

        df, _ = fill_vitals(df)
        df = add_news2(df)

        pid = df["patient_id"].iloc[0] if "patient_id" in df else "unknown_patient"
        total_hours = len(df)
        limit = min(total_hours, max_hours) if max_hours is not None else total_hours

        vitals_array = df[VITALS].to_numpy(dtype=np.float32)

        for hr in range(limit):
            current_row = df.iloc[hr]
            current_vitals = {v: float(current_row[v]) for v in VITALS if v in current_row}
            current_news2 = int(current_row.get("news2", 0))

            # Maintain sliding window of up to 12 hours
            window_start = max(0, hr - 11)
            window_buffer = vitals_array[window_start : hr + 1]

            is_last = (hr == limit - 1)

            telemetry = HourlyTelemetry(
                patient_id=pid,
                hour=hr,
                vitals=current_vitals,
                news2=current_news2,
                window_buffer=window_buffer,
                is_last_hour=is_last,
            )

            yield telemetry

            if sleep_dur > 0 and not is_last:
                time.sleep(sleep_dur)


def select_demo_patients(
    cohort_dir: str | Path,
    min_peak: int = 7,
    max_candidates: int = 5,
) -> list[dict[str, Any]]:
    """
    Programmatically scan the cohort to find ideal demo patients whose
    NEWS2 score visibly climbs from calm (< 4) to acute deterioration (>= min_peak).
    """
    files = sorted(Path(cohort_dir).glob("*.psv"))
    candidates: list[dict[str, Any]] = []

    for f in files:
        df = load_patient(f)
        df, _ = fill_vitals(df)
        df = add_news2(df)
        facts = patient_facts(df)
        news = facts.get("news2", {})

        first_score = news.get("first", 0)
        peak_score = news.get("peak", 0)
        crossing_hr = news.get("first_crossing_hour")
        peak_hr = news.get("peak_hour")

        # Select patients who start stable and deteriorate significantly
        if (
            first_score <= 4
            and peak_score >= min_peak
            and crossing_hr is not None
            and crossing_hr > 0
        ):
            candidates.append({
                "patient_id": facts["patient_id"],
                "file_path": str(f),
                "age": facts.get("age"),
                "sex": facts.get("sex"),
                "icu_hours": facts.get("icu_hours"),
                "initial_news2": first_score,
                "peak_news2": peak_score,
                "first_crossing_hour": crossing_hr,
                "peak_hour": peak_hr,
                "band_at_peak": news.get("band_at_peak"),
            })

            if len(candidates) >= max_candidates:
                break

    return candidates
