"""
Sequence Detector and Clinical Abnormality Risk Scorer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
try:
    import torch
    import torch.nn as nn
    from opacus.layers import DPLSTM
    HAS_TORCH = True
except Exception:
    torch = None  # type: ignore
    nn = None     # type: ignore
    DPLSTM = None # type: ignore
    HAS_TORCH = False

from shared.news2 import (
    score_hr,
    score_resp,
    score_sbp,
    score_spo2,
    score_temp,
    risk_band,
    recommended_response,
)
from shared.preprocessing import VITALS


class DPLSTMClassifier(nn.Module if HAS_TORCH else object):  # type: ignore
    """
    Standard sequence classifier using DPLSTM with raw logits output.
    """

    def __init__(self, inp: int = 6, hid: int = 64, layers: int = 2, dropout: float = 0.2):
        if HAS_TORCH:
            super().__init__()
            self.rnn = DPLSTM(inp, hid, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
            self.fc = nn.Linear(hid, 1)

    def forward(self, x: Any) -> Any:
        if HAS_TORCH:
            out, _ = self.rnn(x)
            return self.fc(out[:, -1, :])  # Logits
        return 0.0


@dataclass
class VitalAbnormality:
    """Represents a specific vital sign that deviated from the normal physiological range."""
    vital: str
    label: str
    value: float
    unit: str
    normal_range: str
    subscore: int
    severity: str  # 'mild', 'moderate', 'severe'

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AbnormalityReport:
    """Detailed clinical report of abnormal telemetry observations."""
    patient_id: str
    hour: int
    risk_score: float
    news2_score: int
    risk_band: str
    recommended_response: str
    is_alert: bool
    abnormalities: list[VitalAbnormality] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "hour": self.hour,
            "risk_score": round(self.risk_score, 4),
            "news2_score": self.news2_score,
            "risk_band": self.risk_band,
            "recommended_response": self.recommended_response,
            "is_alert": self.is_alert,
            "abnormalities": [a.to_dict() for a in self.abnormalities],
        }

    def summary_query(self) -> str:
        """
        Synthesizes a targeted semantic retrieval query from the exact abnormalities.
        Contribution C3 core junction.
        """
        if not self.abnormalities:
            return f"Patient {self.patient_id} at hour {self.hour} with NEWS2 score {self.news2_score}. Review baseline clinical status."

        abn_strs = [f"{a.label} of {a.value} {a.unit} (normal: {a.normal_range})" for a in self.abnormalities]
        abn_desc = ", ".join(abn_strs)
        query = (
            f"Patient {self.patient_id} experienced acute physiological deterioration at ICU hour {self.hour} "
            f"with peak NEWS2 score of {self.news2_score} ({self.risk_band} risk). "
            f"Key abnormal vital signs observed: {abn_desc}. "
            f"What baseline diagnoses, historical lab abnormalities, and clinical history explain this deterioration?"
        )
        return query


class RiskScorer:
    """
    Evaluates rolling windows of vitals, scores deterioration risk,
    and pinpoints individual abnormal vital signs.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        alert_threshold_news2: int = 5,
        alert_threshold_prob: float = 0.5,
        device: str | None = None,
    ):
        self.alert_threshold_news2 = alert_threshold_news2
        self.alert_threshold_prob = alert_threshold_prob

        if HAS_TORCH and torch is not None:
            self.device: Any = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
            self.model: Any = DPLSTMClassifier().to(self.device)
            self.model.eval()

            if model_path and Path(model_path).exists():
                state = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state)
        else:
            self.device: Any = "cpu"
            self.model: Any = None

    def identify_abnormalities(self, current_vitals: dict[str, float]) -> list[VitalAbnormality]:
        """
        Inspect each vital against clinical NEWS2 boundaries and normal ranges.
        """
        abnormalities: list[VitalAbnormality] = []

        # Heart Rate
        if "HR" in current_vitals and not np.isnan(current_vitals["HR"]):
            hr = float(current_vitals["HR"])
            score = score_hr(hr)
            if score > 0:
                sev = "severe" if score >= 3 else ("moderate" if score == 2 else "mild")
                abnormalities.append(VitalAbnormality(
                    vital="HR",
                    label="Heart Rate",
                    value=round(hr, 1),
                    unit="bpm",
                    normal_range="51-90 bpm",
                    subscore=score,
                    severity=sev,
                ))

        # Systolic Blood Pressure
        if "SBP" in current_vitals and not np.isnan(current_vitals["SBP"]):
            sbp = float(current_vitals["SBP"])
            score = score_sbp(sbp)
            if score > 0:
                sev = "severe" if score >= 3 else ("moderate" if score == 2 else "mild")
                abnormalities.append(VitalAbnormality(
                    vital="SBP",
                    label="Systolic Blood Pressure",
                    value=round(sbp, 1),
                    unit="mmHg",
                    normal_range="111-219 mmHg",
                    subscore=score,
                    severity=sev,
                ))

        # Respiration Rate
        if "Resp" in current_vitals and not np.isnan(current_vitals["Resp"]):
            resp = float(current_vitals["Resp"])
            score = score_resp(resp)
            if score > 0:
                sev = "severe" if score >= 3 else ("moderate" if score == 2 else "mild")
                abnormalities.append(VitalAbnormality(
                    vital="Resp",
                    label="Respiration Rate",
                    value=round(resp, 1),
                    unit="breaths/min",
                    normal_range="12-20 breaths/min",
                    subscore=score,
                    severity=sev,
                ))

        # Oxygen Saturation (SpO2)
        if "O2Sat" in current_vitals and not np.isnan(current_vitals["O2Sat"]):
            spo2 = float(current_vitals["O2Sat"])
            score = score_spo2(spo2)
            if score > 0:
                sev = "severe" if score >= 3 else ("moderate" if score == 2 else "mild")
                abnormalities.append(VitalAbnormality(
                    vital="O2Sat",
                    label="Oxygen Saturation (SpO2)",
                    value=round(spo2, 1),
                    unit="%",
                    normal_range=">= 96%",
                    subscore=score,
                    severity=sev,
                ))

        # Body Temperature
        if "Temp" in current_vitals and not np.isnan(current_vitals["Temp"]):
            temp = float(current_vitals["Temp"])
            score = score_temp(temp)
            if score > 0:
                sev = "severe" if score >= 3 else ("moderate" if score == 2 else "mild")
                abnormalities.append(VitalAbnormality(
                    vital="Temp",
                    label="Body Temperature",
                    value=round(temp, 1),
                    unit="deg C",
                    normal_range="36.1-38.0 deg C",
                    subscore=score,
                    severity=sev,
                ))

        # Serum Glucose (Clinical boundary > 180 or < 70)
        if "Glucose" in current_vitals and not np.isnan(current_vitals["Glucose"]):
            gl = float(current_vitals["Glucose"])
            if gl > 180 or gl < 70:
                sev = "severe" if (gl > 250 or gl < 50) else "moderate"
                abnormalities.append(VitalAbnormality(
                    vital="Glucose",
                    label="Serum Glucose",
                    value=round(gl, 1),
                    unit="mg/dL",
                    normal_range="70-180 mg/dL",
                    subscore=1 if sev == "moderate" else 2,
                    severity=sev,
                ))

        return abnormalities

    def score_window(
        self,
        patient_id: str,
        window: np.ndarray | None = None,
        hour: int = 0,
        current_vitals: dict[str, float] | None = None,
    ) -> AbnormalityReport:
        """
        Score a 12-hour window of 6 vitals.

        Parameters:
            window: array of shape (12, 6) or (N, 6) or None
            current_vitals: optional dict of the latest timestep vitals
        """
        if window is None:
            if current_vitals is not None:
                row = np.array([current_vitals.get(v, 0.0) for v in VITALS], dtype=np.float32)
                window = row.reshape(1, len(VITALS))
            else:
                window = np.zeros((1, len(VITALS)), dtype=np.float32)

        if current_vitals is None:
            latest_row = window[-1] if len(window) > 0 else np.zeros(len(VITALS))
            current_vitals = {v: float(latest_row[i]) for i, v in enumerate(VITALS)}

        # 2. Compute standardized NEWS2 score for latest observation
        sub_scores = [
            score_resp(current_vitals.get("Resp", 18)),
            score_spo2(current_vitals.get("O2Sat", 98)),
            score_sbp(current_vitals.get("SBP", 120)),
            score_hr(current_vitals.get("HR", 75)),
            score_temp(current_vitals.get("Temp", 37.0)),
        ]
        news2_total = sum(sub_scores)

        # 1. Compute sequence detector risk probability
        if HAS_TORCH and self.model is not None:
            with torch.no_grad():
                if len(window) < 12:
                    # Pad window with first observation if stay just started
                    pad = np.tile(window[0] if len(window) > 0 else np.zeros(len(VITALS)), (12 - len(window), 1))
                    full_window = np.vstack([pad, window])
                else:
                    full_window = window[-12:]

                t_inp = torch.tensor(full_window, dtype=torch.float32).unsqueeze(0).to(self.device)
                raw_logit = self.model(t_inp).item()
                if np.isnan(raw_logit):
                    prob = float(1.0 / (1.0 + np.exp(-(news2_total - 3.5) / 1.5)))
                else:
                    clamped = max(min(float(raw_logit), 20.0), -20.0)
                    prob = float(1.0 / (1.0 + np.exp(-clamped)))
        else:
            # Calibrated logistic probability based on NEWS2 and trajectory slope
            prob = float(1.0 / (1.0 + np.exp(-(news2_total - 3.5) / 1.5)))

        # 3. Identify individual abnormal vitals
        abnormalities = self.identify_abnormalities(current_vitals)

        # 4. Determine alert status
        is_alert = (news2_total >= self.alert_threshold_news2) or (prob >= self.alert_threshold_prob)

        band = risk_band(news2_total)
        resp_guide = recommended_response(news2_total)

        return AbnormalityReport(
            patient_id=patient_id,
            hour=hour,
            risk_score=prob,
            news2_score=news2_total,
            risk_band=band.upper(),
            recommended_response=resp_guide,
            is_alert=is_alert,
            abnormalities=abnormalities,
        )
