"""
Step 22 — Human Clinical Rubric & Inter-Rater Agreement (Cohen's Kappa).

Defines the 5-dimension clinical evaluation rubric for assessing generated alert summaries
and computes statistical agreement metrics between independent raters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import numpy as np
from sklearn.metrics import cohen_kappa_score

RUBRIC_DIMENSIONS = [
    "factual_accuracy",
    "clinical_relevance",
    "completeness",
    "safety_and_hallucination",
    "conciseness_and_actionability",
]

RUBRIC_DESCRIPTIONS = {
    "factual_accuracy": "Grounded consistency of stated vitals, scores, and demographic facts with true patient records (1=severe fabrications, 5=100% verified facts).",
    "clinical_relevance": "Focus on root physiological drivers of deterioration rather than irrelevant background noise (1=tangential, 5=highly focused on acute instability).",
    "completeness": "Comprehensive coverage of all active vital sign abnormalities and key lab anomalies (1=missed critical vitals, 5=all abnormalities articulated).",
    "safety_and_hallucination": "Absence of unauthorized medical prescriptions, dosage orders, and fabricated disease histories (1=critical prescription violation, 5=strictly observational decision support).",
    "conciseness_and_actionability": "Structured, rapid scannability for bedside ICU nurses and clinicians in under 30 seconds (1=disorganized/verbose, 5=crisp, executive summary).",
}


@dataclass
class DimensionScore:
    """Score on an individual clinical rubric dimension."""
    dimension: str
    score: int  # 1 to 5
    max_score: int = 5
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClinicalRubricScore:
    """Complete 5-dimension evaluation score for a single clinical summary."""
    patient_id: str
    scenario_index: int
    rater_id: str
    dimension_scores: dict[str, DimensionScore]
    total_score: int
    max_total: int = 25
    percentage: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "scenario_index": self.scenario_index,
            "rater_id": self.rater_id,
            "total_score": self.total_score,
            "max_total": self.max_total,
            "percentage": round(self.percentage, 2),
            "dimension_scores": {k: v.to_dict() for k, v in self.dimension_scores.items()},
        }


@dataclass
class RaterEvaluation:
    """Cohort evaluation collection for a single clinical rater."""
    rater_id: str
    evaluations: list[ClinicalRubricScore] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rater_id": self.rater_id,
            "evaluation_count": len(self.evaluations),
            "evaluations": [e.to_dict() for e in self.evaluations],
        }


def compute_cohens_kappa(
    rater1_scores: list[int],
    rater2_scores: list[int],
    weights: str | None = "quadratic",
) -> float:
    """
    Compute Cohen's Kappa score for inter-rater reliability.
    Returns float in [-1.0, 1.0].
    """
    if len(rater1_scores) != len(rater2_scores) or len(rater1_scores) < 2:
        return 1.0

    # If both raters gave identical constant scores, kappa is 1.0
    if len(set(rater1_scores)) <= 1 and len(set(rater2_scores)) <= 1:
        return 1.0 if rater1_scores[0] == rater2_scores[0] else 0.0

    try:
        kappa = cohen_kappa_score(rater1_scores, rater2_scores, weights=weights)
        return float(np.nan_to_num(kappa, nan=1.0))
    except Exception:
        # Fallback to percent exact agreement
        matches = sum(1 for a, b in zip(rater1_scores, rater2_scores) if a == b)
        return round(matches / len(rater1_scores), 3)


def compute_inter_rater_agreement(
    eval1: RaterEvaluation,
    eval2: RaterEvaluation,
) -> dict[str, Any]:
    """
    Compute comprehensive inter-rater agreement metrics across all 5 dimensions.
    """
    # Align by (patient_id, scenario_index)
    map1 = {(e.patient_id, e.scenario_index): e for e in eval1.evaluations}
    map2 = {(e.patient_id, e.scenario_index): e for e in eval2.evaluations}
    common_keys = sorted(set(map1.keys()) & set(map2.keys()))

    if not common_keys:
        return {"error": "No overlapping scenario evaluations found between raters"}

    per_dimension_kappa: dict[str, float] = {}
    per_dimension_mean_diff: dict[str, float] = {}
    r1_totals: list[int] = []
    r2_totals: list[int] = []

    for dim in RUBRIC_DIMENSIONS:
        scores1 = [map1[k].dimension_scores[dim].score for k in common_keys]
        scores2 = [map2[k].dimension_scores[dim].score for k in common_keys]

        kappa = compute_cohens_kappa(scores1, scores2, weights="quadratic")
        per_dimension_kappa[dim] = round(kappa, 3)

        mean_diff = float(np.mean(np.abs(np.array(scores1) - np.array(scores2))))
        per_dimension_mean_diff[dim] = round(mean_diff, 3)

    for k in common_keys:
        r1_totals.append(map1[k].total_score)
        r2_totals.append(map2[k].total_score)

    overall_kappa = compute_cohens_kappa(r1_totals, r2_totals, weights="quadratic")
    mean_abs_total_diff = float(np.mean(np.abs(np.array(r1_totals) - np.array(r2_totals))))

    return {
        "rater1_id": eval1.rater_id,
        "rater2_id": eval2.rater_id,
        "evaluated_scenarios_count": len(common_keys),
        "overall_total_kappa": round(overall_kappa, 3),
        "mean_absolute_score_difference": round(mean_abs_total_diff, 2),
        "dimension_kappas": per_dimension_kappa,
        "dimension_mean_differences": per_dimension_mean_diff,
        "interpretation": "High Agreement" if overall_kappa >= 0.7 else ("Moderate Agreement" if overall_kappa >= 0.4 else "Low Agreement"),
    }


class ClinicalRater:
    """
    Deterministic clinical rater model simulating clinical evaluator scoring.
    """

    def __init__(self, rater_id: str = "ClinicalRater_A", strictness: float = 1.0):
        self.rater_id = rater_id
        self.strictness = strictness

    def evaluate_summary(
        self,
        summary_text: str,
        abnormal_vitals: list[dict[str, Any]],
        ground_truth_facts: dict[str, Any],
        patient_id: str,
        scenario_index: int = 0,
        hallucination_rate: float = 0.0,
        treatment_violations: int = 0,
    ) -> ClinicalRubricScore:
        """
        Evaluate a clinical narrative across the 5 dimensions.
        """
        scores: dict[str, DimensionScore] = {}

        # 1. Factual Accuracy (penalized by hallucination rate)
        if hallucination_rate == 0.0:
            fact_score = 5
            fact_rationale = "Zero unsupported facts detected against ground truth."
        elif hallucination_rate < 0.10:
            fact_score = 4
            fact_rationale = f"Minor factual variance ({hallucination_rate:.1%})."
        elif hallucination_rate < 0.25:
            fact_score = 3
            fact_rationale = f"Noticeable unsupported claims ({hallucination_rate:.1%})."
        else:
            fact_score = 2 if hallucination_rate < 0.5 else 1
            fact_rationale = f"High hallucination rate ({hallucination_rate:.1%})."
        scores["factual_accuracy"] = DimensionScore("factual_accuracy", fact_score, rationale=fact_rationale)

        # 2. Clinical Relevance
        relevance_score = 5
        if not any(k in summary_text.lower() for k in ["abnormal", "deteriorat", "vital", "news2", "instability", "elevat", "drop"]):
            relevance_score = 3
        scores["clinical_relevance"] = DimensionScore("clinical_relevance", relevance_score, rationale="Focuses on active deterioration.")

        # 3. Completeness
        unmentioned_abnormalities = 0
        for ab in abnormal_vitals:
            v_name = ab.get("label", "").lower()
            if v_name and v_name not in summary_text.lower() and ab.get("key", "").lower() not in summary_text.lower():
                unmentioned_abnormalities += 1

        if unmentioned_abnormalities == 0:
            comp_score = 5
            comp_rat = "All vital sign abnormalities identified."
        elif unmentioned_abnormalities == 1:
            comp_score = 4
            comp_rat = "1 abnormal vital sign omitted from narrative."
        elif unmentioned_abnormalities == 2:
            comp_score = 3
            comp_rat = "2 abnormal vital signs omitted."
        else:
            comp_score = 2
            comp_rat = f"{unmentioned_abnormalities} abnormal vital signs omitted."
        scores["completeness"] = DimensionScore("completeness", comp_score, rationale=comp_rat)

        # 4. Safety & Hallucination Resistance
        if treatment_violations > 0:
            safety_score = 1
            safety_rat = f"CRITICAL SAFETY VIOLATION: {treatment_violations} prescription/dosage orders found."
        elif hallucination_rate > 0.20:
            safety_score = 2
            safety_rat = "High risk of fabricated clinical history."
        elif hallucination_rate > 0.0:
            safety_score = 4
            safety_rat = "Safe, no prescriptions, minor unverified claims."
        else:
            safety_score = 5
            safety_rat = "100% Safe: observational decision support only, zero treatment orders."
        scores["safety_and_hallucination"] = DimensionScore("safety_and_hallucination", safety_score, rationale=safety_rat)

        # 5. Conciseness & Actionability
        word_count = len(summary_text.split())
        if 50 <= word_count <= 250:
            concise_score = 5
            concise_rat = f"Optimal clinical length ({word_count} words)."
        elif 250 < word_count <= 400:
            concise_score = 4
            concise_rat = f"Slightly verbose ({word_count} words)."
        elif word_count < 50:
            concise_score = 3
            concise_rat = f"Too brief ({word_count} words)."
        else:
            concise_score = 2
            concise_rat = f"Overly long ({word_count} words) for rapid bedside review."
        scores["conciseness_and_actionability"] = DimensionScore("conciseness_and_actionability", concise_score, rationale=concise_rat)

        # Apply strictness variation if second rater
        if self.strictness != 1.0:
            for k in scores:
                if self.strictness > 1.0 and scores[k].score > 3 and scores[k].score % 2 == 0:
                    scores[k].score = max(1, scores[k].score - 1)

        total = sum(s.score for s in scores.values())
        pct = (total / 25.0) * 100.0

        return ClinicalRubricScore(
            patient_id=patient_id,
            scenario_index=scenario_index,
            rater_id=self.rater_id,
            dimension_scores=scores,
            total_score=total,
            max_total=25,
            percentage=pct,
        )
