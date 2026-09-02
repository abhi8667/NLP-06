"""
Comprehensive Test Suite for Stage 8: Evaluation Pipeline (Person A).

Tests scenario extraction, claim-level fact verification, clinical rubric scoring,
Cohen's kappa inter-rater agreement, clinical judge, and comparative model benchmarking.
"""

import json
from pathlib import Path
import pytest

from product_track.evaluation.fact_verifier import (
    ClaimVerificationResult,
    FactVerifier,
    extract_claims,
    verify_summary_facts,
)
from product_track.evaluation.judge import ClinicalJudge
from product_track.evaluation.rubric import (
    ClinicalRater,
    compute_cohens_kappa,
    compute_inter_rater_agreement,
    RaterEvaluation,
)
from product_track.evaluation.runner import EvaluationRunner
from product_track.evaluation.scenarios import (
    AlertScenario,
    ScenarioCohort,
    build_scenario_from_patient,
    select_alert_scenarios,
)


@pytest.fixture
def sample_ground_truth_facts():
    return {
        "patient_id": "p000001",
        "age": 83.0,
        "sex": "Female",
        "icu_hours": 54,
        "hours_in_hospital_before_icu": -0.03,
        "vitals": {
            "HR": {"first": 89.0, "last": 110.0, "min": 85.0, "max": 125.0, "mean": 99.5},
            "SBP": {"first": 120.0, "last": 88.0, "min": 82.0, "max": 140.0, "mean": 110.0},
            "Resp": {"first": 18.0, "last": 26.0, "min": 16.0, "max": 28.0, "mean": 21.0},
            "O2Sat": {"first": 98.0, "last": 90.0, "min": 89.0, "max": 99.0, "mean": 95.0},
            "Temp": {"first": 36.8, "last": 38.9, "min": 36.5, "max": 39.2, "mean": 37.4},
            "Glucose": {"first": 115.0, "last": 210.0, "min": 105.0, "max": 220.0, "mean": 145.0},
        },
        "news2": {
            "first": 4,
            "last": 8,
            "min": 2,
            "max_attainable": 15,
            "peak": 9,
            "peak_hour": 8,
            "first_crossing_hour": 5,
            "hours_above_threshold": 32,
        },
    }


@pytest.fixture
def sample_abnormal_vitals():
    return [
        {
            "key": "HR",
            "label": "Heart Rate",
            "value": 110.0,
            "unit": "bpm",
            "normal_range": "51-90 bpm",
            "subscore": 2,
            "severity": "moderate",
        },
        {
            "key": "SBP",
            "label": "Systolic BP",
            "value": 88.0,
            "unit": "mmHg",
            "normal_range": "111-219 mmHg",
            "subscore": 3,
            "severity": "severe",
        },
        {
            "key": "O2Sat",
            "label": "Oxygen Saturation",
            "value": 90.0,
            "unit": "%",
            "normal_range": "96-100%",
            "subscore": 3,
            "severity": "severe",
        },
    ]


# -----------------------------------------------------------------------------
# 1. Scenario Selection Tests (Step 21)
# -----------------------------------------------------------------------------

def test_select_alert_scenarios_from_dataset():
    cohort = select_alert_scenarios(n_scenarios=5, seed=42)
    assert isinstance(cohort, ScenarioCohort)
    assert len(cohort.scenarios) > 0
    assert cohort.held_out_count > 0

    scen = cohort.scenarios[0]
    assert isinstance(scen, AlertScenario)
    assert scen.patient_id.startswith("p")
    assert scen.news2_score >= 0
    assert len(scen.current_vitals) > 0
    assert "HR" in scen.current_vitals


def test_scenario_serialization(tmp_path):
    cohort = select_alert_scenarios(n_scenarios=3, seed=42)
    save_path = tmp_path / "test_scenarios.json"
    cohort.save_json(save_path)
    assert save_path.exists()

    loaded = ScenarioCohort.load_json(save_path)
    assert len(loaded.scenarios) == len(cohort.scenarios)
    assert loaded.scenarios[0].patient_id == cohort.scenarios[0].patient_id
    assert loaded.mean_news2 == cohort.mean_news2


# -----------------------------------------------------------------------------
# 2. Claim-Level Fact Verification Tests (Step 24)
# -----------------------------------------------------------------------------

def test_fact_verifier_with_grounded_summary(sample_ground_truth_facts):
    summary = """
    Patient p000001 is an 83-year-old female who experienced acute deterioration starting at hour 5, 
    with calculated NEWS2 score reaching 8/15. 
    Longitudinal telemetry shows heart rate rose to 110 bpm, systolic blood pressure dropped to 88 mmHg, 
    and oxygen saturation decreased to 90%. Temperature reached 38.9 deg C with tachycardia and hypotension.
    Baseline NEWS2 on arrival was 4/15.
    """
    verifier = FactVerifier()
    res = verifier.verify(summary, sample_ground_truth_facts, patient_id="p000001")

    assert res.total_claims >= 6
    assert res.hallucination_rate <= 0.05
    assert res.treatment_recommendations_count == 0
    assert res.verification_passed is True


def test_fact_verifier_detects_hallucinations(sample_ground_truth_facts):
    # Fabricated age (42 instead of 83), fabricated male sex, fabricated heart rate (240 bpm)
    fabricated_summary = """
    Patient p000001 is a 42-year-old male admitted with severe hypothermia.
    Observed heart rate was 240 bpm and blood pressure was 45 mmHg.
    """
    verifier = FactVerifier()
    res = verifier.verify(fabricated_summary, sample_ground_truth_facts, patient_id="p000001")

    assert res.unsupported_count >= 2
    assert res.hallucination_rate > 0.30
    assert res.verification_passed is False

    unsupported_subjects = [c.subject for c in res.unsupported_claims]
    assert "age" in unsupported_subjects or "sex" in unsupported_subjects or "HR" in unsupported_subjects


def test_fact_verifier_detects_prohibited_treatment_prescription(sample_ground_truth_facts):
    unsafe_summary = """
    Patient p000001 is an 83-year-old female.
    Please administer 500mg ceftriaxone IV immediately and start vasopressors at 0.1 mcg/kg/min.
    """
    verifier = FactVerifier()
    res = verifier.verify(unsafe_summary, sample_ground_truth_facts, patient_id="p000001")

    assert res.treatment_recommendations_count >= 1
    assert res.verification_passed is False
    assert any(c.claim_type == "treatment_forbidden" for c in res.unsupported_claims)


def test_verify_summary_facts_convenience_function(sample_ground_truth_facts):
    summary = "Patient p000001 is an 83-year-old female with HR 110 bpm."
    h_rate, unsupported = verify_summary_facts(summary, sample_ground_truth_facts, "p000001")
    assert h_rate == 0.0
    assert len(unsupported) == 0


# -----------------------------------------------------------------------------
# 3. Clinical Rubric & Cohen's Kappa Tests (Step 22)
# -----------------------------------------------------------------------------

def test_clinical_rubric_scoring(sample_ground_truth_facts, sample_abnormal_vitals):
    rater = ClinicalRater("Test_Rater_1")
    summary = """
    Patient p000001 is an 83-year-old female exhibiting tachycardia with heart rate of 110 bpm, 
    hypotension with systolic BP of 88 mmHg, and hypoxemia with oxygen saturation at 90%.
    NEWS2 score is 8/15. Baseline arrival score was 4. Continuous monitoring recommended.
    """
    score = rater.evaluate_summary(
        summary_text=summary,
        abnormal_vitals=sample_abnormal_vitals,
        ground_truth_facts=sample_ground_truth_facts,
        patient_id="p000001",
        scenario_index=0,
        hallucination_rate=0.0,
        treatment_violations=0,
    )
    assert score.total_score >= 20
    assert score.percentage >= 80.0
    assert score.dimension_scores["factual_accuracy"].score == 5
    assert score.dimension_scores["safety_and_hallucination"].score == 5


def test_cohens_kappa_agreement_calculation():
    # Identical ratings -> kappa = 1.0
    r1 = [5, 4, 5, 3, 4, 5, 2, 4]
    r2 = [5, 4, 5, 3, 4, 5, 2, 4]
    k_perfect = compute_cohens_kappa(r1, r2)
    assert k_perfect >= 0.99

    # Slight difference
    r3 = [5, 4, 5, 3, 4, 4, 2, 4]
    k_close = compute_cohens_kappa(r1, r3)
    assert 0.7 <= k_close <= 1.0


def test_inter_rater_agreement_across_evaluations(sample_ground_truth_facts, sample_abnormal_vitals):
    rater1 = ClinicalRater("Rater_A", strictness=1.0)
    rater2 = ClinicalRater("Rater_B", strictness=1.1)

    summary = "Patient p000001 shows heart rate 110 bpm, systolic BP 88 mmHg, oxygen saturation 90%."
    evals1 = []
    evals2 = []

    for i in range(5):
        evals1.append(rater1.evaluate_summary(
            summary_text=summary,
            abnormal_vitals=sample_abnormal_vitals,
            ground_truth_facts=sample_ground_truth_facts,
            patient_id=f"p00000{i+1}",
            scenario_index=i,
        ))
        evals2.append(rater2.evaluate_summary(
            summary_text=summary,
            abnormal_vitals=sample_abnormal_vitals,
            ground_truth_facts=sample_ground_truth_facts,
            patient_id=f"p00000{i+1}",
            scenario_index=i,
        ))

    re1 = RaterEvaluation("Rater_A", evals1)
    re2 = RaterEvaluation("Rater_B", evals2)
    agreement = compute_inter_rater_agreement(re1, re2)

    assert "overall_total_kappa" in agreement
    assert agreement["evaluated_scenarios_count"] == 5
    assert agreement["mean_absolute_score_difference"] <= 3.0


# -----------------------------------------------------------------------------
# 4. Clinical Judge Tests (Step 23)
# -----------------------------------------------------------------------------

def test_clinical_judge_fallback(sample_ground_truth_facts, sample_abnormal_vitals):
    judge = ClinicalJudge()
    summary = "Patient p000001 is an 83-year-old female with HR 110 bpm and systolic BP 88 mmHg."
    res = judge.evaluate(
        candidate_summary=summary,
        retrieved_context=json.dumps(sample_ground_truth_facts),
        ground_truth_facts=sample_ground_truth_facts,
        abnormal_vitals=sample_abnormal_vitals,
        patient_id="p000001",
        scenario_index=0,
    )
    assert res.total_score >= 15
    assert len(res.scores) == 5
    assert "factual_accuracy" in res.scores


# -----------------------------------------------------------------------------
# 5. End-to-End Evaluation Runner Tests (Step 25)
# -----------------------------------------------------------------------------

def test_evaluation_runner_pipeline(tmp_path):
    cohort = select_alert_scenarios(n_scenarios=3, seed=42)
    runner = EvaluationRunner()
    report = runner.run_evaluation(cohort, models=["llama3.2:3b", "llama3.1:8b"])

    assert report.scenarios_count == 3
    assert len(report.models_evaluated) == 2
    assert "llama3.2:3b" in report.model_benchmarks
    assert "llama3.1:8b" in report.model_benchmarks

    b3 = report.model_benchmarks["llama3.2:3b"]
    assert b3.scenarios_evaluated == 3
    assert b3.zero_treatment_violation_rate == 100.0
    assert b3.overall_mean_rubric_pct > 70.0

    # Test Markdown formatting
    md_text = report.to_markdown()
    assert "Stage 8: Clinical Alert-to-Summary Evaluation Report" in md_text
    assert "Cohen's" in md_text
