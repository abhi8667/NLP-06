"""
Stage 8: Evaluation Pipeline for Alert-to-Summary Bridge (Contribution C3).

Provides scenario selection, claim-level programmatic fact verification,
5-dimension clinical rubric scoring with Cohen's kappa, LLM-as-judge evaluation,
and edge model comparison (Llama 3.2 3B vs Llama 3.1 8B).
"""

from .fact_verifier import (
    Claim,
    ClaimVerificationResult,
    FactVerifier,
    extract_claims,
    verify_summary_facts,
)
from .judge import ClinicalJudge, JudgeResult
from .rubric import (
    ClinicalRubricScore,
    DimensionScore,
    RaterEvaluation,
    compute_cohens_kappa,
    compute_inter_rater_agreement,
)
from .runner import EvaluationReport, EvaluationRunner
from .scenarios import AlertScenario, ScenarioCohort, select_alert_scenarios

__all__ = [
    "AlertScenario",
    "ScenarioCohort",
    "select_alert_scenarios",
    "Claim",
    "ClaimVerificationResult",
    "FactVerifier",
    "extract_claims",
    "verify_summary_facts",
    "ClinicalRubricScore",
    "DimensionScore",
    "RaterEvaluation",
    "compute_cohens_kappa",
    "compute_inter_rater_agreement",
    "ClinicalJudge",
    "JudgeResult",
    "EvaluationReport",
    "EvaluationRunner",
]
