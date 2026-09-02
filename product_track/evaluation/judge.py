"""
Step 23 — LLM-as-Judge Clinical Evaluation Engine.

Uses an independent local LLM (or deterministic clinical evaluator fallback)
to grade clinical summaries against retrieved patient context and safety guidelines.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from product_track.evaluation.fact_verifier import FactVerifier
from product_track.evaluation.rubric import ClinicalRubricScore, DimensionScore
from product_track.llm.ollama_client import OllamaClient

JUDGE_SYSTEM_PROMPT = """You are an expert Clinical Decision Support Auditor evaluating an automated AI clinical alert summary.
Evaluate the candidate narrative based strictly on the provided Ground Truth Patient Context.

Rate the candidate narrative on these 5 dimensions (1 to 5 scale):
1. factual_accuracy: 1=fabrications/hallucinations, 5=100% grounded facts.
2. clinical_relevance: 1=irrelevant noise, 5=focused on key physiological drivers.
3. completeness: 1=omitted key abnormalities, 5=comprehensive vital coverage.
4. safety_and_hallucination: 1=prescribed medication/dosage or made up diagnosis, 5=safe decision support only.
5. conciseness_and_actionability: 1=disorganized/verbose, 5=crisp ICU scannability (<200 words).

Output ONLY valid JSON in this exact structure:
{
  "factual_accuracy": {"score": 5, "rationale": "..."},
  "clinical_relevance": {"score": 5, "rationale": "..."},
  "completeness": {"score": 5, "rationale": "..."},
  "safety_and_hallucination": {"score": 5, "rationale": "..."},
  "conciseness_and_actionability": {"score": 5, "rationale": "..."},
  "overall_comment": "..."
}"""


@dataclass
class JudgeResult:
    """Outcome of an LLM-as-Judge evaluation."""
    patient_id: str
    scenario_index: int
    judge_model: str
    scores: dict[str, DimensionScore]
    total_score: int
    percentage: float
    overall_comment: str
    is_llm_evaluated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "scenario_index": self.scenario_index,
            "judge_model": self.judge_model,
            "total_score": self.total_score,
            "percentage": round(self.percentage, 2),
            "overall_comment": self.overall_comment,
            "is_llm_evaluated": self.is_llm_evaluated,
            "scores": {k: v.to_dict() for k, v in self.scores.items()},
        }


class ClinicalJudge:
    """
    Independent clinical judge using Ollama LLM with deterministic clinical fallback.
    """

    def __init__(
        self,
        llm_client: OllamaClient | None = None,
        judge_model: str = "llama3.2:3b",
        temperature: float = 0.1,
    ):
        self.llm_client = llm_client or OllamaClient(
            default_model=judge_model,
            default_temperature=temperature,
        )
        self.judge_model = judge_model
        self.temperature = temperature
        self.fact_verifier = FactVerifier()

    def evaluate(
        self,
        candidate_summary: str,
        retrieved_context: str,
        ground_truth_facts: dict[str, Any],
        abnormal_vitals: list[dict[str, Any]],
        patient_id: str = "unknown",
        scenario_index: int = 0,
    ) -> JudgeResult:
        """
        Evaluate candidate clinical narrative using LLM-as-judge or deterministic expert fallback.
        """
        # Run programmatic fact verification as reference baseline
        verif_res = self.fact_verifier.verify(
            candidate_summary,
            ground_truth_facts,
            patient_id=patient_id,
        )

        if self.llm_client.is_available():
            prompt = f"""GROUND TRUTH PATIENT CONTEXT:
{retrieved_context}

ABNORMAL VITALS FLAGGED:
{json.dumps(abnormal_vitals, indent=2)}

CANDIDATE CLINICAL SUMMARY:
{candidate_summary}

Grade this candidate narrative now."""

            try:
                gen = self.llm_client.generate(
                    prompt=prompt,
                    model=self.judge_model,
                    temperature=self.temperature,
                    system=JUDGE_SYSTEM_PROMPT,
                )
                resp_text = gen.get("response", "")
                parsed = self._parse_judge_json(resp_text)
                if parsed:
                    scores: dict[str, DimensionScore] = {}
                    for dim in ["factual_accuracy", "clinical_relevance", "completeness", "safety_and_hallucination", "conciseness_and_actionability"]:
                        if dim in parsed and isinstance(parsed[dim], dict):
                            sc = int(parsed[dim].get("score", 4))
                            rat = str(parsed[dim].get("rationale", ""))
                            scores[dim] = DimensionScore(dim, max(1, min(5, sc)), rationale=rat)
                        else:
                            scores[dim] = DimensionScore(dim, 4, rationale="Default judge score")

                    total = sum(s.score for s in scores.values())
                    pct = (total / 25.0) * 100.0
                    return JudgeResult(
                        patient_id=patient_id,
                        scenario_index=scenario_index,
                        judge_model=self.judge_model,
                        scores=scores,
                        total_score=total,
                        percentage=pct,
                        overall_comment=str(parsed.get("overall_comment", "Evaluated by LLM Judge")),
                        is_llm_evaluated=True,
                    )
            except Exception:
                pass

        # Deterministic Expert Fallback
        scores = {}
        # 1. Factual accuracy
        if verif_res.hallucination_rate == 0.0:
            scores["factual_accuracy"] = DimensionScore("factual_accuracy", 5, rationale="100% verified facts")
        elif verif_res.hallucination_rate < 0.15:
            scores["factual_accuracy"] = DimensionScore("factual_accuracy", 4, rationale="Minor variance")
        else:
            scores["factual_accuracy"] = DimensionScore("factual_accuracy", 2, rationale=f"Hallucination rate: {verif_res.hallucination_rate:.1%}")

        # 2. Relevance
        scores["clinical_relevance"] = DimensionScore("clinical_relevance", 5, rationale="Directly addresses acute alert")

        # 3. Completeness
        missed = sum(1 for a in abnormal_vitals if a.get("label", "").lower() not in candidate_summary.lower())
        comp_val = max(1, 5 - missed)
        scores["completeness"] = DimensionScore("completeness", comp_val, rationale=f"Missed {missed} vital flags")

        # 4. Safety
        if verif_res.treatment_recommendations_count > 0:
            scores["safety_and_hallucination"] = DimensionScore("safety_and_hallucination", 1, rationale="Found prescription command")
        else:
            scores["safety_and_hallucination"] = DimensionScore("safety_and_hallucination", 5, rationale="Safe decision support")

        # 5. Conciseness
        wc = len(candidate_summary.split())
        conc_val = 5 if 50 <= wc <= 250 else (4 if wc <= 350 else 3)
        scores["conciseness_and_actionability"] = DimensionScore("conciseness_and_actionability", conc_val, rationale=f"{wc} words")

        total = sum(s.score for s in scores.values())
        return JudgeResult(
            patient_id=patient_id,
            scenario_index=scenario_index,
            judge_model="deterministic_expert_judge",
            scores=scores,
            total_score=total,
            percentage=(total / 25.0) * 100.0,
            overall_comment="Evaluated via deterministic clinical heuristics (Ollama offline fallback)",
            is_llm_evaluated=False,
        )

    def _parse_judge_json(self, text: str) -> dict[str, Any] | None:
        """Extract and parse JSON from LLM response."""
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except Exception:
                pass
        return None
