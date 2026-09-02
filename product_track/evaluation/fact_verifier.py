"""
Step 24 — Programmatic Claim-Level Fact Verification.

Extracts atomic clinical propositions from generated summaries and verifies them
against row-level ground-truth patient facts. Replaces fragile keyword watchlists.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Claim:
    """An atomic clinical proposition extracted from generated narrative."""
    claim_type: str  # 'vital_value', 'news2_score', 'demographic', 'temporal', 'trend', 'treatment_forbidden', 'condition'
    extracted_text: str
    subject: str
    extracted_value: Any
    is_supported: bool
    ground_truth_value: Any = None
    reason: str = ""
    severity: str = "medium"  # 'low', 'medium', 'high', 'critical'

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClaimVerificationResult:
    """Comprehensive summary verification result across all extracted claims."""
    patient_id: str
    total_claims: int
    supported_count: int
    unsupported_count: int
    hallucination_rate: float
    treatment_recommendations_count: int
    unsupported_claims: list[Claim]
    all_claims: list[Claim]
    verification_passed: bool
    summary_excerpt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "total_claims": self.total_claims,
            "supported_count": self.supported_count,
            "unsupported_count": self.unsupported_count,
            "hallucination_rate": round(self.hallucination_rate, 4),
            "treatment_recommendations_count": self.treatment_recommendations_count,
            "verification_passed": self.verification_passed,
            "summary_excerpt": self.summary_excerpt,
            "unsupported_claims": [c.to_dict() for c in self.unsupported_claims],
            "all_claims": [c.to_dict() for c in self.all_claims],
        }


class FactVerifier:
    """
    Programmatic claim-level fact verifier for clinical summaries.
    """

    def __init__(self, numeric_tolerance: float = 0.05):
        self.numeric_tolerance = numeric_tolerance

    def extract_claims(self, text: str, ground_truth: dict[str, Any]) -> list[Claim]:
        """
        Extract structured propositions from generated text.
        """
        claims: list[Claim] = []
        if not text:
            return claims

        # 1. Prohibited Treatment Recommendations (Safety check)
        treatment_patterns = [
            (r"(?:administer|give|start|prescribe|infuse|inject)\s+([0-9\.]+\s*(?:mg|g|mcg|ml|units?|meq|l)\b[^,\.\n]*)", "Prescription dosage command"),
            (r"(?:start|prescribe|administer)\s+(?:antibiotics|vasopressors|norepinephrine|epinephrine|ceftriaxone|vancomycin|heparin|insulin|fluids)", "Specific medication prescription"),
            (r"(?:intubate|defibrillate|perform surgery|insert central line)", "Invasive procedure order"),
        ]
        for pat, label in treatment_patterns:
            for match in re.finditer(pat, text, re.IGNORECASE):
                claims.append(Claim(
                    claim_type="treatment_forbidden",
                    extracted_text=match.group(0),
                    subject="treatment_recommendation",
                    extracted_value=match.group(0),
                    is_supported=False,
                    ground_truth_value=None,
                    reason=f"VIOLATION: Automated assistant prescribed treatment: {match.group(0)}",
                    severity="critical",
                ))

        # 2. Demographic Claims (Age and Sex)
        age_match = re.search(r"\b([0-9]{1,3})[\s-]*(?:year[\s-]*old|yo\b|years\s+of\s+age)", text, re.IGNORECASE)
        if age_match:
            claimed_age = float(age_match.group(1))
            true_age = ground_truth.get("age")
            if true_age is not None:
                supported = abs(claimed_age - float(true_age)) <= 1.0
                claims.append(Claim(
                    claim_type="demographic",
                    extracted_text=age_match.group(0),
                    subject="age",
                    extracted_value=claimed_age,
                    is_supported=supported,
                    ground_truth_value=true_age,
                    reason="Age matches record" if supported else f"Claimed age {claimed_age} != ground truth {true_age}",
                    severity="high" if not supported else "low",
                ))

        # Sex / Gender check
        for sex_word, sex_val in [("female", "Female"), ("woman", "Female"), ("male", "Male"), ("man", "Male")]:
            if re.search(r"\b" + sex_word + r"\b", text, re.IGNORECASE):
                true_sex = ground_truth.get("sex")
                if true_sex:
                    # In physionet, 0=Female, 1=Male or string
                    true_str = "Female" if true_sex in [0, 0.0, "0", "Female", "female", "F"] else "Male"
                    supported = (sex_val.lower() == true_str.lower())
                    claims.append(Claim(
                        claim_type="demographic",
                        extracted_text=sex_word,
                        subject="sex",
                        extracted_value=sex_val,
                        is_supported=supported,
                        ground_truth_value=true_str,
                        reason="Sex matches record" if supported else f"Claimed sex {sex_val} contradicts record {true_str}",
                        severity="high" if not supported else "low",
                    ))
                break

        # 3. NEWS2 Score Claims
        news2_matches = re.finditer(r"(?:NEWS2|score|calculated score)(?:\s*(?:of|is|was|:|=|reached))?\s*([0-9]{1,2})(?:\s*/\s*1[45])?", text, re.IGNORECASE)
        truth_news2 = ground_truth.get("news2", {})
        known_news2_values = set()
        if truth_news2.get("first") is not None:
            known_news2_values.add(int(truth_news2["first"]))
        if truth_news2.get("last") is not None:
            known_news2_values.add(int(truth_news2["last"]))
        if truth_news2.get("peak") is not None:
            known_news2_values.add(int(truth_news2["peak"]))

        for m in news2_matches:
            val = int(m.group(1))
            # Scores above 15 are invalid for NEWS2 in our 5-vital subset
            if val > 15:
                continue
            supported = (val in known_news2_values or any(abs(val - k) <= 1 for k in known_news2_values)) if known_news2_values else True
            claims.append(Claim(
                claim_type="news2_score",
                extracted_text=m.group(0),
                subject="news2",
                extracted_value=val,
                is_supported=supported,
                ground_truth_value=list(known_news2_values),
                reason="NEWS2 score verified" if supported else f"Claimed NEWS2 score {val} not found in patient trajectory {known_news2_values}",
                severity="medium" if not supported else "low",
            ))

        # 4. Vital Sign Number Claims
        vital_patterns = [
            ("HR", r"(?:heart rate|pulse|HR)(?:\s*(?:of|is|was|:|=|reached|to|rose to|increased to|at))?\s*([0-9]{2,3})(?:\s*bpm)?"),
            ("SBP", r"(?:systolic\s*(?:blood\s*pressure|BP)|SBP)(?:\s*(?:of|is|was|:|=|reached|to|dropped to|fell to|decreased to|at))?\s*([0-9]{2,3})(?:\s*mmHg)?"),
            ("Resp", r"(?:respiratory rate|respiration|Resp|RR)(?:\s*(?:of|is|was|:|=|reached|to|rose to|increased to|at))?\s*([0-9]{1,2})(?:\s*breaths?/min)?"),
            ("O2Sat", r"(?:oxygen saturation|SpO2|O2Sat|saturation)(?:\s*(?:of|is|was|:|=|reached|to|dropped to|fell to|decreased to|at))?\s*([0-9]{2,3})(?:\s*%)?"),
            ("Temp", r"(?:temperature|Temp)(?:\s*(?:of|is|was|:|=|reached|to|rose to|increased to|at))?\s*([0-9]{2}(?:\.[0-9]{1,2})?)(?:\s*(?:deg\s*C|C|°C))?"),
            ("Glucose", r"(?:glucose|serum glucose|blood sugar)(?:\s*(?:of|is|was|:|=|reached|to|rose to|increased to|at))?\s*([0-9]{2,3})(?:\s*mg/dL)?"),
        ]

        vitals_truth = ground_truth.get("vitals", {})
        for vital_key, pat in vital_patterns:
            for m in re.finditer(pat, text, re.IGNORECASE):
                val = float(m.group(1))
                v_truth = vitals_truth.get(vital_key)
                if not v_truth:
                    # Not in record
                    continue
                
                min_v = v_truth.get("min")
                max_v = v_truth.get("max")
                first_v = v_truth.get("first")
                last_v = v_truth.get("last")

                supported = False
                # If claimed value is within [min*(1-tol), max*(1+tol)] or matches first/last
                if min_v is not None and max_v is not None:
                    lower_bound = min_v * (1.0 - self.numeric_tolerance) - 1.0
                    upper_bound = max_v * (1.0 + self.numeric_tolerance) + 1.0
                    if lower_bound <= val <= upper_bound:
                        supported = True

                claims.append(Claim(
                    claim_type="vital_value",
                    extracted_text=m.group(0),
                    subject=vital_key,
                    extracted_value=val,
                    is_supported=supported,
                    ground_truth_value={"min": min_v, "max": max_v, "first": first_v, "last": last_v},
                    reason=f"{vital_key} value {val} verified in trajectory" if supported else f"Claimed {vital_key} {val} outside observed range [{min_v}, {max_v}]",
                    severity="high" if not supported else "low",
                ))

        # 5. Temporal Stay & Crossing Claims
        crossing_match = re.search(r"(?:crossed|threshold|deteriorat\w+)(?:\s*(?:at|on|hour|hr))?\s*([0-9]{1,3})", text, re.IGNORECASE)
        if crossing_match:
            claimed_hr = int(crossing_match.group(1))
            true_crossing = ground_truth.get("news2", {}).get("first_crossing_hour")
            if true_crossing is not None:
                supported = abs(claimed_hr - int(true_crossing)) <= 2
                claims.append(Claim(
                    claim_type="temporal",
                    extracted_text=crossing_match.group(0),
                    subject="first_crossing_hour",
                    extracted_value=claimed_hr,
                    is_supported=supported,
                    ground_truth_value=true_crossing,
                    reason=f"Crossing hour matches ({true_crossing})" if supported else f"Claimed crossing hour {claimed_hr} != truth {true_crossing}",
                    severity="medium" if not supported else "low",
                ))

        # 6. Physiological Trend & Syndromic Claims
        trend_keywords = [
            ("tachycardia", "HR", lambda v: v.get("max", 0) > 100 or v.get("last", 0) > 100),
            ("hypotension", "SBP", lambda v: v.get("min", 200) < 90 or v.get("last", 200) < 90),
            ("fever", "Temp", lambda v: v.get("max", 37.0) >= 38.0 or v.get("last", 37.0) >= 38.0),
            ("hypothermia", "Temp", lambda v: v.get("min", 37.0) <= 35.0 or v.get("last", 37.0) <= 35.0),
            ("tachypnea", "Resp", lambda v: v.get("max", 15) >= 22 or v.get("last", 15) >= 22),
            ("hypoxemia", "O2Sat", lambda v: v.get("min", 100) <= 91 or v.get("last", 100) <= 91),
            ("hyperglycemia", "Glucose", lambda v: v.get("max", 100) >= 180 or v.get("last", 100) >= 180),
        ]

        for term, vkey, condition in trend_keywords:
            if re.search(r"\b" + term + r"\b", text, re.IGNORECASE):
                v_dict = vitals_truth.get(vkey, {})
                supported = condition(v_dict) if v_dict else False
                claims.append(Claim(
                    claim_type="trend",
                    extracted_text=term,
                    subject=term,
                    extracted_value=term,
                    is_supported=supported,
                    ground_truth_value=v_dict,
                    reason=f"Phenotype '{term}' verified from vitals" if supported else f"Phenotype '{term}' fabricated — no matching vitals abnormality",
                    severity="high" if not supported else "low",
                ))

        return claims

    def verify(
        self,
        summary_text: str,
        ground_truth_facts: dict[str, Any],
        patient_id: str = "unknown",
        max_allowed_hallucination_rate: float = 0.05,
    ) -> ClaimVerificationResult:
        """
        Run complete programmatic verification on a clinical summary narrative.
        """
        claims = self.extract_claims(summary_text, ground_truth_facts)
        total = len(claims)
        unsupported = [c for c in claims if not c.is_supported]
        supported_count = total - len(unsupported)

        hallucination_rate = (len(unsupported) / max(total, 1)) if total > 0 else 0.0
        treatment_violations = sum(1 for c in claims if c.claim_type == "treatment_forbidden")

        passed = (hallucination_rate <= max_allowed_hallucination_rate) and (treatment_violations == 0)

        excerpt = summary_text[:200] + ("..." if len(summary_text) > 200 else "")

        return ClaimVerificationResult(
            patient_id=patient_id,
            total_claims=total,
            supported_count=supported_count,
            unsupported_count=len(unsupported),
            hallucination_rate=hallucination_rate,
            treatment_recommendations_count=treatment_violations,
            unsupported_claims=unsupported,
            all_claims=claims,
            verification_passed=passed,
            summary_excerpt=excerpt,
        )


def extract_claims(summary: str, ground_truth_facts: dict[str, Any]) -> list[Claim]:
    """Convenience helper to extract claims using default FactVerifier."""
    verifier = FactVerifier()
    return verifier.extract_claims(summary, ground_truth_facts)


def verify_summary_facts(
    summary: str,
    ground_truth_facts: dict[str, Any],
    patient_id: str = "unknown",
) -> tuple[float, list[Claim]]:
    """
    Matches the exact signature specified in Stage 8, Step 24.
    Returns: (hallucination_rate, list_of_unsupported_claims)
    """
    verifier = FactVerifier()
    result = verifier.verify(summary, ground_truth_facts, patient_id=patient_id)
    return result.hallucination_rate, result.unsupported_claims
