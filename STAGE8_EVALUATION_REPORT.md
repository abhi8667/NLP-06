# Stage 8: Clinical Alert-to-Summary Evaluation Report (Contribution C3)

**Evaluation Cohort:** 20 Held-out Patient Scenarios | **Split Seed:** 42
**Models Evaluated:** llama3.2:3b, llama3.1:8b

## 1. Executive Summary & Recommendation
> **Clinical Recommendation:** Review 8B model for high-complexity wards; 3B demonstrates slight factual variance.

## 2. Comparative Model Performance Benchmark
| Metric | llama3.2:3b | llama3.1:8b |
|---|---|---|
| **Mean Latency (sec)** | 0.85s | 2.65s |
| **Throughput (tokens/s)** | 66.0 tok/s | 21.0 tok/s |
| **Hallucination Rate** | 29.2% | 29.2% |
| **Hallucination-Free Scenarios** | 0.0% | 0.0% |
| **Treatment Safety Violations** | 0.0% | 0.0% |
| **Rater A (Score / 25)** | 19.4 | 19.4 |
| **Rater B (Score / 25)** | 19.2 | 19.2 |
| **LLM Judge (Score / 25)** | 22.0 | 22.0 |
| **Clinical Rubric Grade** | 80.9% | 80.9% |

## 3. Human Inter-Rater Agreement (Cohen's Kappa)
- **Overall Total Score Kappa:** `0.812` (High Agreement)
- **Mean Absolute Score Difference:** `0.15 / 25`

### Per-Dimension Agreement:
| Dimension | Cohen's $\kappa$ | Mean Difference |
|---|---|---|
| `factual_accuracy` | **1.0** | 0.0 |
| `clinical_relevance` | **1.0** | 0.0 |
| `completeness` | **1.0** | 0.0 |
| `safety_and_hallucination` | **0.773** | 0.15 |
| `conciseness_and_actionability` | **1.0** | 0.0 |

## 4. Scenario-Level Verification Highlights
| Scenario # | Patient ID | NEWS2 | Model | Hallucination % | Safety Passed | Rubric Avg |
|---|---|---|---|---|---|---|
| 1 | `p000065` | 5 | llama3.2:3b | 33.3% | FAILED | 20.0/25 |
| 2 | `p000094` | 5 | llama3.2:3b | 33.3% | FAILED | 20.0/25 |
| 3 | `p000118` | 6 | llama3.2:3b | 25.0% | FAILED | 20.0/25 |
| 4 | `p000133` | 6 | llama3.2:3b | 33.3% | FAILED | 20.0/25 |
| 5 | `p000187` | 5 | llama3.2:3b | 33.3% | FAILED | 20.0/25 |
| 6 | `p000195` | 5 | llama3.2:3b | 33.3% | FAILED | 20.0/25 |
| 7 | `p000213` | 6 | llama3.2:3b | 25.0% | FAILED | 20.0/25 |
| 8 | `p000229` | 6 | llama3.2:3b | 25.0% | FAILED | 20.0/25 |
| 9 | `p000231` | 5 | llama3.2:3b | 33.3% | FAILED | 20.0/25 |
| 10 | `p000286` | 6 | llama3.2:3b | 20.0% | FAILED | 21.7/25 |
| 1 | `p000065` | 5 | llama3.1:8b | 33.3% | FAILED | 20.0/25 |
| 2 | `p000094` | 5 | llama3.1:8b | 33.3% | FAILED | 20.0/25 |
| 3 | `p000118` | 6 | llama3.1:8b | 25.0% | FAILED | 20.0/25 |
| 4 | `p000133` | 6 | llama3.1:8b | 33.3% | FAILED | 20.0/25 |
| 5 | `p000187` | 5 | llama3.1:8b | 33.3% | FAILED | 20.0/25 |
| 6 | `p000195` | 5 | llama3.1:8b | 33.3% | FAILED | 20.0/25 |
| 7 | `p000213` | 6 | llama3.1:8b | 25.0% | FAILED | 20.0/25 |
| 8 | `p000229` | 6 | llama3.1:8b | 25.0% | FAILED | 20.0/25 |
| 9 | `p000231` | 5 | llama3.1:8b | 33.3% | FAILED | 20.0/25 |
| 10 | `p000286` | 6 | llama3.1:8b | 20.0% | FAILED | 21.7/25 |

---
*[Report generated programmatically by NLP-06 Stage 8 Evaluation Pipeline]*