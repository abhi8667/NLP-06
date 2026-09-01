"""
End-to-end Grounded Clinical RAG Pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from product_track.llm.ollama_client import OllamaClient
from product_track.llm.prompt_builder import build_clinical_rag_prompt
from product_track.rag.vector_store import PatientVectorStore, RetrievedChunk


class ClinicalRAGPipeline:
    """
    End-to-end pipeline providing isolated, grounded retrieval and generation.
    """

    def __init__(
        self,
        vector_store: PatientVectorStore | None = None,
        llm_client: OllamaClient | None = None,
        default_model: str = "llama3.2:3b",
        default_temperature: float = 0.2,
    ):
        self.vector_store = vector_store or PatientVectorStore()
        self.llm_client = llm_client or OllamaClient(
            default_model=default_model,
            default_temperature=default_temperature,
        )

    def answer_question(
        self,
        patient_id: str,
        question: str,
        n_chunks: int = 4,
        distance_threshold: float | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """
        Retrieve patient-scoped records, format strict prompt, and generate answer.
        """
        # 1. Retrieve isolated chunks for patient
        retrieved_chunks = self.vector_store.query(
            patient_id=patient_id,
            query_text=question,
            n_results=n_chunks,
            distance_threshold=distance_threshold,
        )

        # 2. Build refusal-enforcing prompt
        prompt = build_clinical_rag_prompt(retrieved_chunks, question)

        # 3. Local inference via Ollama
        gen_res = self.llm_client.generate(
            prompt=prompt,
            model=model,
        )

        return {
            "patient_id": patient_id,
            "question": question,
            "answer": gen_res["response"],
            "model": gen_res["model"],
            "retrieved_chunks": [c.to_dict() for c in retrieved_chunks],
            "total_duration_s": gen_res["total_duration_s"],
            "eval_rate_tok_s": gen_res["eval_rate_tok_s"],
            "eval_count": gen_res["eval_count"],
        }
