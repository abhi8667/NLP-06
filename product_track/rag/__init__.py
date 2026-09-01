"""
RAG and Vector Retrieval module for Person A (Product Track).
Ensures strict per-patient privacy boundaries and local vector indexing.
"""

from .vector_store import (
    PatientVectorStore,
    RetrievedChunk,
    chunk_text,
)

__all__ = [
    "PatientVectorStore",
    "RetrievedChunk",
    "chunk_text",
]
