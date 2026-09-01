"""
Local ChromaDB vector store with strict per-patient isolation.

Ensures that RAG context retrieval cannot cross patient boundaries by enforcing
hard filters and patient-scoped namespaces.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api import ClientAPI
from chromadb.utils import embedding_functions

from product_track.knowledge_base import ClinicalNote, generate_patient_notes


@dataclass
class RetrievedChunk:
    """Represents a single retrieved text chunk with provenance metadata."""
    content: str
    patient_id: str
    note_type: str
    title: str
    chunk_id: str
    distance: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 100) -> list[str]:
    """
    Split text into overlapping character chunks cleanly by paragraph/sentence breaks.
    Default chunk_size is 1200 chars to keep complete clinical notes and diagnostic tables intact.
    """
    text = text.strip()
    if not text:
        return []

    # If text fits in one chunk, return directly
    if len(text) <= chunk_size:
        return [text]

    # Split by paragraphs / markdown sections first
    sections = [s.strip() for s in re.split(r"\n\n+", text) if s.strip()]
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_len = 0

    for sec in sections:
        if current_len + len(sec) + 2 <= chunk_size:
            current_chunk.append(sec)
            current_len += len(sec) + 2
        else:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
            # If a single section is larger than chunk_size, split by lines
            if len(sec) > chunk_size:
                lines = sec.split("\n")
                sub_chunk: list[str] = []
                sub_len = 0
                for line in lines:
                    if sub_len + len(line) + 1 <= chunk_size:
                        sub_chunk.append(line)
                        sub_len += len(line) + 1
                    else:
                        if sub_chunk:
                            chunks.append("\n".join(sub_chunk))
                        sub_chunk = [line]
                        sub_len = len(line)
                if sub_chunk:
                    chunks.append("\n".join(sub_chunk))
                current_chunk = []
                current_len = 0
            else:
                current_chunk = [sec]
                current_len = len(sec)

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks if chunks else [text]


class PatientVectorStore:
    """
    Local ChromaDB store enforcing strict patient-level isolation.
    """

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        collection_name: str = "patient_clinical_records",
    ):
        self.persist_dir = str(persist_dir) if persist_dir else None
        if self.persist_dir:
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            self.client: ClientAPI = chromadb.PersistentClient(path=self.persist_dir)
        else:
            self.client: ClientAPI = chromadb.Client()

        # Local default embedding function (ONNX miniLM / fast local)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
        )

    def index_patient_notes(
        self,
        patient_id: str,
        notes: list[ClinicalNote],
        chunk_size: int = 1200,
        overlap: int = 100,
    ) -> int:
        """
        Index all notes for a specific patient.
        Deletes any existing documents for this patient first to ensure clean state.
        """
        if not patient_id or not isinstance(patient_id, str):
            raise ValueError(f"Invalid patient_id: {patient_id}")

        # Clear existing records for this patient
        self.clear_patient(patient_id)

        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        ids: list[str] = []

        for note in notes:
            chunks = chunk_text(note.content, chunk_size=chunk_size, overlap=overlap)
            for idx, chunk in enumerate(chunks):
                doc_id = f"{patient_id}_{note.note_type}_{idx}"
                documents.append(chunk)
                metadatas.append({
                    "patient_id": patient_id,
                    "note_type": note.note_type,
                    "title": note.title,
                    "chunk_index": idx,
                })
                ids.append(doc_id)

        if documents:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )

        return len(documents)

    def index_patient_from_psv(self, psv_path: str | Path) -> tuple[str, int]:
        """Load a patient PSV file, generate notes, and index them."""
        notes, facts = generate_patient_notes(psv_path)
        pid = facts["patient_id"]
        count = self.index_patient_notes(pid, notes)
        return pid, count

    def query(
        self,
        patient_id: str,
        query_text: str,
        n_results: int = 4,
        distance_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        """
        Query the vector store with HARD isolation on patient_id.
        Under no circumstances will chunks belonging to other patients be returned.

        If distance_threshold is provided, chunks with distance > distance_threshold
        are discarded as non-relevant.
        """
        if not patient_id or not isinstance(patient_id, str):
            raise ValueError("Query must specify a valid patient_id for scoped retrieval.")

        # Query with explicit where clause
        raw_res = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where={"patient_id": patient_id},
        )

        results: list[RetrievedChunk] = []
        docs = raw_res.get("documents", [[]])[0]
        metas = raw_res.get("metadatas", [[]])[0]
        ids = raw_res.get("ids", [[]])[0]
        distances = raw_res.get("distances", [[]])[0] if raw_res.get("distances") else [None] * len(docs)

        for doc, meta, doc_id, dist in zip(docs, metas, ids, distances):
            # Enforce hard isolation check
            if not meta or meta.get("patient_id") != patient_id:
                continue

            # Optional distance cutoff for strict relevance
            if distance_threshold is not None and dist is not None and dist > distance_threshold:
                continue

            results.append(RetrievedChunk(
                content=doc,
                patient_id=meta["patient_id"],
                note_type=meta.get("note_type", "unknown"),
                title=meta.get("title", ""),
                chunk_id=doc_id,
                distance=dist,
            ))

        return results

    def clear_patient(self, patient_id: str) -> None:
        """Remove all indexed chunks for a patient."""
        try:
            self.collection.delete(where={"patient_id": patient_id})
        except Exception:
            pass

    def count_for_patient(self, patient_id: str) -> int:
        """Count indexed chunks for a specific patient."""
        res = self.collection.get(where={"patient_id": patient_id})
        return len(res.get("ids", []))
