"""
Prompt builder for grounded clinical question answering and summarization.
"""

from __future__ import annotations

from product_track.rag import RetrievedChunk


CLINICAL_SYSTEM_INSTRUCTION = """You are a clinical decision support assistant.
You have access only to the following information about this patient.
Do not infer, assume, or generate information not present in the context.
If the context does not contain enough to answer safely, explicitly say that the record does not contain this information."""


def build_clinical_rag_prompt(chunks: list[RetrievedChunk] | list[str], question: str) -> str:
    """
    Assemble patient context and query into a strict refusal-enforcing prompt.
    """
    chunk_texts: list[str] = []
    for c in chunks:
        if isinstance(c, RetrievedChunk):
            header = f"[{c.title or c.note_type}]" if c.title else ""
            chunk_texts.append(f"{header}\n{c.content}".strip())
        else:
            chunk_texts.append(str(c).strip())

    if not chunk_texts:
        context_block = "(No clinical records retrieved for this query.)"
    else:
        context_block = "\n\n---\n\n".join(chunk_texts)

    prompt = f"""{CLINICAL_SYSTEM_INSTRUCTION}

PATIENT CONTEXT:
{context_block}

QUESTION:
{question}

RESPONSE:"""
    return prompt
