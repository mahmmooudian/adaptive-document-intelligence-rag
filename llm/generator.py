# Grounded answer generation module
from typing import Dict, List, Optional


SYSTEM_PROMPT = """
You are an evidence-grounded document assistant.

Your job is to answer the user's question using only the provided sources.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts.
3. If the provided evidence is insufficient, clearly say that the available
   document evidence is not sufficient to answer the question.
4. Cite supporting sources using their exact identifiers, for example:
   [SOURCE_1]
5. Do not invent source identifiers.
6. If multiple sources support the answer, cite all relevant sources.
7. Keep the answer clear, concise, and faithful to the evidence.
""".strip()


class GroundedAnswerGenerator:
    """
    Provider-agnostic grounded answer generator.

    In the demo version, this class prepares the prompt and can later
    be connected to any cloud or local LLM provider.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.provider = provider
        self.model_name = model_name

    def build_prompt(
        self,
        question: str,
        context: str,
    ) -> str:
        """
        Build a grounded RAG prompt.
        """

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not context or not context.strip():
            return (
                f"{SYSTEM_PROMPT}\n\n"
                f"User Question:\n{question}\n\n"
                "Available Evidence:\n"
                "No relevant evidence was retrieved."
            )

        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"User Question:\n{question}\n\n"
            f"Available Evidence:\n{context}\n\n"
            "Answer the question using only the evidence above."
        )

    def generate(
        self,
        question: str,
        context: str,
    ) -> Dict:
        """
        Generate an answer.

        The actual LLM API call is intentionally not implemented yet.
        This keeps the demo architecture provider-independent.

        Returns a structured placeholder response containing the prompt.
        """

        prompt = self.build_prompt(
            question=question,
            context=context,
        )

        return {
            "status": "provider_not_configured",
            "answer": (
                "LLM provider is not configured yet. "
                "The retrieval pipeline is ready and the grounded prompt "
                "has been generated successfully."
            ),
            "provider": self.provider,
            "model": self.model_name,
            "prompt": prompt,
        }


def extract_used_citations(
    answer: str,
    sources: List[Dict],
) -> List[Dict]:
    """
    Identify which source identifiers were referenced in the answer.
    """

    if not answer:
        return []

    used_sources = []

    for source in sources:
        source_id = source.get("source_id")

        if source_id and f"[{source_id}]" in answer:
            used_sources.append(source)

    return used_sources
