# LLM context construction module
from typing import Dict, List, Tuple


class ContextBuilder:
    """
    Build a compact, citation-aware context for the LLM.
    """

    def __init__(
        self,
        max_context_words: int = 1200,
    ):
        self.max_context_words = max_context_words

    def build(
        self,
        evidence: List[Dict],
    ) -> Tuple[str, List[Dict]]:
        """
        Build the final LLM context from reranked evidence.

        Returns
        -------
        Tuple[str, List[Dict]]
            - Formatted context text
            - Source metadata used for citations
        """

        if not evidence:
            return "", []

        context_parts = []
        sources = []
        used_words = 0

        for source_index, item in enumerate(
            evidence,
            start=1,
        ):
            text = item.get("text", "").strip()

            if not text:
                continue

            words = text.split()
            remaining_budget = (
                self.max_context_words - used_words
            )

            if remaining_budget <= 0:
                break

            if len(words) > remaining_budget:
                text = " ".join(
                    words[:remaining_budget]
                )

            source_id = f"SOURCE_{source_index}"

            source_metadata = {
                "source_id": source_id,
                "chunk_id": item.get("chunk_id"),
                "page_number": item.get("page_number"),
                "filename": item.get("filename"),
                "hybrid_score": item.get("hybrid_score"),
                "reranker_score": item.get(
                    "reranker_score"
                ),
            }

            sources.append(source_metadata)

            context_block = self._format_source(
                source_id=source_id,
                text=text,
                metadata=source_metadata,
            )

            context_parts.append(context_block)

            used_words += len(text.split())

        context = "\n\n".join(context_parts)

        return context, sources

    @staticmethod
    def _format_source(
        source_id: str,
        text: str,
        metadata: Dict,
    ) -> str:
        """
        Format a source block for the LLM.
        """

        filename = (
            metadata.get("filename")
            or "Unknown document"
        )

        page_number = metadata.get(
            "page_number"
        )

        if page_number is not None:
            source_location = (
                f"{filename} | Page {page_number}"
            )
        else:
            source_location = filename

        return (
            f"[{source_id}]\n"
            f"Source: {source_location}\n"
            f"Content:\n{text}"
        )


def build_citation_map(
    sources: List[Dict],
) -> Dict[str, Dict]:
    """
    Convert source metadata into a lookup dictionary.

    Example:
    SOURCE_1 -> document/page/chunk metadata
    """

    return {
        source["source_id"]: source
        for source in sources
    }
