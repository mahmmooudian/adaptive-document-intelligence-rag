# Retrieval reranking module
from typing import List, Dict, Optional

import numpy as np
from sentence_transformers import CrossEncoder


DEFAULT_RERANKER_MODEL = (
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
)


class Reranker:
    """
    Cross-encoder reranker for improving retrieval precision.

    It receives candidate passages from the hybrid retriever
    and scores each passage directly against the user query.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: Optional[str] = None,
    ):
        self.model_name = model_name

        self.model = CrossEncoder(
            model_name,
            device=device,
        )

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Rerank retrieved candidates.

        Parameters
        ----------
        query:
            User query.

        candidates:
            Candidate chunks returned by the hybrid retriever.

        top_k:
            Number of final passages to return.

        Returns
        -------
        List[Dict]
            Reranked candidates sorted by relevance.
        """

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if not candidates:
            return []

        pairs = [
            [query, candidate["text"]]
            for candidate in candidates
        ]

        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
        )

        scores = np.asarray(
            scores,
            dtype=float,
        ).reshape(-1)

        reranked_results = []

        for candidate, score in zip(
            candidates,
            scores,
        ):
            item = candidate.copy()

            item["reranker_score"] = float(score)
            item["reranker_model"] = self.model_name

            reranked_results.append(item)

        reranked_results.sort(
            key=lambda item: item["reranker_score"],
            reverse=True,
        )

        top_k = min(
            top_k,
            len(reranked_results),
        )

        final_results = reranked_results[:top_k]

        for rank, item in enumerate(
            final_results,
            start=1,
        ):
            item["final_rank"] = rank

        return final_results
