# Hybrid retrieval module
from typing import List, Dict, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from core.embeddings import EmbeddingEngine


class HybridRetriever:
    """
    Hybrid retrieval engine combining:
    - Dense semantic retrieval using embeddings
    - Sparse lexical retrieval using BM25
    """

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        vector_weight: float = 0.65,
        bm25_weight: float = 0.35,
    ):
        self.embedding_engine = embedding_engine

        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

        self.chunks: List[Dict] = []

        self.embedding_matrix = None
        self.bm25 = None
        self.tokenized_corpus = None

    def build_index(
        self,
        embedded_chunks: List[Dict],
    ) -> None:
        """
        Build in-memory dense and BM25 indexes.
        """

        if not embedded_chunks:
            raise ValueError("No chunks provided for indexing.")

        self.chunks = embedded_chunks

        # Build dense embedding matrix
        self.embedding_matrix = np.vstack(
            [
                chunk["embedding"]
                for chunk in embedded_chunks
            ]
        )

        # Build BM25 corpus
        self.tokenized_corpus = [
            self._tokenize(chunk["text"])
            for chunk in embedded_chunks
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_corpus
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> List[Dict]:
        """
        Run hybrid retrieval.

        Steps:
        1. Encode query
        2. Dense vector retrieval
        3. BM25 retrieval
        4. Score normalization
        5. Weighted score fusion
        6. Return top-k results
        """

        if not self.chunks:
            raise RuntimeError(
                "Retriever index has not been built."
            )

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        query_embedding = (
            self.embedding_engine.encode_query(query)
        )

        vector_scores = self._vector_search(
            query_embedding
        )

        bm25_scores = self._bm25_search(query)

        normalized_vector_scores = (
            self._normalize_scores(vector_scores)
        )

        normalized_bm25_scores = (
            self._normalize_scores(bm25_scores)
        )

        hybrid_scores = (
            self.vector_weight
            * normalized_vector_scores
            +
            self.bm25_weight
            * normalized_bm25_scores
        )

        candidate_k = min(
            candidate_k,
            len(self.chunks),
        )

        candidate_indices = np.argsort(
            hybrid_scores
        )[::-1][:candidate_k]

        results = []

        for rank, index in enumerate(
            candidate_indices[:top_k],
            start=1,
        ):
            chunk = self.chunks[index].copy()

            chunk.pop("embedding", None)

            chunk["rank"] = rank

            chunk["vector_score"] = float(
                normalized_vector_scores[index]
            )

            chunk["bm25_score"] = float(
                normalized_bm25_scores[index]
            )

            chunk["hybrid_score"] = float(
                hybrid_scores[index]
            )

            results.append(chunk)

        return results

    def _vector_search(
        self,
        query_embedding: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate cosine similarity.

        Embeddings are already normalized,
        therefore dot product equals cosine similarity.
        """

        return np.dot(
            self.embedding_matrix,
            query_embedding,
        )

    def _bm25_search(
        self,
        query: str,
    ) -> np.ndarray:
        """
        Calculate BM25 scores.
        """

        tokenized_query = self._tokenize(query)

        scores = self.bm25.get_scores(
            tokenized_query
        )

        return np.asarray(
            scores,
            dtype=float,
        )

    @staticmethod
    def _normalize_scores(
        scores: np.ndarray,
    ) -> np.ndarray:
        """
        Min-max normalize retrieval scores.
        """

        scores = np.asarray(
            scores,
            dtype=float,
        )

        minimum = scores.min()
        maximum = scores.max()

        if maximum == minimum:
            return np.zeros_like(scores)

        return (
            scores - minimum
        ) / (
            maximum - minimum
        )

    @staticmethod
    def _tokenize(
        text: str,
    ) -> List[str]:
        """
        Lightweight tokenizer for BM25.

        The text has already passed through
        the cleaning and normalization pipeline.
        """

        return [
            token.lower()
            for token in text.split()
            if token.strip()
        ]
