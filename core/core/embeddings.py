# Embedding generation module
from typing import List, Dict, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

class EmbeddingEngine:
    """
    Local embedding engine for document chunks and user queries.

    The implementation is intentionally lightweight for the demo version
    and can later be replaced by a production embedding model or API.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
    ):
        self.model_name = model_name

        self.model = SentenceTransformer(
            model_name,
            device=device,
        )

    def encode_texts(
        self,
        texts: List[str],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Convert a list of texts into embedding vectors.
        """

        if not texts:
            return np.empty((0, 0))

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )

        return embeddings

    def encode_query(
        self,
        query: str,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Convert a single user query into an embedding vector.
        """

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )

        return embedding[0]

    def embed_chunks(
        self,
        chunks: List[Dict],
    ) -> List[Dict]:
        """
        Generate embeddings for a list of chunk dictionaries.

        Expected chunk format:
        {
            "chunk_id": 1,
            "text": "...",
            "word_count": 180
        }
        """

        if not chunks:
            return []

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = self.encode_texts(texts)

        embedded_chunks = []

        for chunk, vector in zip(chunks, embeddings):
            item = chunk.copy()

            item["embedding"] = vector
            item["embedding_model"] = self.model_name

            embedded_chunks.append(item)

        return embedded_chunks
