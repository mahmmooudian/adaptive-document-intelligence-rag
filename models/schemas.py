# Shared project schemas
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DocumentMetadata:
    """
    Basic metadata describing an uploaded document.
    """

    filename: str
    file_type: str
    page_count: Optional[int] = None
    character_count: int = 0


@dataclass
class Chunk:
    """
    Standard representation of a document chunk.
    """

    chunk_id: int
    text: str
    word_count: int

    filename: Optional[str] = None
    page_number: Optional[int] = None

    embedding_model: Optional[str] = None


@dataclass
class RetrievalResult:
    """
    Standard representation of a retrieved chunk.
    """

    chunk_id: int
    text: str

    rank: Optional[int] = None
    final_rank: Optional[int] = None

    filename: Optional[str] = None
    page_number: Optional[int] = None

    vector_score: Optional[float] = None
    bm25_score: Optional[float] = None
    hybrid_score: Optional[float] = None
    reranker_score: Optional[float] = None


@dataclass
class SourceReference:
    """
    Citation metadata used by the context builder
    and final answer generator.
    """

    source_id: str

    chunk_id: Optional[int] = None
    filename: Optional[str] = None
    page_number: Optional[int] = None

    hybrid_score: Optional[float] = None
    reranker_score: Optional[float] = None


@dataclass
class PipelineMetrics:
    """
    Runtime information used for the technical demo.
    """

    document_pages: Optional[int] = None
    extracted_characters: int = 0
    generated_chunks: int = 0

    parsing_time: float = 0.0
    cleaning_time: float = 0.0
    chunking_time: float = 0.0
    embedding_time: float = 0.0
    retrieval_time: float = 0.0
    reranking_time: float = 0.0

    total_processing_time: float = 0.0


@dataclass
class PipelineResponse:
    """
    Final structured result returned by the RAG pipeline.
    """

    question: str
    answer: str

    mode: str = "Fast / TurboRAG"

    sources: List[SourceReference] = field(
        default_factory=list
    )

    metrics: Optional[PipelineMetrics] = None
