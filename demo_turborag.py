import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pymupdf

from docx import Document
from dotenv import load_dotenv
from groq import Groq
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


# ============================================================
# TURBORAG TECHNICAL PROOF OF CONCEPT v3.1
# STABLE GROQ EDITION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(
    dotenv_path=ENV_PATH,
    override=True,
)

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY",
    "",
).strip()

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b",
).strip()


# ============================================================
# CONFIG
# ============================================================

EMBEDDING_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

RERANKER_MODEL = (
    "cross-encoder/"
    "mmarco-mMiniLMv2-L12-H384-v1"
)

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
}

VECTOR_WEIGHT = 0.65
BM25_WEIGHT = 0.35

RETRIEVAL_TOP_K = 20
RERANK_TOP_K = 10

TARGET_CHUNK_WORDS = 180
MAX_CHUNK_WORDS = 260
CHUNK_OVERLAP_WORDS = 40

MAX_CONTEXT_WORDS = 1200

MIN_RERANKER_SCORE = -5.0
MIN_HYBRID_SCORE = 0.25

DEDUP_THRESHOLD = 0.65


# ============================================================
# DOCUMENT PARSER
# ============================================================

def parse_document(
    file_path: str,
) -> Dict:

    path = Path(
        file_path
    )

    if not path.exists():

        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    extension = (
        path.suffix.lower()
    )

    if extension not in SUPPORTED_EXTENSIONS:

        raise ValueError(
            f"Unsupported file format: {extension}"
        )

    if extension == ".pdf":
        return parse_pdf(
            path
        )

    if extension == ".docx":
        return parse_docx(
            path
        )

    return parse_txt(
        path
    )


def parse_pdf(
    path: Path,
) -> Dict:

    pages: List[Dict] = []
    full_text_parts: List[str] = []

    document = pymupdf.open(
        path
    )

    try:

        for page_number, page in enumerate(
            document,
            start=1,
        ):

            text = (
                page
                .get_text("text")
                .strip()
            )

            pages.append(
                {
                    "page_number":
                        page_number,
                    "text":
                        text,
                }
            )

            if text:

                full_text_parts.append(
                    text
                )

    finally:

        document.close()

    full_text = "\n\n".join(
        full_text_parts
    )

    return {
        "filename":
            path.name,
        "file_type":
            "pdf",
        "page_count":
            len(pages),
        "character_count":
            len(full_text),
        "text":
            full_text,
        "pages":
            pages,
    }


def parse_docx(
    path: Path,
) -> Dict:

    document = Document(
        path
    )

    paragraphs: List[Dict] = []
    full_text_parts: List[str] = []

    for paragraph_number, paragraph in enumerate(
        document.paragraphs,
        start=1,
    ):

        text = (
            paragraph.text.strip()
        )

        if not text:
            continue

        paragraphs.append(
            {
                "paragraph_number":
                    paragraph_number,
                "text":
                    text,
            }
        )

        full_text_parts.append(
            text
        )

    full_text = "\n\n".join(
        full_text_parts
    )

    return {
        "filename":
            path.name,
        "file_type":
            "docx",
        "page_count":
            None,
        "character_count":
            len(full_text),
        "text":
            full_text,
        "paragraphs":
            paragraphs,
    }


def parse_txt(
    path: Path,
) -> Dict:

    text = path.read_text(
        encoding="utf-8",
        errors="ignore",
    ).strip()

    return {
        "filename":
            path.name,
        "file_type":
            "txt",
        "page_count":
            None,
        "character_count":
            len(text),
        "text":
            text,
    }


# ============================================================
# CLEANING
# ============================================================

def normalize_persian_characters(
    text: str,
) -> str:

    replacements = {
        "ي": "ی",
        "ى": "ی",
        "ك": "ک",
        "ؤ": "و",
        "ۀ": "ه",
        "ة": "ه",
    }

    for source, target in replacements.items():

        text = text.replace(
            source,
            target,
        )

    return text


def remove_document_noise(
    text: str,
) -> str:

    # Standalone page numbers
    text = re.sub(
        r"(?mi)^\s*(page\s*)?\d+\s*$",
        "",
        text,
    )

    # Known footer in our synthetic PDF
    text = re.sub(
        (
            r"(?mi)^Synthetic RAG "
            r"Evaluation Document\s*-\s*"
            r"AWR-2047\s*$"
        ),
        "",
        text,
    )

    # Repeated separators
    text = re.sub(
        r"[-_=]{4,}",
        " ",
        text,
    )

    return text


def normalize_whitespace(
    text: str,
) -> str:

    text = (
        text
        .replace(
            "\r\n",
            "\n",
        )
        .replace(
            "\r",
            "\n",
        )
        .replace(
            "\t",
            " ",
        )
    )

    text = re.sub(
        r"[ ]{2,}",
        " ",
        text,
    )

    text = re.sub(
        r"\s+([،؛:,.!?؟])",
        r"\1",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def fix_broken_lines(
    text: str,
) -> str:

    lines = text.split(
        "\n"
    )

    output: List[str] = []

    current_line = ""

    for line in lines:

        line = (
            line.strip()
        )

        if not line:

            if current_line:

                output.append(
                    current_line.strip()
                )

                current_line = ""

            output.append(
                ""
            )

            continue

        if current_line:

            current_line += (
                " " + line
            )

        else:

            current_line = (
                line
            )

        if re.search(
            r"[.!?؟:]$",
            line,
        ):

            output.append(
                current_line.strip()
            )

            current_line = ""

    if current_line:

        output.append(
            current_line.strip()
        )

    return "\n".join(
        output
    )


def clean_text(
    text: Optional[str],
) -> str:

    if not text:
        return ""

    text = (
        normalize_persian_characters(
            text
        )
    )

    text = (
        remove_document_noise(
            text
        )
    )

    text = (
        fix_broken_lines(
            text
        )
    )

    text = (
        normalize_whitespace(
            text
        )
    )

    return text


# ============================================================
# CHUNKING
# ============================================================

def word_count(
    text: str,
) -> int:

    return len(
        text.split()
    )


def split_into_paragraphs(
    text: str,
) -> List[str]:

    return [
        paragraph.strip()

        for paragraph
        in text.split(
            "\n\n"
        )

        if paragraph.strip()
    ]


def make_chunk(
    text: str,
    filename: str,
    page_number: Optional[int],
) -> Dict:

    return {
        "text":
            text.strip(),

        "word_count":
            word_count(
                text
            ),

        "filename":
            filename,

        "page_number":
            page_number,
    }


def create_page_chunks(
    page_text: str,
    filename: str,
    page_number: Optional[int],
    target_words: int = TARGET_CHUNK_WORDS,
    max_words: int = MAX_CHUNK_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> List[Dict]:

    paragraphs = (
        split_into_paragraphs(
            page_text
        )
    )

    chunks: List[Dict] = []

    current_parts: List[str] = []

    current_count = 0

    for paragraph in paragraphs:

        paragraph_count = (
            word_count(
                paragraph
            )
        )

        # ====================================================
        # Oversized paragraph
        # ====================================================

        if paragraph_count > max_words:

            if current_parts:

                chunk_text = (
                    "\n\n".join(
                        current_parts
                    ).strip()
                )

                if chunk_text:

                    chunks.append(
                        make_chunk(
                            chunk_text,
                            filename,
                            page_number,
                        )
                    )

                current_parts = []
                current_count = 0

            words = (
                paragraph.split()
            )

            start = 0

            while start < len(
                words
            ):

                end = min(
                    start + max_words,
                    len(words),
                )

                segment = " ".join(
                    words[
                        start:end
                    ]
                ).strip()

                if segment:

                    chunks.append(
                        make_chunk(
                            segment,
                            filename,
                            page_number,
                        )
                    )

                if end >= len(
                    words
                ):
                    break

                start = max(
                    end - overlap_words,
                    start + 1,
                )

            continue

        projected_size = (
            current_count
            + paragraph_count
        )

        if (
            current_parts
            and projected_size > max_words
        ):

            chunk_text = (
                "\n\n".join(
                    current_parts
                ).strip()
            )

            if chunk_text:

                chunks.append(
                    make_chunk(
                        chunk_text,
                        filename,
                        page_number,
                    )
                )

            previous_words = (
                chunk_text.split()
            )

            overlap_text = " ".join(
                previous_words[
                    -overlap_words:
                ]
            ).strip()

            current_parts = (
                [overlap_text]
                if overlap_text
                else []
            )

            current_count = (
                word_count(
                    overlap_text
                )
                if overlap_text
                else 0
            )

        current_parts.append(
            paragraph
        )

        current_count += (
            paragraph_count
        )

        if current_count >= target_words:

            chunk_text = (
                "\n\n".join(
                    current_parts
                ).strip()
            )

            if chunk_text:

                chunks.append(
                    make_chunk(
                        chunk_text,
                        filename,
                        page_number,
                    )
                )

            previous_words = (
                chunk_text.split()
            )

            overlap_text = " ".join(
                previous_words[
                    -overlap_words:
                ]
            ).strip()

            current_parts = (
                [overlap_text]
                if overlap_text
                else []
            )

            current_count = (
                word_count(
                    overlap_text
                )
                if overlap_text
                else 0
            )

    if current_parts:

        chunk_text = (
            "\n\n".join(
                current_parts
            ).strip()
        )

        if chunk_text:

            chunks.append(
                make_chunk(
                    chunk_text,
                    filename,
                    page_number,
                )
            )

    return chunks


def create_document_chunks(
    document: Dict,
) -> List[Dict]:

    chunks: List[Dict] = []

    if (
        document["file_type"] == "pdf"
        and document.get(
            "pages"
        )
    ):

        for page in document[
            "pages"
        ]:

            cleaned_page = (
                clean_text(
                    page["text"]
                )
            )

            if not cleaned_page:
                continue

            page_chunks = (
                create_page_chunks(
                    page_text=
                        cleaned_page,

                    filename=
                        document[
                            "filename"
                        ],

                    page_number=
                        page[
                            "page_number"
                        ],
                )
            )

            chunks.extend(
                page_chunks
            )

    else:

        cleaned_text = (
            clean_text(
                document[
                    "text"
                ]
            )
        )

        if cleaned_text:

            chunks.extend(
                create_page_chunks(
                    page_text=
                        cleaned_text,

                    filename=
                        document[
                            "filename"
                        ],

                    page_number=
                        None,
                )
            )

    for chunk_id, chunk in enumerate(
        chunks,
        start=1,
    ):

        chunk[
            "chunk_id"
        ] = chunk_id

    return chunks


# ============================================================
# EMBEDDING ENGINE
# ============================================================

class EmbeddingEngine:

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
    ):

        print(
            "\nLoading embedding model..."
        )

        self.model_name = (
            model_name
        )

        self.model = (
            SentenceTransformer(
                model_name
            )
        )

        print(
            "Embedding model ready."
        )

    def encode_texts(
        self,
        texts: List[str],
    ) -> np.ndarray:

        if not texts:

            return np.empty(
                (0, 0)
            )

        return (
            self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
        )

    def encode_query(
        self,
        query: str,
    ) -> np.ndarray:

        result = (
            self.model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )

        return (
            result[0]
        )


# ============================================================
# HYBRID RETRIEVER
# ============================================================

class HybridRetriever:

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        vector_weight: float = VECTOR_WEIGHT,
        bm25_weight: float = BM25_WEIGHT,
    ):

        if not np.isclose(
            vector_weight
            + bm25_weight,
            1.0,
        ):

            raise ValueError(
                "vector_weight + "
                "bm25_weight must equal 1.0"
            )

        self.embedding_engine = (
            embedding_engine
        )

        self.vector_weight = (
            vector_weight
        )

        self.bm25_weight = (
            bm25_weight
        )

        self.chunks: List[
            Dict
        ] = []

        self.embeddings: Optional[
            np.ndarray
        ] = None

        self.bm25: Optional[
            BM25Okapi
        ] = None

    @staticmethod
    def tokenize(
        text: str,
    ) -> List[str]:

        return re.findall(
            r"\b[\w\-\.]+\b",
            text.lower(),
            flags=re.UNICODE,
        )

    @staticmethod
    def normalize(
        values: np.ndarray,
    ) -> np.ndarray:

        values = np.asarray(
            values,
            dtype=float,
        )

        if values.size == 0:

            return values

        minimum = (
            values.min()
        )

        maximum = (
            values.max()
        )

        if np.isclose(
            maximum,
            minimum,
        ):

            return np.zeros_like(
                values
            )

        return (
            values - minimum
        ) / (
            maximum - minimum
        )

    def build_index(
        self,
        chunks: List[Dict],
    ) -> None:

        if not chunks:

            raise ValueError(
                "No usable chunks were generated."
            )

        self.chunks = (
            chunks
        )

        texts = [
            chunk["text"]

            for chunk
            in chunks
        ]

        print(
            "\nGenerating embeddings..."
        )

        self.embeddings = (
            self.embedding_engine
            .encode_texts(
                texts
            )
        )

        tokenized_corpus = [

            self.tokenize(
                chunk[
                    "text"
                ]
            )

            for chunk
            in chunks
        ]

        self.bm25 = (
            BM25Okapi(
                tokenized_corpus
            )
        )

        print(
            f"Index ready: "
            f"{len(chunks)} chunks"
        )

    def search(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> List[Dict]:

        if (
            not self.chunks
            or self.embeddings is None
            or self.bm25 is None
        ):

            raise RuntimeError(
                "Retriever index has not been built."
            )

        query = (
            query.strip()
        )

        if not query:
            return []

        query_vector = (
            self.embedding_engine
            .encode_query(
                query
            )
        )

        vector_raw = np.dot(
            self.embeddings,
            query_vector,
        )

        bm25_raw = np.asarray(
            self.bm25.get_scores(
                self.tokenize(
                    query
                )
            ),
            dtype=float,
        )

        vector_scores = (
            self.normalize(
                vector_raw
            )
        )

        bm25_scores = (
            self.normalize(
                bm25_raw
            )
        )

        hybrid_scores = (
            self.vector_weight
            * vector_scores

            + self.bm25_weight
            * bm25_scores
        )

        top_k = min(
            top_k,
            len(
                self.chunks
            ),
        )

        indices = (
            np.argsort(
                hybrid_scores
            )[::-1][
                :top_k
            ]
        )

        results: List[
            Dict
        ] = []

        for rank, index in enumerate(
            indices,
            start=1,
        ):

            item = (
                self.chunks[
                    index
                ].copy()
            )

            item[
                "retrieval_rank"
            ] = rank

            item[
                "vector_score"
            ] = float(
                vector_scores[
                    index
                ]
            )

            item[
                "bm25_score"
            ] = float(
                bm25_scores[
                    index
                ]
            )

            item[
                "hybrid_score"
            ] = float(
                hybrid_scores[
                    index
                ]
            )

            results.append(
                item
            )

        return results


# ============================================================
# RERANKER
# ============================================================

class Reranker:

    def __init__(
        self,
        model_name: str = RERANKER_MODEL,
    ):

        print(
            "\nLoading reranker..."
        )

        self.model_name = (
            model_name
        )

        self.model = (
            CrossEncoder(
                model_name
            )
        )

        print(
            "Reranker ready."
        )

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = RERANK_TOP_K,
    ) -> List[Dict]:

        if not candidates:
            return []

        pairs = [

            [
                query,
                candidate[
                    "text"
                ],
            ]

            for candidate
            in candidates
        ]

        scores = (
            self.model.predict(
                pairs,
                show_progress_bar=False,
            )
        )

        scores = np.asarray(
            scores,
            dtype=float,
        ).reshape(
            -1
        )

        results: List[
            Dict
        ] = []

        for candidate, score in zip(
            candidates,
            scores,
        ):

            item = (
                candidate.copy()
            )

            item[
                "reranker_score"
            ] = float(
                score
            )

            results.append(
                item
            )

        results.sort(
            key=lambda item:
                item[
                    "reranker_score"
                ],
            reverse=True,
        )

        return results[
            :min(
                top_k,
                len(results),
            )
        ]


# ============================================================
# LOW-VALUE CONTENT DETECTION
# ============================================================

def looks_like_low_value_chunk(
    text: str,
) -> bool:

    normalized = (
        text.lower()
        .strip()
    )

    if not normalized:
        return True

    question_marks = (
        normalized.count(
            "?"
        )
        + normalized.count(
            "؟"
        )
    )

    words = (
        normalized.split()
    )

    if len(words) < 5:
        return True

    # Large block of suggested questions
    if (
        question_marks >= 3
        and len(words) < 140
    ):

        return True

    low_value_phrases = [
        "suggested local retrieval questions",
        "suggested multi-evidence questions",
        "table of contents",
        "contents",
        "index",
    ]

    for phrase in (
        low_value_phrases
    ):

        if phrase in normalized:
            return True

    return False


# ============================================================
# QUALITY GATE
# ============================================================

def filter_low_quality_evidence(
    evidence: List[Dict],
    min_reranker_score:
        float = MIN_RERANKER_SCORE,
    min_hybrid_score:
        float = MIN_HYBRID_SCORE,
) -> List[Dict]:

    filtered: List[
        Dict
    ] = []

    for item in evidence:

        if looks_like_low_value_chunk(
            item.get(
                "text",
                "",
            )
        ):

            continue

        reranker_score = float(
            item.get(
                "reranker_score",
                -999.0,
            )
        )

        hybrid_score = float(
            item.get(
                "hybrid_score",
                0.0,
            )
        )

        if (
            reranker_score
            >= min_reranker_score
            or hybrid_score
            >= min_hybrid_score
        ):

            filtered.append(
                item
            )

    return filtered


# ============================================================
# DEDUPLICATION
# ============================================================

def normalize_for_dedup(
    text: str,
) -> str:

    text = (
        text.lower()
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    text = re.sub(
        r"[^\w\s]",
        "",
        text,
        flags=re.UNICODE,
    )

    return (
        text.strip()
    )


def jaccard_similarity(
    text_a: str,
    text_b: str,
) -> float:

    words_a = set(
        normalize_for_dedup(
            text_a
        ).split()
    )

    words_b = set(
        normalize_for_dedup(
            text_b
        ).split()
    )

    if (
        not words_a
        or not words_b
    ):

        return 0.0

    union = (
        words_a
        | words_b
    )

    if not union:
        return 0.0

    intersection = (
        words_a
        & words_b
    )

    return (
        len(intersection)
        / len(union)
    )


def deduplicate_evidence(
    evidence: List[Dict],
    similarity_threshold:
        float = DEDUP_THRESHOLD,
) -> List[Dict]:

    unique: List[
        Dict
    ] = []

    for candidate in evidence:

        is_duplicate = any(

            jaccard_similarity(
                candidate[
                    "text"
                ],
                selected[
                    "text"
                ],
            )
            >= similarity_threshold

            for selected
            in unique
        )

        if not is_duplicate:

            unique.append(
                candidate
            )

    return unique


# ============================================================
# QUERY TYPE / ADAPTIVE EVIDENCE
# ============================================================

def classify_query_complexity(
    question: str,
) -> str:

    q = (
        question.lower()
        .strip()
    )

    analytical_markers = [
        "why",
        "how",
        "compare",
        "explain",
        "relationship",
        "difference",
        "impact",
        "reason",
        "because",
        "what caused",
        "what factors",
        "چرا",
        "چگونه",
        "توضیح",
        "مقایسه",
        "تفاوت",
        "علت",
        "دلایل",
        "تاثیر",
        "تأثیر",
    ]

    connector_markers = [
        " and ",
        " also ",
        " as well as ",
        " both ",
        " و ",
        " همچنین ",
    ]

    if any(
        marker in q
        for marker
        in analytical_markers
    ):

        return "analytical"

    if any(
        marker in q
        for marker
        in connector_markers
    ):

        return "multi_fact"

    if len(
        q.split()
    ) <= 12:

        return "simple"

    return "standard"


def select_adaptive_evidence(
    question: str,
    evidence: List[Dict],
) -> Tuple[
    List[Dict],
    str,
]:

    query_type = (
        classify_query_complexity(
            question
        )
    )

    if query_type == "simple":

        max_evidence = 2

    elif query_type == "multi_fact":

        max_evidence = 3

    elif query_type == "analytical":

        max_evidence = 4

    else:

        max_evidence = 3

    selected = evidence[
        :max_evidence
    ]

    return (
        selected,
        query_type,
    )


# ============================================================
# CONTEXT BUILDER
# ============================================================

def build_context(
    evidence: List[Dict],
    max_words: int = MAX_CONTEXT_WORDS,
) -> Tuple[
    str,
    List[Dict],
]:

    context_parts: List[
        str
    ] = []

    sources: List[
        Dict
    ] = []

    used_words = 0

    for item in evidence:

        text = item.get(
            "text",
            "",
        ).strip()

        if not text:
            continue

        words = (
            text.split()
        )

        remaining = (
            max_words
            - used_words
        )

        if remaining <= 0:
            break

        if len(words) > remaining:

            text = " ".join(
                words[
                    :remaining
                ]
            )

        source_id = (
            f"SOURCE_"
            f"{len(sources) + 1}"
        )

        filename = item.get(
            "filename",
            "Unknown",
        )

        page_number = item.get(
            "page_number"
        )

        chunk_id = item.get(
            "chunk_id",
            "Unknown",
        )

        location_lines = [
            f"Document: {filename}",
        ]

        if page_number is not None:

            location_lines.append(
                f"Page: {page_number}"
            )

        location_lines.append(
            f"Chunk: {chunk_id}"
        )

        context_parts.append(
            f"[{source_id}]\n"
            + "\n".join(
                location_lines
            )
            + "\nContent:\n"
            + text
        )

        sources.append(
            {
                "source_id":
                    source_id,

                "filename":
                    filename,

                "page_number":
                    page_number,

                "chunk_id":
                    chunk_id,

                "hybrid_score":
                    item.get(
                        "hybrid_score"
                    ),

                "reranker_score":
                    item.get(
                        "reranker_score"
                    ),
            }
        )

        used_words += (
            len(
                text.split()
            )
        )

    return (
        "\n\n".join(
            context_parts
        ),
        sources,
    )


# ============================================================
# PROMPT
# ============================================================

def build_prompt(
    question: str,
    context: str,
) -> str:

    return f"""
You are an evidence-grounded document intelligence assistant.

Answer the user's question using ONLY the supplied document evidence.

STRICT RULES:

1. Do not use outside knowledge.
2. Do not invent facts.
3. Every important factual claim must have a citation.
4. Citations MUST use ASCII square brackets exactly like [SOURCE_1].
5. Never use decorative citation brackets such as 【SOURCE_1】.
6. Never invent a SOURCE identifier.
7. Prefer direct evidence over inference.
8. Use the minimum number of sources necessary to support the answer.
9. Do not cite multiple sources that merely repeat the same fact unless necessary.
10. For simple factual questions, answer briefly and directly.
11. For analytical questions, combine the relevant evidence carefully.
12. If evidence is insufficient, respond exactly:
The available document evidence is insufficient to answer this question.
13. Answer in the same language as the user's question whenever practical.

USER QUESTION:

{question}

DOCUMENT EVIDENCE:

{context}

FINAL ANSWER:
""".strip()


# ============================================================
# GROQ GENERATOR
# ============================================================

class GroundedAnswerGenerator:

    def __init__(
        self,
        api_key: str,
        model_name: str = GROQ_MODEL,
    ):

        if not api_key:

            raise ValueError(
                "GROQ_API_KEY was not found."
            )

        self.model_name = (
            model_name
        )

        self.client = Groq(
            api_key=
                api_key
        )

    def generate(
        self,
        question: str,
        context: str,
    ) -> Dict:

        if not context.strip():

            return {
                "answer":
                    (
                        "The available document "
                        "evidence is insufficient "
                        "to answer this question."
                    ),

                "generation_time":
                    0.0,

                "model":
                    self.model_name,

                "prompt_tokens":
                    None,

                "completion_tokens":
                    None,

                "total_tokens":
                    None,
            }

        prompt = (
            build_prompt(
                question=
                    question,

                context=
                    context,
            )
        )

        start = (
            time.perf_counter()
        )

        response = (
            self.client
            .chat
            .completions
            .create(
                model=
                    self.model_name,

                messages=[
                    {
                        "role":
                            "system",

                        "content":
                            (
                                "You are a precise "
                                "evidence-grounded "
                                "document intelligence "
                                "assistant."
                            ),
                    },
                    {
                        "role":
                            "user",

                        "content":
                            prompt,
                    },
                ],

                temperature=0,

                max_completion_tokens=
                    500,

                stream=False,
            )
        )

        generation_time = (
            time.perf_counter()
            - start
        )

        message = (
            response
            .choices[0]
            .message
        )

        answer = (
            message.content
            or ""
        ).strip()

        if not answer:

            raise RuntimeError(
                "Groq returned an empty answer."
            )

        usage = getattr(
            response,
            "usage",
            None,
        )

        return {
            "answer":
                answer,

            "generation_time":
                generation_time,

            "model":
                self.model_name,

            "prompt_tokens":
                (
                    getattr(
                        usage,
                        "prompt_tokens",
                        None,
                    )

                    if usage is not None

                    else None
                ),

            "completion_tokens":
                (
                    getattr(
                        usage,
                        "completion_tokens",
                        None,
                    )

                    if usage is not None

                    else None
                ),

            "total_tokens":
                (
                    getattr(
                        usage,
                        "total_tokens",
                        None,
                    )

                    if usage is not None

                    else None
                ),
        }


# ============================================================
# CITATIONS
# ============================================================

def canonicalize_citation_brackets(
    answer: str,
) -> str:

    if not answer:
        return answer

    answer = re.sub(
        r"【\s*(SOURCE_\d+)\s*】",
        r"[\1]",
        answer,
        flags=re.IGNORECASE,
    )

    return answer


def extract_citations(
    answer: str,
    sources: List[Dict],
) -> List[Dict]:

    if not answer:
        return []

    answer = (
        canonicalize_citation_brackets(
            answer
        )
    )

    source_lookup = {

        source[
            "source_id"
        ]:
            source

        for source
        in sources
    }

    citation_ids = re.findall(
        r"\[\s*(SOURCE_\d+)\s*\]",
        answer,
        flags=re.IGNORECASE,
    )

    used_sources: List[
        Dict
    ] = []

    seen = set()

    for raw_source_id in (
        citation_ids
    ):

        source_id = (
            raw_source_id.upper()
        )

        if source_id in seen:
            continue

        source = (
            source_lookup.get(
                source_id
            )
        )

        if source is not None:

            used_sources.append(
                source
            )

            seen.add(
                source_id
            )

    return used_sources


# ============================================================
# CONFIDENCE
# ============================================================

def estimate_confidence(
    evidence: List[Dict],
    used_sources: List[Dict],
) -> Tuple[
    str,
    float,
]:

    if not evidence:

        return (
            "Low",
            0.0,
        )

    top = (
        evidence[0]
    )

    hybrid = float(
        top.get(
            "hybrid_score",
            0.0,
        )
    )

    reranker = float(
        top.get(
            "reranker_score",
            -10.0,
        )
    )

    retrieval_component = float(
        np.clip(
            hybrid,
            0.0,
            1.0,
        )
    )

    reranker_component = float(
        1.0
        / (
            1.0
            + np.exp(
                -(
                    reranker
                    + 3.0
                )
            )
        )
    )

    citation_component = min(
        len(
            used_sources
        )
        / max(
            len(
                evidence
            ),
            1,
        ),
        1.0,
    )

    score = (
        0.45
        * retrieval_component

        + 0.40
        * reranker_component

        + 0.15
        * citation_component
    )

    score = float(
        np.clip(
            score,
            0.0,
            1.0,
        )
    )

    if score >= 0.70:

        label = (
            "High"
        )

    elif score >= 0.45:

        label = (
            "Medium"
        )

    else:

        label = (
            "Low"
        )

    return (
        label,
        score,
    )


# ============================================================
# DISPLAY
# ============================================================

def print_document_summary(
    document: Dict,
    chunks: List[Dict],
    model_load_time: float,
    parsing_time: float,
    chunking_time: float,
    indexing_time: float,
    total_processing_time: float,
) -> None:

    print(
        "\nDOCUMENT"
    )

    print(
        f"File: "
        f"{document['filename']}"
    )

    print(
        f"Type: "
        f"{document['file_type']}"
    )

    print(
        f"Pages: "
        f"{document['page_count']}"
    )

    print(
        f"Characters: "
        f"{document['character_count']:,}"
    )

    print(
        f"Chunks: "
        f"{len(chunks)}"
    )

    average_chunk = (

        sum(
            chunk[
                "word_count"
            ]

            for chunk
            in chunks
        )

        / len(
            chunks
        )

        if chunks

        else 0.0
    )

    print(
        f"Average chunk size: "
        f"{average_chunk:.1f} words"
    )

    print(
        "\nPERFORMANCE"
    )

    print(
        f"Model load time: "
        f"{model_load_time:.3f}s"
    )

    print(
        f"Parsing time: "
        f"{parsing_time:.3f}s"
    )

    print(
        f"Chunking time: "
        f"{chunking_time:.3f}s"
    )

    print(
        f"Embedding/index time: "
        f"{indexing_time:.3f}s"
    )

    print(
        f"Document processing time: "
        f"{total_processing_time:.3f}s"
    )


def print_evidence(
    evidence: List[Dict],
) -> None:

    print(
        "\nTOP EVIDENCE"
    )

    if not evidence:

        print(
            "No evidence retrieved."
        )

        return

    for index, item in enumerate(
        evidence,
        start=1,
    ):

        preview = (
            item.get(
                "text",
                "",
            )[:500]
            .replace(
                "\n",
                " ",
            )
        )

        print(
            "\n"
            + "-" * 60
        )

        print(
            f"SOURCE_{index}"
        )

        print(
            f"Document: "
            f"{item.get('filename')}"
        )

        if item.get(
            "page_number"
        ) is not None:

            print(
                f"Page: "
                f"{item.get('page_number')}"
            )

        print(
            f"Chunk ID: "
            f"{item.get('chunk_id')}"
        )

        print(
            f"Vector score: "
            f"{item.get('vector_score', 0.0):.4f}"
        )

        print(
            f"BM25 score: "
            f"{item.get('bm25_score', 0.0):.4f}"
        )

        print(
            f"Hybrid score: "
            f"{item.get('hybrid_score', 0.0):.4f}"
        )

        print(
            f"Reranker score: "
            f"{item.get('reranker_score', 0.0):.4f}"
        )

        print(
            "\nText:"
        )

        print(
            preview
            + (
                "..."

                if len(
                    item.get(
                        "text",
                        "",
                    )
                ) > 500

                else ""
            )
        )


def print_final_answer(
    generation_result: Dict,
    confidence_label: str,
    confidence_score: float,
    used_sources: List[Dict],
    retrieval_time: float,
    reranking_time: float,
    total_query_time: float,
    query_type: str,
) -> None:

    answer = (
        canonicalize_citation_brackets(
            generation_result[
                "answer"
            ]
        )
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "FINAL GROUNDED ANSWER"
    )

    print(
        "=" * 60
    )

    print(
        "\n"
        + answer
    )

    print(
        "\n"
        + "-" * 60
    )

    print(
        "ANSWER METRICS"
    )

    print(
        "-" * 60
    )

    print(
        "Provider: Groq"
    )

    print(
        f"Model: "
        f"{generation_result['model']}"
    )

    print(
        f"Query type: "
        f"{query_type}"
    )

    print(
        f"Evidence used: "
        f"{len(used_sources)}"
    )

    print(
        f"Confidence: "
        f"{confidence_label} "
        f"({confidence_score:.2f})"
    )

    print(
        f"Retrieval time: "
        f"{retrieval_time:.3f}s"
    )

    print(
        f"Reranking time: "
        f"{reranking_time:.3f}s"
    )

    print(
        f"Generation time: "
        f"{generation_result['generation_time']:.3f}s"
    )

    print(
        f"Total query time: "
        f"{total_query_time:.3f}s"
    )

    if generation_result[
        "prompt_tokens"
    ] is not None:

        print(
            f"Prompt tokens: "
            f"{generation_result['prompt_tokens']}"
        )

    if generation_result[
        "completion_tokens"
    ] is not None:

        print(
            f"Completion tokens: "
            f"{generation_result['completion_tokens']}"
        )

    if generation_result[
        "total_tokens"
    ] is not None:

        print(
            f"Total tokens: "
            f"{generation_result['total_tokens']}"
        )

    print(
        "\nCITATIONS"
    )

    if not used_sources:

        print(
            "No explicit citation was detected."
        )

        return

    for source in (
        used_sources
    ):

        page_number = (
            source.get(
                "page_number"
            )
        )

        if page_number is None:

            print(
                f"- "
                f"[{source['source_id']}] "
                f"{source['filename']} "
                f"| Chunk "
                f"{source['chunk_id']}"
            )

        else:

            print(
                f"- "
                f"[{source['source_id']}] "
                f"{source['filename']} "
                f"| Page "
                f"{page_number} "
                f"| Chunk "
                f"{source['chunk_id']}"
            )


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(
    file_path: str,
) -> None:

    print(
        "\n"
        + "=" * 60
    )

    print(
        "TURBORAG TECHNICAL DEMO v3.1"
    )

    print(
        "ADAPTIVE EVIDENCE GROQ EDITION"
    )

    print(
        "=" * 60
    )

    model_start = (
        time.perf_counter()
    )

    embedding_engine = (
        EmbeddingEngine()
    )

    reranker = (
        Reranker()
    )

    answer_generator = (
        GroundedAnswerGenerator(
            api_key=
                GROQ_API_KEY,

            model_name=
                GROQ_MODEL,
        )
    )

    model_load_time = (
        time.perf_counter()
        - model_start
    )

    processing_start = (
        time.perf_counter()
    )

    parsing_start = (
        time.perf_counter()
    )

    document = (
        parse_document(
            file_path
        )
    )

    parsing_time = (
        time.perf_counter()
        - parsing_start
    )

    chunking_start = (
        time.perf_counter()
    )

    chunks = (
        create_document_chunks(
            document
        )
    )

    chunking_time = (
        time.perf_counter()
        - chunking_start
    )

    retriever = (
        HybridRetriever(
            embedding_engine=
                embedding_engine
        )
    )

    indexing_start = (
        time.perf_counter()
    )

    retriever.build_index(
        chunks
    )

    indexing_time = (
        time.perf_counter()
        - indexing_start
    )

    total_processing_time = (
        time.perf_counter()
        - processing_start
    )

    print_document_summary(
        document=
            document,

        chunks=
            chunks,

        model_load_time=
            model_load_time,

        parsing_time=
            parsing_time,

        chunking_time=
            chunking_time,

        indexing_time=
            indexing_time,

        total_processing_time=
            total_processing_time,
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "DOCUMENT READY"
    )

    print(
        "=" * 60
    )

    print(
        "\nMode: Fast / TurboRAG"
    )

    print(
        "LLM Provider: Groq"
    )

    print(
        f"LLM Model: "
        f"{GROQ_MODEL}"
    )

    while True:

        print(
            "\n"
            + "-" * 60
        )

        question = input(
            "\nAsk a question "
            "(or type 'exit'): "
        ).strip()

        if question.lower() in {
            "exit",
            "quit",
        }:

            print(
                "\nDemo finished."
            )

            break

        if not question:
            continue

        query_start = (
            time.perf_counter()
        )

        try:

            retrieval_start = (
                time.perf_counter()
            )

            candidates = (
                retriever.search(
                    query=
                        question,

                    top_k=
                        RETRIEVAL_TOP_K,
                )
            )

            retrieval_time = (
                time.perf_counter()
                - retrieval_start
            )

            reranking_start = (
                time.perf_counter()
            )

            reranked = (
                reranker.rerank(
                    query=
                        question,

                    candidates=
                        candidates,

                    top_k=
                        RERANK_TOP_K,
                )
            )

            reranking_time = (
                time.perf_counter()
                - reranking_start
            )

            quality_evidence = (
                filter_low_quality_evidence(
                    evidence=
                        reranked,

                    min_reranker_score=
                        MIN_RERANKER_SCORE,

                    min_hybrid_score=
                        MIN_HYBRID_SCORE,
                )
            )

            deduped_evidence = (
                deduplicate_evidence(
                    evidence=
                        quality_evidence,

                    similarity_threshold=
                        DEDUP_THRESHOLD,
                )
            )

            (
                evidence,
                query_type,
            ) = (
                select_adaptive_evidence(
                    question=
                        question,

                    evidence=
                        deduped_evidence,
                )
            )

            (
                context,
                sources,
            ) = (
                build_context(
                    evidence=
                        evidence,

                    max_words=
                        MAX_CONTEXT_WORDS,
                )
            )

            generation_result = (
                answer_generator.generate(
                    question=
                        question,

                    context=
                        context,
                )
            )

            generation_result[
                "answer"
            ] = (
                canonicalize_citation_brackets(
                    generation_result[
                        "answer"
                    ]
                )
            )

            used_sources = (
                extract_citations(
                    answer=
                        generation_result[
                            "answer"
                        ],

                    sources=
                        sources,
                )
            )

            (
                confidence_label,
                confidence_score,
            ) = (
                estimate_confidence(
                    evidence=
                        evidence,

                    used_sources=
                        used_sources,
                )
            )

            total_query_time = (
                time.perf_counter()
                - query_start
            )

            print(
                "\nRETRIEVAL RESULTS"
            )

            print(
                f"Hybrid candidates: "
                f"{len(candidates)}"
            )

            print(
                f"Reranked candidates: "
                f"{len(reranked)}"
            )

            print(
                f"After quality gate: "
                f"{len(quality_evidence)}"
            )

            print(
                f"After deduplication: "
                f"{len(deduped_evidence)}"
            )

            print(
                f"Adaptive evidence: "
                f"{len(evidence)}"
            )

            print(
                f"Query type: "
                f"{query_type}"
            )

            print(
                f"Retrieval time: "
                f"{retrieval_time:.3f}s"
            )

            print(
                f"Reranking time: "
                f"{reranking_time:.3f}s"
            )

            print_evidence(
                evidence
            )

            print_final_answer(
                generation_result=
                    generation_result,

                confidence_label=
                    confidence_label,

                confidence_score=
                    confidence_score,

                used_sources=
                    used_sources,

                retrieval_time=
                    retrieval_time,

                reranking_time=
                    reranking_time,

                total_query_time=
                    total_query_time,

                query_type=
                    query_type,
            )

        except KeyboardInterrupt:

            print(
                "\nQuery cancelled."
            )

            continue

        except Exception as error:

            print(
                "\nQUERY ERROR"
            )

            print(
                "-" * 60
            )

            print(
                f"{type(error).__name__}: "
                f"{error}"
            )

            print(
                "\nThe document remains loaded. "
                "You may ask another question "
                "or type 'exit'."
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    print(
        "\nAdaptive Document "
        "Intelligence RAG"
    )

    print(
        "TurboRAG Technical "
        "Proof of Concept v3.1"
    )

    print(
        "Adaptive Evidence Groq Edition"
    )

    print(
        "\nSupported files: "
        "PDF, DOCX, TXT"
    )

    if not GROQ_API_KEY:

        print(
            "\nERROR:"
        )

        print(
            "GROQ_API_KEY was not found."
        )

        print(
            f"\nExpected .env path:\n"
            f"{ENV_PATH}"
        )

        print(
            "\n.env example:"
        )

        print(
            "GROQ_API_KEY=your_real_key"
        )

        print(
            "GROQ_MODEL="
            "openai/gpt-oss-20b"
        )

        raise SystemExit(
            1
        )

    file_path = input(
        "\nEnter document path: "
    ).strip()

    file_path = (
        file_path
        .strip('"')
        .strip("'")
    )

    try:

        run_pipeline(
            file_path
        )

    except KeyboardInterrupt:

        print(
            "\n\nProgram stopped by user."
        )

    except Exception as error:

        print(
            "\nSTARTUP ERROR"
        )

        print(
            "-" * 60
        )

        print(
            f"{type(error).__name__}: "
            f"{error}"
        )