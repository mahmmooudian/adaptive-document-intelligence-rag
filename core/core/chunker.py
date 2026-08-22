# Semantic chunking module
from typing import Dict, List


def split_into_paragraphs(text: str) -> List[str]:
    """
    Split cleaned text into meaningful paragraphs.
    """

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    return paragraphs


def estimate_word_count(text: str) -> int:
    """
    Estimate chunk size using word count.

    This is intentionally lightweight for the demo version.
    """

    return len(text.split())


def create_chunks(
    text: str,
    target_words: int = 180,
    max_words: int = 260,
    overlap_words: int = 40,
) -> List[Dict]:
    """
    Create context-aware chunks using paragraph boundaries.

    The algorithm:
    1. Splits text into paragraphs.
    2. Groups related consecutive paragraphs until target size.
    3. Prevents chunks from exceeding max_words.
    4. Adds overlap between consecutive chunks.

    Parameters
    ----------
    text:
        Cleaned input text.

    target_words:
        Preferred approximate chunk size.

    max_words:
        Maximum allowed chunk size.

    overlap_words:
        Number of words copied from the previous chunk.

    Returns
    -------
    List[Dict]
        List of chunk objects with metadata.
    """

    if not text or not text.strip():
        return []

    paragraphs = split_into_paragraphs(text)

    chunks: List[Dict] = []
    current_parts: List[str] = []
    current_word_count = 0

    for paragraph in paragraphs:
        paragraph_word_count = estimate_word_count(paragraph)

        # If a paragraph itself is larger than max_words,
        # split it into smaller windows.
        if paragraph_word_count > max_words:
            words = paragraph.split()

            start = 0

            while start < len(words):
                end = min(start + max_words, len(words))

                segment = " ".join(words[start:end])

                if current_parts:
                    chunk_text = "\n\n".join(current_parts).strip()

                    chunks.append(
                        {
                            "text": chunk_text,
                            "word_count": estimate_word_count(chunk_text),
                        }
                    )

                    current_parts = []
                    current_word_count = 0

                chunks.append(
                    {
                        "text": segment,
                        "word_count": estimate_word_count(segment),
                    }
                )

                if end == len(words):
                    break

                start = max(end - overlap_words, start + 1)

            continue

        projected_size = current_word_count + paragraph_word_count

        if current_parts and projected_size > max_words:
            chunk_text = "\n\n".join(current_parts).strip()

            chunks.append(
                {
                    "text": chunk_text,
                    "word_count": estimate_word_count(chunk_text),
                }
            )

            previous_words = chunk_text.split()

            overlap_text = " ".join(
                previous_words[-overlap_words:]
            )

            current_parts = []

            if overlap_text:
                current_parts.append(overlap_text)
                current_word_count = estimate_word_count(overlap_text)
            else:
                current_word_count = 0

        current_parts.append(paragraph)
        current_word_count += paragraph_word_count

        if current_word_count >= target_words:
            chunk_text = "\n\n".join(current_parts).strip()

            chunks.append(
                {
                    "text": chunk_text,
                    "word_count": estimate_word_count(chunk_text),
                }
            )

            previous_words = chunk_text.split()

            overlap_text = " ".join(
                previous_words[-overlap_words:]
            )

            current_parts = []

            if overlap_text:
                current_parts.append(overlap_text)
                current_word_count = estimate_word_count(overlap_text)
            else:
                current_word_count = 0

    if current_parts:
        chunk_text = "\n\n".join(current_parts).strip()

        if chunk_text:
            chunks.append(
                {
                    "text": chunk_text,
                    "word_count": estimate_word_count(chunk_text),
                }
            )

    # Add chunk IDs
    for index, chunk in enumerate(chunks, start=1):
        chunk["chunk_id"] = index

    return chunks
