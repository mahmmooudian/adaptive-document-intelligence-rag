# Document parsing module
from pathlib import Path
from typing import Dict, List

import fitz  # PyMuPDF
from docx import Document


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def parse_document(file_path: str) -> Dict:
    """
    Parse a supported document and return normalized document data.

    Supported formats:
    - PDF
    - DOCX
    - TXT
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format: {extension}. "
            f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    if extension == ".pdf":
        return _parse_pdf(path)

    if extension == ".docx":
        return _parse_docx(path)

    if extension == ".txt":
        return _parse_txt(path)

    raise ValueError("Unsupported document format.")


def _parse_pdf(path: Path) -> Dict:
    """
    Extract text page-by-page from a PDF file.
    """

    pages: List[Dict] = []
    full_text_parts: List[str] = []

    document = fitz.open(path)

    try:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            pages.append(
                {
                    "page_number": page_number,
                    "text": text,
                }
            )

            if text:
                full_text_parts.append(text)

    finally:
        document.close()

    full_text = "\n\n".join(full_text_parts)

    return {
        "filename": path.name,
        "file_type": "pdf",
        "page_count": len(pages),
        "character_count": len(full_text),
        "text": full_text,
        "pages": pages,
    }


def _parse_docx(path: Path) -> Dict:
    """
    Extract paragraph text from a DOCX file.
    """

    document = Document(path)

    paragraphs: List[Dict] = []
    full_text_parts: List[str] = []

    for index, paragraph in enumerate(document.paragraphs, start=1):
        text = paragraph.text.strip()

        if not text:
            continue

        paragraphs.append(
            {
                "paragraph_number": index,
                "text": text,
            }
        )

        full_text_parts.append(text)

    full_text = "\n\n".join(full_text_parts)

    return {
        "filename": path.name,
        "file_type": "docx",
        "page_count": None,
        "character_count": len(full_text),
        "text": full_text,
        "paragraphs": paragraphs,
    }


def _parse_txt(path: Path) -> Dict:
    """
    Extract plain text from a TXT file.
    """

    text = path.read_text(encoding="utf-8", errors="ignore").strip()

    return {
        "filename": path.name,
        "file_type": "txt",
        "page_count": None,
        "character_count": len(text),
        "text": text,
    }
