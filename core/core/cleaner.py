# Text cleaning and normalization module
import re
from typing import Optional


def normalize_persian_characters(text: str) -> str:
    """
    Normalize common Arabic characters to Persian equivalents.
    """

    replacements = {
        "ي": "ی",
        "ى": "ی",
        "ك": "ک",
        "ؤ": "و",
        "ۀ": "ه",
        "ة": "ه",
    }

    for source, target in replacements.items():
        text = text.replace(source, target)

    return text


def normalize_whitespace(text: str) -> str:
    """
    Normalize spaces, tabs, and line breaks.
    """

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Replace tabs with spaces
    text = text.replace("\t", " ")

    # Remove excessive spaces
    text = re.sub(r"[ ]{2,}", " ", text)

    # Remove spaces before punctuation
    text = re.sub(r"\s+([،؛:,.!?؟])", r"\1", text)

    # Normalize excessive empty lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def fix_broken_lines(text: str) -> str:
    """
    Join lines that were broken during PDF extraction.

    Empty lines are preserved as paragraph boundaries.
    """

    lines = text.split("\n")

    cleaned_lines = []
    current_line = ""

    for line in lines:
        line = line.strip()

        if not line:
            if current_line:
                cleaned_lines.append(current_line.strip())
                current_line = ""

            cleaned_lines.append("")
            continue

        if current_line:
            current_line += " " + line
        else:
            current_line = line

        # End current paragraph when sentence appears complete
        if re.search(r"[.!?؟:]$", line):
            cleaned_lines.append(current_line.strip())
            current_line = ""

    if current_line:
        cleaned_lines.append(current_line.strip())

    return "\n".join(cleaned_lines)


def remove_repeated_noise(text: str) -> str:
    """
    Remove simple document noise patterns.
    """

    # Remove standalone page numbers
    text = re.sub(
        r"(?m)^\s*(page\s*)?\d+\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove excessive separator characters
    text = re.sub(r"[-_=]{4,}", " ", text)

    return text


def clean_text(
    text: Optional[str],
    normalize_persian: bool = True,
) -> str:
    """
    Main cleaning pipeline.

    Parameters
    ----------
    text:
        Raw extracted document text.

    normalize_persian:
        Normalize common Persian/Arabic character differences.

    Returns
    -------
    str
        Cleaned text ready for chunking.
    """

    if not text:
        return ""

    cleaned = text

    if normalize_persian:
        cleaned = normalize_persian_characters(cleaned)

    cleaned = remove_repeated_noise(cleaned)
    cleaned = fix_broken_lines(cleaned)
    cleaned = normalize_whitespace(cleaned)

    return cleaned
