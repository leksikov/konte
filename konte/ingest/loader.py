"""Document loader module for PDF, TXT, and MD files."""

from pathlib import Path

import aiofiles
import structlog
from pypdf import PdfReader

logger = structlog.get_logger()

_TEXT_SUFFIXES = frozenset({".txt", ".md"})
_PDF_SUFFIX = ".pdf"


def load_txt(file_path: Path) -> str:
    """Load content from a TXT file.

    Args:
        file_path: Path to the TXT file.

    Returns:
        File content as string.

    Raises:
        FileNotFoundError: If file does not exist.
    """
    file_path = Path(file_path)
    logger.debug("document_loading", path=str(file_path), file_type="txt")
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    content = file_path.read_text(encoding="utf-8")
    logger.debug("document_loaded", path=str(file_path), content_length=len(content))
    return content


def load_md(file_path: Path) -> str:
    """Load content from a Markdown file.

    Args:
        file_path: Path to the MD file.

    Returns:
        File content as string (markdown preserved).

    Raises:
        FileNotFoundError: If file does not exist.
    """
    return load_txt(file_path)


def load_pdf(file_path: Path) -> str:
    """Load content from a PDF file.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Extracted text content as string.

    Raises:
        FileNotFoundError: If file does not exist.
    """
    file_path = Path(file_path)
    logger.debug("document_loading", path=str(file_path), file_type="pdf")
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    reader = PdfReader(file_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    logger.debug("pdf_pages_loaded", path=str(file_path), page_count=len(pages))
    content = "\n\n".join(pages)
    logger.debug("document_loaded", path=str(file_path), content_length=len(content))
    return content


def load_document(file_path: Path) -> str:
    """Load document content with auto-detection of file type.

    Args:
        file_path: Path to the document file.

    Returns:
        Document content as string.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file type is not supported.
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == _PDF_SUFFIX:
        return load_pdf(file_path)
    if suffix in _TEXT_SUFFIXES:
        return load_txt(file_path)
    raise ValueError(f"Unsupported file type: {suffix}")


async def load_txt_async(file_path: Path) -> str:
    """Async version of load_txt.

    Args:
        file_path: Path to the TXT file.

    Returns:
        File content as string.

    Raises:
        FileNotFoundError: If file does not exist.
    """
    file_path = Path(file_path)
    logger.debug("document_loading", path=str(file_path), file_type="txt")
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    async with aiofiles.open(file_path, encoding="utf-8") as f:
        content = await f.read()
    logger.debug("document_loaded", path=str(file_path), content_length=len(content))
    return content


async def load_md_async(file_path: Path) -> str:
    """Async version of load_md.

    Args:
        file_path: Path to the MD file.

    Returns:
        File content as string.

    Raises:
        FileNotFoundError: If file does not exist.
    """
    return await load_txt_async(file_path)


async def load_document_async(file_path: Path) -> str:
    """Async version of load_document.

    PDF extraction is CPU-bound with no async pypdf equivalent, so it runs
    inline; only text files gain from the async path.

    Args:
        file_path: Path to the document file.

    Returns:
        Document content as string.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file type is not supported.
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == _PDF_SUFFIX:
        return load_pdf(file_path)
    if suffix in _TEXT_SUFFIXES:
        return await load_txt_async(file_path)
    raise ValueError(f"Unsupported file type: {suffix}")
