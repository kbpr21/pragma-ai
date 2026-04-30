import logging
from pathlib import Path
from typing import List

import httpx

from pragma.ingestion.loader import DocumentSegment

logger = logging.getLogger(__name__)


def load_html_file(path: Path) -> List[DocumentSegment]:
    """Load HTML file."""
    html_content = path.read_text(encoding="utf-8")
    return _parse_html(html_content, str(path))


def load_html_url(url: str) -> List[DocumentSegment]:
    """Load HTML from URL."""
    try:
        response = httpx.get(url, timeout=30.0)
        response.raise_for_status()
        return _parse_html(response.text, url)
    except httpx.HTTPError as e:
        logger.warning(f"Failed to fetch URL {url}: {e}")
        return []


def _parse_html(html_content: str, source: str) -> List[DocumentSegment]:
    """Parse HTML content and extract text."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.warning(
            "beautifulsoup4 not installed. Install: pip install beautifulsoup4"
        )
        return []

    soup = BeautifulSoup(html_content, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    main_content = (
        soup.find("article")
        or soup.find("main")
        or soup.find("div", {"role": "main"})
        or soup.body
    )

    if not main_content:
        return []

    for unwanted in main_content(["nav", "footer", "header"]):
        unwanted.decompose()

    text = main_content.get_text(separator="\n", strip=True)

    if len(text) < 50:
        return []

    return [
        DocumentSegment(
            content=text,
            source=source,
            doc_type="html",
            metadata={"url": source}
            if source.startswith("http")
            else {"filename": Path(source).name},
        )
    ]
