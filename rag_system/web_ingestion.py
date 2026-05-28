from html.parser import HTMLParser
from typing import List, Dict, Any
from urllib.parse import quote_plus
import re
import requests


DEFAULT_TIMEOUT = 15


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"} and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self._parts.append(text)

    def get_text(self) -> str:
        return " ".join(self._parts)


def _extract_html_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    return parser.get_text()


def fetch_url_text(url: str, timeout: int = DEFAULT_TIMEOUT, max_chars: int = 8000) -> str:
    """Fetch and extract readable text from a public URL."""
    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "rag-system/0.1 (+https://github.com/)"}
    )
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()
    if "text/html" not in content_type and "text/plain" not in content_type:
        return ""

    if "text/plain" in content_type:
        return response.text[:max_chars]

    return _extract_html_text(response.text)[:max_chars]


def url_to_document(url: str) -> Dict[str, Any]:
    text = fetch_url_text(url)
    return {
        "content": text,
        "metadata": {
            "source": url,
            "type": "url"
        }
    }


def search_web(query: str, max_results: int = 5, timeout: int = DEFAULT_TIMEOUT) -> List[str]:
    """Return public web URLs using DuckDuckGo HTML search response."""
    search_url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
    response = requests.get(
        search_url,
        timeout=timeout,
        headers={"User-Agent": "rag-system/0.1 (+https://github.com/)"}
    )
    response.raise_for_status()

    matches = re.findall(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"', response.text)
    links: List[str] = []
    for href in matches:
        if href.startswith("http"):
            links.append(href)
        if len(links) >= max_results:
            break
    return links


def web_search_to_documents(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    documents: List[Dict[str, Any]] = []
    for url in search_web(query, max_results=max_results):
        try:
            doc = url_to_document(url)
            if doc["content"].strip():
                documents.append(doc)
        except Exception:
            continue
    return documents
