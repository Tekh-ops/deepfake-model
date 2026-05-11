"""
factcheck/ddg_checker.py
─────────────────────────
DuckDuckGo search for claim verification.
No API key required. No rate limits for moderate use.
"""

from config import DDG_MAX_RESULTS


def _ddgs_available() -> bool:
    try:
        import duckduckgo_search  # noqa: F401
        return True
    except ImportError:
        return False


def search_web(query: str, max_results: int = DDG_MAX_RESULTS) -> list[dict]:
    """
    General DuckDuckGo web search.
    Returns list of {title, url, snippet} dicts or [{error: ...}] on failure.
    """
    if not _ddgs_available():
        return [{"error": "duckduckgo-search not installed. Run: pip install duckduckgo-search"}]

    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results,
                                     safesearch="moderate"))
        return [
            {
                "title":   r.get("title", ""),
                "url":     r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in results
        ]
    except Exception as e:
        return [{"error": str(e)}]


def search_news(query: str, max_results: int = DDG_MAX_RESULTS) -> list[dict]:
    """
    DuckDuckGo news-specific search.
    Returns list of {title, url, source, date, body} dicts.
    """
    if not _ddgs_available():
        return [{"error": "duckduckgo-search not installed"}]

    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.news(query, max_results=max_results))
        return [
            {
                "title":  r.get("title", ""),
                "url":    r.get("url", ""),
                "source": r.get("source", ""),
                "date":   r.get("date", ""),
                "body":   r.get("body", ""),
            }
            for r in results
        ]
    except Exception as e:
        return [{"error": str(e)}]
