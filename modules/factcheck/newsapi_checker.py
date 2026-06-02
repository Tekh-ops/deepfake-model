"""
factcheck/newsapi_checker.py
─────────────────────────────
NewsAPI.org search and basic claim verification.
Requires NEWSAPI_KEY in .env — get a free key at https://newsapi.org/register
Degrades gracefully if key is missing.
"""

import os
import requests
from datetime import datetime, timedelta

from config import NEWSAPI_KEY, NEWS_DAYS_BACK, NEWS_MAX_ARTICLES


class NewsAPIChecker:
    BASE_URL = "https://newsapi.org/v2/everything"

    def __init__(self):
        self.api_key = NEWSAPI_KEY

    def _available(self) -> bool:
        return bool(self.api_key and self.api_key != "your_newsapi_key_here")

    def search(self, query: str, days_back: int = NEWS_DAYS_BACK,
               max_articles: int = NEWS_MAX_ARTICLES) -> list[dict]:
        """Search NewsAPI for articles matching a query."""
        if not self._available():
            return [{"error": "NEWSAPI_KEY not configured in .env"}]

        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        params = {
            "q":        query[:200],
            "from":     from_date,
            "sortBy":   "relevancy",
            "pageSize": max_articles,
            "language": "en",
            "apiKey":   self.api_key,
        }
        try:
            r = requests.get(self.BASE_URL, params=params, timeout=10)
            r.raise_for_status()
            articles = r.json().get("articles", [])
            return [
                {
                    "title":       a.get("title", ""),
                    "source":      a.get("source", {}).get("name", ""),
                    "url":         a.get("url", ""),
                    "published":   a.get("publishedAt", ""),
                    "description": a.get("description", ""),
                }
                for a in articles
            ]
        except Exception as e:
            return [{"error": str(e)}]

    def check_claim(self, claim_text: str) -> dict:
        """
        Search for a claim and return a verdict dict.
        Returns UNVERIFIABLE if no articles found or key missing.
        """
        articles = self.search(claim_text)

        if not articles:
            return {"verdict": "UNVERIFIABLE", "articles": [],
                    "reason": "No relevant news found"}

        if "error" in articles[0]:
            return {"verdict": "UNVERIFIABLE", "articles": [],
                    "reason": articles[0]["error"]}

        return {
            "verdict":       "REFERENCED_IN_NEWS",
            "articles":      articles[:3],
            "article_count": len(articles),
        }


# Module-level singleton
_checker: NewsAPIChecker | None = None

def get_newsapi_checker() -> NewsAPIChecker:
    global _checker
    if _checker is None:
        _checker = NewsAPIChecker()
    return _checker
