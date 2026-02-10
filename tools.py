"""
Search tools for the research assistant.

4 tools with tenacity retry logic:
1. DuckDuckGo -- general web search
2. Wikipedia -- encyclopedia articles
3. ArXiv -- academic paper search
4. Web Scraper -- extract content from URLs

Each tool returns a list of SearchResult objects for downstream processing.
"""

import logging
import re
from typing import Optional

import arxiv
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from langchain_community.utilities import WikipediaAPIWrapper
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from models import SearchResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. DuckDuckGo Search
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def search_duckduckgo(query: str, max_results: int = 5) -> list[SearchResult]:
    """Search DuckDuckGo and return structured results."""
    try:
        results = []
        with DDGS() as ddgs:
            # text() returns an iterator of dicts {title, href, body}
            ddg_gen = ddgs.text(query, max_results=max_results)
            if ddg_gen:
                for r in ddg_gen:
                    results.append(SearchResult(
                        title=r.get("title", "No Title"),
                        url=r.get("href", ""),
                        snippet=r.get("body", "")[:500],
                        source="duckduckgo",
                    ))
        
        if not results:
             return [SearchResult(
                title="DuckDuckGo Search",
                url="",
                snippet=f"No results found for {query}",
                source="duckduckgo",
            )]
        return results
    except Exception as e:
        logger.warning("DuckDuckGo search failed: %s", e)
        raise


# ---------------------------------------------------------------------------
# 2. Wikipedia Search
# ---------------------------------------------------------------------------

_wiki = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=1000)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def search_wikipedia(query: str, max_results: int = 3) -> list[SearchResult]:
    """Search Wikipedia for relevant articles."""
    try:
        raw = _wiki.run(query)
        if not raw or raw.strip().lower().startswith("no good"):
            return []

        # Split by "Page:" or "Summary:" markers if present
        pages = re.split(r"(?=Page:\s)", raw)
        results = []
        for page_text in pages[:max_results]:
            page_text = page_text.strip()
            if not page_text:
                continue
            # Extract title if present
            title_match = re.match(r"Page:\s*(.+?)(?:\n|Summary:)", page_text)
            title = title_match.group(1).strip() if title_match else "Wikipedia Article"
            # Extract summary
            summary_match = re.search(r"Summary:\s*(.+)", page_text, re.DOTALL)
            snippet = summary_match.group(1).strip()[:500] if summary_match else page_text[:500]

            results.append(SearchResult(
                title=title,
                url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                snippet=snippet,
                source="wikipedia",
            ))

        if not results:
            results.append(SearchResult(
                title="Wikipedia",
                url="",
                snippet=raw[:500],
                source="wikipedia",
            ))
        return results
    except Exception as e:
        logger.warning("Wikipedia search failed: %s", e)
        raise


# ---------------------------------------------------------------------------
# 3. ArXiv Search
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def search_arxiv(query: str, max_results: int = 5) -> list[SearchResult]:
    """Search ArXiv for academic papers."""
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        results = []
        for paper in client.results(search):
            results.append(SearchResult(
                title=paper.title,
                url=paper.entry_id,
                snippet=paper.summary[:500],
                source="arxiv",
            ))
        return results
    except Exception as e:
        logger.warning("ArXiv search failed: %s", e)
        raise


# ---------------------------------------------------------------------------
# 4. Web Scraper
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=retry_if_exception_type((requests.RequestException,)),
    reraise=True,
)
def scrape_url(url: str) -> Optional[SearchResult]:
    """Scrape a URL and extract text content."""
    try:
        headers = {"User-Agent": "ResearchAssistant/1.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove script and style elements
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else url
        text = soup.get_text(separator=" ", strip=True)[:1000]

        return SearchResult(
            title=title,
            url=url,
            snippet=text,
            source="web_scraper",
        )
    except Exception as e:
        logger.warning("Web scrape failed for %s: %s", url, e)
        return None


# ---------------------------------------------------------------------------
# Aggregate search: run all tools, collect results
# ---------------------------------------------------------------------------

def run_all_searches(query: str) -> list[SearchResult]:
    """
    Run all search tools and aggregate results.
    Each tool is called with error isolation -- a single tool failure
    does not block the others.
    """
    all_results: list[SearchResult] = []
    tool_errors: list[str] = []

    # DuckDuckGo
    try:
        ddg_results = search_duckduckgo(query)
        all_results.extend(ddg_results)
        logger.info("DuckDuckGo returned %d results", len(ddg_results))
    except Exception as e:
        tool_errors.append(f"duckduckgo: {e}")
        logger.error("DuckDuckGo failed after retries: %s", e)

    # Wikipedia
    try:
        wiki_results = search_wikipedia(query)
        all_results.extend(wiki_results)
        logger.info("Wikipedia returned %d results", len(wiki_results))
    except Exception as e:
        tool_errors.append(f"wikipedia: {e}")
        logger.error("Wikipedia failed after retries: %s", e)

    # ArXiv
    try:
        arxiv_results = search_arxiv(query)
        all_results.extend(arxiv_results)
        logger.info("ArXiv returned %d results", len(arxiv_results))
    except Exception as e:
        tool_errors.append(f"arxiv: {e}")
        logger.error("ArXiv failed after retries: %s", e)

    if tool_errors:
        logger.warning("Tool errors: %s", "; ".join(tool_errors))

    logger.info("Total results aggregated: %d", len(all_results))
    return all_results