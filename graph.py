"""
LangGraph 4-step research workflow.

Pipeline: search → filter → synthesize → validate
- search:      Queries all 4 tools, collects raw SearchResults
- filter:      LLM deduplicates and ranks results by relevance
- synthesize:  LLM generates a structured ResearchSummary from filtered results
- validate:    Pydantic validation with conditional retry (max 2 retries)
"""

import json
import logging
import time
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from models import (
    Citation,
    Finding,
    QueryMetadata,
    ResearchSummary,
    SearchResult,
    SourceType,
)
from tools import run_all_searches

logger = logging.getLogger(__name__)

# ---------- LLM -------------------------------------------------------
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.2, max_tokens=4096)

MAX_VALIDATION_RETRIES = 2


# ---------- State definition -------------------------------------------

class ResearchState(TypedDict):
    """Typed state passed through the graph."""
    query: str
    search_results: list[dict]
    filtered_results: list[dict]
    synthesis_raw: str
    validated_output: dict | None
    tools_used: list[str]
    sources_found: int
    sources_used: int
    retry_count: int
    start_time: float
    error: str


# ---------- Node functions ---------------------------------------------

def search_node(state: ResearchState) -> dict:
    """Run all search tools and collect raw results."""
    query = state["query"]
    logger.info("[search] Searching for: %s", query)

    results: list[SearchResult] = run_all_searches(query)

    tools_used = list({r.source for r in results})
    results_dicts = [r.model_dump() for r in results]

    logger.info("[search] Found %d results from %s", len(results), tools_used)

    return {
        "search_results": results_dicts,
        "tools_used": tools_used,
        "sources_found": len(results),
    }


def filter_node(state: ResearchState) -> dict:
    """LLM-powered deduplication and relevance ranking."""
    query = state["query"]
    results = state["search_results"]

    if not results:
        logger.warning("[filter] No search results to filter")
        return {"filtered_results": [], "sources_used": 0}

    results_text = "\n\n".join(
        f"[{i+1}] Source: {r['source']} | Title: {r['title']}\n{r['snippet']}"
        for i, r in enumerate(results)
    )

    filter_prompt = f"""You are a research assistant. Given the query and search results below,
select the most relevant and non-redundant results. Remove duplicates and low-quality results.
Return a JSON array of indices (1-based) of the results to keep, ordered by relevance.

Query: {query}

Search Results:
{results_text}

Return ONLY a JSON array of indices, e.g. [1, 3, 5, 7]. No other text."""

    response = llm.invoke([
        SystemMessage(content="You are a precise research filter. Return only valid JSON."),
        HumanMessage(content=filter_prompt),
    ])

    try:
        indices = json.loads(response.content)
        filtered = [results[i - 1] for i in indices if 1 <= i <= len(results)]
    except (json.JSONDecodeError, IndexError, TypeError):
        logger.warning("[filter] Could not parse LLM filter response, keeping all results")
        filtered = results

    logger.info("[filter] Kept %d of %d results", len(filtered), len(results))
    return {"filtered_results": filtered, "sources_used": len(filtered)}


def synthesize_node(state: ResearchState) -> dict:
    """LLM generates a structured research summary."""
    query = state["query"]
    filtered = state["filtered_results"]

    if not filtered:
        logger.warning("[synthesize] No filtered results to synthesize")
        return {"synthesis_raw": "", "error": "No search results available"}

    sources_text = "\n\n".join(
        f"[{i+1}] Source: {r['source']} | Title: {r['title']} | URL: {r.get('url', 'N/A')}\n{r['snippet']}"
        for i, r in enumerate(filtered)
    )

    synthesis_prompt = f"""You are a research assistant. Synthesize the following sources into a structured research summary.

Query: {query}

Sources:
{sources_text}

Return ONLY valid JSON (no markdown, no code fences) matching this exact schema:
{{
    "topic": "high-level topic name",
    "query": "{query}",
    "summary": "comprehensive synthesis paragraph (at least 50 characters)",
    "findings": [
        {{
            "claim": "key finding or claim",
            "evidence": "supporting evidence from sources",
            "confidence": 0.85,
            "citations": [
                {{
                    "author": "author name or Unknown",
                    "title": "source title",
                    "url": "source url or empty string",
                    "year": null,
                    "source_type": "paper|article|wiki|web"
                }}
            ]
        }}
    ],
    "sources": [
        {{
            "author": "author name or Unknown",
            "title": "source title",
            "url": "source url or empty string",
            "year": null,
            "source_type": "paper|article|wiki|web"
        }}
    ],
    "tools_used": {json.dumps(state.get("tools_used", []))},
    "metadata": {{
        "query_time_seconds": 0,
        "sources_found": {state.get("sources_found", 0)},
        "sources_used": {state.get("sources_used", 0)},
        "tools_used": {json.dumps(state.get("tools_used", []))},
        "parse_success": true,
        "retry_count": {state.get("retry_count", 0)}
    }}
}}

CRITICAL: Return ONLY the JSON object. No explanation, no markdown fences, no extra text.
Include at least 2-3 findings with citations. Set confidence between 0.0 and 1.0 based on evidence strength.
Use source_type values: "paper" for arxiv, "wiki" for wikipedia, "web" for duckduckgo, "article" for news."""

    response = llm.invoke([
        SystemMessage(content="You output only valid JSON. Never use markdown code fences."),
        HumanMessage(content=synthesis_prompt),
    ])

    raw_text = response.content.strip()

    # Strip markdown code fences if present
    if raw_text.startswith("```"):
        lines = raw_text.split("\n")
        # Remove first line (```json) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw_text = "\n".join(lines).strip()

    logger.info("[synthesize] Generated %d chars of synthesis", len(raw_text))
    return {"synthesis_raw": raw_text}


def validate_node(state: ResearchState) -> dict:
    """Validate synthesis against Pydantic schema."""
    raw = state.get("synthesis_raw", "")
    retry_count = state.get("retry_count", 0)

    if not raw:
        logger.error("[validate] Empty synthesis, cannot validate")
        return {"validated_output": None, "error": "Empty synthesis"}

    try:
        data = json.loads(raw)

        # Inject timing metadata
        elapsed = round(time.time() - state.get("start_time", time.time()), 3)
        if "metadata" in data:
            data["metadata"]["query_time_seconds"] = elapsed
            data["metadata"]["retry_count"] = retry_count
            data["metadata"]["parse_success"] = True

        # Validate with Pydantic
        summary = ResearchSummary.model_validate(data)
        logger.info("[validate] Pydantic validation passed")
        return {"validated_output": summary.model_dump(), "error": ""}

    except (json.JSONDecodeError, Exception) as e:
        logger.warning("[validate] Validation failed (attempt %d): %s", retry_count + 1, e)
        return {
            "validated_output": None,
            "retry_count": retry_count + 1,
            "error": str(e),
        }


# ---------- Conditional edge -------------------------------------------

def should_retry(state: ResearchState) -> str:
    """Decide whether to retry synthesis or finish."""
    if state.get("validated_output") is not None:
        return "end"
    if state.get("retry_count", 0) >= MAX_VALIDATION_RETRIES:
        logger.error("[graph] Max retries reached, returning with error")
        return "end"
    logger.info("[graph] Retrying synthesis (attempt %d)", state.get("retry_count", 0) + 1)
    return "retry"


# ---------- Build the graph --------------------------------------------

def build_research_graph() -> StateGraph:
    """Construct the 4-step research workflow graph."""
    graph = StateGraph(ResearchState)

    # Add nodes
    graph.add_node("search", search_node)
    graph.add_node("filter", filter_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("validate", validate_node)

    # Linear edges: search → filter → synthesize → validate
    graph.add_edge("search", "filter")
    graph.add_edge("filter", "synthesize")
    graph.add_edge("synthesize", "validate")

    # Conditional edge: validate → END or validate → synthesize (retry)
    graph.add_conditional_edges(
        "validate",
        should_retry,
        {
            "end": END,
            "retry": "synthesize",
        },
    )

    # Entry point
    graph.set_entry_point("search")

    return graph.compile()


def run_research(query: str) -> dict:
    """Execute the full research pipeline for a query."""
    graph = build_research_graph()

    initial_state: ResearchState = {
        "query": query,
        "search_results": [],
        "filtered_results": [],
        "synthesis_raw": "",
        "validated_output": None,
        "tools_used": [],
        "sources_found": 0,
        "sources_used": 0,
        "retry_count": 0,
        "start_time": time.time(),
        "error": "",
    }

    result = graph.invoke(initial_state)
    return result
