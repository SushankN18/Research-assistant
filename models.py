"""
Pydantic v2 models for structured, validated research outputs.

5 nested models with 22+ validated fields covering citations,
findings with confidence scores, query metadata, and full research summaries.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class SourceType(str, Enum):
    """Type of source where information was found."""
    PAPER = "paper"
    ARTICLE = "article"
    WIKI = "wiki"
    WEB = "web"


class SearchResult(BaseModel):
    """Raw search result from a single tool invocation."""
    title: str = Field(..., description="Title of the search result")
    url: str = Field(default="", description="URL of the source")
    snippet: str = Field(..., description="Text snippet or summary from the source")
    source: str = Field(..., description="Tool that produced this result (e.g., duckduckgo, wikipedia, arxiv)")

    class Config:
        frozen = True


class Citation(BaseModel):
    """Structured citation for a source used in research synthesis."""
    author: str = Field(default="Unknown", description="Author or publisher name")
    title: str = Field(..., description="Title of the cited work")
    url: str = Field(default="", description="URL to the source")
    year: Optional[int] = Field(default=None, description="Publication year")
    source_type: SourceType = Field(default=SourceType.WEB, description="Category of source")

    @field_validator("year")
    @classmethod
    def validate_year(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and (v < 1900 or v > datetime.now().year + 1):
            raise ValueError(f"Year {v} is not in valid range [1900, {datetime.now().year + 1}]")
        return v


class Finding(BaseModel):
    """A single research finding with evidence and confidence score."""
    claim: str = Field(..., description="The key finding or claim")
    evidence: str = Field(..., description="Supporting evidence or context")
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Confidence score from 0.0 (speculative) to 1.0 (well-supported)"
    )
    citations: list[Citation] = Field(default_factory=list, description="Sources backing this finding")


class QueryMetadata(BaseModel):
    """Metrics captured during query processing."""
    query_time_seconds: float = Field(..., ge=0.0, description="Total wall-clock time for the query")
    sources_found: int = Field(..., ge=0, description="Total search results retrieved across all tools")
    sources_used: int = Field(..., ge=0, description="Sources that made it into the final synthesis")
    tools_used: list[str] = Field(default_factory=list, description="Names of tools invoked")
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="ISO timestamp of query execution"
    )
    parse_success: bool = Field(default=True, description="Whether Pydantic validation succeeded on first attempt")
    retry_count: int = Field(default=0, ge=0, description="Number of synthesis retries before validation passed")


class ResearchSummary(BaseModel):
    """Complete research output with validated structure."""
    topic: str = Field(..., description="High-level research topic")
    query: str = Field(..., description="Original user query")
    summary: str = Field(..., min_length=50, description="Synthesized research summary (min 50 chars)")
    findings: list[Finding] = Field(
        ..., min_length=1,
        description="Key findings extracted from sources"
    )
    sources: list[Citation] = Field(
        default_factory=list,
        description="All citations used in the summary"
    )
    tools_used: list[str] = Field(default_factory=list, description="Tools invoked during research")
    metadata: QueryMetadata = Field(..., description="Query performance and provenance metadata")
