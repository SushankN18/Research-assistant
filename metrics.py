"""
Metrics tracking for the research assistant.

Logs per-query metrics (latency, parse rates, source counts, tool usage)
to a JSONL file and provides aggregate statistics.
"""

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)

METRICS_FILE = Path(__file__).parent / "metrics_log.jsonl"


@dataclass
class QueryMetrics:
    """Metrics for a single query execution."""
    query: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    total_seconds: float = 0.0
    sources_found: int = 0
    sources_used: int = 0
    tools_used: list[str] = field(default_factory=list)
    parse_success: bool = False
    retry_count: int = 0
    error: str = ""

    def finalize(self) -> None:
        """Calculate total_seconds from start/end time."""
        if self.end_time > 0 and self.start_time > 0:
            self.total_seconds = round(self.end_time - self.start_time, 3)


class MetricsTracker:
    """
    Tracks and persists query metrics.

    Usage:
        tracker = MetricsTracker()
        with tracker.track_query("my research question") as metrics:
            # ... do work ...
            metrics.sources_found = 15
            metrics.parse_success = True
        tracker.print_summary()
    """

    def __init__(self, log_file: Path = METRICS_FILE):
        self.log_file = log_file
        self.queries: list[QueryMetrics] = []

    @contextmanager
    def track_query(self, query: str) -> Generator[QueryMetrics, None, None]:
        """Context manager that times a query and logs metrics."""
        metrics = QueryMetrics(query=query, start_time=time.time())
        try:
            yield metrics
        except Exception as e:
            metrics.error = str(e)
            raise
        finally:
            metrics.end_time = time.time()
            metrics.finalize()
            self.queries.append(metrics)
            self._write_to_log(metrics)

    def _write_to_log(self, metrics: QueryMetrics) -> None:
        """Append metrics to JSONL log file."""
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                record = asdict(metrics)
                f.write(json.dumps(record) + "\n")
        except IOError as e:
            logger.error("Failed to write metrics log: %s", e)

    def get_summary(self) -> dict:
        """Return aggregate statistics across all tracked queries."""
        if not self.queries:
            return {"total_queries": 0}

        successful = [q for q in self.queries if q.parse_success]
        total = len(self.queries)

        avg_time = sum(q.total_seconds for q in self.queries) / total
        avg_sources = sum(q.sources_found for q in self.queries) / total
        parse_rate = len(successful) / total * 100 if total > 0 else 0.0
        total_retries = sum(q.retry_count for q in self.queries)

        # Tool usage distribution
        tool_counts: dict[str, int] = {}
        for q in self.queries:
            for tool in q.tools_used:
                tool_counts[tool] = tool_counts.get(tool, 0) + 1

        return {
            "total_queries": total,
            "successful_queries": len(successful),
            "parse_success_rate_pct": round(parse_rate, 1),
            "avg_query_time_seconds": round(avg_time, 2),
            "avg_sources_per_query": round(avg_sources, 1),
            "total_retries": total_retries,
            "tool_usage": tool_counts,
        }

    def print_summary(self) -> None:
        """Print a formatted summary table."""
        summary = self.get_summary()
        if summary["total_queries"] == 0:
            print("No queries tracked yet.")
            return

        print("\n" + "=" * 55)
        print("  RESEARCH ASSISTANT -- METRICS SUMMARY")
        print("=" * 55)
        print(f"  Total Queries:          {summary['total_queries']}")
        print(f"  Successful Parses:      {summary['successful_queries']}")
        print(f"  Parse Success Rate:     {summary['parse_success_rate_pct']}%")
        print(f"  Avg Query Time:         {summary['avg_query_time_seconds']}s")
        print(f"  Avg Sources/Query:      {summary['avg_sources_per_query']}")
        print(f"  Total Retries:          {summary['total_retries']}")
        print("-" * 55)
        print("  Tool Usage:")
        for tool, count in summary.get("tool_usage", {}).items():
            print(f"    {tool:20s}  {count} calls")
        print("=" * 55 + "\n")
