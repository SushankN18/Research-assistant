"""
AI Research Assistant — Main Entry Point

Interactive CLI that runs research queries through the LangGraph
4-step workflow (search → filter → synthesize → validate)
with full metrics tracking.
"""

import json
import logging
import sys

from dotenv import load_dotenv

from graph import run_research
from metrics import MetricsTracker
from models import ResearchSummary

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def format_output(result: dict) -> str:
    """Pretty-format the research output for terminal display."""
    output = result.get("validated_output")
    if output is None:
        error = result.get("error", "Unknown error")
        return f"\nResearch failed: {error}\n\nRaw synthesis:\n{result.get('synthesis_raw', 'N/A')[:500]}"

    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("  RESEARCH SUMMARY")
    lines.append("=" * 60)
    lines.append(f"\nTopic: {output.get('topic', 'N/A')}")
    lines.append(f"Query: {output.get('query', 'N/A')}")
    lines.append(f"\nSummary:\n{output.get('summary', 'N/A')}")

    # Findings
    findings = output.get("findings", [])
    if findings:
        lines.append(f"\nKey Findings ({len(findings)}):")
        for i, f in enumerate(findings, 1):
            conf = f.get("confidence", 0)
            conf_bar = "█" * int(conf * 10) + "░" * (10 - int(conf * 10))
            lines.append(f"\n  {i}. {f.get('claim', 'N/A')}")
            lines.append(f"     Evidence: {f.get('evidence', 'N/A')[:200]}")
            lines.append(f"     Confidence: [{conf_bar}] {conf:.0%}")
            cites = f.get("citations", [])
            if cites:
                for c in cites[:3]:
                    lines.append(f"     * {c.get('title', 'N/A')} ({c.get('source_type', 'web')})")

    # Sources
    sources = output.get("sources", [])
    if sources:
        lines.append(f"\nSources ({len(sources)}):")
        for s in sources:
            url = s.get("url", "")
            url_str = f" -- {url}" if url else ""
            lines.append(f"  * {s.get('title', 'N/A')}{url_str}")

    # Metadata
    meta = output.get("metadata", {})
    if meta:
        lines.append(f"\nPerformance:")
        lines.append(f"  Query time:    {meta.get('query_time_seconds', 0):.1f}s")
        lines.append(f"  Sources found: {meta.get('sources_found', 0)}")
        lines.append(f"  Sources used:  {meta.get('sources_used', 0)}")
        lines.append(f"  Tools used:    {', '.join(meta.get('tools_used', []))}")
        lines.append(f"  Parse success: {'Yes' if meta.get('parse_success') else 'No'}")
        retries = meta.get("retry_count", 0)
        if retries > 0:
            lines.append(f"  Retries:       {retries}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def main():
    """Run the research assistant interactively."""
    print("\n" + "=" * 60)
    print("  AI Research Assistant")
    print("  Powered by Claude 3.5 Sonnet + LangGraph")
    print("  4-step workflow: search -> filter -> synthesize -> validate")
    print("=" * 60)

    tracker = MetricsTracker()

    while True:
        try:
            query = input("\nEnter your research query (or 'quit' to exit): ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query or query.lower() in ("quit", "exit", "q"):
            break

        print(f"\nResearching: {query}")
        print("   Running search -> filter -> synthesize -> validate...\n")

        with tracker.track_query(query) as metrics:
            result = run_research(query)

            # Update metrics from result
            metrics.sources_found = result.get("sources_found", 0)
            metrics.sources_used = result.get("sources_used", 0)
            metrics.tools_used = result.get("tools_used", [])
            metrics.retry_count = result.get("retry_count", 0)
            metrics.parse_success = result.get("validated_output") is not None

        # Display output
        print(format_output(result))

        # Save validated output to JSON
        if result.get("validated_output"):
            out_file = f"research_output.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result["validated_output"], f, indent=2, ensure_ascii=False)
            print(f"\nOutput saved to {out_file}")

    # Print session metrics
    tracker.print_summary()
    print("Goodbye!")


if __name__ == "__main__":
    main()