# AI Research Assistant

Multi-step research agent powered by **Claude 3.5 Sonnet** and **LangGraph**, automating literature search and research synthesis with validated structured outputs.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph Workflow                        │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌─────────────┐  ┌──────────┐│
│  │  SEARCH  │→ │  FILTER  │→ │ SYNTHESIZE  │→ │ VALIDATE ││
│  │ 4 tools  │  │ LLM rank │  │ LLM summary │  │ Pydantic ││
│  └──────────┘  └──────────┘  └─────────────┘  └────┬─────┘│
│                                    ↑                │      │
│                                    └── retry ───────┘      │
└─────────────────────────────────────────────────────────────┘
```

**Workflow Steps:**
1. **Search** — queries 4 tools (DuckDuckGo, Wikipedia, ArXiv, Web Scraper) with tenacity retry logic
2. **Filter** — LLM deduplicates and ranks results by relevance
3. **Synthesize** — LLM generates structured research summary with citations and confidence scores
4. **Validate** — Pydantic v2 validation with conditional retry (max 2 retries on parse failure)

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Claude 3.5 Sonnet (Anthropic API) |
| Orchestration | LangGraph (state machine) |
| Validation | Pydantic v2 (5 nested models, 22+ fields) |
| Search | DuckDuckGo, Wikipedia, ArXiv, BeautifulSoup |
| Reliability | Tenacity (exponential backoff retries) |
| Metrics | Custom JSONL tracker |

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:
```
ANTHROPIC_API_KEY=your-key-here
```

## Usage

### Interactive Mode
```bash
python main.py
```

### Benchmark (collect metrics for resume)
```bash
# Run all 5 benchmark queries
python benchmark.py

# Run a subset
python benchmark.py 3
```

## Output Schema

```
ResearchSummary
├── topic: str
├── query: str
├── summary: str (min 50 chars)
├── findings: list[Finding]
│   ├── claim: str
│   ├── evidence: str
│   ├── confidence: float (0.0 – 1.0)
│   └── citations: list[Citation]
│       ├── author: str
│       ├── title: str
│       ├── url: str
│       ├── year: int | None
│       └── source_type: paper | article | wiki | web
├── sources: list[Citation]
├── tools_used: list[str]
└── metadata: QueryMetadata
    ├── query_time_seconds: float
    ├── sources_found: int
    ├── sources_used: int
    ├── tools_used: list[str]
    ├── parse_success: bool
    └── retry_count: int
```

## Project Structure

```
├── main.py          # Interactive CLI entry point
├── graph.py         # LangGraph 4-step workflow
├── models.py        # 5 Pydantic v2 models (22+ fields)
├── tools.py         # 4 search tools with retry logic
├── metrics.py       # JSONL metrics tracker
├── benchmark.py     # Benchmark suite for real metrics
└── requirements.txt # Dependencies
```