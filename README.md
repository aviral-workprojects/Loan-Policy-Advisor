# 🏦 Loan Policy Advisor

> A production-grade AI-powered loan eligibility and comparison system.  
> **RAG + Deterministic Rule Engine + Multi-source Knowledge + Semantic Cache**

---

## Architecture

```
User Query
  │
  ├─► [0] Semantic Cache          → Return instantly if near-identical query seen before
  │
  ├─► [1] Query Understanding     → Claude Haiku extracts intent + entities (JSON)
  │           └─ intent: eligibility | comparison | general
  │           └─ banks: [Axis, ICICI, …]
  │           └─ entities: {age, monthly_income, credit_score, …}
  │
  ├─► [2] Router (MoE-style)      → Determine which data sources to query
  │           └─ bank mentioned   → bank-specific + aggregator
  │           └─ comparison       → all banks + aggregators
  │           └─ regulatory       → RBI only
  │           └─ general          → hybrid (all sources)
  │
  ├─► [3] Retrieval (FAISS RAG)   → Semantic search across embedded documents
  │           └─ sentence-transformers/all-MiniLM-L6-v2
  │           └─ FAISS IndexFlatIP (cosine similarity)
  │           └─ Filtered by bank/doc_type
  │
  ├─► [4] Fallback Search         → Tavily/SERP API if retrieved docs < threshold
  │
  ├─► [5] Rule Engine (DSL)       → DETERMINISTIC eligibility evaluation
  │           └─ YAML rules per bank
  │           └─ Zero LLM involvement
  │           └─ Returns: eligible, failed conditions, missing fields
  │
  ├─► [6] Context Fusion          → Merge + deduplicate + prioritize sources
  │           └─ Priority: Bank rules > RBI > Aggregators
  │
  ├─► [7] LLM Explanation         → Claude Sonnet generates final response
  │           └─ Uses ONLY retrieved context (no hallucination)
  │           └─ Returns structured JSON with confidence score
  │
  └─► [8] Cache Store + Response
```

---

## Quick Start

### 1. Clone & install

```bash
git clone <repo>
cd loan_advisor
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env with your Anthropic API key
```

### 3. Bootstrap the knowledge index

```bash
python bootstrap.py
```

This scans all files in `data/`, chunks and embeds them, and saves a FAISS index to `models/`.

### 4. Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Try it immediately (no API key needed)

```bash
# Rule engine demo — 100% deterministic, no LLM required
python cli.py --rule-only
```

---

## API Usage

### POST /query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Am I eligible for Axis Bank loan? Age 28, salary 40k, CIBIL 750, salaried 2 years"}'
```

Response:
```json
{
  "decision": "Eligible",
  "summary": "You meet all Axis Bank personal loan eligibility criteria.",
  "detailed_explanation": "Based on your profile (age 28, income ₹40,000/month, CIBIL 750)...",
  "recommendations": ["Consider Axis for best rate given your profile", "..."],
  "banks_compared": ["Axis"],
  "rule_results": [
    {
      "bank": "Axis",
      "eligible": true,
      "summary": "✅ Eligible for Axis personal loan",
      "passed": ["Minimum age is 21 years", "Minimum monthly income is ₹15,000", ...],
      "failed": [],
      "missing": []
    }
  ],
  "confidence": 0.92,
  "sources_cited": ["Axis eligibility rules", "Paisabazaar comparison data"],
  "latency_ms": 1240,
  "from_cache": false
}
```

### GET /banks

Returns all banks with available rule files.

### POST /rebuild-index

Re-scan `data/` and rebuild FAISS index after adding new documents.

### DELETE /cache

Clear the semantic response cache.

---

## Project Structure

```
loan_advisor/
│
├── config.py               ← All settings (models, paths, thresholds)
├── pipeline.py             ← Main orchestrator
├── bootstrap.py            ← One-time index builder
├── cli.py                  ← Command-line demo tool
│
├── api/
│   └── main.py             ← FastAPI app with all endpoints
│
├── services/
│   ├── rule_engine.py      ← Deterministic DSL evaluator (YAML rules)
│   ├── retrieval.py        ← FAISS vector store + semantic search
│   ├── llm.py              ← Query parsing + explanation generation
│   ├── router.py           ← MoE-style query routing
│   ├── fusion.py           ← Context merge + deduplication
│   ├── search.py           ← Fallback web search (Tavily/SERP)
│   └── cache.py            ← Semantic response cache
│
├── rules/                  ← Bank eligibility rules (YAML)
│   ├── axis.yaml
│   ├── icici.yaml
│   ├── hdfc.yaml
│   └── sbi.yaml
│
├── data/                   ← Knowledge base documents
│   ├── axis/               ← Text/HTML scraped from Axis Bank
│   ├── icici/              ← Text/HTML scraped from ICICI
│   ├── hdfc_pdfs/          ← HDFC PDF documents
│   ├── sbi_pdfs/           ← SBI PDF documents
│   ├── paisabazaar/        ← Aggregator comparison data
│   ├── bankbazaar/         ← Aggregator insights
│   └── rbi/                ← RBI regulatory PDFs/text
│
├── models/                 ← Auto-generated FAISS index (after bootstrap)
│   ├── index.faiss
│   └── chunks.pkl
│
├── cache/                  ← Semantic cache storage (auto-generated)
│
└── tests/
    └── test_pipeline.py    ← Full test suite
```

---

## Adding New Data

1. Drop `.txt` or `.pdf` files into the appropriate `data/<source>/` folder
2. Run `python bootstrap.py` (or `POST /rebuild-index`)
3. No code changes needed

---

## Adding a New Bank

1. Create `rules/<bankname>.yaml` with the eligibility rules
2. Add knowledge files to `data/<bankname>/`
3. Run `python bootstrap.py`

The bank is automatically detected in routing and rule evaluation.

---

## Adding New Rules to Existing Bank

Edit the YAML file:

```yaml
bank: Axis
loan_type: personal
logic: all   # "all" = AND, "any" = OR

rules:
  - id: my_new_rule
    field: loan_amount        # must match profile dict key
    operator: "<="            # >=, <=, >, <, ==, !=, in, not_in
    value: 4000000            # ₹40 lakh max
    message: "Loan amount must not exceed ₹40 lakh"
```

Supported operators: `>=`, `<=`, `>`, `<`, `==`, `!=`, `in`, `not_in`

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` or `openai` | `anthropic` |
| `ANTHROPIC_API_KEY` | Anthropic API key | — |
| `OPENAI_API_KEY` | OpenAI API key | — |
| `FALLBACK_PROVIDER` | `tavily` or `serp` | `tavily` |
| `TAVILY_API_KEY` | Tavily search API key | — |
| `SERP_API_KEY` | SERP API key | — |

---

## Design Decisions

| Decision | Rationale |
|---|---|
| **Deterministic rule engine** | Zero hallucination risk for eligibility decisions |
| **Small model for parsing** | Claude Haiku is fast + cheap for JSON extraction |
| **Large model only for explanation** | Claude Sonnet only runs after decisions are made |
| **Semantic cache** | Avoids redundant LLM calls for similar queries |
| **YAML rules** | Non-technical stakeholders can update rules without code changes |
| **Early exit** | Rule engine runs before LLM; failed rule = cheaper response |
| **Source priority** | Bank > RBI > Aggregator prevents aggregator data overriding official rules |

---

## Tests

```bash
# Rule engine tests (no API key needed)
python cli.py --rule-only

# Full test suite (requires pytest)
python -m pytest tests/ -v

# Manual integration test
python cli.py "Compare HDFC and SBI for 30k salary, 720 CIBIL, age 28"
```
