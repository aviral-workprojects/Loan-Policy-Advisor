"""
Central configuration for Loan Policy Advisor.
All environment-based settings are read here.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root (same folder as this file)
load_dotenv(Path(__file__).parent / ".env")

# --- Paths ---
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "cache"
RULES_DIR = BASE_DIR / "rules"
FAISS_INDEX_DIR = BASE_DIR / "models"

# --- LLM ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")

# Provider: "anthropic" | "openai" | "groq"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

# Model names per provider
# Groq models: llama-3.3-70b-versatile, llama-3.1-8b-instant, mixtral-8x7b-32768, gemma2-9b-it
_GROQ_PARSE_MODEL   = os.getenv("GROQ_PARSE_MODEL",   "llama-3.1-8b-instant")     # fast + cheap
_GROQ_EXPLAIN_MODEL = os.getenv("GROQ_EXPLAIN_MODEL", "llama-3.3-70b-versatile")  # powerful

_ANTHROPIC_PARSE_MODEL   = "claude-haiku-4-5-20251001"
_ANTHROPIC_EXPLAIN_MODEL = "claude-sonnet-4-6"

_OPENAI_PARSE_MODEL   = "gpt-4o-mini"
_OPENAI_EXPLAIN_MODEL = "gpt-4o"

# Auto-select models based on provider
if LLM_PROVIDER == "groq":
    PARSE_MODEL   = _GROQ_PARSE_MODEL
    EXPLAIN_MODEL = _GROQ_EXPLAIN_MODEL
elif LLM_PROVIDER == "openai":
    PARSE_MODEL   = _OPENAI_PARSE_MODEL
    EXPLAIN_MODEL = _OPENAI_EXPLAIN_MODEL
else:  # anthropic
    PARSE_MODEL   = _ANTHROPIC_PARSE_MODEL
    EXPLAIN_MODEL = _ANTHROPIC_EXPLAIN_MODEL

# --- Embeddings ---
EMBEDDING_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM     = 384
CHUNK_SIZE        = 512
CHUNK_OVERLAP     = 64

# --- Retrieval ---
TOP_K             = 5
MIN_DOCS_THRESHOLD = 2   # below this → trigger fallback search

# --- Reranker ---
# Set USE_RERANKER=false in env to disable without touching code
USE_RERANKER      = os.getenv("USE_RERANKER", "true").lower() == "true"

# Model served by NVIDIA API Catalog or local NIM
RERANKER_MODEL    = os.getenv("RERANKER_MODEL", "nvidia/llama-nemotron-rerank-1b-v2")

# Backend: "nvidia_api" → NVIDIA API Catalog (needs NVIDIA_API_KEY)
#          "nim"        → local NIM container  (no key needed)
RERANKER_BACKEND  = os.getenv("RERANKER_BACKEND", "nvidia_api")

# NVIDIA API Catalog key (get free at build.nvidia.com)
NVIDIA_API_KEY    = os.getenv("NVIDIA_API_KEY", "")

# Local NIM base URL (used when RERANKER_BACKEND="nim")
NIM_BASE_URL      = os.getenv("NIM_BASE_URL", "http://localhost:8001")

# After FAISS retrieval, fetch this many candidates for the reranker to score
RERANKER_FETCH_K  = int(os.getenv("RERANKER_FETCH_K", "10"))

# Final number of docs returned after reranking
RERANKER_TOP_N    = int(os.getenv("RERANKER_TOP_N", "3"))

# Seconds before the reranker HTTP call times out
RERANKER_TIMEOUT  = int(os.getenv("RERANKER_TIMEOUT", "10"))

# Queries shorter than this word count skip reranking (too simple to need it)
RERANKER_MIN_QUERY_WORDS = int(os.getenv("RERANKER_MIN_QUERY_WORDS", "4"))

# --- Search Fallback ---
TAVILY_API_KEY    = os.getenv("TAVILY_API_KEY", "")
SERP_API_KEY      = os.getenv("SERP_API_KEY", "")
FALLBACK_PROVIDER = os.getenv("FALLBACK_PROVIDER", "tavily")  # "tavily" | "serp"

# --- Semantic Cache ---
CACHE_SIMILARITY_THRESHOLD = 0.92   # cosine sim above this → cache hit
CACHE_MAX_ENTRIES = 500

# --- API ---
API_HOST = "0.0.0.0"
API_PORT = 8000

# --- Sources (for scraping bootstrap) ---
BANK_URLS = {
    "Axis":  "https://www.axisbank.com/retail/loans/personal-loan/personal-loan-eligibility",
    "ICICI": "https://www.icicibank.com/personal-banking/loans/personal-loan/eligibility",
}
AGGREGATOR_URLS = {
    "Paisabazaar": "https://www.paisabazaar.com/personal-loan/",
    "BankBazaar":  "https://www.bankbazaar.com/personal-loan.html",
}

# --- NVIDIA NemoRetriever Embeddings ---
# Set NVIDIA_EMBED_BACKEND="nvidia_api" to use NVIDIA API instead of sentence-transformers
NVIDIA_EMBED_BACKEND = os.getenv("NVIDIA_EMBED_BACKEND", "local")   # "local" | "nvidia_api"
NVIDIA_EMBED_MODEL   = os.getenv("NVIDIA_EMBED_MODEL",
                                  "nvidia/llama-3_2-nemoretriever-300m-embed-v2")
# Dimension for NemoRetriever 300M embed model
NVIDIA_EMBED_DIM     = 1024

# --- MoE Routing ---
# Word count threshold below which we skip RAG (pure rule engine path).
# Lowered from 6 → 3: queries like "ROI Axis personal loan" (4 words) or
# "minimum income requirement ICICI" (4 words) must reach the retrieval layer
# or they silently fail to return rate / policy information.
# Only truly trivial 1–3 word inputs (e.g. "help", "hello") skip RAG.
MOE_SIMPLE_QUERY_WORDS = int(os.getenv("MOE_SIMPLE_QUERY_WORDS", "3"))

# --- PDF Pipeline (NVIDIA Page Elements) ---
# Processed JSON output from pdf_pipeline.py is stored here
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"

# NVIDIA Page Elements API endpoint
# Model: nemoretriever-page-elements-v3
# Docs: https://build.nvidia.com/nvidia/nemoretriever-page-elements-v3
NVIDIA_PAGE_ELEMENTS_URL = "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-page-elements-v3"

# Enable structured PDF processing via NVIDIA API
# If False → falls back to plain text extraction (existing behaviour)
USE_NVIDIA_PDF = os.getenv("USE_NVIDIA_PDF", "false").lower() == "true"

# Max retries for NVIDIA API calls in the PDF pipeline
PDF_API_MAX_RETRIES = int(os.getenv("PDF_API_MAX_RETRIES", "3"))
PDF_API_RETRY_DELAY = float(os.getenv("PDF_API_RETRY_DELAY", "2.0"))  # seconds

# Chunk size limits for structured content (words)
PDF_SECTION_CHUNK_MAX  = int(os.getenv("PDF_SECTION_CHUNK_MAX", "400"))
PDF_TABLE_CHUNK_MAX    = int(os.getenv("PDF_TABLE_CHUNK_MAX",   "300"))