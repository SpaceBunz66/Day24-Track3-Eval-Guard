"""Shared configuration for Lab 24: Eval + Guardrail Stack."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENROUTER_API_KEY = (
    os.getenv("OPENROUTER_API_KEY", "")
    or os.getenv("OPENROUTE_API_KEY", "")
)
HF_TOKEN = os.getenv("HF_TOKEN", "")  # Optional: for HuggingFace models

# --- Chat model provider ---
LLM_PROVIDER = os.getenv(
    "LLM_PROVIDER",
    "openrouter",
).lower()
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")
DEFAULT_CHAT_MODEL = os.getenv(
    "DEFAULT_CHAT_MODEL",
    OPENROUTER_MODEL if LLM_PROVIDER == "openrouter" else "gpt-4o-mini",
)

# --- Qdrant (same as Day 18) ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab24_production"

# --- Embedding (same as Day 18) ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# --- Chunking (same as Day 18) ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search (same as Day 18) ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set_50q.json")
ANSWERS_PATH = os.path.join(os.path.dirname(__file__), "answers_50q.json")
HUMAN_LABELS_PATH = os.path.join(os.path.dirname(__file__), "human_labels_10q.json")
ADVERSARIAL_SET_PATH = os.path.join(os.path.dirname(__file__), "adversarial_set_20.json")
GUARDRAILS_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "guardrails")

# --- LLM Judge ---
JUDGE_MODEL = os.getenv("JUDGE_MODEL", DEFAULT_CHAT_MODEL)


def active_llm_api_key() -> str:
    """Return the API key for the selected chat provider."""
    if LLM_PROVIDER == "openrouter":
        return OPENROUTER_API_KEY or OPENAI_API_KEY
    return OPENAI_API_KEY


def openai_client_kwargs() -> dict:
    """Return kwargs for the OpenAI SDK, including OpenRouter base_url."""
    api_key = active_llm_api_key()
    kwargs = {"api_key": api_key} if api_key else {}
    if LLM_PROVIDER == "openrouter":
        kwargs["base_url"] = OPENROUTER_BASE_URL
    return kwargs

# --- Guardrail latency budget ---
LATENCY_BUDGET_P95_MS = 500  # target: full guard stack P95 < 500ms
PRESIDIO_LANGUAGE = "en"    # Presidio base language; custom VN recognizers added via PatternRecognizer
