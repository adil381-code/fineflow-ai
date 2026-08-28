# app/config.py
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent
DATA_DIR      = ROOT / "data"
RAW_DIR       = DATA_DIR / "raw"
DOCS_TXT      = DATA_DIR / "docs_txt"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"
for _d in (DATA_DIR, RAW_DIR, DOCS_TXT, CHROMA_DB_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── OpenAI ───────────────────────────────────────────────────────────────────
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL         = os.getenv("OPENAI_MODEL", "gpt-4o")             # main answers
OPENAI_SMALL_MODEL   = os.getenv("OPENAI_SMALL_MODEL", "gpt-4o-mini")  # query rewrite + summaries
OPENAI_API_URL       = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
OPENAI_EMBED_MODEL   = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_EMBED_API_URL = os.getenv("OPENAI_EMBED_API_URL", "https://api.openai.com/v1/embeddings")

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.55"))
LLM_MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS", "110"))

# ── Retrieval ────────────────────────────────────────────────────────────────
TOP_K           = int(os.getenv("TOP_K", "6"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))

CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "2200"))

# ── Memory ───────────────────────────────────────────────────────────────────
CHAT_HISTORY_TURNS  = int(os.getenv("CHAT_HISTORY_TURNS", "12"))   # user+bot pairs sent to the model
SUMMARY_EVERY_TURNS = int(os.getenv("SUMMARY_EVERY_TURNS", "8"))   # rolling summary cadence
FOLLOWUP_EVERY      = int(os.getenv("FOLLOWUP_EVERY", "2"))        # force a follow-up question at least every N replies

# ── MySQL ────────────────────────────────────────────────────────────────────
MYSQL_HOST     = os.getenv("MYSQL_HOST", "")
MYSQL_USER     = os.getenv("MYSQL_USER", "")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "")
MYSQL_PORT     = int(os.getenv("MYSQL_PORT", "3306"))

# ── API ──────────────────────────────────────────────────────────────────────
ADMIN_TOKEN        = os.getenv("ADMIN_TOKEN", "")   # required for /admin/* when set
CORS_ORIGINS       = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "20"))   # messages per session per minute