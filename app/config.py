from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# Root paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DOCS_TXT = DATA_DIR / "docs_txt"
CHUNKS_TEXT = DATA_DIR / "chunks_text.json"
CHUNKS_META = DATA_DIR / "chunks_meta.json"
FAISS_INDEX = DATA_DIR / "faiss.index"

# OpenAI API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-ada-002")
OPENAI_EMBED_API_URL = "https://api.openai.com/v1/embeddings"

# Retrieval settings
CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "220"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "40"))
TOP_K = int(os.getenv("TOP_K", "5"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))

# Rate limiting and chat memory
MIN_SECONDS_BETWEEN_CALLS = float(os.getenv("MIN_SECONDS_BETWEEN_CALLS", "0.25"))
CHAT_HISTORY_TURNS = int(os.getenv("CHAT_HISTORY_TURNS", "20"))

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)
DOCS_TXT.mkdir(parents=True, exist_ok=True)