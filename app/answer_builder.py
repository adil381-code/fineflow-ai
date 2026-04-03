# app/answer_builder.py
"""
Nova — FineFlow Answer Builder (Hybrid RAG + GPT-4o + Persona Engine)
with exact FAQ matching for core questions.
"""

import os
import time
import json
import re
import threading
from typing import List, Dict, Any, Optional

import requests
import numpy as np
import faiss  # <--- IMPORTANT: added missing import
from app.logger import logger

# ----------------------------
# Config (from app.config or env)
# ----------------------------
try:
    from app.config import (
        OPENAI_API_KEY, OPENAI_MODEL, OPENAI_API_URL,
        TOP_K, CONFIDENCE_THRESHOLD, MIN_SECONDS_BETWEEN_CALLS, CHAT_HISTORY_TURNS
    )
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
    OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    TOP_K = int(os.getenv("TOP_K", "5"))
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))
    MIN_SECONDS_BETWEEN_CALLS = float(os.getenv("MIN_SECONDS_BETWEEN_CALLS", "0.25"))
    CHAT_HISTORY_TURNS = int(os.getenv("CHAT_HISTORY_TURNS", "20"))

SUPPORT_EMAIL = os.getenv("SUPPORT_EMAIL", "support@fineflow.com")
SALES_EMAIL = os.getenv("SALES_EMAIL", "sales@fineflow.com")

# ----------------------------
# Retriever import (defensive)
# ----------------------------
try:
    from app.retriever import load_index, search as rag_search_impl, rerank_hits
except Exception as e:
    logger.warning("Could not import retriever (or rerank_hits): %s", e)
    try:
        from app.retriever import load_index, search as rag_search_impl
    except Exception:
        def load_index():
            return None, None, [], []
        def rag_search_impl(q, top_k=TOP_K):
            return []
    def rerank_hits(hits, q):
        return hits

# ----------------------------
# Optional Redis session backing
# ----------------------------
_redis_client = None
try:
    REDIS_URL = os.getenv("REDIS_URL", "")
    if REDIS_URL:
        import redis
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        _redis_client.ping()
        logger.info("Redis session store enabled (REDIS_URL provided).")
    else:
        _redis_client = None
except Exception as e:
    logger.warning("Redis unavailable; falling back to in-memory sessions: %s", e)
    _redis_client = None

# ----------------------------
# Persona & prompts
# ----------------------------
NOVA_UNIVERSAL_PROMPT = (
    "You are Nova, the intelligent, calm, and highly capable assistant behind Fine Flow — "
    "an advanced fleet fine management system.\n"
    "Be human, professional, and helpful. Never break immersion or say you are an AI.\n"
    "**Always give very concise answers.** Use 1‑2 sentences unless the user explicitly asks for more detail.\n"
    "When listing multiple items (like pricing plans), use bullet points.\n"
    "Never add extra commentary or explanations beyond what is asked.\n"
    "Always use the retrieved document excerpts to ground your answers. "
    "Do NOT print internal filenames (e.g. 'faq.txt') to end users.\n"
    "Output plain text only – no markdown formatting like asterisks (**) or backticks."
)

SALES_NOVA_PROMPT = (
    "You are Sales Nova. Tone: confident, benefit-driven, persuasive but honest. "
    "Lead with pain points and ROI, end with a clear call to action (book demo / contact sales). "
    "Be extremely concise – 1‑2 sentences. Use bullet points only if listing."
)

TECH_NOVA_PROMPT = (
    "You are Tech Nova. Tone: calm, expert, non-condescending. Explain system behaviour, integrations, security, "
    "and troubleshooting without revealing proprietary internals. Keep it to 1‑2 sentences."
)

LEGAL_NOVA_PROMPT = (
    "You are Legal Nova. Explain timelines and procedural info about fines/appeals. Always include this disclaimer:\n"
    "\"Please note: I can't provide legal advice. I can only explain how Fine Flow works and what the general process usually involves.\"\n"
    "Do NOT give specific legal advice or predictive outcomes. Keep it short (1‑2 sentences)."
)

def build_system_prompt(mode: str) -> str:
    base = NOVA_UNIVERSAL_PROMPT
    mode = (mode or "universal").lower()
    if mode == "sales":
        return base + "\n" + SALES_NOVA_PROMPT
    if mode == "tech":
        return base + "\n" + TECH_NOVA_PROMPT
    if mode == "legal":
        return base + "\n" + LEGAL_NOVA_PROMPT
    return base

# ----------------------------
# Session / cache / rate-limit
# ----------------------------
_LOCK = threading.Lock()
_SESSION_STORE: Dict[str, List[Dict[str, str]]] = {}
_LAST_CALL_TS: Dict[str, float] = {}

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
LLM_CACHE_PATH = os.path.join(DATA_DIR, "llm_prompt_cache.json")
LLM_COUNTER_PATH = os.path.join(DATA_DIR, "llm_calls.json")
FAQ_PATH = os.path.join(DATA_DIR, "faq.json")

os.makedirs(DATA_DIR, exist_ok=True)

try:
    _LLM_CACHE = json.loads(open(LLM_CACHE_PATH, "r", encoding="utf8").read()) if os.path.exists(LLM_CACHE_PATH) else {}
except Exception:
    _LLM_CACHE = {}

try:
    _LLM_COUNTER = json.loads(open(LLM_COUNTER_PATH, "r", encoding="utf8").read()) if os.path.exists(LLM_COUNTER_PATH) else {"calls": 0}
except Exception:
    _LLM_COUNTER = {"calls": 0}

def _save_llm_cache():
    try:
        with open(LLM_CACHE_PATH, "w", encoding="utf8") as f:
            json.dump(_LLM_CACHE, f, ensure_ascii=False, indent=2)
    except Exception:
        logger.exception("Failed to save LLM cache")

def _increment_llm_calls():
    try:
        _LLM_COUNTER["calls"] = _LLM_COUNTER.get("calls", 0) + 1
        with open(LLM_COUNTER_PATH, "w", encoding="utf8") as f:
            json.dump(_LLM_COUNTER, f, ensure_ascii=False)
    except Exception:
        logger.exception("Failed to increment LLM call counter")

def _get_session(session_id: str) -> List[Dict[str, str]]:
    if _redis_client:
        key = f"nova:session:{session_id}"
        try:
            raw = _redis_client.get(key)
            if not raw:
                return []
            return json.loads(raw)
        except Exception:
            logger.exception("Redis read failed for session; falling back to empty session.")
            return []
    with _LOCK:
        return _SESSION_STORE.setdefault(session_id, [])

def _add_to_session(session_id: str, role: str, content: str):
    record = {"role": role, "content": content}
    if _redis_client:
        key = f"nova:session:{session_id}"
        try:
            cur = _get_session(session_id)
            cur.append(record)
            max_len = CHAT_HISTORY_TURNS * 4
            if len(cur) > max_len:
                cur = cur[-max_len:]
            _redis_client.set(key, json.dumps(cur), ex=60 * 60 * 24)
            return
        except Exception:
            logger.exception("Redis write failed; falling back to in-memory session.")
    with _LOCK:
        hist = _SESSION_STORE.setdefault(session_id, [])
        hist.append(record)
        max_len = CHAT_HISTORY_TURNS * 4
        if len(hist) > max_len:
            _SESSION_STORE[session_id] = hist[-max_len:]

def enforce_rate_limit(session_id: str):
    now = time.time()
    last = _LAST_CALL_TS.get(session_id, 0.0)
    wait = MIN_SECONDS_BETWEEN_CALLS - (now - last)
    if wait > 0:
        time.sleep(wait)
    _LAST_CALL_TS[session_id] = time.time()

# ----------------------------
# Intent detection + helpers
# ----------------------------
_EXPAND_KW = [
    "expand", "explain in detail", "explain more", "in detail", "detailed",
    "long answer", "step by step", "how does it work", "how it works",
    "tell me more", "explain the process", "what is the process"
]
_SHORT_KW = ["short", "brief", "concise", "in short", "summary", "one line", "one sentence"]
_FACTUAL_KW = ["price", "pricing", "cost", "how much", "monthly", "per month", "plan", "plans", "subscription", "license", "fee", "terms", "contract", "sla", "gdpr", "data protection"]
_SUPPORT_KW = ["password", "login", "log in", "forgot password", "reset", "sign in", "how to use", "get started", "setup", "onboarding"]
_EMOTIONAL_KW = ["sad", "upset", "angry", "frustrated", "wtf", "shit", "damn", "hate", "lost memory", "did you forget"]

def user_requests_expansion(q: str) -> bool:
    t = (q or "").lower()
    return any(kw in t for kw in _EXPAND_KW)

def user_requests_short(q: str) -> bool:
    t = (q or "").lower()
    return any(kw in t for kw in _SHORT_KW)

def is_factual_query(q: str) -> bool:
    t = (q or "").lower()
    if not any(kw in t for kw in _FACTUAL_KW):
        return False
    if re.search(r"\b(\$|£|€|\d+|how much|per month|monthly|cost|price)\b", t):
        return True
    return False

def is_support_query(q: str) -> bool:
    t = (q or "").lower()
    return any(kw in t for kw in _SUPPORT_KW)

def is_emotional(q: str) -> bool:
    t = (q or "").lower()
    return any(kw in t for kw in _EMOTIONAL_KW)

def detect_mode(q: str) -> str:
    t = (q or "").lower()
    if any(w in t for w in ["appeal", "pcn", "penalty charge", "fine deadline", "what happens if i ignore"]):
        return "legal"
    if any(w in t for w in ["secure", "security", "encryption", "gdpr", "api", "integrate", "integration", "tech stack", "database", "ocr", "csv", "dropbox"]):
        return "tech"
    if any(w in t for w in ["why should i use", "why use", "benefit", "roi", "return on investment", "better than", "competitor", "cost", "pricing", "price", "trial", "subscription"]):
        return "sales"
    return "universal"

def shorten_text(text: str, max_sentences: int) -> str:
    if not text:
        return text
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) <= max_sentences:
        return " ".join(parts).strip()
    out = " ".join(parts[:max_sentences]).strip()
    if not out.endswith((".", "!", "?")):
        out += "..."
    return out

def strip_markdown(text: str) -> str:
    if not text:
        return text
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text

# ----------------------------
# FAQ Matcher (exact Q&A) 
# ----------------------------
_FAQ_QUESTIONS: List[str] = []
_FAQ_ANSWERS: List[str] = []
_FAQ_EMBEDDINGS: Optional[np.ndarray] = None
_EMBED_MODEL = None  # will be set from retriever

def load_faq():
    global _FAQ_QUESTIONS, _FAQ_ANSWERS, _FAQ_EMBEDDINGS, _EMBED_MODEL
    if not os.path.exists(FAQ_PATH):
        logger.warning("FAQ file not found at %s – FAQ matching disabled.", FAQ_PATH)
        return
    try:
        with open(FAQ_PATH, "r", encoding="utf8") as f:
            faq_data = json.load(f)
        _FAQ_QUESTIONS = [item["question"] for item in faq_data]
        _FAQ_ANSWERS = [item["answer"] for item in faq_data]

        # Import embedding model from retriever (must be loaded)
        try:
            from app.retriever import _embed_model as embed_model
            _EMBED_MODEL = embed_model
        except Exception:
            logger.exception("Could not get embedding model for FAQ matching")
            return

        if _EMBED_MODEL is None:
            logger.warning("Embedding model not available – FAQ matching disabled.")
            return

        # Compute embeddings for FAQ questions
        logger.info("Computing embeddings for %d FAQ questions...", len(_FAQ_QUESTIONS))
        batch_size = 32
        embs = []
        for i in range(0, len(_FAQ_QUESTIONS), batch_size):
            batch = _FAQ_QUESTIONS[i:i+batch_size]
            emb = _EMBED_MODEL.encode(batch, convert_to_numpy=True)
            if emb.dtype != np.float32:
                emb = emb.astype("float32")
            embs.append(emb)
        _FAQ_EMBEDDINGS = np.vstack(embs)
        # Normalise for cosine similarity
        faiss.normalize_L2(_FAQ_EMBEDDINGS)   # now faiss is imported
        logger.info("FAQ embeddings ready.")
    except Exception as e:
        logger.exception("Failed to load FAQ: %s", e)
        _FAQ_QUESTIONS = []
        _FAQ_ANSWERS = []
        _FAQ_EMBEDDINGS = None

def match_faq(query: str, threshold: float = 0.80) -> Optional[str]:
    """Return exact answer if query matches a FAQ question with similarity > threshold."""
    if _FAQ_EMBEDDINGS is None or _EMBED_MODEL is None:
        return None
    # embed query
    q_emb = _EMBED_MODEL.encode([query], convert_to_numpy=True)
    if q_emb.dtype != np.float32:
        q_emb = q_emb.astype("float32")
    faiss.normalize_L2(q_emb)
    # compute cosine similarity
    sim = np.dot(_FAQ_EMBEDDINGS, q_emb.T).flatten()
    best_idx = np.argmax(sim)
    best_score = sim[best_idx]
    if best_score > threshold:
        logger.info("FAQ match with score %.4f for: %s", best_score, query)
        return _FAQ_ANSWERS[best_idx]
    return None

# ----------------------------
# Retriever load at startup
# ----------------------------
try:
    from app.retriever import load_index
    _EMBED_MODEL, _FAISS_INDEX, _CHUNKS, _METAS = load_index()
    logger.info("Retriever loaded at startup: chunks=%d", len(_CHUNKS) if _CHUNKS else 0)
    # Now load FAQ (depends on _EMBED_MODEL)
    load_faq()
except Exception as e:
    logger.warning("Initial load_index failed: %s", e)
    _EMBED_MODEL = None
    _FAISS_INDEX = None
    _CHUNKS = []
    _METAS = []

def rag_search(query: str, top_k: int = TOP_K):
    try:
        hits = rag_search_impl(query, top_k=top_k)
        try:
            hits = rerank_hits(hits, query)
        except Exception:
            pass
        return hits
    except Exception as e:
        logger.exception("rag_search_impl failed: %s", e)
        return []

def synthesize_from_rag(hits: List[Dict[str, Any]]) -> str:
    if not hits:
        return ""
    sentences = []
    for h in hits[:3]:
        chunk = (h.get("chunk") or h.get("text") or "").strip()
        if not chunk:
            continue
        first = re.split(r"(?<=[.!?])\s+", chunk)[0].strip()
        if first:
            sentences.append(first)
    if not sentences:
        return ""
    # limit to 2 sentences for conciseness
    return " ".join(dict.fromkeys(sentences[:2]))

# ----------------------------
# LLM call
# ----------------------------
HEADERS = {"Authorization": f"Bearer {GROQ_API_KEY}"} if OPENAI_API_KEY else {}

def llm_call(system: str, user: str, max_tokens: int = 250, retries: int = 3, timeout: int = 30) -> str:
    prompt_key = f"SYSTEM={system}\nUSER={user}"
    key = str(abs(hash(prompt_key)))
    if key in _LLM_CACHE:
        return _LLM_CACHE[key]
    if not GROQ_API_KEY:
        logger.warning("No API key configured; skipping LLM call.")
        return ""
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.18,
        "max_tokens": max_tokens,
        "top_p": 1,
    }
    backoff = 1.0
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(
                GROQ_API_URL,
                headers={**HEADERS, "Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
            )
            if r.status_code == 429:
                logger.warning("LLM rate limited (429), attempt=%d", attempt)
                time.sleep(backoff)
                backoff *= 2
                continue
            r.raise_for_status()
            j = r.json()
            if isinstance(j, dict):
                choices = j.get("choices")
                if isinstance(choices, list) and choices:
                    msg = choices[0].get("message") or {}
                    content = msg.get("content")
                    if isinstance(content, str):
                        ans = content.strip()
                        _LLM_CACHE[key] = ans
                        _save_llm_cache()
                        _increment_llm_calls()
                        return ans
            logger.warning("LLM returned unexpected JSON shape")
            return ""
        except requests.RequestException as e:
            logger.warning("LLM HTTP error attempt=%d: %s", attempt, e)
            time.sleep(backoff)
            backoff *= 2
        except Exception as e:
            logger.exception("LLM unexpected error attempt=%d: %s", attempt, e)
            time.sleep(backoff)
            backoff *= 2
    logger.error("LLM exhausted retries")
    return ""

# ----------------------------
# Prompt builder
# ----------------------------
def build_user_prompt(question: str, intent: str, rag_hits: List[Dict[str, Any]], session_history: List[Dict[str, str]]) -> str:
    history_text = ""
    if session_history:
        last = session_history[-6:]
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in last])
    retrieved_block = ""
    for i, h in enumerate((rag_hits or [])[:min(len(rag_hits), TOP_K)]):
        meta = h.get("meta") or {}
        source = meta.get("source") or meta.get("title") or f"doc_{i+1}"
        chunk = (h.get("chunk") or h.get("text") or "").replace("\n", " ")
        snippet = chunk[:800].strip()
        if snippet:
            retrieved_block += f"[{i+1}] SOURCE_META: {source}\n{snippet}\n\n"
    user_prompt = f"""
User question:
{question}

Intent: {intent}

Conversation history (most recent):
{history_text or '<none>'}

Retrieved document excerpts (top {min(len(rag_hits), TOP_K)}):
{retrieved_block or '<none>'}

Instructions to Nova:
- Use retrieved excerpts to ground factual answers. Do NOT print backend filenames.
- If the user asks for a hard factual item (pricing, plan names, exact fees, contract terms, legal advice) and there is no supporting excerpt above, do NOT invent numbers. Instead point to support/sales.
- **Be extremely concise:** answer in 1‑2 sentences. Use bullet points only for lists (e.g., pricing plans).
- If the question is a common FAQ, match the tone and exact phrasing from the FAQ.
- Do not add explanations unless the user asks "how" or "why". Just give the direct answer.
Answer now:
"""
    return user_prompt

# ----------------------------
# Main builder
# ----------------------------
def _sanitize_user_answer(final_answer: str) -> str:
    try:
        final_answer = re.sub(r"\s*\(Source:\s*[^\)]+\)", "", final_answer, flags=re.IGNORECASE)
        final_answer = re.sub(r"\bSource:\s*[^\.\n]+", "", final_answer, flags=re.IGNORECASE)
        final_answer = re.sub(r"SOURCE_META:\s*[^\.\n]+", "", final_answer, flags=re.IGNORECASE)
        final_answer = re.sub(r"\s{2,}", " ", final_answer).strip()
    except Exception:
        return final_answer
    return final_answer

def build_response(question: str, session_id: str = "default", allow_tools: bool = True) -> Dict[str, Any]:
    q = (question or "").strip()
    if not session_id:
        session_id = "default"

    try:
        _add_to_session(session_id, "user", q)
    except Exception:
        logger.exception("Failed to add user turn to session")

    enforce_rate_limit(session_id)

    if not q:
        reply = "I didn't catch that — what would you like to know about Fine Flow?"
        try:
            _add_to_session(session_id, "assistant", reply)
        except Exception:
            pass
        return {"answer": reply, "confidence": 1.0, "sources": [], "tools": {"recommended": None, "executed": None}, "flag": False}

    lower_q = q.lower()
    if is_emotional(lower_q):
        canned = "I'm sorry this has been frustrating. Tell me what's wrong and I'll walk you through it."
        try:
            _add_to_session(session_id, "assistant", canned)
        except Exception:
            pass
        return {"answer": canned, "confidence": 1.0, "sources": [], "tools": {"recommended": None, "executed": None}, "flag": False}

    if re.search(r"\b(hi|hello|hey|good morning|good afternoon|good evening)\b", lower_q):
        canned = "Hi, I'm Nova from Fine Flow. How can I help you today?"
        try:
            _add_to_session(session_id, "assistant", canned)
        except Exception:
            pass
        return {"answer": canned, "confidence": 1.0, "sources": [], "tools": {"recommended": None, "executed": None}, "flag": False}

    if any(x in lower_q for x in ["who are you", "what is nova", "who is nova"]):
        canned = "I'm Nova, Fine Flow's assistant. I help fleets manage fines and appeals."
        try:
            _add_to_session(session_id, "assistant", canned)
        except Exception:
            pass
        return {"answer": canned, "confidence": 1.0, "sources": [], "tools": {"recommended": None, "executed": None}, "flag": False}

    if is_support_query(lower_q):
        reply = f"Use the 'Forgot password' link on the login page or contact support at {SUPPORT_EMAIL} for quick help."
        try:
            _add_to_session(session_id, "assistant", reply)
        except Exception:
            pass
        return {"answer": reply, "confidence": 1.0, "sources": [], "tools": {"recommended": "contact_support", "executed": None}, "flag": False}

    # --- FAQ MATCHING (threshold lowered to 0.80) ---
    faq_answer = match_faq(q, threshold=0.80)
    if faq_answer:
        try:
            _add_to_session(session_id, "assistant", faq_answer)
        except Exception:
            pass
        return {"answer": faq_answer, "confidence": 1.0, "sources": [], "tools": {"recommended": None, "executed": None}, "flag": False}

    # --- RAG search ---
    try:
        rag_hits = rag_search(q, top_k=TOP_K)
    except Exception as e:
        logger.exception("RAG search failed: %s", e)
        rag_hits = []

    hits_count = len(rag_hits) if rag_hits else 0
    top_score = float(rag_hits[0].get("score", 0.0)) if hits_count else 0.0
    logger.info("RAG search results: hits=%d top_score=%.4f (q=%s)", hits_count, top_score, q[:120])

    factual = is_factual_query(q)
    mode = detect_mode(q)

    if factual and not rag_hits:
        reply = f"I don't have that exact detail. Please contact support at {SUPPORT_EMAIL} or sales at {SALES_EMAIL} for precise figures."
        logger.info("Factual fallback (no docs): %s", q[:120])
        try:
            _add_to_session(session_id, "assistant", reply)
        except Exception:
            pass
        return {"answer": reply, "confidence": 0.0, "sources": [], "tools": {"recommended": "contact_support", "executed": None}, "flag": True}

    history = _get_session(session_id)
    system_prompt = build_system_prompt(mode)
    user_prompt = build_user_prompt(q, "factual" if factual else "conversational", rag_hits, history)

    final_raw = ""
    if OPENAI_API_KEY:
        try:
            llm_out = llm_call(system_prompt, user_prompt)
            if llm_out:
                final_raw = llm_out.strip()
            else:
                logger.warning("LLM returned empty; falling back to synthesizer.")
                final_raw = synthesize_from_rag(rag_hits) or ""
        except Exception as e:
            logger.exception("LLM call failed: %s", e)
            final_raw = synthesize_from_rag(rag_hits) or ""
    else:
        final_raw = synthesize_from_rag(rag_hits) or ""

    want_expand = user_requests_expansion(q)
    want_short = user_requests_short(q)
    if want_expand:
        final_answer = final_raw
    elif want_short:
        final_answer = shorten_text(final_raw, max_sentences=1)
    else:
        final_answer = shorten_text(final_raw, max_sentences=2)   # default to 2 sentences

    if not final_answer.strip():
        if factual:
            final_answer = f"I don't have that detail. Contact {SUPPORT_EMAIL} for exact information."
            flag = True
        else:
            synth = synthesize_from_rag(rag_hits)
            if synth:
                final_answer = shorten_text(synth, max_sentences=2)
                flag = False
            else:
                final_answer = f"I don't have documents on that. Contact {SUPPORT_EMAIL} for help."
                flag = False
    else:
        flag = (top_score < CONFIDENCE_THRESHOLD)

    final_answer = _sanitize_user_answer(final_answer)
    final_answer = strip_markdown(final_answer)

    if len(final_answer) > 500:
        truncated = final_answer[:497]
        last_period = truncated.rfind('.')
        last_exclamation = truncated.rfind('!')
        last_question = truncated.rfind('?')
        last_boundary = max(last_period, last_exclamation, last_question)
        if last_boundary > 400:
            final_answer = truncated[:last_boundary + 1]
        else:
            final_answer = truncated + "..."

    try:
        _add_to_session(session_id, "assistant", final_answer)
    except Exception:
        pass

    sources = []
    for h in (rag_hits or [])[:TOP_K]:
        meta = h.get("meta") or {}
        title = meta.get("title") or meta.get("source") or "doc"
        sources.append({"title": title, "score": float(h.get("score", 0.0))})

    return {
        "answer": final_answer,
        "confidence": float(top_score),
        "sources": sources,
        "tools": {"recommended": None, "executed": None},
        "flag": bool(flag),
    }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
def answer_sync(q: str, session_id: str = "default") -> Dict[str, Any]:
    try:
        out = build_response(q, session_id=session_id)
        out["confidence"] = float(out.get("confidence", 0.0))
        return out
    except Exception:
        logger.exception("answer_sync failed")
        return {
            "answer": "Internal error answering your question.",
            "confidence": 0.0,
            "sources": [],
            "tools": {},
            "flag": True,
        }

logger.info(
    "Nova answer_builder (Hybrid RAG + GPT-4o + Persona Engine) loaded. LLM enabled=%s, model=%s",
    bool(OPENAI_API_KEY),
    OPENAI_MODEL,
)