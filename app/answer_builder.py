# app/answer_builder.py
"""
Nova — FineFlow Answer Builder (Hybrid RAG + GPT-4o + Persona Engine)
with structured facts fallback and ChromaDB retriever.
"""

import os
import time
import json
import re
import threading
from typing import List, Dict, Any, Optional
from pathlib import Path

import requests
import numpy as np

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
# Structured Facts
# ----------------------------
STRUCTURED_FACTS_PATH = Path(__file__).parent / "data" / "structured_facts.json"
_STRUCTURED_FACTS = {}
if STRUCTURED_FACTS_PATH.exists():
    with open(STRUCTURED_FACTS_PATH, 'r', encoding='utf8') as f:
        _STRUCTURED_FACTS = json.load(f)

def get_fact_from_structured(query: str) -> Optional[str]:
    """Return precise answer from structured facts if query matches."""
    q = query.lower()
    plans = _STRUCTURED_FACTS.get("plans", {})
    # Plan details
    for plan_name, details in plans.items():
        if plan_name.lower() in q:
            if "price" in q or "cost" in q or "month" in q:
                return f"The {plan_name} plan costs £{details['price']}/month."
            if "credit" in q:
                return f"The {plan_name} plan includes {details['credits']} credits."
            if "vehicle" in q and "limit" in q:
                return f"The {plan_name} plan supports {details['vehicle_limit']} vehicles."
            # Generic plan info
            if any(kw in q for kw in ["plan", "subscription"]):
                return (f"The {plan_name} plan: £{details['price']}/month, "
                        f"{details['credits']} credits, {details['vehicle_limit']} vehicles.")
    # All plans list
    if any(kw in q for kw in ["all plans", "list plans", "available plans"]):
        lines = ["Fine Flow subscription plans:"]
        for name, det in plans.items():
            lines.append(f"- {name}: £{det['price']}/month, {det['credits']} credits, {det['vehicle_limit']} vehicles")
        return "\n".join(lines)
    # Per-fine rate
    if "per fine" in q or "per-fine" in q:
        return f"The per-fine rate is £{_STRUCTURED_FACTS.get('per_fine_rate')}."
    # Overage
    if "overage" in q:
        return f"Overage rate is £{_STRUCTURED_FACTS.get('overage_rate')} per fine."
    # Vehicle overage charge
    if ("exceed" in q or "over" in q) and "vehicle" in q and "limit" in q:
        return f"Exceeding the vehicle limit incurs a £{_STRUCTURED_FACTS.get('vehicle_overage_charge')} charge per vehicle."
    # Referral rewards
    if "refer" in q:
        # Vehicle count based reward
        match = re.search(r'(\d+)\s*vehicle', q)
        if match:
            count = int(match.group(1))
            rewards = _STRUCTURED_FACTS.get("referral_rewards", {})
            for rng, val in rewards.items():
                if '-' in rng:
                    low, high = map(int, rng.split('-'))
                    if low <= count <= high:
                        return f"For a fleet of {count} vehicles, you'll receive {val} credits."
                elif rng.endswith('+'):
                    low = int(rng[:-1])
                    if count >= low:
                        return f"For a fleet of {count} vehicles, you'll receive {val} credits."
        # Tiers
        if "tier" in q:
            tiers = _STRUCTURED_FACTS.get("referral_tiers", {})
            lines = ["Referral Tiers:"]
            for name, info in tiers.items():
                lines.append(f"- {name}: {info['referrals']} referrals → {info['reward']}")
            return "\n".join(lines)
        # New joiner reward
        if "new" in q or "join" in q:
            return f"New companies joining with a referral code receive £{_STRUCTURED_FACTS.get('new_joiner_reward')} worth of credits."
    # Elite vehicle limit override (important for memory consistency)
    if "elite" in q and "vehicle" in q and "limit" in q:
        return "The Elite plan has an unlimited vehicle limit."
    return None

# ----------------------------
# Retriever import
# ----------------------------
try:
    from app.retriever import search as rag_search_impl, rerank_hits, build_index, load_index
except Exception as e:
    logger.warning("Could not import retriever: %s", e)
    def rag_search_impl(q, top_k=TOP_K): return []
    def rerank_hits(hits, q): return hits
    def build_index(force_rebuild=False): pass
    def load_index(): return None, None, [], []

# ----------------------------
# Redis session (optional)
# ----------------------------
_redis_client = None
try:
    REDIS_URL = os.getenv("REDIS_URL", "")
    if REDIS_URL:
        import redis
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        _redis_client.ping()
        logger.info("Redis session store enabled.")
except Exception as e:
    logger.warning("Redis unavailable; using in-memory sessions: %s", e)

# ----------------------------
# Persona & prompts
# ----------------------------
NOVA_UNIVERSAL_PROMPT = (
    "You are Nova, the intelligent, calm, and highly capable assistant behind Fine Flow — "
    "an advanced fleet fine management system.\n"
    "Be human, professional, and helpful. Never break immersion or say you are an AI.\n"
    "**Always give concise answers.** Use 1‑2 sentences unless the user explicitly asks for more detail.\n"
    "When listing multiple items (like pricing plans or referral tiers), **always include the complete list** "
    "as provided in the retrieved excerpts. Use bullet points.\n"
    "Never add extra commentary or explanations beyond what is asked.\n"
    "Always use the retrieved document excerpts to ground your answers. "
    "Do NOT print internal filenames.\n"
    "Output plain text only – no markdown formatting."
)

SALES_NOVA_PROMPT = (
    "You are Sales Nova. Tone: confident, benefit-driven, persuasive but honest. "
    "Lead with pain points and ROI, end with a clear call to action (book demo / contact sales). "
    "Be concise. Use bullet points only if listing."
)

TECH_NOVA_PROMPT = (
    "You are Tech Nova. Tone: calm, expert, non-condescending. Explain system behaviour, integrations, security, "
    "and troubleshooting without revealing proprietary internals. Keep it short."
)

LEGAL_NOVA_PROMPT = (
    "You are Legal Nova. Explain timelines and procedural info about fines/appeals. Always include this disclaimer:\n"
    "\"Please note: I can't provide legal advice. I can only explain how Fine Flow works and what the general process usually involves.\"\n"
    "Do NOT give specific legal advice or predictive outcomes."
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
        pass

def _increment_llm_calls():
    try:
        _LLM_COUNTER["calls"] += 1
        with open(LLM_COUNTER_PATH, "w", encoding="utf8") as f:
            json.dump(_LLM_COUNTER, f, ensure_ascii=False)
    except Exception:
        pass

def _get_session(session_id: str) -> List[Dict[str, str]]:
    if _redis_client:
        key = f"nova:session:{session_id}"
        try:
            raw = _redis_client.get(key)
            return json.loads(raw) if raw else []
        except Exception:
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
            _redis_client.set(key, json.dumps(cur), ex=60*60*24)
            return
        except Exception:
            pass
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
# OpenAI Embedding helpers (for FAQ)
# ----------------------------
def get_openai_embedding_single(text: str) -> np.ndarray:
    if not OPENAI_API_KEY:
        raise RuntimeError("OpenAI API key not configured")
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "text-embedding-3-small", "input": [text]}
    r = requests.post("https://api.openai.com/v1/embeddings", headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    return np.array(data['data'][0]['embedding'], dtype=np.float32)

def get_openai_embedding_batch(texts: List[str]) -> np.ndarray:
    if not OPENAI_API_KEY:
        raise RuntimeError("OpenAI API key not configured")
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "text-embedding-3-small", "input": texts}
    r = requests.post("https://api.openai.com/v1/embeddings", headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    embeddings = [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
    return np.array(embeddings, dtype=np.float32)

# ----------------------------
# FAQ Matcher
# ----------------------------
_FAQ_QUESTIONS: List[str] = []
_FAQ_ANSWERS: List[str] = []
_FAQ_EMBEDDINGS: Optional[np.ndarray] = None

def load_faq():
    global _FAQ_QUESTIONS, _FAQ_ANSWERS, _FAQ_EMBEDDINGS
    if not os.path.exists(FAQ_PATH):
        return
    try:
        with open(FAQ_PATH, "r", encoding="utf8") as f:
            faq_data = json.load(f)
        _FAQ_QUESTIONS = [item["question"] for item in faq_data]
        _FAQ_ANSWERS = [item["answer"] for item in faq_data]
        if not OPENAI_API_KEY:
            return
        logger.info("Computing embeddings for %d FAQ questions...", len(_FAQ_QUESTIONS))
        batch_size = 100
        embs = []
        for i in range(0, len(_FAQ_QUESTIONS), batch_size):
            batch = _FAQ_QUESTIONS[i:i+batch_size]
            emb = get_openai_embedding_batch(batch)
            embs.append(emb)
        _FAQ_EMBEDDINGS = np.vstack(embs)
        # Normalize for cosine similarity
        from numpy.linalg import norm
        _FAQ_EMBEDDINGS = _FAQ_EMBEDDINGS / norm(_FAQ_EMBEDDINGS, axis=1, keepdims=True)
        logger.info("FAQ embeddings ready.")
    except Exception as e:
        logger.exception("Failed to load FAQ: %s", e)

def match_faq(query: str, threshold: float = 0.80) -> Optional[str]:
    if _FAQ_EMBEDDINGS is None or not OPENAI_API_KEY:
        return None
    try:
        q_emb = get_openai_embedding_single(query)
        q_emb = q_emb / np.linalg.norm(q_emb)
        sim = np.dot(_FAQ_EMBEDDINGS, q_emb)
        best_idx = np.argmax(sim)
        best_score = sim[best_idx]
        if best_score > threshold:
            return _FAQ_ANSWERS[best_idx]
    except Exception:
        pass
    return None

# ----------------------------
# Intent detection
# ----------------------------
_EXPAND_KW = ["expand", "explain in detail", "explain more", "in detail", "detailed", "step by step", "how does it work", "tell me more"]
_SHORT_KW = ["short", "brief", "concise", "in short", "summary", "one line", "one sentence"]
_FACTUAL_KW = ["price", "pricing", "cost", "how much", "monthly", "plan", "subscription", "fee"]
_SUPPORT_KW = ["password", "login", "forgot", "reset", "sign in", "setup", "onboarding"]
_EMOTIONAL_KW = ["sad", "upset", "angry", "frustrated", "wtf", "shit", "damn", "hate"]

def user_requests_expansion(q: str) -> bool:
    return any(kw in q.lower() for kw in _EXPAND_KW)
def user_requests_short(q: str) -> bool:
    return any(kw in q.lower() for kw in _SHORT_KW)
def is_factual_query(q: str) -> bool:
    return any(kw in q.lower() for kw in _FACTUAL_KW) or bool(re.search(r"\b(\$|£|€|\d+|how much|per month|monthly|cost|price)\b", q.lower()))
def is_support_query(q: str) -> bool:
    return any(kw in q.lower() for kw in _SUPPORT_KW)
def is_emotional(q: str) -> bool:
    return any(kw in q.lower() for kw in _EMOTIONAL_KW)
def detect_mode(q: str) -> str:
    t = q.lower()
    if any(w in t for w in ["appeal", "pcn", "penalty", "fine deadline"]): return "legal"
    if any(w in t for w in ["security", "encryption", "gdpr", "api", "integrate", "tech"]): return "tech"
    if any(w in t for w in ["why use", "benefit", "roi", "better than", "competitor", "cost", "pricing", "trial"]): return "sales"
    return "universal"

def shorten_text(text: str, max_sentences: int, preserve_lists: bool = True) -> str:
    if not text:
        return text
    # If bullet list present, preserve structure
    if preserve_lists and re.search(r'(^|\n)\s*[-*•]', text):
        # Extract bullet items
        items = re.split(r'\n\s*[-*•]\s*', text.strip())
        if len(items) > 1:
            # First part before first bullet might be intro, skip or keep?
            # Simple: return all items if few, else truncate
            if len(items) <= 5:
                return "\n- " + "\n- ".join(items).strip()
            else:
                return "\n- " + "\n- ".join(items[:5]).strip() + "\n- ..."
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) <= max_sentences:
        return " ".join(parts)
    return " ".join(parts[:max_sentences]) + "..."

def strip_markdown(text: str) -> str:
    if not text: return text
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text

# ----------------------------
# LLM call
# ----------------------------
HEADERS = {"Authorization": f"Bearer {OPENAI_API_KEY}"} if OPENAI_API_KEY else {}

def llm_call(system: str, user: str, max_tokens: int = 350, retries: int = 3) -> str:
    prompt_key = f"SYSTEM={system}\nUSER={user}"
    key = str(abs(hash(prompt_key)))
    if key in _LLM_CACHE:
        return _LLM_CACHE[key]
    if not OPENAI_API_KEY:
        return ""
    payload = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0.18,
        "max_tokens": max_tokens,
    }
    backoff = 1.0
    for attempt in range(retries):
        try:
            r = requests.post(OPENAI_API_URL, headers={**HEADERS, "Content-Type": "application/json"}, json=payload, timeout=30)
            if r.status_code == 429:
                time.sleep(backoff)
                backoff *= 2
                continue
            r.raise_for_status()
            ans = r.json()["choices"][0]["message"]["content"].strip()
            _LLM_CACHE[key] = ans
            _save_llm_cache()
            _increment_llm_calls()
            return ans
        except Exception as e:
            logger.warning("LLM attempt %d failed: %s", attempt, e)
            time.sleep(backoff)
            backoff *= 2
    return ""

# ----------------------------
# Prompt builder
# ----------------------------
def build_user_prompt(question: str, intent: str, rag_hits: List[Dict], history: List[Dict]) -> str:
    hist_text = "\n".join([f"{m['role']}: {m['content']}" for m in history[-6:]]) if history else "<none>"
    retrieved = ""
    for i, h in enumerate(rag_hits[:TOP_K]):
        meta = h.get("meta", {})
        src = meta.get("title", meta.get("source", f"doc_{i+1}"))
        chunk = h.get("chunk", "").replace("\n", " ")[:800]
        retrieved += f"[{i+1}] SOURCE: {src}\n{chunk}\n\n"
    return f"""
User question:
{question}

Intent: {intent}

Conversation history (recent):
{hist_text}

Retrieved excerpts (top {len(rag_hits[:TOP_K])}):
{retrieved or '<none>'}

Instructions:
- Use excerpts to ground answers. Do not invent numbers.
- Be concise: 1‑2 sentences unless asked for detail.
- For lists, provide all items from excerpts using bullet points.
- If factual info missing, direct to support/sales.
Answer now:
"""

def synthesize_from_rag(hits: List[Dict]) -> str:
    if not hits:
        return ""
    sentences = []
    for h in hits[:3]:
        chunk = h.get("chunk", "").strip()
        if chunk:
            first = re.split(r"(?<=[.!?])\s+", chunk)[0]
            sentences.append(first)
    return " ".join(dict.fromkeys(sentences[:2]))

def _sanitize_user_answer(ans: str) -> str:
    ans = re.sub(r"\s*\(Source:\s*[^\)]+\)", "", ans, flags=re.IGNORECASE)
    ans = re.sub(r"\bSource:\s*[^\.\n]+", "", ans, flags=re.IGNORECASE)
    ans = re.sub(r"SOURCE_META:\s*[^\.\n]+", "", ans, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", ans).strip()

# ----------------------------
# Main builder
# ----------------------------
def build_response(question: str, session_id: str = "default") -> Dict[str, Any]:
    q = question.strip()
    if not session_id:
        session_id = "default"

    _add_to_session(session_id, "user", q)
    enforce_rate_limit(session_id)

    # Quick canned responses
    if not q:
        reply = "I didn't catch that — what would you like to know about Fine Flow?"
        _add_to_session(session_id, "assistant", reply)
        return {"answer": reply, "confidence": 1.0, "sources": [], "flag": False}
    lower_q = q.lower()
    if is_emotional(lower_q):
        reply = "I'm sorry this has been frustrating. Tell me what's wrong and I'll walk you through it."
        _add_to_session(session_id, "assistant", reply)
        return {"answer": reply, "confidence": 1.0, "sources": [], "flag": False}
    if re.search(r"\b(hi|hello|hey|good morning|good afternoon|good evening)\b", lower_q):
        reply = "Hi, I'm Nova from Fine Flow. How can I help you today?"
        _add_to_session(session_id, "assistant", reply)
        return {"answer": reply, "confidence": 1.0, "sources": [], "flag": False}
    if any(x in lower_q for x in ["who are you", "what is nova", "who is nova"]):
        reply = "I'm Nova, Fine Flow's assistant. I help fleets manage fines and appeals."
        _add_to_session(session_id, "assistant", reply)
        return {"answer": reply, "confidence": 1.0, "sources": [], "flag": False}
    if is_support_query(lower_q):
        reply = f"Use the 'Forgot password' link on the login page or contact support at {SUPPORT_EMAIL}."
        _add_to_session(session_id, "assistant", reply)
        return {"answer": reply, "confidence": 1.0, "sources": [], "flag": False}

    # Structured facts check (override RAG)
    fact_answer = get_fact_from_structured(q)
    if fact_answer:
        _add_to_session(session_id, "assistant", fact_answer)
        return {"answer": fact_answer, "confidence": 1.0, "sources": [], "flag": False}

    # FAQ matching
    faq_answer = match_faq(q, threshold=0.80)
    if faq_answer:
        _add_to_session(session_id, "assistant", faq_answer)
        return {"answer": faq_answer, "confidence": 1.0, "sources": [], "flag": False}

    # RAG search
    try:
        rag_hits = rag_search_impl(q, top_k=TOP_K)
        rag_hits = rerank_hits(rag_hits, q)
    except Exception:
        rag_hits = []
    top_score = rag_hits[0]["score"] if rag_hits else 0.0

    factual = is_factual_query(q)
    mode = detect_mode(q)
    history = _get_session(session_id)

    if factual and not rag_hits:
        reply = f"I don't have that exact detail. Please contact support at {SUPPORT_EMAIL} or sales at {SALES_EMAIL}."
        _add_to_session(session_id, "assistant", reply)
        return {"answer": reply, "confidence": 0.0, "sources": [], "flag": True}

    system_prompt = build_system_prompt(mode)
    user_prompt = build_user_prompt(q, "factual" if factual else "conversational", rag_hits, history)

    final_raw = ""
    if OPENAI_API_KEY:
        llm_out = llm_call(system_prompt, user_prompt)
        final_raw = llm_out if llm_out else synthesize_from_rag(rag_hits)
    else:
        final_raw = synthesize_from_rag(rag_hits)

    want_expand = user_requests_expansion(q)
    want_short = user_requests_short(q)
    if want_expand:
        final_answer = final_raw
    elif want_short:
        final_answer = shorten_text(final_raw, max_sentences=1)
    else:
        final_answer = shorten_text(final_raw, max_sentences=2)

    if not final_answer.strip():
        final_answer = f"I don't have documents on that. Contact {SUPPORT_EMAIL} for help."
        flag = True
    else:
        flag = (top_score < CONFIDENCE_THRESHOLD)

    final_answer = _sanitize_user_answer(final_answer)
    final_answer = strip_markdown(final_answer)
    if len(final_answer) > 500:
        final_answer = final_answer[:497] + "..."

    _add_to_session(session_id, "assistant", final_answer)

    sources = [{"title": h.get("meta", {}).get("title", "doc"), "score": h["score"]} for h in rag_hits[:TOP_K]]
    return {"answer": final_answer, "confidence": top_score, "sources": sources, "flag": flag}

def answer_sync(q: str, session_id: str = "default") -> Dict[str, Any]:
    try:
        return build_response(q, session_id)
    except Exception:
        logger.exception("answer_sync failed")
        return {"answer": "Internal error.", "confidence": 0.0, "sources": [], "flag": True}

logger.info("Nova answer_builder loaded. LLM enabled=%s, model=%s", bool(OPENAI_API_KEY), OPENAI_MODEL)