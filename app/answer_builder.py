# app/answer_builder.py
"""
FineFlow Nova — Answer Builder Production Final v7
=========================================================
Fixes applied vs v6:
- Concise responses: 2-4 sentences max from KB, never a wall of text
- Real conversation memory: follow-up questions stored and acted on
- Single-word queries handled intelligently (pricing, fines, appeals etc)
- Topic lock enforced: HTML, ML, pizza — all hard redirected
- Correct payment answer: NO, Fine Flow does NOT pay fines automatically
- Pricing correct: Essential £99/50v, Core £199/100v, Elite £499/unlimited
- Sales-oriented tone: always ends with a qualifying question
- Rudeness handled gracefully
- Follow-up questions asked proactively to qualify the user
"""

import json
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from app.config import (
    CHAT_HISTORY_TURNS,
    CONFIDENCE_THRESHOLD,
    OPENAI_API_KEY,
    OPENAI_API_URL,
    OPENAI_MODEL,
    TOP_K,
)
from app.logger import logger
from app.retriever import rerank_hits, search as rag_search

# ---------------------------------------------------------------------------
# Knowledge Base loader
# ---------------------------------------------------------------------------
KB_PATH = Path(__file__).parent.parent / "data" / "fineflow_kb.json"
_kb: List[Dict[str, Any]] = []


def _load_kb() -> None:
    global _kb
    if not KB_PATH.exists():
        logger.warning("KB file not found at %s", KB_PATH)
        return
    try:
        with open(KB_PATH, "r", encoding="utf-8") as f:
            _kb = json.load(f)
        logger.info("Loaded %d KB entries", len(_kb))
    except Exception:
        logger.exception("Failed to load KB")


_load_kb()


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------
def _normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Clean output — strip markdown symbols
# ---------------------------------------------------------------------------
def _clean(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"_(.*?)_", r"\1", text)
    text = text.replace("→", "to")
    text = text.replace("->", "to")
    text = text.replace("±", "plus or minus")
    text = text.replace("`", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# KB matching — exact substring + token overlap + keyword match
# ---------------------------------------------------------------------------
def _kb_match(query: str) -> Optional[Dict[str, Any]]:
    nq = _normalise(query)
    q_tokens = set(nq.split())
    best_entry: Optional[Dict[str, Any]] = None
    best_score = 0.0

    for entry in _kb:
        searchable = _normalise(entry.get("question", ""))
        keywords_raw = entry.get("keywords", [])
        if keywords_raw:
            searchable += " " + " ".join(_normalise(k) for k in keywords_raw)

        if nq in searchable:
            return entry

        s_tokens = set(searchable.split())
        if not s_tokens:
            continue
        overlap = len(q_tokens & s_tokens)
        score = overlap / max(len(q_tokens), 1)
        if score > best_score:
            best_score = score
            best_entry = entry

    if best_score >= 0.45 and best_entry:
        return best_entry
    return None


# ---------------------------------------------------------------------------
# Session memory
# ---------------------------------------------------------------------------
_SESSION: Dict[str, List[Dict[str, str]]] = {}
_SESSION_CONTEXT: Dict[str, Optional[Dict[str, Any]]] = {}
_LOCK = threading.Lock()


def _get_history(sid: str) -> List[Dict[str, str]]:
    with _LOCK:
        return list(_SESSION.get(sid, []))


def _add_turn(sid: str, role: str, content: str) -> None:
    with _LOCK:
        hist = _SESSION.setdefault(sid, [])
        hist.append({"role": role, "content": content})
        max_turns = CHAT_HISTORY_TURNS * 2
        if len(hist) > max_turns:
            _SESSION[sid] = hist[-max_turns:]


def _set_ctx(sid: str, entry: Optional[Dict[str, Any]]) -> None:
    with _LOCK:
        _SESSION_CONTEXT[sid] = entry


def _get_ctx(sid: str) -> Optional[Dict[str, Any]]:
    with _LOCK:
        return _SESSION_CONTEXT.get(sid)


# ---------------------------------------------------------------------------
# Intent sets
# ---------------------------------------------------------------------------
_AFFIRMATIVE = {
    "yes", "yeah", "yep", "sure", "ok", "okay", "go on", "please",
    "continue", "tell me more", "go ahead", "more", "explain",
    "yes please", "show me", "absolutely", "of course", "do it",
    "go for it", "sounds good", "great", "definitely", "of course",
}
_NEGATIVE = {
    "no", "nope", "nah", "no thanks", "not now", "skip", "never mind",
    "nevermind", "no need", "dont", "don t", "not really",
}
_GREETINGS = {
    "hi", "hello", "hey", "hiya", "howdy", "good morning",
    "good afternoon", "good evening", "morning", "afternoon",
    "hi there", "hey there", "hello there", "hi nova", "hey nova",
}
_SOCIAL = {
    "how are you", "how are you doing", "how r u", "how r you",
    "how are u", "hows it going", "how is it going", "whats up",
    "what s up", "sup", "you ok", "you good",
}
_THANKS = {
    "thanks", "thank you", "thank u", "cheers", "that helps",
    "that helped", "okay thanks", "ok thanks", "great thanks",
    "perfect", "brilliant", "nice one", "lovely",
}
_RUDE = {
    "you dumb", "you are dumb", "ur dumb", "stupid", "idiot",
    "useless", "rubbish", "garbage", "terrible", "awful",
    "this is rubbish", "this is garbage", "you suck",
}

# Single-word / short topic triggers — map to KB categories
_TOPIC_SHORTCUTS: Dict[str, str] = {
    "pricing":    "how much does fine flow cost",
    "price":      "how much does fine flow cost",
    "cost":       "how much does fine flow cost",
    "plans":      "how much does fine flow cost",
    "packages":   "how much does fine flow cost",
    "fines":      "what happens to a fine once it comes into the system",
    "fine":       "what happens to a fine once it comes into the system",
    "appeals":    "can you help me appeal a fine",
    "appeal":     "can you help me appeal a fine",
    "billing":    "how does billing work",
    "dashboard":  "what does the dashboard show",
    "drivers":    "how do i add drivers to the system",
    "driver":     "how does fine flow know which driver a fine belongs to",
    "matching":   "how does fine flow know which driver a fine belongs to",
    "security":   "is my data safe with fine flow",
    "gdpr":       "is my data safe with fine flow",
    "referral":   "is there a referral programme",
    "referrals":  "is there a referral programme",
    "contact":    "how do i contact fine flow",
    "reports":    "what reports can i export from fine flow",
    "export":     "what reports can i export from fine flow",
    "statuses":   "what do the fine statuses mean",
    "status":     "what do the fine statuses mean",
    "overdue":    "what happens if i don t deal with a fine in time",
    "email":      "how does fine flow get my fines in the first place",
    "gmail":      "how do i connect my gmail to fine flow",
    "payg":       "is there a pay as you go option",
    "overage":    "what is the overage charge",
    "savings":    "how much time and money can fine flow save me",
    "features":   "what are fine flows key features",
    "roles":      "what are the different user roles in fine flow",
}

# ---------------------------------------------------------------------------
# Off-topic guard
# ---------------------------------------------------------------------------
_OFF_TOPIC_PATTERNS = [
    r"\b(html|css|javascript|python|java|php|sql|react|angular|vue|node|django|flask)\b",
    r"\b(machine learning|deep learning|neural network|artificial intelligence|ai model)\b",
    r"\b(recipe|cook|food|restaurant|pizza|burger|coffee)\b",
    r"\b(movie|film|song|music|sport|football|cricket|weather|news)\b",
    r"\b(write me a poem|tell me a joke|essay|story|translate|proofread)\b",
    r"\b(chatgpt|openai|google|bing|alexa|siri|amazon)\b",
    r"\b(coding|programming|developer|debug|github|docker|kubernetes)\b",
    r"\b(database|mysql|mongodb|server|deploy|devops|backend|frontend)\b",
]
_OFF_TOPIC_COMPILED = [re.compile(p, re.IGNORECASE) for p in _OFF_TOPIC_PATTERNS]


def _is_off_topic(query: str) -> bool:
    for pattern in _OFF_TOPIC_COMPILED:
        if pattern.search(query):
            return True
    return False


def _is_affirmative(q: str) -> bool:
    return _normalise(q) in _AFFIRMATIVE

def _is_negative(q: str) -> bool:
    return _normalise(q) in _NEGATIVE

def _is_greeting(q: str) -> bool:
    return _normalise(q) in _GREETINGS

def _is_social(q: str) -> bool:
    return _normalise(q) in _SOCIAL

def _is_thanks(q: str) -> bool:
    return _normalise(q) in _THANKS

def _is_rude(q: str) -> bool:
    nq = _normalise(q)
    return any(r in nq for r in _RUDE)

def _is_pricing_confusion(q: str) -> bool:
    nq = q.lower().strip()
    if re.match(r"^[£$]?\s*2\s*\??$", nq):
        return True
    return bool(re.search(r"£\s*2\s+(fee|for|charge)", nq))

def _is_vague(q: str) -> bool:
    nq = _normalise(q)
    all_special = _AFFIRMATIVE | _NEGATIVE | _GREETINGS | _SOCIAL | _THANKS | _RUDE
    # Only truly vague if very short AND not a known topic shortcut
    return len(nq.split()) <= 2 and nq not in all_special and nq not in _TOPIC_SHORTCUTS


# ---------------------------------------------------------------------------
# System prompt — strict, concise, sales-oriented
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are Nova, the AI assistant for Fine Flow — a UK fleet fine (PCN) management platform.

CRITICAL RULES — follow every one, no exceptions:

1. TOPIC LOCK: You ONLY answer questions about Fine Flow. If someone asks about coding, HTML, recipes, machine learning, general knowledge, or anything unrelated, respond with exactly: "I can only help with Fine Flow questions. Is there anything about fines, pricing, appeals or the platform I can help with?"

2. RESPONSE LENGTH: Keep answers to 2-4 sentences maximum. Never write paragraphs. One clear idea, then ask a follow-up question to keep the conversation going.

3. FOLLOW-UP QUESTIONS: Always end your response with one short qualifying question. Examples: "How many vehicles do you run?", "Would you like to know more about how appeals work?", "What is your current monthly fine volume?"

4. SALES TONE: You are helpful, warm, confident and concise. You guide the user toward understanding the value of Fine Flow. Never be robotic or list-heavy.

5. PAYMENT ANSWER — CRITICAL: Fine Flow does NOT automatically pay fines. It does NOT log into authority websites. Payment is always completed manually by the user on the authority's own site. Never say Fine Flow pays fines or handles payment automatically.

6. NO MARKDOWN: No asterisks, no bullet points, no bold, no headers. Plain conversational English only.

7. NO FILLER: Never start with "Certainly!", "Great question!", "Of course!" Just answer directly.

CORRECT PRICING — never change these numbers:
Essential: £99 per month, up to 50 vehicles.
Core: £199 per month, up to 100 vehicles.
Elite: £499 per month, unlimited vehicles.
All plans: £0.75 per fine within monthly allowance.
Overage: £2.50 per fine beyond the plan limit.
Pay-as-you-go (no subscription): £2.75 per fine.
There is NO £2.00 fee anywhere.

KEY FACTS:
Fine Flow monitors Gmail every minute for incoming fine emails. It extracts fine details using AI, matches fines to drivers using vehicle logs, tracks deadlines, manages appeals and sends appeal letters to authorities. It does NOT pay fines. All plans include full platform access — no locked features.

Contact: +47 32 28 50 00 | ff.sales@fineflow.com

CONVERSATION STYLE EXAMPLE:
User: "how much does it cost"
Nova: "Fine Flow starts at £99 a month for fleets up to 50 vehicles, rising to £199 for up to 100 and £499 for unlimited. Every plan includes the full platform with no locked features. How many vehicles do you run?"

User: "what is html"
Nova: "I can only help with Fine Flow questions. Is there anything about fines, pricing, appeals or the platform I can help with?"
"""


# ---------------------------------------------------------------------------
# OpenAI call — temperature 0.0
# ---------------------------------------------------------------------------
def _call_openai(messages: List[Dict[str, str]], max_tokens: int = 200) -> Optional[str]:
    if not OPENAI_API_KEY:
        return None
    try:
        r = requests.post(
            OPENAI_API_URL,
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENAI_MODEL,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": max_tokens,
            },
            timeout=20,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        logger.exception("OpenAI call failed")
        return None


# ---------------------------------------------------------------------------
# Shorten KB answer to 2-3 sentences max for concise output
# ---------------------------------------------------------------------------
def _shorten(text: str, max_sentences: int = 3) -> str:
    """Take first N sentences from a KB answer to keep responses tight."""
    if not text:
        return text
    # Split on sentence endings
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return " ".join(sentences[:max_sentences]).strip()


# ---------------------------------------------------------------------------
# Core response builder
# ---------------------------------------------------------------------------
def build_response(query: str, session_id: str = "default") -> Dict[str, Any]:
    query = query.strip()
    session_id = session_id or "default"

    if not query:
        return {"answer": "Ask me anything about Fine Flow.", "confidence": 1.0}

    nq = _normalise(query)

    # --- Greeting ---
    if _is_greeting(query):
        _set_ctx(session_id, None)
        return {
            "answer": "I'm Nova. Ask me anything — I'll help you manage fines, resolve issues, and keep everything moving.",
            "confidence": 1.0,
        }

    # --- Social ---
    if _is_social(query):
        return {
            "answer": "Doing well, thanks. What can I help you with today — pricing, fines, appeals?",
            "confidence": 1.0,
        }

    # --- Thanks ---
    if _is_thanks(query):
        return {
            "answer": "Happy to help. Is there anything else you would like to know?",
            "confidence": 1.0,
        }

    # --- Rudeness — calm and redirect ---
    if _is_rude(query):
        return {
            "answer": "No problem — let me try again. What specifically would you like to know about Fine Flow? Pricing, how fines work, or something else?",
            "confidence": 1.0,
        }

    # --- Negative ---
    if _is_negative(query):
        _add_turn(session_id, "user", query)
        answer = "No problem. What else can I help you with?"
        _add_turn(session_id, "assistant", answer)
        _set_ctx(session_id, None)
        return {"answer": answer, "confidence": 1.0}

    # --- Pricing confusion (£2 fee that does not exist) ---
    if _is_pricing_confusion(query):
        answer = "There is no £2 fee. Within your plan allowance it is £0.75 per fine. Beyond your allowance it is £2.50. The pay-as-you-go option with no subscription is £2.75 per fine. Which of these would you like more detail on?"
        _add_turn(session_id, "user", query)
        _add_turn(session_id, "assistant", answer)
        return {"answer": answer, "confidence": 1.0}

    # --- Off-topic guard ---
    if _is_off_topic(query):
        answer = "I can only help with Fine Flow questions. Is there anything about fines, pricing, appeals or the platform I can help with?"
        _add_turn(session_id, "user", query)
        _add_turn(session_id, "assistant", answer)
        return {"answer": answer, "confidence": 1.0}

    # --- Single-word / short topic shortcut ---
    if nq in _TOPIC_SHORTCUTS:
        query = _TOPIC_SHORTCUTS[nq]
        nq = _normalise(query)

    # --- Vague (genuinely unclear, not a topic shortcut) ---
    if _is_vague(query):
        last_ctx = _get_ctx(session_id)
        if last_ctx:
            topic = last_ctx.get("category", "this topic").replace("_", " ")
            answer = f"Could you be a bit more specific? I can go deeper on {topic}, or help with something else."
        else:
            answer = "Could you give me a bit more detail? I can help with fines, pricing, appeals, billing, drivers or the dashboard."
        _add_turn(session_id, "user", query)
        _add_turn(session_id, "assistant", answer)
        return {"answer": answer, "confidence": 1.0}

    # --- Affirmative follow-up ---
    if _is_affirmative(query):
        last_ctx = _get_ctx(session_id)
        if last_ctx:
            follow_up = last_ctx.get("follow_up")
            cat = last_ctx.get("category", "")

            if cat == "core_overview":
                answer = (
                    "Fine Flow works in stages: it monitors your Gmail inbox every minute, extracts fine details automatically using AI, "
                    "then matches each fine to the responsible driver using your vehicle logs. "
                    "From there, drivers confirm or dispute, and appeals are managed end to end. "
                    "Which stage would you like to go deeper on?"
                )
                _add_turn(session_id, "user", query)
                _add_turn(session_id, "assistant", answer)
                _set_ctx(session_id, None)
                return {"answer": answer, "confidence": 1.0}

            if follow_up:
                clean_fu = _clean(follow_up)
                _add_turn(session_id, "user", query)
                _add_turn(session_id, "assistant", clean_fu)
                _set_ctx(session_id, None)
                return {"answer": clean_fu, "confidence": 1.0}

        answer = "What would you like to know more about? Fines, pricing, appeals, billing or the dashboard?"
        _add_turn(session_id, "user", query)
        _add_turn(session_id, "assistant", answer)
        return {"answer": answer, "confidence": 1.0}

    # --- KB match ---
    _add_turn(session_id, "user", query)
    kb_entry = _kb_match(query)
    if kb_entry:
        # Shorten the answer to keep it concise — 3 sentences max
        raw_answer = _clean(kb_entry["answer"])
        short_answer = _shorten(raw_answer, max_sentences=3)
        follow_up_q = kb_entry.get("follow_up")

        if follow_up_q:
            display = f"{short_answer}\n\n{_clean(follow_up_q)}"
        else:
            display = short_answer

        _add_turn(session_id, "assistant", display)
        _set_ctx(session_id, kb_entry)
        return {"answer": display, "confidence": 1.0}

    # --- RAG retrieval ---
    docs: List[Dict[str, Any]] = []
    try:
        raw = rag_search(query, top_k=TOP_K)
        docs = rerank_hits(raw, query)
    except Exception:
        logger.exception("RAG search failed")

    strong_docs = [d for d in docs if d.get("score", 0) >= CONFIDENCE_THRESHOLD]
    context_chunks = [d["chunk"][:500] for d in strong_docs[:2]]
    context = "\n\n---\n\n".join(context_chunks) if context_chunks else ""

    # --- OpenAI fallback ---
    history = _get_history(session_id)
    history_without_last = history[:-1] if history else []

    messages: List[Dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    messages.extend(history_without_last[-8:])  # Last 4 turns only to keep context tight

    user_content = query
    if context:
        user_content = f"[Fine Flow context]\n{context}\n\n[User question]\n{query}"
    messages.append({"role": "user", "content": user_content})

    answer = _call_openai(messages, max_tokens=180)

    if not answer:
        answer = "That is a good question. Your Fine Flow dashboard or the support team at ff.sales@fineflow.com can help with anything account-specific."

    answer = _clean(answer)
    _add_turn(session_id, "assistant", answer)
    _set_ctx(session_id, None)

    return {
        "answer": answer,
        "confidence": strong_docs[0]["score"] if strong_docs else 0.3,
        "sources": [
            {"source": d["meta"].get("source", ""), "score": round(d["score"], 3)}
            for d in strong_docs[:2]
        ],
    }


# ---------------------------------------------------------------------------
# Public sync wrapper
# ---------------------------------------------------------------------------
def answer_sync(q: str, session_id: str = "default") -> Dict[str, Any]:
    try:
        return build_response(q, session_id)
    except Exception:
        logger.exception("Crash in answer_sync")
        return {"answer": "Something went wrong. Please try again.", "confidence": 0.0}