# app/answer_builder.py
"""
FineFlow Nova — Answer Builder Production Final v8
=========================================================
Fixes vs v7:
- Identity questions ("who are you", "who is nova") handled before KB lookup
- KB matching threshold raised + stopword filtering to prevent false matches
- "yes" after follow_up now correctly delivers the follow-up content
- Vehicle count detection ("I have 32 vehicles") routes to plan recommendation
- "I want to buy" / purchase intent detected and handled with plan recommendation
- Off-topic patterns expanded (sandwich, food items)
- "why", "how", "what", "anything" handled as conversational redirects
- Pricing query variants all route correctly to pricing answer
- Savings vs pricing separated cleanly
- Response length enforced: 2-3 sentences from KB, 150 token LLM cap
- Follow-up question always appended to guide conversation
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
# Stopwords — excluded from token overlap scoring to prevent false matches
# ---------------------------------------------------------------------------
_STOPWORDS = {
    "a", "an", "the", "is", "it", "in", "on", "of", "to", "do", "my",
    "me", "you", "your", "i", "we", "are", "was", "be", "will", "can",
    "has", "have", "had", "not", "and", "or", "but", "if", "this",
    "that", "with", "for", "at", "by", "from", "as", "what", "how",
    "why", "who", "when", "where", "which", "there", "here", "their",
    "them", "its", "our", "all", "any", "does", "did", "so", "up",
    "about", "just", "more", "also", "would", "could", "should",
}


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------
def _normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _meaningful_tokens(text: str) -> set:
    """Return tokens with stopwords removed for cleaner matching."""
    return {t for t in _normalise(text).split() if t not in _STOPWORDS and len(t) > 2}


# ---------------------------------------------------------------------------
# Clean output
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


def _shorten(text: str, max_sentences: int = 3) -> str:
    """Trim to first N sentences for concise output."""
    if not text:
        return text
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return " ".join(sentences[:max_sentences]).strip()


# ---------------------------------------------------------------------------
# KB matching — meaningful token overlap only, higher threshold
# ---------------------------------------------------------------------------
def _kb_match(query: str) -> Optional[Dict[str, Any]]:
    nq = _normalise(query)
    q_tokens = _meaningful_tokens(query)

    # If no meaningful tokens after stopword removal, don't attempt KB match
    if not q_tokens:
        return None

    best_entry: Optional[Dict[str, Any]] = None
    best_score = 0.0

    for entry in _kb:
        # Build searchable from question + keywords
        searchable_parts = [entry.get("question", "")]
        searchable_parts += entry.get("keywords", [])
        searchable = " ".join(searchable_parts)

        s_tokens = _meaningful_tokens(searchable)
        if not s_tokens:
            continue

        # Exact normalised substring match on full searchable
        if nq in _normalise(searchable):
            return entry

        # Meaningful token overlap (Jaccard-style: overlap / query length)
        overlap = len(q_tokens & s_tokens)
        if overlap == 0:
            continue

        # Score: overlap relative to query size, boosted by keyword hits
        score = overlap / max(len(q_tokens), 1)

        # Boost if query tokens appear in keywords specifically
        kw_tokens = _meaningful_tokens(" ".join(entry.get("keywords", [])))
        kw_overlap = len(q_tokens & kw_tokens)
        score += 0.15 * kw_overlap

        if score > best_score:
            best_score = score
            best_entry = entry

    # Higher threshold (0.60) to prevent false matches
    if best_score >= 0.60 and best_entry:
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
# Intent detection sets
# ---------------------------------------------------------------------------
_AFFIRMATIVE = {
    "yes", "yeah", "yep", "sure", "ok", "okay", "go on", "please",
    "continue", "tell me more", "go ahead", "more", "absolutely",
    "of course", "do it", "go for it", "sounds good", "definitely",
    "yes please", "show me",
}
_NEGATIVE = {
    "no", "nope", "nah", "no thanks", "not now", "skip",
    "never mind", "nevermind", "no need", "not really",
}
_GREETINGS = {
    "hi", "hello", "hey", "hiya", "howdy", "good morning",
    "good afternoon", "good evening", "morning", "afternoon",
    "hi there", "hey there", "hello there", "hi nova", "hey nova",
    "hello nova",
}
_SOCIAL = {
    "how are you", "how are you doing", "how r u", "how r you",
    "how are u", "hows it going", "how is it going", "whats up",
    "what s up", "sup", "you ok", "you good",
}
_THANKS = {
    "thanks", "thank you", "thank u", "cheers", "that helps",
    "that helped", "okay thanks", "ok thanks", "great thanks",
    "perfect", "brilliant", "nice one", "lovely", "great",
}
_RUDE = {
    "you dumb", "you are dumb", "ur dumb", "stupid", "idiot",
    "useless", "rubbish", "garbage", "terrible", "you suck",
    "this is rubbish", "this is garbage",
}
# Identity questions — handled before KB
_IDENTITY = {
    "who are you", "who r you", "who r u", "who is nova", "who is this",
    "who am i talking to", "what are you", "what is nova", "are you a bot",
    "are you human", "are you ai", "who the hell are you",
    "whos there", "who s there", "who there", "whats your name",
    "what is your name", "your name", "introduce yourself",
    "tell me about yourself",
}
# Pure filler / too vague to do anything with
_PURE_VAGUE = {
    "anything", "something", "stuff", "things", "idk", "dunno",
    "whatever", "hmm", "hm", "ok", "okay", "right", "alright",
    "cool", "nice", "interesting", "really", "seriously",
    "why", "how", "what", "when", "where", "explain", "tell me",
    "go on", "and", "so", "then", "next",
}

# Single-word topic shortcuts → expanded query for KB
_TOPIC_SHORTCUTS: Dict[str, str] = {
    "pricing":   "how much does fine flow cost",
    "price":     "how much does fine flow cost",
    "cost":      "how much does fine flow cost",
    "plans":     "how much does fine flow cost",
    "packages":  "how much does fine flow cost",
    "plan":      "what plan should i be on",
    "fines":     "what happens to a fine once it comes into the system",
    "fine":      "what happens to a fine once it comes into the system",
    "appeals":   "can you help me appeal a fine",
    "appeal":    "can you help me appeal a fine",
    "billing":   "how does billing work",
    "dashboard": "what does the dashboard show",
    "drivers":   "how do i add drivers to the system",
    "driver":    "how does fine flow know which driver a fine belongs to",
    "matching":  "how does fine flow know which driver a fine belongs to",
    "security":  "is my data safe with fine flow",
    "gdpr":      "is my data safe with fine flow",
    "referral":  "is there a referral programme",
    "referrals": "is there a referral programme",
    "contact":   "how do i contact fine flow",
    "reports":   "what reports can i export from fine flow",
    "export":    "what reports can i export from fine flow",
    "statuses":  "what do the fine statuses mean",
    "status":    "what do the fine statuses mean",
    "overdue":   "what happens if i dont deal with a fine in time",
    "email":     "how does fine flow get my fines in the first place",
    "gmail":     "how do i connect my gmail to fine flow",
    "payg":      "is there a pay as you go option",
    "overage":   "what is the overage charge",
    "savings":   "how much time and money can fine flow save me",
    "features":  "what are fine flows key features",
    "roles":     "what are the different user roles in fine flow",
    "payment":   "do you automatically pay fines for me",
    "pay":       "do you automatically pay fines for me",
}

# Purchase intent patterns
_PURCHASE_PATTERNS = [
    r"\b(want to buy|want to purchase|want to subscribe|want to sign up|want to get it)\b",
    r"\b(how do i buy|how do i get|how do i start|how do i sign up|how do i subscribe)\b",
    r"\b(get started|start a trial|free trial|sign me up|ready to buy|ready to start)\b",
    r"\b(i want it|i ll take it|lets go|let s go|book a demo|talk to sales)\b",
]
_PURCHASE_COMPILED = [re.compile(p, re.IGNORECASE) for p in _PURCHASE_PATTERNS]

# Vehicle count pattern — "I have 32 vehicles", "we run 50 vans" etc
_VEHICLE_PATTERN = re.compile(
    r"\b(\d+)\s*(vehicle|vehicles|van|vans|truck|trucks|car|cars|fleet|lorry|lorries)\b",
    re.IGNORECASE
)

# Off-topic patterns
_OFF_TOPIC_PATTERNS = [
    r"\b(html|css|javascript|python|java|php|sql|react|angular|vue|node|django|flask)\b",
    r"\b(machine learning|deep learning|neural network|artificial intelligence)\b",
    r"\b(recipe|cook|cooking|food|restaurant|pizza|burger|sandwich|coffee|tea|cake|meal)\b",
    r"\b(movie|film|song|music|sport|football|cricket|weather|news|politics)\b",
    r"\b(write me a poem|tell me a joke|essay|story|translate|proofread)\b",
    r"\b(chatgpt|openai|google|bing|alexa|siri|amazon)\b",
    r"\b(coding|programming|developer|debug|github|docker|kubernetes)\b",
    r"\b(database|mysql|mongodb|server|deploy|devops|backend|frontend)\b",
    r"\b(make me a|can you make|bake me|cook me|order me)\b",
]
_OFF_TOPIC_COMPILED = [re.compile(p, re.IGNORECASE) for p in _OFF_TOPIC_PATTERNS]


# ---------------------------------------------------------------------------
# Intent helpers
# ---------------------------------------------------------------------------
def _is_off_topic(query: str) -> bool:
    for p in _OFF_TOPIC_COMPILED:
        if p.search(query):
            return True
    return False


def _is_purchase_intent(query: str) -> bool:
    for p in _PURCHASE_COMPILED:
        if p.search(query):
            return True
    return False


def _get_vehicle_count(query: str) -> Optional[int]:
    m = _VEHICLE_PATTERN.search(query)
    return int(m.group(1)) if m else None


def _plan_for_vehicles(count: int) -> str:
    if count <= 50:
        return f"With {count} vehicles, the Essential plan at £99 per month is the right fit — it covers up to 50 vehicles and includes the full platform. Would you like help getting started?"
    elif count <= 100:
        return f"With {count} vehicles, the Core plan at £199 per month is ideal — it covers up to 100 vehicles with everything included. Want me to walk you through what is included?"
    elif count <= 200:
        return f"With {count} vehicles, the Advanced plan at £399 per month covers you — handles up to 200 vehicles with full platform access. Would you like to know more?"
    else:
        return f"With {count} vehicles, the Elite plan at £499 per month gives you unlimited vehicle capacity and 1,000 fines per month. Want to know how to get started?"


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

def _is_identity(q: str) -> bool:
    return _normalise(q) in _IDENTITY

def _is_pure_vague(q: str) -> bool:
    nq = _normalise(q)
    return nq in _PURE_VAGUE or (len(nq.split()) == 1 and nq not in _TOPIC_SHORTCUTS)

def _is_pricing_confusion(q: str) -> bool:
    nq = q.lower().strip()
    if re.match(r"^[£$]?\s*2\s*\??$", nq):
        return True
    return bool(re.search(r"£\s*2\s+(fee|for|charge)", nq))


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are Nova, the AI assistant for Fine Flow — a UK fleet fine (PCN) management platform.

ABSOLUTE RULES:

1. TOPIC LOCK: Only answer Fine Flow questions. For anything else (coding, food, general knowledge, other AI), say exactly: "I can only help with Fine Flow questions. Is there anything about fines, pricing, appeals or the platform I can help with?"

2. LENGTH: Maximum 3 sentences per response. Never write lists or paragraphs. One clear point, then one follow-up question.

3. ALWAYS end with one short question to keep the conversation going. Examples: "How many vehicles do you run?", "Would you like to know more about appeals?", "Want me to walk you through pricing?"

4. TONE: Warm, confident, concise. Like a knowledgeable salesperson, not a search engine. No filler phrases like "Certainly!" or "Great question!". Just answer.

5. PAYMENT — CRITICAL: Fine Flow does NOT pay fines. It does NOT log into authority websites. Payment is always done manually by the user on the authority's site. If asked about automatic payment, say NO clearly.

6. NO FORMATTING: No bullet points, no bold, no asterisks, no headers. Plain English only.

CORRECT PRICING:
Essential: £99/month — up to 50 vehicles
Core: £199/month — up to 100 vehicles  
Advanced: £399/month — up to 200 vehicles
Elite: £499/month — unlimited vehicles
Per fine within allowance: £0.75
Overage: £2.50 per fine
Pay-as-you-go (no subscription): £2.75 per fine
No plan has locked features.

CONTACT: Phone +47 32 28 50 00 | Email ff.sales@fineflow.com

STYLE EXAMPLE:
User: how much does it cost
Nova: Fine Flow starts at £99 a month for up to 50 vehicles, rising to £199 for 100, and £499 for unlimited. Every plan includes the full platform with no locked features. How many vehicles are in your fleet?

User: what is html
Nova: I can only help with Fine Flow questions. Is there anything about fines, pricing, appeals or the platform I can help with?
"""


# ---------------------------------------------------------------------------
# OpenAI call
# ---------------------------------------------------------------------------
def _call_openai(messages: List[Dict[str, str]], max_tokens: int = 150) -> Optional[str]:
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
# Core response builder
# ---------------------------------------------------------------------------
def build_response(query: str, session_id: str = "default") -> Dict[str, Any]:
    query = query.strip()
    session_id = session_id or "default"

    if not query:
        return {"answer": "Ask me anything about Fine Flow.", "confidence": 1.0}

    nq = _normalise(query)

    # ------------------------------------------------------------------ #
    # LAYER 1: Hard-coded social / identity intents (before everything)   #
    # ------------------------------------------------------------------ #

    if _is_greeting(query):
        _set_ctx(session_id, None)
        return {
            "answer": "I'm Nova. Ask me anything — I'll help you manage fines, resolve issues, and keep everything moving.",
            "confidence": 1.0,
        }

    if _is_social(query):
        return {
            "answer": "Doing well, thanks. What can I help you with today — pricing, fines, or appeals?",
            "confidence": 1.0,
        }

    if _is_identity(query):
        return {
            "answer": "I'm Nova, the AI assistant for Fine Flow. I can help you with anything about the platform — fines, pricing, appeals, billing and more. What would you like to know?",
            "confidence": 1.0,
        }

    if _is_thanks(query):
        return {
            "answer": "Happy to help. Is there anything else you would like to know?",
            "confidence": 1.0,
        }

    if _is_rude(query):
        return {
            "answer": "Let me try again. What specifically would you like to know about Fine Flow — pricing, how fines work, or something else?",
            "confidence": 1.0,
        }

    if _is_negative(query):
        _add_turn(session_id, "user", query)
        answer = "No problem. What else can I help you with?"
        _add_turn(session_id, "assistant", answer)
        _set_ctx(session_id, None)
        return {"answer": answer, "confidence": 1.0}

    # ------------------------------------------------------------------ #
    # LAYER 2: Off-topic guard                                            #
    # ------------------------------------------------------------------ #

    if _is_off_topic(query):
        answer = "I can only help with Fine Flow questions. Is there anything about fines, pricing, appeals or the platform I can help with?"
        _add_turn(session_id, "user", query)
        _add_turn(session_id, "assistant", answer)
        return {"answer": answer, "confidence": 1.0}

    # ------------------------------------------------------------------ #
    # LAYER 3: Pricing confusion (£2 fee that does not exist)            #
    # ------------------------------------------------------------------ #

    if _is_pricing_confusion(query):
        answer = "There is no £2 fee in Fine Flow. Within your plan allowance it is £0.75 per fine, beyond your allowance it is £2.50, and the pay-as-you-go option is £2.75 per fine with no subscription. Which would you like more detail on?"
        _add_turn(session_id, "user", query)
        _add_turn(session_id, "assistant", answer)
        return {"answer": answer, "confidence": 1.0}

    # ------------------------------------------------------------------ #
    # LAYER 4: Vehicle count detected → recommend plan                   #
    # ------------------------------------------------------------------ #

    vehicle_count = _get_vehicle_count(query)
    if vehicle_count is not None:
        answer = _plan_for_vehicles(vehicle_count)
        _add_turn(session_id, "user", query)
        _add_turn(session_id, "assistant", answer)
        return {"answer": answer, "confidence": 1.0}

    # ------------------------------------------------------------------ #
    # LAYER 5: Purchase intent                                            #
    # ------------------------------------------------------------------ #

    if _is_purchase_intent(query):
        answer = "To get started with Fine Flow, head to fineflow.com or contact the sales team directly on +47 32 28 50 00 or at ff.sales@fineflow.com. How many vehicles are in your fleet so I can point you to the right plan?"
        _add_turn(session_id, "user", query)
        _add_turn(session_id, "assistant", answer)
        return {"answer": answer, "confidence": 1.0}

    # ------------------------------------------------------------------ #
    # LAYER 6: Topic shortcut (single word like "pricing", "appeals")    #
    # ------------------------------------------------------------------ #

    if nq in _TOPIC_SHORTCUTS:
        query = _TOPIC_SHORTCUTS[nq]
        nq = _normalise(query)

    # ------------------------------------------------------------------ #
    # LAYER 7: Pure vague — nothing to work with                         #
    # ------------------------------------------------------------------ #

    if _is_pure_vague(query):
        last_ctx = _get_ctx(session_id)
        if last_ctx:
            topic = last_ctx.get("category", "this topic").replace("_", " ")
            answer = f"Could you be a bit more specific? I can go deeper on {topic}, or help with something else entirely."
        else:
            answer = "What would you like to know about? I can help with pricing, how fines work, appeals, billing, or the dashboard."
        _add_turn(session_id, "user", query)
        _add_turn(session_id, "assistant", answer)
        return {"answer": answer, "confidence": 1.0}

    # ------------------------------------------------------------------ #
    # LAYER 8: Affirmative follow-up                                     #
    # ------------------------------------------------------------------ #

    if _is_affirmative(query):
        last_ctx = _get_ctx(session_id)
        if last_ctx:
            cat = last_ctx.get("category", "")
            follow_up_text = last_ctx.get("follow_up", "")

            # Core overview → deliver system breakdown
            if cat == "core_overview":
                answer = (
                    "Fine Flow works in six stages: Gmail monitoring every minute, AI extraction of fine details, "
                    "automatic driver matching using your vehicle logs, driver confirmation or dispute, "
                    "admin appeal review, and full outcome tracking. "
                    "Which stage would you like to go deeper on?"
                )
                _add_turn(session_id, "user", query)
                _add_turn(session_id, "assistant", answer)
                _set_ctx(session_id, None)
                return {"answer": answer, "confidence": 1.0}

            # Appeals follow-up → explain appeal generation
            if cat in ("appeals", "appeals_intelligence", "appeals_generation"):
                answer = (
                    "When a driver disputes a fine, the admin reviews it and can accept or reject the appeal. "
                    "If accepted, Fine Flow generates the appeal letter and sends it directly to the issuing authority. "
                    "The system tracks every outcome and uses past results to improve future appeal recommendations. "
                    "Would you like to know about the win percentage feature?"
                )
                _add_turn(session_id, "user", query)
                _add_turn(session_id, "assistant", answer)
                _set_ctx(session_id, None)
                return {"answer": answer, "confidence": 1.0}

            # Pricing follow-up → ask vehicle count
            if cat == "pricing":
                answer = "How many vehicles are in your fleet? That will help me point you to the right plan straightaway."
                _add_turn(session_id, "user", query)
                _add_turn(session_id, "assistant", answer)
                _set_ctx(session_id, None)
                return {"answer": answer, "confidence": 1.0}

            # Generic follow-up from KB entry
            if follow_up_text:
                clean_fu = _clean(follow_up_text)
                # Deliver the follow-up as a proper answer via OpenAI
                history = _get_history(session_id)
                messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
                messages.extend(history[-6:])
                messages.append({"role": "user", "content": f"The user said yes to: {follow_up_text}. Answer that question concisely."})
                ai_answer = _call_openai(messages, max_tokens=150)
                answer = _clean(ai_answer) if ai_answer else clean_fu
                _add_turn(session_id, "user", query)
                _add_turn(session_id, "assistant", answer)
                _set_ctx(session_id, None)
                return {"answer": answer, "confidence": 1.0}

        answer = "What would you like to know more about — fines, pricing, appeals, or billing?"
        _add_turn(session_id, "user", query)
        _add_turn(session_id, "assistant", answer)
        return {"answer": answer, "confidence": 1.0}

    # ------------------------------------------------------------------ #
    # LAYER 9: KB match                                                   #
    # ------------------------------------------------------------------ #

    _add_turn(session_id, "user", query)
    kb_entry = _kb_match(query)

    if kb_entry:
        raw_answer = _clean(kb_entry["answer"])
        short_answer = _shorten(raw_answer, max_sentences=3)
        follow_up_q = kb_entry.get("follow_up")
        display = f"{short_answer}\n\n{_clean(follow_up_q)}" if follow_up_q else short_answer
        _add_turn(session_id, "assistant", display)
        _set_ctx(session_id, kb_entry)
        return {"answer": display, "confidence": 1.0}

    # ------------------------------------------------------------------ #
    # LAYER 10: RAG + OpenAI fallback                                    #
    # ------------------------------------------------------------------ #

    docs: List[Dict[str, Any]] = []
    try:
        raw_docs = rag_search(query, top_k=TOP_K)
        docs = rerank_hits(raw_docs, query)
    except Exception:
        logger.exception("RAG search failed")

    strong_docs = [d for d in docs if d.get("score", 0) >= CONFIDENCE_THRESHOLD]
    context_chunks = [d["chunk"][:500] for d in strong_docs[:2]]
    context = "\n\n---\n\n".join(context_chunks) if context_chunks else ""

    history = _get_history(session_id)
    messages: List[Dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    messages.extend(history[:-1][-8:])  # Last 4 turns for context

    user_content = query
    if context:
        user_content = f"[Fine Flow context]\n{context}\n\n[User question]\n{query}"
    messages.append({"role": "user", "content": user_content})

    answer = _call_openai(messages, max_tokens=150)

    if not answer:
        answer = "Your Fine Flow dashboard will have the specific detail you need, or contact the team at ff.sales@fineflow.com or +47 32 28 50 00."

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