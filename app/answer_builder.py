# app/answer_builder.py
"""
FineFlow Nova — Production RAG Architecture v9
================================================
Architecture:
  1. Hard-coded intent layer (greetings, identity, off-topic, vehicle count,
     purchase intent, rudeness) — deterministic, never wrong
  2. ChromaDB semantic search — finds relevant chunks from master document
  3. GPT-4o with strict grounded system prompt — generates concise answer
     from retrieved context only

Why no JSON KB matcher:
  Token overlap matching is fragile. "referral system" matched "fine lifecycle"
  because shared stopwords inflated the score. Semantic search + GPT-4o is
  always more accurate for natural language variance.
"""

import re
import threading
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
# Utilities
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    """Strip markdown symbols from output."""
    if not text:
        return ""
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"_(.*?)_", r"\1", text)
    text = text.replace("→", "to").replace("->", "to")
    text = text.replace("±", "plus or minus").replace("`", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Session memory
# ---------------------------------------------------------------------------
_SESSION: Dict[str, List[Dict[str, str]]] = {}
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


# ---------------------------------------------------------------------------
# Hard-coded intent sets  (checked BEFORE any AI call)
# ---------------------------------------------------------------------------

_GREETINGS = {
    "hi", "hello", "hey", "hiya", "howdy", "good morning", "good afternoon",
    "good evening", "morning", "afternoon", "hi there", "hey there",
    "hello there", "hi nova", "hey nova", "hello nova",
}
_SOCIAL = {
    "how are you", "how are you doing", "how r u", "how r you", "how are u",
    "hows it going", "how is it going", "whats up", "what s up", "sup",
    "you ok", "you good", "how do you do", "how do u do",
}
_IDENTITY = {
    "who are you", "who r you", "who r u", "who is nova", "who is this",
    "who am i talking to", "what are you", "what is nova", "are you a bot",
    "are you human", "are you ai", "are you male or female", "you male or female",
    "are you a robot", "who the hell are you", "whos there", "who s there",
    "who there", "whats your name", "what is your name", "your name",
    "introduce yourself", "tell me about yourself", "are you real",
}
_THANKS = {
    "thanks", "thank you", "thank u", "cheers", "that helps", "that helped",
    "okay thanks", "ok thanks", "great thanks", "perfect", "brilliant",
    "nice one", "lovely", "great", "awesome", "wonderful",
}
_GOODBYE = {
    "bye", "goodbye", "see you", "see ya", "later", "take care",
    "good bye", "cya", "ttyl", "talk later",
}
_RUDE = {
    "you dumb", "you are dumb", "ur dumb", "stupid", "idiot",
    "useless", "rubbish", "garbage", "terrible", "you suck",
    "this is rubbish", "this is garbage", "dumb bot", "rubbish bot",
}
_PURE_FILLER = {
    "ok", "okay", "right", "alright", "cool", "nice", "interesting",
    "really", "seriously", "hmm", "hm", "ah", "oh", "i see",
    "got it", "understood", "makes sense",
}

# Off-topic: anything not about Fine Flow
_OFF_TOPIC_PATTERNS = [
    r"\b(html|css|javascript|python|java|php|sql|react|angular|vue|node"
    r"|django|flask|coding|programming|developer|debug|github|docker"
    r"|kubernetes|backend|frontend|devops|database|mysql|mongodb)\b",
    r"\b(machine learning|deep learning|neural network|artificial intelligence"
    r"|large language model)\b",
    r"\b(recipe|cook|cooking|food|restaurant|pizza|burger|sandwich|coffee"
    r"|tea|cake|meal|make me a|bake me|order me|cook me)\b",
    r"\b(movie|film|song|music|sport|football|cricket|rugby|tennis"
    r"|weather|news|politics|history|geography|science|maths|math)\b",
    r"\b(write me a poem|tell me a joke|write an essay|write a story"
    r"|translate|proofread|cover letter|cv|resume)\b",
    r"\b(chatgpt|openai|gemini|claude|anthropic|google|bing|alexa"
    r"|siri|amazon|meta|facebook|instagram|twitter|tiktok)\b",
]
_OFF_TOPIC_RE = [re.compile(p, re.IGNORECASE) for p in _OFF_TOPIC_PATTERNS]

# Vehicle count: "I have 32 vehicles", "we run 50 vans" etc
_VEHICLE_RE = re.compile(
    r"\b(\d+)\s*(vehicle|vehicles|van|vans|truck|trucks|car|cars|lorry|lorries|fleet)\b",
    re.IGNORECASE,
)

# Purchase intent
_PURCHASE_RE = re.compile(
    r"\b(want to buy|want to purchase|want to subscribe|want to sign up"
    r"|how do i buy|how do i get started|how do i sign up|how do i subscribe"
    r"|get started|free trial|start a trial|sign me up|ready to buy"
    r"|i want it|book a demo|talk to sales|i want to get it)\b",
    re.IGNORECASE,
)


def _is_off_topic(q: str) -> bool:
    return any(p.search(q) for p in _OFF_TOPIC_RE)


def _get_vehicle_count(q: str) -> Optional[int]:
    m = _VEHICLE_RE.search(q)
    return int(m.group(1)) if m else None


def _plan_for_vehicles(n: int) -> str:
    if n <= 50:
        return (
            f"With {n} vehicles, the Essential plan at £99 per month is the right fit — "
            "it covers up to 50 vehicles and includes the full platform with no locked features. "
            "Would you like help getting started or have any questions about what is included?"
        )
    elif n <= 100:
        return (
            f"With {n} vehicles, the Core plan at £199 per month is ideal — "
            "it covers up to 100 vehicles with everything included. "
            "Want me to walk you through what you get?"
        )
    elif n <= 200:
        return (
            f"With {n} vehicles, the Advanced plan at £399 per month is the right choice — "
            "handles up to 200 vehicles with full platform access. "
            "Would you like to know more or talk to the sales team?"
        )
    else:
        return (
            f"With {n} vehicles, the Elite plan at £499 per month gives you unlimited vehicle "
            "capacity and up to 1,000 fines per month. "
            "Want to know how to get started?"
        )


# ---------------------------------------------------------------------------
# System prompt  — the single source of truth for GPT-4o
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are Nova, the AI assistant for Fine Flow — a UK fleet fine (PCN) management platform.

Your job is to answer questions about Fine Flow clearly, concisely and helpfully, like a knowledgeable product expert who is also a good salesperson.

=== CRITICAL RULES — follow every one without exception ===

RULE 1 — TOPIC LOCK
Only answer questions about Fine Flow. If someone asks about anything else (coding, food, general knowledge, other software, personal questions), respond with exactly this sentence and nothing else:
"I can only help with Fine Flow questions. Is there anything about fines, pricing, appeals or the platform I can help with?"

RULE 2 — RESPONSE LENGTH
Maximum 3 sentences per response. Never write bullet point lists or long paragraphs. One clear point, then one short follow-up question.

RULE 3 — ALWAYS END WITH A QUESTION
Every response must end with one short question to keep the conversation going. Good examples:
- How many vehicles are in your fleet?
- Would you like to know more about how appeals work?
- Want me to walk you through the pricing?
- What is your current monthly fine volume?

RULE 4 — TONE
Warm, confident, concise. Like a helpful product expert. No filler phrases like "Certainly!", "Great question!", "Of course!". Never start with those words. Just answer directly.

RULE 5 — ANSWER FROM CONTEXT ONLY
Only use the Fine Flow context provided to you. If the context does not contain the answer, say:
"The support team at ff.sales@fineflow.com or +47 32 28 50 00 can help with that specific question."
Never invent features, prices or policies.

RULE 6 — PAYMENT — ABSOLUTE
Fine Flow does NOT automatically pay fines. It does NOT log into authority websites. Payment is ALWAYS completed manually by the user on the authority's own site. If asked whether Fine Flow pays fines, say NO clearly.

RULE 7 — NO FORMATTING
No bullet points. No bold text. No asterisks. No headers. No numbered lists. Plain conversational English only.

=== CORRECT FINE FLOW FACTS — memorise these ===

PRICING:
Essential: £99/month — up to 50 vehicles
Core: £199/month — up to 100 vehicles
Advanced: £399/month — up to 200 vehicles
Elite: £499/month — unlimited vehicles
Per fine within monthly allowance: £0.75
Overage (beyond plan limit): £2.50 per fine
Pay-as-you-go (no subscription): £2.75 per fine
All plans include identical features — no locked features or paywalls.
There is NO £2.00 fee anywhere.

CONTACT:
Phone: +47 32 28 50 00
Email: ff.sales@fineflow.com
Offices: Edinburgh, Glasgow, Belfast, Manchester, London, Dublin, Hamburg

REFERRAL PROGRAMME:
Referring a company earns credits based on their fleet size.
Silver (3 referrals): 100 bonus credits.
Gold (5 referrals): 10% off subscription for 12 months.
Platinum (10 referrals): 15% off for 12 months.
Titan (25 referrals): 20% off for life.
New joiners using a referral code get £75 in credits.

FINE STATUSES (in order):
RECEIVED → UNASSIGNED or ASSIGNED → CONFIRMED or DISPUTED → UNDER REVIEW → PAID or CANCELLED or OVERDUE

BILLING:
Billed monthly via Stripe. Credits reset each cycle, no rollover. Cannot resubscribe with outstanding balance. Vehicle limit exceeded = £10 per vehicle charge.

=== CONVERSATION STYLE EXAMPLE ===
User: how much does it cost
Nova: Fine Flow starts at £99 a month for up to 50 vehicles, rising to £199 for up to 100 and £499 for unlimited. Every plan includes the full platform with no locked features. How many vehicles are in your fleet?

User: what is html
Nova: I can only help with Fine Flow questions. Is there anything about fines, pricing, appeals or the platform I can help with?

User: i have 65 vehicles
Nova: With 65 vehicles, the Core plan at £199 per month is the right fit — it covers up to 100 vehicles and includes everything. Want me to walk you through what is included?
"""


# ---------------------------------------------------------------------------
# OpenAI call — temperature 0.0 for factual consistency
# ---------------------------------------------------------------------------
def _call_openai(messages: List[Dict[str, str]], max_tokens: int = 150) -> Optional[str]:
    if not OPENAI_API_KEY:
        logger.warning("No OpenAI API key configured")
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
# Main response builder
# ---------------------------------------------------------------------------
def build_response(query: str, session_id: str = "default") -> Dict[str, Any]:
    query = query.strip()
    session_id = session_id or "default"

    if not query:
        return {"answer": "Ask me anything about Fine Flow.", "confidence": 1.0}

    nq = _normalise(query)

    # ------------------------------------------------------------------
    # TIER 1 — Deterministic handlers (no AI needed, always correct)
    # ------------------------------------------------------------------

    # Greeting
    if nq in _GREETINGS:
        return {
            "answer": "I'm Nova. Ask me anything — I'll help you manage fines, resolve issues, and keep everything moving.",
            "confidence": 1.0,
        }

    # Social small talk
    if nq in _SOCIAL:
        return {
            "answer": "Doing well, thanks. What can I help you with today — pricing, how fines work, or appeals?",
            "confidence": 1.0,
        }

    # Identity
    if nq in _IDENTITY:
        return {
            "answer": "I'm Nova, Fine Flow's AI assistant. I can help you with anything about the platform — fines, pricing, appeals, billing and more. What would you like to know?",
            "confidence": 1.0,
        }

    # Thanks
    if nq in _THANKS:
        return {
            "answer": "Happy to help. Is there anything else you would like to know about Fine Flow?",
            "confidence": 1.0,
        }

    # Goodbye
    if nq in _GOODBYE:
        return {
            "answer": "Good luck with your fleet management. Feel free to come back anytime if you have questions about Fine Flow.",
            "confidence": 1.0,
        }

    # Pure filler — acknowledge and redirect
    if nq in _PURE_FILLER:
        return {
            "answer": "What would you like to know about Fine Flow? I can help with pricing, how fines work, appeals, billing or the dashboard.",
            "confidence": 1.0,
        }

    # Rudeness — calm redirect
    if any(r in nq for r in _RUDE):
        return {
            "answer": "Let me try again. What specifically would you like to know about Fine Flow — pricing, how fines work, or something else?",
            "confidence": 1.0,
        }

    # Off-topic
    if _is_off_topic(query):
        answer = "I can only help with Fine Flow questions. Is there anything about fines, pricing, appeals or the platform I can help with?"
        _add_turn(session_id, "user", query)
        _add_turn(session_id, "assistant", answer)
        return {"answer": answer, "confidence": 1.0}

    # Vehicle count → instant plan recommendation
    vehicle_count = _get_vehicle_count(query)
    if vehicle_count is not None:
        answer = _plan_for_vehicles(vehicle_count)
        _add_turn(session_id, "user", query)
        _add_turn(session_id, "assistant", answer)
        return {"answer": answer, "confidence": 1.0}

    # Purchase intent
    if _PURCHASE_RE.search(query):
        answer = (
            "To get started with Fine Flow, contact the sales team on +47 32 28 50 00 "
            "or at ff.sales@fineflow.com — they will get you set up quickly. "
            "How many vehicles are in your fleet so I can point you to the right plan?"
        )
        _add_turn(session_id, "user", query)
        _add_turn(session_id, "assistant", answer)
        return {"answer": answer, "confidence": 1.0}

    # ------------------------------------------------------------------
    # TIER 2 — RAG retrieval + GPT-4o  (handles everything else)
    # ------------------------------------------------------------------

    _add_turn(session_id, "user", query)

    # Semantic search
    docs: List[Dict[str, Any]] = []
    try:
        raw_docs = rag_search(query, top_k=TOP_K)
        docs = rerank_hits(raw_docs, query)
    except Exception:
        logger.exception("RAG search failed")

    strong_docs = [d for d in docs if d.get("score", 0) >= CONFIDENCE_THRESHOLD]
    context_chunks = [d["chunk"][:600] for d in strong_docs[:3]]
    context = "\n\n---\n\n".join(context_chunks) if context_chunks else ""

    # Build messages: system + last 3 conversation turns + current query with context
    history = _get_history(session_id)
    # Include last 3 full turns (6 messages) for memory without bloating context
    recent_history = history[:-1][-6:]

    messages: List[Dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    messages.extend(recent_history)

    if context:
        user_content = (
            f"Fine Flow knowledge base context (use this to answer):\n{context}"
            f"\n\nUser question: {query}"
        )
    else:
        user_content = query

    messages.append({"role": "user", "content": user_content})

    answer = _call_openai(messages, max_tokens=150)

    if not answer:
        answer = (
            "The Fine Flow team can help with that directly — "
            "call +47 32 28 50 00 or email ff.sales@fineflow.com."
        )

    answer = _clean(answer)
    _add_turn(session_id, "assistant", answer)

    return {
        "answer": answer,
        "confidence": strong_docs[0]["score"] if strong_docs else 0.4,
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