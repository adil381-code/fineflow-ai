# app/answer_builder.py
"""
FineFlow Nova — Production Final v10
======================================
Architecture:
  Tier 1 — Deterministic intent handlers (greetings, identity, off-topic,
            vehicle count, purchase intent, filler) — no AI, always correct.
  Tier 2 — RAG (ChromaDB semantic search) + GPT-4o with strict system prompt.
            Session history passed on every call for real memory.

Key fixes vs v9:
  - Bare number ("85") after vehicle context detected via session memory
  - "will it pay fines automatically" → CRITICAL rule enforced via few-shot example in prompt
  - "referral system" → RAG now finds correct chunk (restructured KB doc)
  - "yes" after recommendation → session stores last_topic, GPT-4o uses it
  - Off-topic regex no longer fires on "referral system" or similar Fine Flow terms
  - All session context passed to every GPT-4o call
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
    if not text:
        return ""
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"_(.*?)_", r"\1", text)
    text = text.replace("→", "to").replace("->", "to")
    text = text.replace("±", "plus or minus").replace("`", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _norm(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Session memory  — stores conversation history + last assistant topic
# ---------------------------------------------------------------------------
_SESSION: Dict[str, List[Dict[str, str]]] = {}
_SESSION_META: Dict[str, Dict[str, Any]] = {}   # stores last_topic, awaiting_vehicle_count
_LOCK = threading.Lock()


def _get_history(sid: str) -> List[Dict[str, str]]:
    with _LOCK:
        return list(_SESSION.get(sid, []))


def _add_turn(sid: str, role: str, content: str) -> None:
    with _LOCK:
        hist = _SESSION.setdefault(sid, [])
        hist.append({"role": role, "content": content})
        keep = CHAT_HISTORY_TURNS * 2
        if len(hist) > keep:
            _SESSION[sid] = hist[-keep:]


def _set_meta(sid: str, key: str, value: Any) -> None:
    with _LOCK:
        _SESSION_META.setdefault(sid, {})[key] = value


def _get_meta(sid: str, key: str) -> Any:
    with _LOCK:
        return _SESSION_META.get(sid, {}).get(key)


def _clear_meta(sid: str) -> None:
    with _LOCK:
        _SESSION_META[sid] = {}


# ---------------------------------------------------------------------------
# Intent sets — all normalised, checked with exact match
# ---------------------------------------------------------------------------

_GREETINGS = {
    "hi", "hello", "hey", "hiya", "howdy", "good morning", "good afternoon",
    "good evening", "morning", "afternoon", "hi there", "hey there",
    "hello there", "hi nova", "hey nova", "hello nova", "yo", "sup",
}
_SOCIAL = {
    "how are you", "how are you doing", "how r u", "how r you", "how are u",
    "hows it going", "how is it going", "whats up", "what s up",
    "you ok", "you good", "how do you do",
}
_IDENTITY = {
    "who are you", "who r you", "who r u", "who is nova", "who is this",
    "who am i talking to", "what are you", "what is nova", "are you a bot",
    "are you human", "are you ai", "are you male or female",
    "you male or female", "are you a robot", "who the hell are you",
    "whos there", "who s there", "who there", "whats your name",
    "what is your name", "your name", "introduce yourself",
    "tell me about yourself", "are you real", "are you a person",
}
_THANKS = {
    "thanks", "thank you", "thank u", "cheers", "that helps", "that helped",
    "okay thanks", "ok thanks", "great thanks", "perfect", "brilliant",
    "nice one", "lovely", "great", "awesome", "wonderful", "thank you so much",
    "many thanks",
}
_GOODBYE = {
    "bye", "goodbye", "see you", "see ya", "later", "take care",
    "good bye", "cya", "ttyl", "talk later", "farewell",
}
_RUDE = {
    "you dumb", "you are dumb", "ur dumb", "stupid", "idiot",
    "useless", "rubbish", "garbage", "terrible", "you suck",
    "this is rubbish", "this is garbage", "dumb bot", "rubbish bot",
    "you re useless", "ur useless", "waste of time",
}
# Single-word pure filler — acknowledge and redirect
_PURE_FILLER = {
    "ok", "okay", "right", "alright", "cool", "nice", "interesting",
    "really", "seriously", "hmm", "hm", "ah", "oh", "i see",
    "got it", "understood", "makes sense", "noted",
}

# ---------------------------------------------------------------------------
# Off-topic: ONLY non-Fine-Flow topics. Must NOT catch Fine Flow terminology.
# ---------------------------------------------------------------------------
_OFF_TOPIC_PATTERNS = [
    # Programming languages and tools
    r"\b(html|css|javascript|typescript|python|java|php|sql|react|angular"
    r"|vue|node\.?js|django|flask|laravel|rails|spring|kubernetes|docker"
    r"|github|git|devops|backend|frontend|fullstack|api development)\b",
    # Generic AI/ML (not Fine Flow AI features)
    r"\b(machine learning|deep learning|neural network|large language model"
    r"|generative ai|train a model|fine.?tune|llm|gpt|bert)\b",
    # Food
    r"\b(recipe|cook|cooking|food|restaurant|pizza|burger|sandwich|coffee"
    r"|tea|cake|meal|bake|order food|make me a|bake me|cook me)\b",
    # Entertainment / general knowledge
    r"\b(movie|film|song|music|football|cricket|rugby|tennis|golf"
    r"|weather forecast|news today|politics|history lesson|geography"
    r"|capital city|who invented|when was .+ born)\b",
    # Writing tasks unrelated to Fine Flow
    r"\b(write me a poem|tell me a joke|write an essay|write a story"
    r"|translate this|proofread|cover letter|write my cv|write my resume)\b",
    # Competitor AI products
    r"\b(chatgpt|openai|gemini|claude|anthropic|google bard|bing ai"
    r"|alexa|siri|cortana|amazon echo)\b",
]
_OFF_TOPIC_RE = [re.compile(p, re.IGNORECASE) for p in _OFF_TOPIC_PATTERNS]


def _is_off_topic(q: str) -> bool:
    return any(p.search(q) for p in _OFF_TOPIC_RE)


# ---------------------------------------------------------------------------
# Vehicle count detection — handles "65 vehicles", "I have 85", "we run 50"
# ---------------------------------------------------------------------------
_VEHICLE_EXPLICIT_RE = re.compile(
    r"\b(\d+)\s*(vehicle|vehicles|van|vans|truck|trucks|car|cars"
    r"|lorry|lorries|fleet|in my fleet|in our fleet)\b",
    re.IGNORECASE,
)
# Also catches "I have 85" / "we have 65" / "we run 50" when context exists
_NUMBER_ONLY_RE = re.compile(r"^\s*(\d+)\s*$")
_HAVE_NUMBER_RE = re.compile(
    r"\b(i have|we have|we run|we got|i got|i manage|we manage|thats|that s|its)\s+(\d+)\b",
    re.IGNORECASE,
)

# Purchase intent
_PURCHASE_RE = re.compile(
    r"\b(want to buy|want to purchase|want to subscribe|want to sign up"
    r"|how do i buy|how do i get started|how do i sign up|how to subscribe"
    r"|get started|free trial|start a trial|sign me up|ready to buy"
    r"|i want it|book a demo|talk to sales|i want to get it|how to start)\b",
    re.IGNORECASE,
)


def _extract_vehicle_count(query: str, session_id: str) -> Optional[int]:
    """Extract vehicle count from query, or from bare number if context implies it."""
    # Explicit mention: "65 vehicles", "we run 85 vans"
    m = _VEHICLE_EXPLICIT_RE.search(query)
    if m:
        return int(m.group(1))

    # "I have 85" or "we have 65"
    m = _HAVE_NUMBER_RE.search(query)
    if m:
        return int(m.group(2))

    # Bare number like "85" — only if we were recently discussing vehicles/pricing
    m = _NUMBER_ONLY_RE.match(query)
    if m:
        last_topic = _get_meta(session_id, "last_topic")
        if last_topic in ("pricing", "vehicles", "plan_recommendation"):
            return int(m.group(1))

    return None


def _plan_for_vehicles(n: int) -> tuple:
    """Return (answer, topic) for a given vehicle count."""
    if n <= 50:
        answer = (
            f"With {n} vehicles, the Essential plan at £99 per month is the right fit — "
            "covers up to 50 vehicles and includes the full platform with no locked features. "
            "Would you like help getting started or have any questions about what is included?"
        )
    elif n <= 100:
        answer = (
            f"With {n} vehicles, the Core plan at £199 per month is ideal — "
            "covers up to 100 vehicles with everything included. "
            "Want me to walk you through what is included?"
        )
    elif n <= 200:
        answer = (
            f"With {n} vehicles, the Advanced plan at £399 per month is the right choice — "
            "handles up to 200 vehicles with full platform access. "
            "Would you like to know more or talk to the sales team?"
        )
    else:
        answer = (
            f"With {n} vehicles, the Elite plan at £499 per month is built for you — "
            "unlimited vehicles and up to 1,000 fines per month included. "
            "Want to know how to get started?"
        )
    return answer, "plan_recommendation"


# ---------------------------------------------------------------------------
# System prompt — single source of truth for GPT-4o
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are Nova, Fine Flow's AI assistant. Fine Flow is a UK fleet fine (PCN) management platform.

== ABSOLUTE RULES ==

1. TOPIC LOCK
Only answer Fine Flow questions. For anything unrelated (coding, food, general knowledge, other AI products), respond with exactly:
"I can only help with Fine Flow questions. Is there anything about fines, pricing, appeals or the platform I can help with?"

2. LENGTH
3 sentences maximum. No bullet lists. No long paragraphs. One clear point, then one question.

3. END WITH A QUESTION
Every response must end with a short question. Examples:
"How many vehicles are in your fleet?"
"Would you like to know more about appeals?"
"Want me to walk you through what is included?"
"What is your current monthly fine volume?"

4. TONE
Warm, direct, confident. Like a knowledgeable product expert. No filler: never start with "Certainly!", "Great question!", "Of course!". Just answer.

5. NO FORMATTING
No bullet points. No bold. No asterisks. No numbered lists. Plain English only.

6. ANSWER FROM CONTEXT
Use the Fine Flow context provided in each message. If context is missing or unclear, use the facts below. Never invent features or prices.

== PAYMENT — READ THIS CAREFULLY ==
Fine Flow does NOT automatically pay fines. Fine Flow NEVER logs into council or authority websites.
Payment portals use bot protection and card verification that makes automation impossible.
Fine Flow does everything except the final payment click — it captures, assigns, tracks and organises everything so payment is fast and simple when YOU do it on the authority's site.
If anyone asks whether Fine Flow pays fines automatically, always say NO.

== CORRECT FINE FLOW FACTS ==

PRICING:
Essential: £99/month — up to 50 vehicles
Core: £199/month — up to 51 to 100 vehicles
Advanced: £399/month — up to 101 to 200 vehicles
Elite: £499/month — unlimited (200+ vehicles)
Per fine within allowance: £0.75
Overage (over monthly limit): £2.50 per fine
Pay-as-you-go (no subscription): £2.75 per fine
No plan has locked features. All plans include the full platform.
There is NO £2.00 fee.

REFERRAL PROGRAMME (yes Fine Flow has one):
Earn credits when a referred company subscribes — 100 credits for 1-25 vehicles, 250 for 26-100, 750 for 101-500, 2000 for 500+.
Tiers: Silver (3 referrals) = 100 bonus credits; Gold (5) = 10% off 12 months; Platinum (10) = 15% off 12 months; Titan (25) = 20% off for life.
New joiners using a referral code get £75 in credits.

CONTACT:
Phone: +47 32 28 50 00
Email: ff.sales@fineflow.com
Offices: Edinburgh, Glasgow, Belfast, Manchester, London, Dublin, Hamburg

BILLING:
Monthly via Stripe. Credits reset each cycle, no rollover. Cannot resubscribe with outstanding balance. £10/vehicle if vehicle limit exceeded.

== EXAMPLES OF CORRECT RESPONSES ==

User: how much does it cost
Nova: Fine Flow starts at £99 a month for up to 50 vehicles, rising to £199 for up to 100 and £499 for unlimited. Every plan includes the full platform with no locked features. How many vehicles are in your fleet?

User: will it pay my fines automatically
Nova: No — Fine Flow does not pay fines automatically. Payment is always done by you on the authority's website, as council portals block automation entirely. What Fine Flow does is organise everything so the payment step takes seconds rather than hours.

User: is there a referral system
Nova: Yes, Fine Flow has a referral programme. When a company signs up using your referral code and subscribes, you earn bonus credits based on their fleet size. You also unlock subscription discounts as you hit milestones — would you like the full breakdown?

User: what is html
Nova: I can only help with Fine Flow questions. Is there anything about fines, pricing, appeals or the platform I can help with?

User: i have 65 vehicles
Nova: With 65 vehicles, the Core plan at £199 per month is ideal — it covers up to 100 vehicles with everything included. Want me to walk you through what is included?

User: yes
Nova: Every Fine Flow plan includes real-time fine alerts, automated driver assignment, full appeal management and compliance reporting. There are no locked features — you get the complete platform from day one. Would you like to know how the appeals system works?
"""


# ---------------------------------------------------------------------------
# OpenAI call
# ---------------------------------------------------------------------------
def _call_openai(messages: List[Dict[str, str]], max_tokens: int = 160) -> Optional[str]:
    if not OPENAI_API_KEY:
        logger.warning("No OpenAI API key set")
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
            timeout=25,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        logger.exception("OpenAI call failed")
        return None


# ---------------------------------------------------------------------------
# RAG retrieval
# ---------------------------------------------------------------------------
def _retrieve_context(query: str) -> str:
    """Semantic search → rerank → return top chunks as context string."""
    try:
        raw = rag_search(query, top_k=TOP_K)
        ranked = rerank_hits(raw, query)
        strong = [d for d in ranked if d.get("score", 0) >= CONFIDENCE_THRESHOLD]
        chunks = [d["chunk"][:700] for d in strong[:3]]
        return "\n\n---\n\n".join(chunks)
    except Exception:
        logger.exception("RAG retrieval failed")
        return ""


# ---------------------------------------------------------------------------
# Build final message list for GPT-4o
# ---------------------------------------------------------------------------
def _build_messages(
    query: str,
    context: str,
    history: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT}]

    # Last 6 messages (3 full turns) for memory — enough context, not bloated
    messages.extend(history[-6:])

    # User message: inject RAG context if available
    if context:
        user_content = (
            f"Fine Flow knowledge base (use this to answer accurately):\n"
            f"{context}\n\n"
            f"User: {query}"
        )
    else:
        user_content = query

    messages.append({"role": "user", "content": user_content})
    return messages


# ---------------------------------------------------------------------------
# Main response builder
# ---------------------------------------------------------------------------
def build_response(query: str, session_id: str = "default") -> Dict[str, Any]:
    query = query.strip()
    session_id = session_id or "default"

    if not query:
        return {"answer": "Ask me anything about Fine Flow.", "confidence": 1.0}

    nq = _norm(query)

    # ==================================================================
    # TIER 1 — Deterministic handlers (fast, no AI, always correct)
    # ==================================================================

    if nq in _GREETINGS:
        _clear_meta(session_id)
        return {
            "answer": "I'm Nova. Ask me anything — I'll help you manage fines, resolve issues, and keep everything moving.",
            "confidence": 1.0,
        }

    if nq in _SOCIAL:
        return {
            "answer": "Doing well, thanks. What can I help you with today — pricing, fines, or appeals?",
            "confidence": 1.0,
        }

    if nq in _IDENTITY:
        return {
            "answer": "I'm Nova, Fine Flow's AI assistant. I can help with anything about the platform — fines, pricing, appeals, billing and more. What would you like to know?",
            "confidence": 1.0,
        }

    if nq in _THANKS:
        return {
            "answer": "Happy to help. Is there anything else you would like to know about Fine Flow?",
            "confidence": 1.0,
        }

    if nq in _GOODBYE:
        return {
            "answer": "Good luck with your fleet management. Come back anytime if you have questions about Fine Flow.",
            "confidence": 1.0,
        }

    if nq in _PURE_FILLER:
        last_topic = _get_meta(session_id, "last_topic")
        if last_topic:
            return {
                "answer": f"What would you like to know more about — I can go deeper on {last_topic.replace('_', ' ')}, or help with something else?",
                "confidence": 1.0,
            }
        return {
            "answer": "What would you like to know about Fine Flow? I can help with pricing, how fines work, appeals, billing or the dashboard.",
            "confidence": 1.0,
        }

    if any(r in nq for r in _RUDE):
        return {
            "answer": "Let me try again. What specifically would you like to know about Fine Flow — pricing, how fines work, or something else?",
            "confidence": 1.0,
        }

    # Off-topic — checked AFTER identity/social so "are you an AI" passes through
    if _is_off_topic(query):
        answer = "I can only help with Fine Flow questions. Is there anything about fines, pricing, appeals or the platform I can help with?"
        _add_turn(session_id, "user", query)
        _add_turn(session_id, "assistant", answer)
        return {"answer": answer, "confidence": 1.0}

    # Vehicle count → instant plan recommendation
    vehicle_count = _extract_vehicle_count(query, session_id)
    if vehicle_count is not None:
        answer, topic = _plan_for_vehicles(vehicle_count)
        _add_turn(session_id, "user", query)
        _add_turn(session_id, "assistant", answer)
        _set_meta(session_id, "last_topic", topic)
        return {"answer": answer, "confidence": 1.0}

    # Purchase intent
    if _PURCHASE_RE.search(query):
        answer = (
            "To get started, contact the sales team on +47 32 28 50 00 or at ff.sales@fineflow.com — "
            "they will get you set up quickly. "
            "How many vehicles are in your fleet so I can point you to the right plan?"
        )
        _add_turn(session_id, "user", query)
        _add_turn(session_id, "assistant", answer)
        _set_meta(session_id, "last_topic", "pricing")
        return {"answer": answer, "confidence": 1.0}

    # ==================================================================
    # TIER 2 — RAG + GPT-4o  (handles all real product questions)
    # ==================================================================

    _add_turn(session_id, "user", query)

    context = _retrieve_context(query)
    history = _get_history(session_id)
    # Pass all history except the turn we just added (GPT sees it as user message)
    messages = _build_messages(query, context, history[:-1])

    answer = _call_openai(messages, max_tokens=160)

    if not answer:
        answer = (
            "The Fine Flow team can help with that directly — "
            "call +47 32 28 50 00 or email ff.sales@fineflow.com."
        )

    answer = _clean(answer)
    _add_turn(session_id, "assistant", answer)

    # Store topic hint from query for follow-up context
    topic_hints = {
        "pric": "pricing", "cost": "pricing", "plan": "pricing", "package": "pricing",
        "appeal": "appeals", "dispute": "appeals", "challenge": "appeals",
        "driver": "driver_management", "assign": "driver_matching",
        "referral": "referral_programme", "refer": "referral_programme",
        "security": "security", "gdpr": "security", "safe": "security",
        "billing": "billing", "invoice": "billing", "stripe": "billing",
        "dashboard": "dashboard", "report": "reports", "export": "reports",
        "gmail": "gmail_connection", "email": "email_ingestion",
        "save": "savings", "time": "savings",
    }
    for hint, topic in topic_hints.items():
        if hint in nq:
            _set_meta(session_id, "last_topic", topic)
            break

    return {
        "answer": answer,
        "confidence": 0.9 if context else 0.5,
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