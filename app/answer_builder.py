# app/answer_builder.py
"""
FineFlow Nova — Answer Builder (Production Final)
- KB-first with two-pass fuzzy matching
- History-aware OpenAI with hard-grounded system prompt
- Proper negative/clarification/pricing-confusion handling
- Zero AI leakage or disclaimers
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
# KB matching — two-pass (exact substring + token overlap)
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

    if best_score >= 0.55 and best_entry:
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
# Intent helpers
# ---------------------------------------------------------------------------
_AFFIRMATIVE = {
    "yes", "yeah", "yep", "sure", "ok", "okay", "go on", "please",
    "continue", "tell me more", "go ahead", "more", "explain",
    "yes please", "show me", "absolutely", "of course", "do it",
}
_NEGATIVE = {
    "no", "nope", "nah", "no thanks", "not now", "skip", "never mind",
    "nevermind", "no need", "dont", "don t", "not really",
}
_GREETINGS = {
    "hi", "hello", "hey", "hiya", "howdy", "good morning",
    "good afternoon", "good evening", "morning", "afternoon",
}


def _is_affirmative(q: str) -> bool:
    return _normalise(q) in _AFFIRMATIVE


def _is_negative(q: str) -> bool:
    return _normalise(q) in _NEGATIVE


def _is_greeting(q: str) -> bool:
    return _normalise(q) in _GREETINGS


def _is_pricing_confusion(q: str) -> bool:
    nq = q.lower().strip()
    patterns = [r"pound\s*2\b", r"2\s*pound", r"£\s*2\b", r"^2\s*\??$", r"what.*2.*fee", r"2.*fee"]
    return any(re.search(p, nq) for p in patterns)


def _is_vague(q: str) -> bool:
    nq = _normalise(q)
    all_special = _AFFIRMATIVE | _NEGATIVE | _GREETINGS
    return len(nq.split()) <= 2 and nq not in all_special


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are Nova, the AI assistant for FineFlow — a UK fleet fine (PCN) management platform.

Answer ONLY using the facts below. Never invent features, statuses, or prices. Never mention Groq, LLaMA, OpenAI, or any AI technology. Never say "as of my last update", "I cannot confirm", or "not specified". If something is genuinely unknown say: "That's not something I have on hand — your FineFlow dashboard or support team can help."

PRODUCT FACTS:

WHAT FINEFLOW DOES:
- Monitors connected email inbox for incoming fine/PCN notices every minute
- Extracts: fine amount, PCN/citation number, vehicle registration, issue date/time, due date, location, violation type
- Matches fine to driver using: vehicle registration + date (within plus or minus 1 day) + time window vs driver logs (CSV or system data)
- No match: UNASSIGNED queue for manual admin review
- Multiple matches: REVIEW_REQUIRED, moved to review queue, all overrides audit-logged
- Full lifecycle tracking with clear statuses
- Does NOT automatically pay fines — payment portals are external authority sites with bot protection

FINE STATUSES:
RECEIVED then UNASSIGNED or ASSIGNED then CONFIRMED or DISPUTED then PAID or CANCELLED or OVERDUE
REVIEW_REQUIRED: admin resolves, then ASSIGNED

APPEAL WORKFLOW:
Driver disputes then admin reviews then accepts (UNDER REVIEW, appeal sent to authority) or rejects (fine returns to active payable state)
Full lifecycle: Submitted then Under Review then Accepted or Rejected then Closed
Outcomes tracked to improve future appeal recommendations.
Win percentage is a guidance signal, not a guarantee.

OVERDUE SCHEDULER:
Runs nightly at midnight. Any RECEIVED or UNASSIGNED or ASSIGNED or CONFIRMED fine past due date becomes OVERDUE.

PRICING (exact, never change):
- Core: 99 pounds per month, up to 50 vehicles, 100 fines included
- Essential: 199 pounds per month, up to 100 vehicles, 300 fines included
- Advanced: 399 pounds per month, up to 200 vehicles, 700 fines included
- Elite: 499 pounds per month, unlimited vehicles, 1000 fines included
- All plans: 0.75 pounds per fine within monthly allowance
- Overage: 2.50 pounds per fine beyond plan limit. There is NO 2.00 pound fee — it does not exist.
- Pay-per-fine (no subscription): 2.75 pounds per fine, prepaid blocks
- The 0.75 pound fee applies ONCE per fine — appeals, reassignments, disputes, payment do NOT add charges
- Monthly allowance resets each period. No rollover. Unused allowance forfeited.

BILLING:
- Overage collected at end of billing cycle
- Cannot resubscribe with outstanding balance — must clear first
- Cancellation: cooldown until end of current billing period

SECURITY AND COMPLIANCE:
- JWT auth 24hr expiry, bcrypt passwords, AES-256-CBC encryption for Gmail tokens
- GDPR compliant — no data sold or shared with third parties
- Payments handled externally via Stripe — FineFlow never stores raw card details

ROLES:
- Admin (internal): full system access
- Company: manage own drivers, fines, appeals, billing
- Driver: view and act on own assigned fines only

RESPONSE RULES:
1. Be concise and product-focused
2. Start Yes or No capability questions immediately with Yes or No
3. Never use "as of my last update", "I cannot confirm", or any AI technology names
4. Use "monthly allowance" or "included fines" — never "credit limit"
5. If user says no or declines a follow-up, acknowledge and offer to help with something else
6. If asked about a 2 pound fee, clarify it does not exist and state the correct 2.50 pound overage
7. If pricing is questioned, confirm the correct figures and direct to support for formal queries
"""

# ---------------------------------------------------------------------------
# System breakdown for "yes" after core_overview
# ---------------------------------------------------------------------------
_SYSTEM_BREAKDOWN = """Here's how FineFlow works from start to finish:

**1. Ingestion**
FineFlow monitors your connected email inbox every minute. When a penalty notice arrives, it is detected automatically — no manual forwarding needed.

**2. Extraction**
The system reads the email or attachment and extracts the key fields: fine amount, PCN reference, vehicle registration, issue date and time, due date, violation type, and location. The fine is created with status **RECEIVED**.

**3. Assignment**
Using your uploaded driver logs (CSV or system data), FineFlow matches the fine to the responsible driver by checking vehicle registration, date (within plus or minus 1 day), and time window. Clear match becomes **ASSIGNED**. No match becomes **UNASSIGNED** placed in admin review queue. Conflicting match becomes **REVIEW_REQUIRED**.

**4. Driver Action**
The assigned driver confirms responsibility (**CONFIRMED**) or raises a dispute (**DISPUTED**) with a reason and optional evidence.

**5. Disputes and Appeals**
The admin reviews disputed fines. Accepted means the appeal is prepared and submitted to the authority, status becomes **UNDER REVIEW**. Rejected means the fine returns to an active payable state. Outcomes are tracked and improve future appeal recommendations.

**6. Tracking and Outcomes**
Every fine has a live status on the dashboard. Fines past their due date are flagged **OVERDUE** nightly. Final outcomes are **PAID**, **CANCELLED** (appeal won), or **OVERDUE**.

Which stage would you like to go deeper on?"""


# ---------------------------------------------------------------------------
# OpenAI call
# ---------------------------------------------------------------------------
def _call_openai(messages: List[Dict[str, str]], max_tokens: int = 350) -> Optional[str]:
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
                "temperature": 0.15,
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
        return {"answer": "Ask me anything about FineFlow.", "confidence": 1.0}

    # Greeting
    if _is_greeting(query):
        _set_ctx(session_id, None)
        return {
            "answer": "I'm Nova. Ask me anything — I'll help you manage fines, resolve issues, and keep everything moving.",
            "confidence": 1.0,
        }

    # Negative
    if _is_negative(query):
        _add_turn(session_id, "user", query)
        answer = "No problem. Is there anything else about FineFlow I can help you with?"
        _add_turn(session_id, "assistant", answer)
        _set_ctx(session_id, None)
        return {"answer": answer, "confidence": 1.0}

    # Pricing confusion — the £2 that doesn't exist
    if _is_pricing_confusion(query):
        answer = (
            "There is no £2 fee in FineFlow. The three fees are:\n\n"
            "• **£0.75** — per fine within your monthly allowance (subscription)\n"
            "• **£2.50** — overage rate when you exceed your monthly allowance\n"
            "• **£2.75** — pay-per-fine rate with no subscription\n\n"
            "Which one would you like more detail on?"
        )
        _add_turn(session_id, "user", query)
        _add_turn(session_id, "assistant", answer)
        return {"answer": answer, "confidence": 1.0}

    # Vague query
    if _is_vague(query):
        last_ctx = _get_ctx(session_id)
        if last_ctx:
            topic = last_ctx.get("category", "this topic").replace("_", " ")
            answer = f"Could you be more specific? I can go deeper on {topic}, or help with something else entirely."
        else:
            answer = "Could you give me a bit more detail? I can help with anything about FineFlow — fines, pricing, appeals, billing, or the dashboard."
        _add_turn(session_id, "user", query)
        _add_turn(session_id, "assistant", answer)
        return {"answer": answer, "confidence": 1.0}

    # Affirmative follow-up
    if _is_affirmative(query):
        last_ctx = _get_ctx(session_id)
        if last_ctx:
            cat = last_ctx.get("category", "")
            follow_up = last_ctx.get("follow_up")

            if cat == "core_overview":
                _add_turn(session_id, "user", query)
                _add_turn(session_id, "assistant", _SYSTEM_BREAKDOWN)
                _set_ctx(session_id, None)
                return {"answer": _SYSTEM_BREAKDOWN, "confidence": 1.0}

            if follow_up:
                _add_turn(session_id, "user", query)
                _add_turn(session_id, "assistant", follow_up)
                _set_ctx(session_id, None)
                return {"answer": follow_up, "confidence": 1.0}

        answer = "Of course — what would you like to know more about? You can ask about fines, pricing, appeals, billing, or the dashboard."
        _add_turn(session_id, "user", query)
        _add_turn(session_id, "assistant", answer)
        return {"answer": answer, "confidence": 1.0}

    # KB match
    _add_turn(session_id, "user", query)
    kb_entry = _kb_match(query)
    if kb_entry:
        answer = kb_entry["answer"]
        follow_up_q = kb_entry.get("follow_up")
        display = f"{answer}\n\n_{follow_up_q}_" if follow_up_q else answer
        _add_turn(session_id, "assistant", display)
        _set_ctx(session_id, kb_entry)
        return {"answer": display, "confidence": 1.0}

    # RAG retrieval
    docs: List[Dict[str, Any]] = []
    try:
        raw = rag_search(query, top_k=TOP_K)
        docs = rerank_hits(raw, query)
    except Exception:
        logger.exception("RAG search failed")

    strong_docs = [d for d in docs if d.get("score", 0) >= CONFIDENCE_THRESHOLD]
    context_chunks = [d["chunk"][:600] for d in strong_docs[:3]]
    context = "\n\n---\n\n".join(context_chunks) if context_chunks else ""

    # OpenAI with full history
    history = _get_history(session_id)
    history_without_last = history[:-1] if history else []

    messages: List[Dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    messages.extend(history_without_last)

    user_content = query
    if context:
        user_content = f"[FineFlow documentation context]\n{context}\n\n[User question]\n{query}"
    messages.append({"role": "user", "content": user_content})

    answer = _call_openai(messages)

    # Fallback
    if not answer:
        if context:
            answer = (
                strong_docs[0]["chunk"][:400].strip()
                + "\n\n_For full details, check your FineFlow dashboard or contact support._"
            )
        else:
            answer = (
                "That's not something I have specific details on. "
                "Your FineFlow dashboard or our support team will be able to help you with that."
            )

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