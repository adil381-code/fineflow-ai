# app/answer_builder.py
"""
FineFlow Nova — Answer Builder (Production Final v5)
- Uses client KB exactly as provided — no KB modifications
- Social, casual, thanks, vague query handling
- History-aware OpenAI fallback with hard-grounded system prompt
- Clean output: no asterisks, arrows, ± symbols, underscores, or markdown
- Professional tone matching client's brief
- Never says "not in documentation" — always gives a useful answer
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
# Clean output — strip markdown and special symbols at render time
# ---------------------------------------------------------------------------
def _clean(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"_(.*?)_", r"\1", text)
    text = text.replace("→", "becomes")
    text = text.replace("->", "becomes")
    text = text.replace("±", "plus or minus")
    text = text.replace("`", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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
# Intent detection
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
    "hi there", "hey there", "hello there",
}
_SOCIAL = {
    "how are you", "how are you doing", "how r u", "how r you",
    "how are u", "hows it going", "how is it going", "whats up",
    "what s up", "sup", "you ok", "you good", "alright",
    "how do you do", "how do u do", "how have you been",
}
_THANKS = {
    "thanks", "thank you", "thank u", "cheers", "great", "awesome",
    "perfect", "brilliant", "nice", "lovely", "fantastic", "helpful",
    "got it", "understood", "makes sense", "that helps", "that helped",
    "cool", "okay thanks", "ok thanks", "great thanks",
}


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


def _is_pricing_confusion(q: str) -> bool:
    """Detect questions about a £2 fee that does not exist in FineFlow."""
    nq = q.lower().strip()
    # Match standalone "2", "£2", "2?" but NOT "2.50" or "2.75"
    if re.match(r"^[£$]?\s*2\s*\??$", nq):
        return True
    patterns = [
        r"£\s*2\s+fee",
        r"£\s*2\s+for",
        r"£\s*2\s+charge",
        r"what.*£\s*2\b(?!\.\d)",
        r"£2 for what",
        r"2 for what",
    ]
    return any(re.search(p, nq) for p in patterns)


def _is_vague(q: str) -> bool:
    nq = _normalise(q)
    all_special = _AFFIRMATIVE | _NEGATIVE | _GREETINGS | _SOCIAL | _THANKS
    return len(nq.split()) <= 2 and nq not in all_special


# ---------------------------------------------------------------------------
# Static rich responses
# ---------------------------------------------------------------------------

_SYSTEM_BREAKDOWN = """Here is how FineFlow works from start to finish:

1. Ingestion
FineFlow monitors your connected email inbox every minute. When a penalty notice arrives it is detected automatically — no manual forwarding needed.

2. Extraction
The system reads the email or attachment and pulls out the key fields: fine amount, PCN reference, vehicle registration, issue date and time, due date, violation type, and location. The fine is created with status RECEIVED.

3. Assignment
Using your uploaded driver logs (CSV or system data), FineFlow matches the fine to the responsible driver by checking vehicle registration, date within plus or minus one day, and time window. A clear match means the fine is ASSIGNED. No match means it is UNASSIGNED and placed in the admin review queue. A conflicting match is flagged as REVIEW_REQUIRED for the admin to resolve.

4. Driver Action
The assigned driver confirms responsibility, setting the status to CONFIRMED, or raises a dispute with a reason and optional evidence, setting it to DISPUTED.

5. Disputes and Appeals
The admin reviews disputed fines. If accepted, the appeal is prepared and submitted to the issuing authority and the status becomes UNDER REVIEW. If rejected, the fine returns to an active payable state. All outcomes are tracked and used to improve future appeal recommendations.

6. Tracking and Outcomes
Every fine has a live status on the dashboard. Fines past their due date are flagged OVERDUE nightly. Final outcomes are PAID, CANCELLED if the appeal was won, or OVERDUE.

Which stage would you like to go deeper on?"""

_PLAN_COMPARISON = """Here is how the plans compare so you can find the right fit:

Core at £99 per month covers fleets up to 50 vehicles with 100 fines included. It is the most cost-efficient entry point for smaller operations.

Essential at £199 per month covers up to 100 vehicles with 300 fines included. A solid choice for mid-sized fleets with steady fine volumes.

Advanced at £399 per month handles up to 200 vehicles and 700 fines per month. Built for larger operations where volume is consistent.

Elite at £499 per month is unlimited vehicles with 1,000 fines included. Designed for high-volume fleets where running out of allowance is not an option.

If your volume is low or unpredictable, the pay-per-fine option at £2.75 per fine with no subscription is worth considering.

All plans charge £0.75 per fine within your allowance and £2.50 per fine if you go over. How many vehicles do you run?"""

# ---------------------------------------------------------------------------
# System prompt for OpenAI fallback
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are Nova, the AI assistant for FineFlow — a UK fleet fine (PCN) management platform.

Your job is to answer questions about FineFlow clearly, confidently and helpfully. You sound like a knowledgeable product expert, not a search engine. You never say things like "not specified in the context", "I cannot confirm", "as of my last update", or "the documentation doesn't mention". If you do not have a specific detail, you give the most useful answer you can and suggest the dashboard or support team for anything account-specific.

FORMATTING — always follow these rules:
- Write in plain professional English. No markdown.
- No asterisks, no underscores, no arrows, no symbols like +/-.
- Do not start responses with numbered lists. Lead with a sentence.
- No filler phrases like "Certainly!", "Great question!", or "Of course!".
- Be concise. One idea per sentence. No repetition.

PRODUCT FACTS — use only these, never invent:

WHAT FINEFLOW DOES:
Monitors connected email inbox every minute for incoming fine or PCN notices. Extracts fine amount, PCN reference, vehicle registration, issue date and time, due date, location, and violation type. Matches fines to drivers using vehicle registration, date within plus or minus one day, and time window against driver logs (CSV or system data). No match puts the fine in the UNASSIGNED queue for manual admin review. Multiple matches create a REVIEW_REQUIRED status. Does not automatically pay fines — payment portals are external authority sites with bot protection.

FINE STATUSES in order:
RECEIVED, then UNASSIGNED or ASSIGNED, then CONFIRMED or DISPUTED, then PAID or CANCELLED or OVERDUE. REVIEW_REQUIRED resolves to ASSIGNED after admin action.

APPEAL WORKFLOW:
Driver disputes. Admin reviews. Accepted means UNDER REVIEW, appeal sent to authority. Rejected means fine returns to active payable state. Full lifecycle: Submitted, Under Review, Accepted or Rejected, Closed. Outcomes tracked to improve future recommendations. Win percentage is guidance only, not a guarantee.

OVERDUE SCHEDULER:
Runs nightly at midnight. Any RECEIVED, UNASSIGNED, ASSIGNED, or CONFIRMED fine past its due date becomes OVERDUE.

PRICING — exact, never change:
Core: £99 per month, up to 50 vehicles, 100 fines included.
Essential: £199 per month, up to 100 vehicles, 300 fines included.
Advanced: £399 per month, up to 200 vehicles, 700 fines included.
Elite: £499 per month, unlimited vehicles, 1,000 fines included.
All plans: £0.75 per fine within monthly allowance.
Overage: £2.50 per fine beyond the plan limit. There is NO £2.00 fee.
Pay-per-fine no subscription: £2.75 per fine in prepaid blocks.
The £0.75 fee applies once per fine only. Appeals, reassignments, disputes, and payment do not create additional charges. Monthly allowance resets each period with no rollover.

BILLING:
Overage collected at end of billing cycle. Cannot resubscribe with outstanding balance. Cancellation triggers a cooldown until end of the current billing period.

SECURITY:
JWT auth 24 hour expiry. Bcrypt passwords. AES-256-CBC for Gmail tokens. GDPR compliant — no data sold or shared. Payments handled externally via Stripe. FineFlow never stores raw card details.

SOCIAL QUERIES:
If the user makes small talk or asks how you are, respond briefly and warmly, then redirect to FineFlow. Keep it human — one or two sentences maximum. Never answer a social question with product information.

UNKNOWN QUERIES:
If something is genuinely outside your knowledge of FineFlow, give the closest useful answer you can, then say the dashboard or support team can confirm specifics. Never refuse to engage.
"""

# ---------------------------------------------------------------------------
# OpenAI call
# ---------------------------------------------------------------------------
def _call_openai(messages: List[Dict[str, str]], max_tokens: int = 400) -> Optional[str]:
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
                "temperature": 0.2,
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

    # Social small talk
    if _is_social(query):
        return {
            "answer": "Doing well, thanks for asking. I'm here whenever you need help with fines, pricing, appeals, or anything FineFlow related.",
            "confidence": 1.0,
        }

    # Thanks / acknowledgement
    if _is_thanks(query):
        return {
            "answer": "Happy to help. Is there anything else you would like to know about FineFlow?",
            "confidence": 1.0,
        }

    # Negative / decline
    if _is_negative(query):
        _add_turn(session_id, "user", query)
        answer = "No problem. Is there anything else about FineFlow I can help you with?"
        _add_turn(session_id, "assistant", answer)
        _set_ctx(session_id, None)
        return {"answer": answer, "confidence": 1.0}

    # Pricing confusion — the £2 fee that does not exist
    if _is_pricing_confusion(query):
        answer = (
            "There is no £2 fee in FineFlow. The fees are:\n\n"
            "- £0.75 per fine within your monthly allowance on any subscription plan\n"
            "- £2.50 per fine when you exceed your monthly allowance\n"
            "- £2.75 per fine on the pay-per-fine option with no subscription\n\n"
            "Which one would you like more detail on?"
        )
        _add_turn(session_id, "user", query)
        _add_turn(session_id, "assistant", answer)
        return {"answer": answer, "confidence": 1.0}

    # Vague short query
    if _is_vague(query):
        last_ctx = _get_ctx(session_id)
        if last_ctx:
            topic = last_ctx.get("category", "this topic").replace("_", " ")
            answer = f"Could you be a bit more specific? I can go deeper on {topic}, or help with something else entirely."
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

            # "yes" after core overview — deliver full step breakdown
            if cat == "core_overview":
                _add_turn(session_id, "user", query)
                _add_turn(session_id, "assistant", _SYSTEM_BREAKDOWN)
                _set_ctx(session_id, None)
                return {"answer": _SYSTEM_BREAKDOWN, "confidence": 1.0}

            # "yes" after plan limit question — deliver plan comparison
            if follow_up and "compare" in _normalise(follow_up):
                _add_turn(session_id, "user", query)
                _add_turn(session_id, "assistant", _PLAN_COMPARISON)
                _set_ctx(session_id, None)
                return {"answer": _PLAN_COMPARISON, "confidence": 1.0}

            # Generic follow-up from KB entry
            if follow_up:
                clean_fu = _clean(follow_up)
                _add_turn(session_id, "user", query)
                _add_turn(session_id, "assistant", clean_fu)
                _set_ctx(session_id, None)
                return {"answer": clean_fu, "confidence": 1.0}

        answer = "Of course — what would you like to know more about? You can ask about fines, pricing, appeals, billing, or the dashboard."
        _add_turn(session_id, "user", query)
        _add_turn(session_id, "assistant", answer)
        return {"answer": answer, "confidence": 1.0}

    # KB match — client KB used exactly as provided, only cleaned at output
    _add_turn(session_id, "user", query)
    kb_entry = _kb_match(query)
    if kb_entry:
        answer = _clean(kb_entry["answer"])
        follow_up_q = kb_entry.get("follow_up")
        display = f"{answer}\n\n{_clean(follow_up_q)}" if follow_up_q else answer
        _add_turn(session_id, "assistant", display)
        _set_ctx(session_id, kb_entry)
        return {"answer": display, "confidence": 1.0}

    # RAG retrieval for questions not in KB
    docs: List[Dict[str, Any]] = []
    try:
        raw = rag_search(query, top_k=TOP_K)
        docs = rerank_hits(raw, query)
    except Exception:
        logger.exception("RAG search failed")

    strong_docs = [d for d in docs if d.get("score", 0) >= CONFIDENCE_THRESHOLD]
    context_chunks = [d["chunk"][:600] for d in strong_docs[:3]]
    context = "\n\n---\n\n".join(context_chunks) if context_chunks else ""

    # OpenAI with full conversation history
    history = _get_history(session_id)
    history_without_last = history[:-1] if history else []

    messages: List[Dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    messages.extend(history_without_last)

    user_content = query
    if context:
        user_content = f"[FineFlow documentation context]\n{context}\n\n[User question]\n{query}"
    messages.append({"role": "user", "content": user_content})

    answer = _call_openai(messages)

    # Fallback if OpenAI fails
    if not answer:
        if context:
            answer = (
                strong_docs[0]["chunk"][:400].strip()
                + "\n\nFor anything account-specific, your FineFlow dashboard or support team can help."
            )
        else:
            answer = (
                "That is a good question. For anything specific to your account or setup, "
                "your FineFlow dashboard will have the detail you need, or the support team can help directly."
            )

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