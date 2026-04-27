# app/answer_builder.py
"""
FineFlow Nova — Answer Builder
Senior-grade RAG chatbot engine with:
  - Structured KB (exact + fuzzy match)
  - History-aware OpenAI calls
  - Grounded system prompt (no hallucination)
  - Correct pricing, correct lifecycle language
  - No implementation leakage (no Groq/LLaMA references)
"""

import json
import threading
import time
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
# Knowledge Base
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


def _normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    import re
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _kb_match(query: str) -> Optional[Dict[str, Any]]:
    """
    Two-pass KB lookup:
      Pass 1 — exact normalised substring match against question/keywords
      Pass 2 — token overlap score (returns best if score >= threshold)
    """
    nq = _normalise(query)
    q_tokens = set(nq.split())

    best_entry: Optional[Dict[str, Any]] = None
    best_score = 0.0

    for entry in _kb:
        # Build a searchable string from question + any keywords field
        searchable = _normalise(entry.get("question", ""))
        keywords_raw = entry.get("keywords", [])
        if keywords_raw:
            searchable += " " + " ".join(_normalise(k) for k in keywords_raw)

        # Pass 1 — direct substring
        if nq in searchable:
            return entry

        # Pass 2 — token overlap
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
# Session memory (per session_id)
# ---------------------------------------------------------------------------
_SESSION: Dict[str, List[Dict[str, str]]] = {}
_LOCK = threading.Lock()

# Track last KB entry per session for follow-up awareness
_SESSION_CONTEXT: Dict[str, Optional[Dict[str, Any]]] = {}


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


def _set_session_context(sid: str, entry: Optional[Dict[str, Any]]) -> None:
    with _LOCK:
        _SESSION_CONTEXT[sid] = entry


def _get_session_context(sid: str) -> Optional[Dict[str, Any]]:
    with _LOCK:
        return _SESSION_CONTEXT.get(sid)


# ---------------------------------------------------------------------------
# OpenAI call — history-aware, grounded
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are Nova, the AI assistant for FineFlow — a UK fleet fine (PCN) management platform.

PRODUCT FACTS (use ONLY these — do not invent anything):

WHAT FINEFLOW DOES:
- Monitors a connected email inbox for incoming fine/PCN notices
- Automatically extracts key fields: fine amount, PCN/citation number, vehicle registration, issue date/time, due date, location, violation type
- Matches each fine to the responsible driver using vehicle registration + date + time against uploaded driver logs (CSV or system data)
- If no match: fine goes to UNASSIGNED queue for manual admin review
- Tracks every fine through a full lifecycle with clear statuses (see below)
- Supports dispute and appeal workflows including evidence upload and outcome tracking
- Does NOT automatically pay fines — payment portals are external authority sites with bot protection

FINE STATUSES (in order):
RECEIVED → UNASSIGNED or ASSIGNED → CONFIRMED or DISPUTED → PAID / CANCELLED / OVERDUE

ASSIGNMENT LOGIC:
1. Vehicle registration must match exactly
2. Fine issue date must be within ±1 day of a driver log entry
3. Fine time must fall within driver log start/end time
If multiple drivers match → status becomes REVIEW_REQUIRED, placed in review queue for admin

APPEAL WORKFLOW:
Driver disputes → admin reviews → accepts (appeal sent to authority, status: UNDER REVIEW) or rejects (fine returns to active/payable state)
System tracks outcomes and improves future appeal recommendations over time.

OVERDUE SCHEDULER:
Runs daily at midnight. Any RECEIVED/UNASSIGNED/ASSIGNED/CONFIRMED fine past its due date → OVERDUE.

PRICING (CORRECT — do not deviate):
- Core: £99/month, up to 50 vehicles, 100 fines included
- Essential: £199/month, up to 100 vehicles, 300 fines included
- Advanced: £399/month, up to 200 vehicles, 700 fines included
- Elite: £499/month, unlimited vehicles, 1,000 fines included
- All plans: £0.75 per fine (within included allowance)
- Overage: £2.50 per fine beyond plan limit
- Pay-per-fine (no subscription): £2.75 per fine, prepaid blocks
- £0.75 fee applies ONCE per fine — appeals, reassignments, disputes do NOT create additional charges
- Credits reset monthly. Unused allowance is forfeited. No rollover.

BILLING RULES:
- Overage charges collected at end of billing cycle
- Cannot resubscribe with outstanding balance
- Cancellation: cooldown until end of current period

SECURITY:
- JWT auth (24hr expiry), bcrypt passwords, AES-256-CBC encryption for Gmail tokens
- GDPR compliant — no data sold or shared with third parties
- Payments handled externally — FineFlow does not store card details

ROLES:
- Admin (internal): full system access
- Company: manage own drivers, fines, appeals, billing
- Driver: view own fines, confirm or dispute assigned fines

REFERRAL:
- 1–25 vehicle fleet referred → 100 credits to referrer
- 26–100 → 250, 101–500 → 750, 500+ → 2,000
- Tiers: Silver (3 refs) = 100 bonus credits; Gold (5) = 10% off 12mo; Platinum (10) = 15% off 12mo; Titan (25) = 20% off lifetime
- New joiner reward: £75 of credits on sign-up with referral code

RULES FOR YOUR RESPONSES:
1. NEVER mention Groq, LLaMA, or any underlying AI technology
2. NEVER invent features, statuses, or pricing not listed above
3. If something is genuinely unknown, say: "That detail isn't something I have on hand — your dashboard or the FineFlow support team can help."
4. Keep responses clear, concise, and product-focused
5. When the user says "yes" or "go on" after a question, provide the next logical breakdown
6. Always give a direct Yes/No on capability questions before explaining
7. DO NOT use "credit limit" terminology — use "included fines" or "monthly allowance"
"""


def _call_openai(
    messages: List[Dict[str, str]],
    max_tokens: int = 400,
) -> Optional[str]:
    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not set — skipping LLM call")
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
# Follow-up detection
# ---------------------------------------------------------------------------

_AFFIRMATIVE = {
    "yes", "yeah", "yep", "sure", "ok", "okay", "go on",
    "please", "continue", "tell me more", "go ahead", "more", "explain",
    "yes please", "show me", "absolutely", "of course",
}


def _is_affirmative(query: str) -> bool:
    return _normalise(query) in _AFFIRMATIVE


# ---------------------------------------------------------------------------
# Step-by-step breakdown answers (for "yes" follow-ups)
# ---------------------------------------------------------------------------

_SYSTEM_BREAKDOWN = """Here's how FineFlow works from start to finish:

**1. Ingestion**
FineFlow monitors your connected email inbox every minute. When a penalty notice arrives, it's detected automatically — no manual forwarding needed.

**2. Extraction (OCR & AI)**
The system reads the email or attachment and extracts the key fields: fine amount, PCN reference, vehicle registration, issue date and time, due date, violation type, and location. The fine is created with status **RECEIVED**.

**3. Assignment**
Using your uploaded driver logs (CSV or system data), FineFlow matches the fine to the responsible driver by checking vehicle registration, date (within ±1 day), and time window. On a clear match → status becomes **ASSIGNED**. If no match or ambiguous → **UNASSIGNED**, placed in a review queue for manual resolution.

**4. Driver Action**
The assigned driver confirms responsibility (**CONFIRMED**) or raises a dispute (**DISPUTED**) with a reason and optional evidence.

**5. Disputes & Appeals**
The admin reviews disputed fines. If the appeal is accepted, it's prepared and submitted to the issuing authority — status becomes **UNDER REVIEW**. If rejected, the fine returns to an active payable state. Outcomes are tracked and used to improve future appeal recommendations.

**6. Tracking & Outcomes**
Every fine has a live status visible on the dashboard. Fines past their due date are automatically flagged **OVERDUE** (checked nightly). Final outcomes are **PAID**, **CANCELLED** (appeal won), or **OVERDUE**.

Want me to go deeper on any specific stage?"""


# ---------------------------------------------------------------------------
# Core response builder
# ---------------------------------------------------------------------------

def build_response(query: str, session_id: str = "default") -> Dict[str, Any]:
    query = query.strip()
    session_id = session_id or "default"

    # ── 1. Empty ────────────────────────────────────────────────────────────
    if not query:
        return {"answer": "Ask me anything about FineFlow.", "confidence": 1.0}

    # ── 2. Greeting ─────────────────────────────────────────────────────────
    if _normalise(query) in {"hi", "hello", "hey", "hiya", "howdy"}:
        _set_session_context(session_id, None)
        return {
            "answer": "I'm Nova. Ask me anything — I'll help you manage fines, resolve issues, and keep everything moving.",
            "confidence": 1.0,
        }

    # ── 3. Follow-up "yes" → expand last KB entry ───────────────────────────
    if _is_affirmative(query):
        last_ctx = _get_session_context(session_id)
        if last_ctx:
            follow_up = last_ctx.get("follow_up")
            cat = last_ctx.get("category", "")

            # Special case: core overview "yes" → full system breakdown
            if cat == "core_overview":
                _add_turn(session_id, "user", query)
                _add_turn(session_id, "assistant", _SYSTEM_BREAKDOWN)
                _set_session_context(session_id, None)
                return {"answer": _SYSTEM_BREAKDOWN, "confidence": 1.0}

            if follow_up:
                _add_turn(session_id, "user", query)
                _add_turn(session_id, "assistant", follow_up)
                _set_session_context(session_id, None)
                return {"answer": follow_up, "confidence": 1.0}

        # No context → treat as generic OpenAI fallback below
        pass

    # ── 4. KB exact/fuzzy match ─────────────────────────────────────────────
    kb_entry = _kb_match(query)
    if kb_entry:
        answer = kb_entry["answer"]
        _add_turn(session_id, "user", query)
        _add_turn(session_id, "assistant", answer)
        _set_session_context(session_id, kb_entry)

        # Append follow-up prompt if present
        follow_up_q = kb_entry.get("follow_up")
        if follow_up_q:
            display = f"{answer}\n\n_{follow_up_q}_"
        else:
            display = answer

        return {"answer": display, "confidence": 1.0}

    # ── 5. RAG retrieval ─────────────────────────────────────────────────────
    _add_turn(session_id, "user", query)

    docs: List[Dict[str, Any]] = []
    try:
        raw_docs = rag_search(query, top_k=TOP_K)
        docs = rerank_hits(raw_docs, query)
    except Exception:
        logger.exception("RAG search failed")

    # Filter by confidence threshold
    strong_docs = [d for d in docs if d.get("score", 0) >= CONFIDENCE_THRESHOLD]
    context_chunks = [d["chunk"][:600] for d in strong_docs[:3]]
    context = "\n\n---\n\n".join(context_chunks) if context_chunks else ""

    # ── 6. Build OpenAI messages with full history ───────────────────────────
    history = _get_history(session_id)
    # Remove the last user turn we just added (will be re-added in messages below)
    history_without_last = history[:-1] if history else []

    messages: List[Dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    messages.extend(history_without_last)

    user_content = query
    if context:
        user_content = (
            f"[Retrieved context from FineFlow documentation]\n{context}\n\n"
            f"[User question]\n{query}"
        )

    messages.append({"role": "user", "content": user_content})

    # ── 7. Call OpenAI ───────────────────────────────────────────────────────
    answer = _call_openai(messages)

    # ── 8. Fallback ──────────────────────────────────────────────────────────
    if not answer:
        if context:
            answer = (
                strong_docs[0]["chunk"][:400].strip()
                + "\n\n_(For full details, check your FineFlow dashboard or contact support.)_"
            )
        else:
            answer = (
                "I don't have that specific detail on hand. "
                "Your FineFlow dashboard or our support team will be able to help you with that."
            )

    _add_turn(session_id, "assistant", answer)
    _set_session_context(session_id, None)

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