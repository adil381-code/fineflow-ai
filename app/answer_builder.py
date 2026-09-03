# app/answer_builder.py
"""
FineFlow Nova — LLM-first conversation engine (streaming, tools, SQL memory)
============================================================================
• No regex intent routing. Every message → model with system prompt (tone rules)
  + CUSTOMER CONTEXT + rolling summary + last N turns + KB excerpts for THIS turn.
• Facts live in the knowledge base. Only tone/behaviour lives here.
• Retrieval query is rewritten with history so "yes" / "london" still hit the right topic.
• Tools:  save_customer_details → profile persisted in MySQL
          escalate_to_team      → asks for email if none on file; logs email_capture + ticket
• Follow-up cadence guaranteed: if the previous reply had no question, this one must end with one.
• Memory is SQL only: chat_history (turns) + session_state (profile/summary). Chroma holds the KB only.
• Streaming: build_response_stream() yields text chunks then a final dict; build_response() wraps it.

v3.6 changes (client KB v2 alignment):
  11. Pricing aligned to the client's MASTER KB v2: FOUR plans - Essential £99 (5-50),
      Core £199 (51-100), Advanced £399 (101-200), Elite £499 (200+) - plus PAYG £2.75.
      Advanced/£399 are now VALID; only the £2.00 fee remains blocked. State.plan()
      thresholds updated. Savings figures £400/£1,200/£4,000 (KB ROI topic) allowed;
      any other £ figure still fails grounding.
  12. Tone exemplars from the client's sample Q&A added to the prompt (voice only -
      the sample file is NOT a knowledge store; the single KB remains the only source).

v3.5 changes:
  10. _hard_guard(): final deterministic output gate. If, even after corrective
      regeneration, the answer still contains the Advanced plan / £399 / £2.00 or a
      £-savings claim, the offending content is rewritten in code — the locked plan
      list replaces bad pricing, the 80% phrase replaces invented savings. These two
      failure classes can no longer reach the user under any model behaviour.

v3.4 changes:
  7. LOCKED PRICING moved into the system prompt (client-confirmed, overrides the KB).
     If KB excerpts mention an Advanced plan / £399 / £2.00 they are declared outdated
     and must be ignored. This makes pricing correct even against a stale KB.
  8. £ savings claims banned entirely — the only savings claim allowed is the
     "up to 80% admin time" phrase. Verifier now blocks "save ... £N" patterns,
     the Advanced plan and £399 in answers (regeneration, not just a log line).
  9. Rule 6: a bare number ("8", "40ish") answers the most recent question Nova
     asked — never reassigned to a different field.

v3.3 changes:
  5. Rule 7/8 extended: "no" to a Nova question means drop that thread and pivot;
     an unanswered question is never re-asked verbatim — rephrase once, then pivot.
  6. Deterministic answer verifier (_verify_answer): catches (a) a reply that repeats
     the previous reply, (b) a follow-up question already asked this session, and
     (c) any £ amount not present in the KB excerpts or customer context (invented
     savings/prices). One corrective regeneration is performed; violations are logged.

v3.2 changes (see chat log):
  1. _clean() now strips hollow-filler sentences ("just let me know", "feel free to ask",
     "I'm here to help with anything...") that the model still leaks despite rule 17.
  2. SALES FLOW rule added to _SYSTEM: pricing questions must end by asking fleet size
     when it isn't known, and must lead with the matching plan when it is.
  3. Rule 10 strengthened: reciting the phone number instead of calling escalate_to_team
     is named as a failure.
  4. Locked-fact tripwire: if a reply mentions the removed "Advanced" plan, £399 or £2.00,
     an ERROR is logged — that means the deployed KB is stale and must be fixed + re-embedded.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import requests

from app.config import (
    OPENAI_API_KEY, OPENAI_API_URL, OPENAI_MODEL, OPENAI_SMALL_MODEL,
    LLM_TEMPERATURE, LLM_MAX_TOKENS,
    TOP_K, CHAT_HISTORY_TURNS, SUMMARY_EVERY_TURNS, FOLLOWUP_EVERY,
    MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE, MYSQL_PORT,
)
from app.logger import logger
from app.retriever import retrieve_context

# ─────────────────────────────────────────────────────────────────────────────
# MySQL layer
# ─────────────────────────────────────────────────────────────────────────────


class _DB:
    def __init__(self) -> None:
        self._conn = None
        self._lock = threading.RLock()

    @property
    def enabled(self) -> bool:
        return bool(MYSQL_HOST)

    def _connect(self) -> None:
        import pymysql
        self._conn = pymysql.connect(
            host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER, password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE, charset="utf8mb4", autocommit=True,
            connect_timeout=5, read_timeout=15, write_timeout=15,
            cursorclass=pymysql.cursors.DictCursor,
        )
        logger.info("MySQL connected")

    def run(self, sql: str, params: tuple = (), fetch: str = "none") -> Any:
        """fetch: 'none' → lastrowid | 'one' → row/None | 'all' → list. None on DB failure."""
        if not self.enabled:
            return None
        with self._lock:
            for attempt in (1, 2):
                try:
                    if self._conn is None:
                        self._connect()
                    else:
                        self._conn.ping(reconnect=True)
                    with self._conn.cursor() as cur:
                        cur.execute(sql, params)
                        if fetch == "one":
                            return cur.fetchone()
                        if fetch == "all":
                            return cur.fetchall() or []
                        return cur.lastrowid
                except Exception as e:
                    logger.warning("MySQL error (attempt %d): %s", attempt, e)
                    try:
                        if self._conn:
                            self._conn.close()
                    except Exception:
                        pass
                    self._conn = None
            return None

    def healthy(self) -> bool:
        return self.run("SELECT 1 AS ok", fetch="one") is not None


db = _DB()

_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(100), email VARCHAR(255) UNIQUE, support_id VARCHAR(100),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS chat_history (
        id INT AUTO_INCREMENT PRIMARY KEY,
        session_id VARCHAR(100) NOT NULL, user_id INT DEFAULT NULL,
        sender VARCHAR(20) NOT NULL, message TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_session (session_id), INDEX idx_user (user_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS tickets (
        id INT AUTO_INCREMENT PRIMARY KEY,
        ticket_number VARCHAR(50) UNIQUE, session_id VARCHAR(100), user_id INT DEFAULT NULL,
        email VARCHAR(255), subject VARCHAR(255), message TEXT,
        status VARCHAR(50) DEFAULT 'OPEN', created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS email_captures (
        id INT AUTO_INCREMENT PRIMARY KEY,
        email VARCHAR(255) NOT NULL, session_id VARCHAR(100), question TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS session_state (
        session_id VARCHAR(100) PRIMARY KEY, user_id INT DEFAULT NULL,
        state LONGTEXT NOT NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_user (user_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
]


def ensure_tables() -> None:
    for stmt in _SCHEMA:
        db.run(stmt)


try:
    ensure_tables()
except Exception as e:  # never block import
    logger.warning("ensure_tables: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# Public DB helpers (used by api.py)
# ─────────────────────────────────────────────────────────────────────────────

def db_find_or_create_user(name: str, email: str, support_id: str = "") -> Tuple[int, bool]:
    row = db.run("SELECT id FROM users WHERE email=%s", (email,), fetch="one")
    if row:
        return int(row["id"]), True
    new_id = db.run("INSERT INTO users (name,email,support_id) VALUES (%s,%s,%s)",
                    (name, email, support_id or ""))
    return (int(new_id), False) if new_id else (0, False)


def db_save_message(session_id: str, sender: str, message: str, user_id: int = 0) -> None:
    db.run("INSERT INTO chat_history (session_id,user_id,sender,message) VALUES (%s,%s,%s,%s)",
           (session_id, user_id or None, sender, message))


def db_load_history(session_id: str = "", user_id: int = 0, limit: int = 40) -> List[Dict]:
    if user_id:
        rows = db.run(
            """SELECT sender,message FROM (SELECT sender,message,id FROM chat_history
               WHERE user_id=%s ORDER BY id DESC LIMIT %s) t ORDER BY id ASC""",
            (user_id, limit), fetch="all")
    else:
        rows = db.run(
            """SELECT sender,message FROM (SELECT sender,message,id FROM chat_history
               WHERE session_id=%s ORDER BY id DESC LIMIT %s) t ORDER BY id ASC""",
            (session_id, limit), fetch="all")
    return [{"sender": r["sender"], "message": r["message"]} for r in (rows or [])]


def db_save_email_capture(email: str, session_id: str, question: str) -> None:
    db.run("INSERT INTO email_captures (email,session_id,question) VALUES (%s,%s,%s)",
           (email, session_id, question))


def db_create_ticket(user_id: int, subject: str, message: str,
                     email: str = "", session_id: str = "") -> str:
    """Race-free ticket numbering from the auto-increment id."""
    rid = db.run(
        "INSERT INTO tickets (session_id,user_id,email,subject,message) VALUES (%s,%s,%s,%s,%s)",
        (session_id, user_id or None, email, subject[:255], message))
    if not rid:
        return "TKT-ERR"
    tkt = f"TKT-{1000 + int(rid)}"
    db.run("UPDATE tickets SET ticket_number=%s WHERE id=%s", (tkt, rid))
    return tkt


def db_migrate_guest(guest_session_id: str, user_id: int) -> None:
    """Guest → logged-in: move chat turns and profile onto the user's session so nothing is lost."""
    if not guest_session_id or user_id <= 0 or guest_session_id.startswith("user_"):
        return
    target = f"user_{user_id}"
    db.run("UPDATE chat_history SET user_id=%s, session_id=%s WHERE session_id=%s",
           (user_id, target, guest_session_id))
    existing = db.run("SELECT state FROM session_state WHERE session_id=%s", (target,), fetch="one")
    guest = db.run("SELECT state FROM session_state WHERE session_id=%s", (guest_session_id,), fetch="one")
    if guest and not existing:
        db.run("INSERT INTO session_state (session_id,user_id,state) VALUES (%s,%s,%s)",
               (target, user_id, guest["state"]))
    if guest:
        db.run("DELETE FROM session_state WHERE session_id=%s", (guest_session_id,))
    with _MEM_LOCK:
        if guest_session_id in _MEM_STATE and target not in _MEM_STATE:
            _MEM_STATE[target] = _MEM_STATE.pop(guest_session_id)
        if guest_session_id in _MEM_HIST:
            _MEM_HIST.setdefault(target, []).extend(_MEM_HIST.pop(guest_session_id))


# ─────────────────────────────────────────────────────────────────────────────
# Conversation state — persisted per session
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class State:
    name: Optional[str] = None
    email: Optional[str] = None
    fleet_size: Optional[int] = None
    monthly_fines: Optional[int] = None
    industry: Optional[str] = None
    pain_points: List[str] = field(default_factory=list)
    current_process: Optional[str] = None
    open_question: Optional[str] = None     # question waiting for an email
    awaiting_email: bool = False
    summary: str = ""                       # rolling summary of older turns
    turns: int = 0
    no_question_streak: int = 0             # consecutive bot replies without a follow-up question
    asked_questions: List[str] = field(default_factory=list)  # follow-up questions already asked this session
    escalated: bool = False                 # a ticket has already been logged with the sales/support team
    last_answer: str = ""                   # exact text of the previous bot reply, to block verbatim repeats

    @classmethod
    def from_json(cls, raw: str) -> "State":
        try:
            d = json.loads(raw or "{}")
            return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        except Exception:
            return cls()

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    def plan(self) -> str:
        if not self.fleet_size:
            return ""
        if self.fleet_size <= 50:
            return "Essential (£99/month)"
        if self.fleet_size <= 100:
            return "Core (£199/month)"
        if self.fleet_size <= 200:
            return "Advanced (£399/month)"
        return "Elite (£499/month)"

    def context_block(self) -> str:
        lines = []
        if self.name:            lines.append(f"- Name: {self.name}")
        if self.email:           lines.append(f"- Email on file: {self.email}")
        if self.fleet_size:      lines.append(f"- Fleet size: {self.fleet_size} vehicles → plan: {self.plan()}")
        if self.monthly_fines:   lines.append(f"- Fines per month: {self.monthly_fines}")
        if self.industry:        lines.append(f"- Industry: {self.industry}")
        if self.current_process: lines.append(f"- Current process: {self.current_process}")
        if self.pain_points:     lines.append(f"- Pain points: {', '.join(self.pain_points)}")
        if self.awaiting_email and self.open_question:
            lines.append(f"- WAITING FOR THEIR EMAIL so this can be passed to the team: \"{self.open_question}\"")
        if self.asked_questions:
            lines.append("- Follow-up questions already asked this session (never ask these again, even reworded): "
                          + " | ".join(self.asked_questions[-8:]))
        if self.escalated:
            lines.append("- A ticket is already logged with the team for this session. Do NOT invite them to "
                          "contact sales/the team again or repeat the sales pitch — they're already in the queue.")
        if self.last_answer:
            lines.append("- Your exact previous reply (do NOT restate this, even reworded, unless the user asks "
                          "a genuinely new question that needs it): \"" + self.last_answer[:300] + "\"")
        return "\n".join(lines) if lines else "(nothing known yet)"


_MEM_STATE: Dict[str, State] = {}
_MEM_HIST: Dict[str, List[Dict]] = {}
_MEM_LOCK = threading.Lock()


def load_state(session_id: str) -> State:
    row = db.run("SELECT state FROM session_state WHERE session_id=%s", (session_id,), fetch="one")
    if row:
        return State.from_json(row["state"])
    with _MEM_LOCK:
        return _MEM_STATE.get(session_id) or State()


def save_state(session_id: str, user_id: int, st: State) -> None:
    with _MEM_LOCK:
        _MEM_STATE[session_id] = st
    db.run(
        """INSERT INTO session_state (session_id,user_id,state) VALUES (%s,%s,%s)
           ON DUPLICATE KEY UPDATE state=VALUES(state), user_id=VALUES(user_id)""",
        (session_id, user_id or None, st.to_json()))


def load_history(session_id: str, user_id: int) -> List[Dict]:
    rows = db_load_history(session_id, user_id, limit=CHAT_HISTORY_TURNS * 2)
    if rows:
        return [{"role": "user" if r["sender"] == "user" else "assistant", "content": r["message"]}
                for r in rows]
    with _MEM_LOCK:
        return list(_MEM_HIST.get(session_id, []))[-CHAT_HISTORY_TURNS * 2:]


def save_turn(session_id: str, user_id: int, role: str, content: str) -> None:
    with _MEM_LOCK:
        h = _MEM_HIST.setdefault(session_id, [])
        h.append({"role": role, "content": content})
        if len(h) > CHAT_HISTORY_TURNS * 4:
            _MEM_HIST[session_id] = h[-CHAT_HISTORY_TURNS * 4:]
    db_save_message(session_id, "user" if role == "user" else "bot", content, user_id)


def _extract_question(answer: str) -> Optional[str]:
    """Pull the trailing question (if any) out of a reply so it can be tracked in state."""
    parts = re.split(r"(?<=[.!?])\s+", (answer or "").strip())
    for p in reversed(parts):
        if p.strip().endswith("?"):
            return p.strip()[:160]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI — non-streaming and streaming (with tool-call assembly)
# ─────────────────────────────────────────────────────────────────────────────

def _headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}


def _openai(messages: List[Dict], model: str, max_tokens: int, temperature: float) -> Optional[str]:
    """Simple text completion (rewrite / summary). None on failure."""
    if not OPENAI_API_KEY:
        return None
    try:
        r = requests.post(OPENAI_API_URL, headers=_headers(), timeout=30,
                          json={"model": model, "messages": messages,
                                "max_tokens": max_tokens, "temperature": temperature})
        r.raise_for_status()
        return (r.json()["choices"][0]["message"].get("content") or "").strip()
    except Exception as e:
        logger.error("OpenAI call failed: %s", e)
        return None


def _openai_stream(messages: List[Dict], tools: List[Dict]
                   ) -> Generator[Tuple[str, Any], None, None]:
    """
    Streams the main answer. Yields ("text", chunk) as tokens arrive and finally
    ("done", {"content": str, "tool_calls": [...]}). Yields ("error", msg) on failure.
    """
    if not OPENAI_API_KEY:
        yield ("error", "OPENAI_API_KEY missing")
        return
    payload = {"model": OPENAI_MODEL, "messages": messages, "max_tokens": LLM_MAX_TOKENS,
               "temperature": LLM_TEMPERATURE, "tools": tools, "tool_choice": "auto", "stream": True}
    text: List[str] = []
    calls: Dict[int, Dict[str, str]] = {}
    try:
        with requests.post(OPENAI_API_URL, headers=_headers(), json=payload,
                           stream=True, timeout=(10, 60)) as r:
            if r.status_code != 200:
                yield ("error", f"HTTP {r.status_code}: {r.text[:300]}")
                return
            for raw in r.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                data = raw[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    text.append(delta["content"])
                    yield ("text", delta["content"])
                for tc in delta.get("tool_calls") or []:
                    slot = calls.setdefault(int(tc.get("index", 0)), {"id": "", "name": "", "args": ""})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["args"] += fn["arguments"]
    except Exception as e:
        yield ("error", str(e))
        return
    tool_calls = [{"id": c["id"] or f"call_{i}", "type": "function",
                   "function": {"name": c["name"], "arguments": c["args"] or "{}"}}
                  for i, c in sorted(calls.items())]
    yield ("done", {"content": "".join(text), "tool_calls": tool_calls})


# ─────────────────────────────────────────────────────────────────────────────
# Prompt — TONE ONLY (client's rules). Facts come from the knowledge base.
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM = """You are Nova, the AI assistant for Fine Flow — a UK fleet fine management platform.

Fine Flow's mission: Turning penalties into progress.
Core promise: Cut admin time by up to 80% and never miss a penalty deadline again.

VOICE - write exactly like the client's own sample answers. Their voice: confident, plain, concrete, slightly conversational, zero fluff. Study these patterns and match them:
- Direct verdict first, reason second: "No - and that's intentional." / "Driver logs are not required."
- Reassurance through concreteness: "nothing gets missed, nothing gets buried in email" / "it's flagged for review - nothing breaks and nothing gets missed."
- Plain mechanics, no hype: "It compares each fine against your uploaded logs - checking the vehicle, date, and time."
- Grounded honesty over salesmanship: "every council reviews appeals differently, so there's no guaranteed result."
- Offer-style follow-ups: "Want a breakdown of how it all works?" / "Want me to walk you through how assignment works?"
Where the knowledge base excerpt already contains a well-formed sentence for the fact you need, use that sentence as-is or near-verbatim - the client wrote it deliberately.

PERSONALITY: You talk like a sharp, friendly colleague texting a customer back — not like a script. Use contractions (it's, you'll, we've). Vary your sentence openers and phrasing turn to turn so nothing feels copy-pasted. React to what the person actually said before moving on. Warm, confident, direct. Never robotic, never apologetic, never salesy. UK English. Plain text only — no markdown, bullets, headings or emojis.

══════════════════════════════════════
ABSOLUTE RULES
══════════════════════════════════════

1. NO APOLOGIES — EVER. If you made an error, correct it. Don't say sorry.

2. LENGTH - match the client's sample answers: 2 to 4 sentences for a real question, structured like theirs - the direct answer first, then the concrete mechanics, then (when useful) one offer-style follow-up. One sentence is fine for simple confirmations. Never exceed 5 sentences and never write a wall of prose; if a topic genuinely needs more (full pricing list, step-by-step setup), use short plain lines. Before sending, check: does every sentence add a new fact or move the conversation? Cut any that don't.

2a. EXPLAIN / DETAIL / FULL STEPS REQUESTS - when the user asks to "explain", "explain in detail", "full steps", "walk me through", or similar: give a COMPLETE but CONDENSED summary, not exhaustive detail. Cover every step that exists, but in one short clause per step - e.g. "1. Email ingestion - Gmail is monitored automatically. 2. Data extraction - AI pulls the fine details. 3. Driver assignment - matched by vehicle and time." Never leave a step unlisted just to add more words to an earlier one. Before finishing, mentally check you've reached the last step and closed the sentence - an answer that stops mid-step or mid-list is worse than a shorter complete one. If the full explanation genuinely cannot fit, offer to break it into parts rather than starting an answer you can't finish. When the request is for full detail on how Fine Flow works overall, NEVER re-serve the overview you already gave - walk the six lifecycle stages instead, one short clause each: email ingestion, data extraction, driver assignment, driver action (confirm/dispute), appeal review, outcome tracking.

3. EXACT WORDING — when describing Fine Flow itself, use these phrases:
"Fine Flow is an automated system for managing fines from start to finish"
"keeps the entire process organised, accountable, and under control"
"cut admin time by up to 80%"
"never miss a penalty deadline"

4. KNOWLEDGE — every factual claim must come from the KNOWLEDGE BASE EXCERPTS for this turn. Stick close to the KB's own wording for names, numbers, feature names, plan names and process steps — light rewording for grammar/flow is fine, but don't paraphrase facts into different language or drop precision to sound casual. Where an excerpt says "always include X" or "be precise about roles", obey it exactly. Never invent features, prices, timelines, integrations or steps — and never calculate or estimate a number yourself (a £ saving, a time saving, a fine count) even from real customer figures like fleet size, unless the KB excerpts give you that exact number or formula to use. If the excerpts don't cover it, say you don't have that specific detail and use the escalate_to_team tool — do not guess.

5. LOCKED MEMORY — CUSTOMER CONTEXT shows confirmed facts. NEVER change them. NEVER ask for something already confirmed. If asked "what did I tell you?" or "what do you know about us" — state EVERY confirmed value exactly (fleet size, fines per month, name, email, industry, pain points), not a subset. Weave them into answers naturally the way a colleague who knows the account would - "With your 8 vehicles you'd be on Essential", "Given the 40-odd fines you handle a month..." - at most one such reference per reply, only where it genuinely sharpens the answer. Never recite the whole profile unprompted; never ask for anything already listed.

6. CONTINUITY — read the conversation before replying. When the user answers a question you asked, treat their reply as that answer even if it's one word ("yes", "london", "5", "gmail"). A bare number ("8", "40ish", "around 60") is ALWAYS the answer to the most recent unanswered question YOU asked — if you asked fleet size two messages ago and they now send "8", that's 8 vehicles, even if other topics came in between. Save it with save_customer_details against the right field, and never reassign it to a different field later. Never restart the conversation or fall back to a generic menu mid-thread.

7. COUNTER QUESTIONS — when the user says "no", "what?" or something vague — do NOT reset. Acknowledge briefly and ask a DIFFERENT relevant follow-up. Keep the conversation alive. "No" in reply to a question you asked means they're declining that thread — DROP it completely and pivot to a different useful topic; re-asking the same question after a "no" is a failure. If they say "yes"/"ok" without actually giving the detail you asked for, do not re-ask the identical question: either ask for the specific missing piece in clearly different words with an example ("Which council was it — Lambeth, TfL...?"), or move on to something else you can help with. The same question NEVER appears twice in a conversation, word for word or reworded.

8. PROGRESSION — when the user's message is just an acknowledgement ("yes", "sure", "yeah", "ok", "okay", "sounds good") or a string of them ("ok yes hmm right", "hmm ok") and not a new question — NEVER answer these with the product overview or any pitch:
- Do NOT repeat what you just said. Check "Your exact previous reply" in CUSTOMER CONTEXT — if your new reply would restate the same facts, plan, price or pitch, even in different words, stop and rewrite it.
- If a ticket is already logged (CUSTOMER CONTEXT shows escalated), do NOT re-offer to contact sales or re-pitch the plan — acknowledge briefly and ask if there's something new you can help with, or stay quiet on next steps since the team already has it.
- Otherwise move the conversation FORWARD: take the next concrete action, or ask ONE specific new diagnostic question — never the same ground already covered.

9. FOLLOW-UP QUESTIONS - informational answers about how Fine Flow works SHOULD end with ONE short offer-style follow-up, exactly like the client's samples: "Want a breakdown of how it all works?", "Want me to walk you through how assignment works?", "Tell me what you're seeing and I can help you prioritise." This is the default, not the exception - skip it only when the previous reply also ended with a question the user hasn't answered yet, or the user asked a closed factual question that's now fully settled. NEVER repeat a question you've already asked this conversation, even reworded - check the history and the "already asked" list in CUSTOMER CONTEXT first. Generic filler closers ("Anything else?", "Want to know more?") count as repeats when the same pattern recurs. Sales-diagnostic questions to rotate when they fit and haven't been asked: fleet size, fines per month, current process, biggest pain point, which stage causes most admin. Never ask for something already in CUSTOMER CONTEXT. If every natural question is used up, close cleanly with no question.

10. UNKNOWN QUESTIONS / ISSUES / CONTACT SUPPORT — call escalate_to_team whenever ANY of these happen:
- the user asks to contact the admin, sales or support team, or how to reach them
- the user reports a problem, issue, error, or that something isn't working (e.g. "I'm having trouble", "it's not connecting", "I have a problem with...")
- the user asks something about Fine Flow you can't answer from the knowledge base
Do NOT just recite the phone number or email and stop there — that leaves them unhelped with no ticket raised. Reciting the contact details INSTEAD of calling escalate_to_team is a failure, every time — the tool is how the team actually finds out. Call the tool. If it says no email is on file, ask for their email in one short, natural sentence ("What's the best email for the team to reach you on?"), then call escalate_to_team again once they give it. NEVER say a query has been passed on, or that "the team will assist you", unless the tool actually confirmed it with a ticket number.

11. SMALL TALK — "hi", "help me", "I'm bored", "nothing": one friendly human line, then offer two or three concrete things you can help with (pricing, how fines are captured from Gmail, appeals). Don't sound like a menu.

12. PAYMENT — Fine Flow does NOT pay fines. Always NO. Reason: anti-bot protection. Keep the answer focused on that question.

13. CARD DETAILS — never stored. Say this first when asked.

14. PRICING - LOCKED to the client's knowledge base. These figures are final:
Essential £99/month (5-50 vehicles) | Core £199/month (51-100) | Advanced £399/month (101-200) | Elite £499/month (200+).
Per fine within allowance £0.75 | overage £2.50 | Pay-as-you-go £2.75 per fine, no subscription, zero lock-in.
There are exactly FOUR subscription plans plus PAYG. When listing plans, ALWAYS include all four tiers AND mention PAYG as an alternative - never omit Advanced or PAYG. Every plan includes identical features - no locked features, no paywalls. There is NO £2.00 fee anywhere in Fine Flow. Never invent a plan, price, fee or capacity beyond this list.

14a. SAVINGS CLAIMS - only the knowledge base's own figures, tied to the right fleet size: small fleet (up to 50 vehicles) over £400/month, medium (51-200) over £1,200/month, large (200+) over £4,000/month, alongside "cut admin time by up to 80%". Never state any other £ savings amount and never calculate one yourself.

14b. SALES FLOW — pricing is a conversation, not a price list. When the user asks about pricing, plans, cost or "which plan":
- If fleet size is NOT in CUSTOMER CONTEXT: answer from the KB, then END the reply by asking how many vehicles are in their fleet. This question is mandatory here even if a follow-up wouldn't otherwise be needed.
- If fleet size IS in CUSTOMER CONTEXT: lead with the one plan that fits their fleet size, by name and price, before anything else. Don't re-list all plans unless they ask for the full lineup.
- Once fleet size and monthly fines are both known and they're deciding, recommend the fitting plan once and offer to put them in touch with the sales team once — never re-pitch after that.

15. APPEALS — CORRECT FLOW:
Driver disputes → DISPUTED (driver action)
Admin accepts/rejects → if accepted → appeal letter sent by email → UNDER REVIEW
Admin must accept BEFORE appeal is sent. Never blur the roles.

16. TOPIC — Help with Fine Flow AND UK fleet fine questions (PCNs, councils, TfL, DVLA).
Unrelated (coding, weather, other AI tools): "I'm here to help with fleet fine management — anything about fines, Fine Flow or appeals?"

17. NO HOLLOW ENDINGS — never use "let me know", "just let me know", "feel free to ask", "don't hesitate", or any close variant of them, anywhere in a reply, not just at the end. These are banned phrases, not a style suggestion — a human colleague doesn't tack "let me know!" onto every message. If you'd reach for one of these, either ask the specific follow-up question you actually mean (rule 9) or just stop talking.

18. Call save_customer_details silently whenever the user shares their name, email, fleet size, monthly fine volume, industry, current process or a pain point.

CONTACT: +47 32 28 50 00 | ff.sales@fineflow.com

PROGRESSION EXAMPLES:
After referrals + "yes" → "Do you have a company in mind to refer?"
After appeals + "yes" → "Is there a specific fine you're looking to appeal?"
After "specific fine" + "yes" → "Which council issued it, and when is the payment deadline?"
After plan + "yes" → "Every plan includes automatic fine capture, driver matching, deadline tracking and full appeal management. Want to get in touch with the sales team?"
After billing + "yes" → "What charge is appearing that you weren't expecting?"
After fines-per-month given + no fleet size → "How many vehicles are in your fleet so I can point you to the right plan?"
"""

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "save_customer_details",
            "description": "Persist facts the user has shared about themselves or their fleet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                    "fleet_size": {"type": "integer", "description": "Number of vehicles"},
                    "monthly_fines": {"type": "integer", "description": "Fines/PCNs per month"},
                    "industry": {"type": "string"},
                    "current_process": {"type": "string",
                                        "description": "How they manage fines today (e.g. spreadsheets)"},
                    "pain_point": {"type": "string",
                                   "description": "One problem they mentioned (e.g. missed deadlines)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_team",
            "description": ("Log a question or request for the Fine Flow team to follow up by email. "
                            "Requires the user's email; if none is on file the tool will ask you to collect it."),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string",
                                 "description": "Clear one-line summary of what the user needs"},
                    "email": {"type": "string", "description": "User's email if they just provided it"},
                },
                "required": ["question"],
            },
        },
    },
]

_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")


def _run_tool(name: str, args: Dict[str, Any], st: State,
              session_id: str, user_id: int) -> Dict[str, Any]:
    if name == "save_customer_details":
        if args.get("name"):
            st.name = str(args["name"]).strip()[:100]
        if args.get("email") and _EMAIL_RE.match(str(args["email"]).strip()):
            st.email = str(args["email"]).strip().lower()
        if isinstance(args.get("fleet_size"), int) and 0 < args["fleet_size"] <= 50000:
            st.fleet_size = args["fleet_size"]
        if isinstance(args.get("monthly_fines"), int) and 0 < args["monthly_fines"] < 100000:
            st.monthly_fines = args["monthly_fines"]
        if args.get("industry"):
            st.industry = str(args["industry"]).strip()[:60]
        if args.get("current_process"):
            st.current_process = str(args["current_process"]).strip()[:200]
        pp = (args.get("pain_point") or "").strip()[:120]
        if pp and pp.lower() not in [x.lower() for x in st.pain_points]:
            st.pain_points.append(pp)
        return {"ok": True, "customer_context": st.context_block()}

    if name == "escalate_to_team":
        question = (args.get("question") or st.open_question or "General enquiry").strip()
        email = (args.get("email") or "").strip().lower() or (st.email or "")
        if not email or not _EMAIL_RE.match(email):
            st.awaiting_email = True
            st.open_question = question
            return {"ok": False, "reason": "no_email_on_file",
                    "instruction": ("Ask the user for their email address in one short, natural sentence "
                                    "so the team can follow up. Do NOT say the query has been passed on yet.")}
        st.email = email
        st.awaiting_email = False
        st.open_question = None
        st.escalated = True
        db_save_email_capture(email, session_id, question)
        tkt = db_create_ticket(user_id, question[:120], question, email=email, session_id=session_id)
        ref = f" as {tkt}" if tkt != "TKT-ERR" else ""
        return {"ok": True, "ticket_number": tkt, "email": email,
                "instruction": (f"Confirm briefly that the query has been logged{ref} and the Fine Flow "
                                f"team will follow up at {email}. Then ask one short follow-up question.")}

    return {"ok": False, "error": f"unknown tool {name}"}


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval query rewrite + rolling summary
# ─────────────────────────────────────────────────────────────────────────────

def _standalone_query(query: str, history: List[Dict], st: State) -> str:
    """Turn 'yes' / 'london' / 'and gmail?' into a self-contained KB search query."""
    if not history or len(query.split()) > 14:
        return query
    convo = "\n".join(f"{m['role'].upper()}: {m['content'][:300]}" for m in history[-6:])
    msgs = [
        {"role": "system", "content":
            "Rewrite the user's latest message as one self-contained search query for a Fine Flow "
            "(UK fleet fine management software) knowledge base, using the conversation for context. "
            "Keep the user's intent; expand pronouns and one-word replies. Match the LITERAL question "
            "being asked now - do not drift to the previous turn's topic. If they ask for more/full "
            "detail on how Fine Flow works, target: fine lifecycle stages ingestion extraction "
            "assignment dispute appeal outcome. Output ONLY the query."},
        {"role": "user", "content":
            f"Summary so far: {st.summary or '(none)'}\n\nConversation:\n{convo}\n\nLatest user message: {query}"},
    ]
    text = (_openai(msgs, OPENAI_SMALL_MODEL, 60, 0.0) or "").strip().strip('"')
    return text if 3 <= len(text) <= 300 else query


def _update_summary(st: State, history: List[Dict]) -> None:
    convo = "\n".join(f"{m['role'].upper()}: {m['content'][:400]}"
                      for m in history[-SUMMARY_EVERY_TURNS * 2:])
    msgs = [
        {"role": "system", "content":
            "Maintain a concise running summary (max 120 words) of a support chat between Nova (Fine Flow "
            "assistant) and a customer. Keep: what they asked about, facts they shared, decisions, open "
            "issues, what Nova last asked. Drop pleasantries. Output only the summary."},
        {"role": "user", "content": f"Previous summary:\n{st.summary or '(none)'}\n\nNew turns:\n{convo}"},
    ]
    text = (_openai(msgs, OPENAI_SMALL_MODEL, 220, 0.2) or "").strip()
    if text:
        st.summary = text[:1200]


# ─────────────────────────────────────────────────────────────────────────────
# Output hygiene
# ─────────────────────────────────────────────────────────────────────────────

def _clean_chunk(t: str) -> str:
    return t.replace("**", "").replace("`", "")


# v3.2: hollow-filler sentences the model still leaks despite rule 17.
# A sentence is dropped when it matches one of these AND carries no real content
# (no question mark, no digits, no specific instruction words).
_HOLLOW_PAT = re.compile(
    r"(just\s+let\s+me\s+know|let\s+me\s+know\b|feel\s+free\s+to\s+ask|"
    r"don'?t\s+hesitate|happy\s+to\s+help|here\s+to\s+help\s+with\s+anything|"
    r"any\s+other\s+questions|if\s+you\s+need\s+(any\s+)?(more\s+|further\s+)?"
    r"(help|details|information|assistance))", re.I)
_SUBSTANCE_PAT = re.compile(r"[\d£?]|which|what|when|where|how\s+many|go\s+to|click|settings", re.I)


def _strip_hollow(text: str) -> str:
    """Remove filler sentences ('If you need more help, just let me know!')."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = []
    for s in sentences:
        if _HOLLOW_PAT.search(s) and not _SUBSTANCE_PAT.search(s):
            continue
        kept.append(s)
    out = " ".join(kept).strip()
    return out if out else text  # never return empty because everything was filler


def _clean(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"(?m)^\s*#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s*[-*•]\s+", "", text)
    text = text.replace("`", "")
    text = _strip_hollow(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# v3.2: locked-fact tripwire. These figures were removed from the KB by client
# decision (§3 of the handoff report). If they appear in an answer, the deployed
# KB is stale — fix the KB file and re-embed (rm -rf data/chroma_db + restart).
_LOCKED_VIOLATION = re.compile(r"£2\.00\b")


_MONEY_RE = re.compile(r"£\s?([\d,]+(?:\.\d+)?)")


def _norm_cmp(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s?]", "", (t or "").lower())).strip()


def _find_violations(answer: str, st: "State", kb_ctx: str) -> List[str]:
    """Deterministic checks the model keeps failing despite prompt rules."""
    v: List[str] = []
    # (a) whole-reply repeat of the previous reply (exact OR near-verbatim)
    import difflib
    if st.last_answer and (
            _norm_cmp(answer) == _norm_cmp(st.last_answer)
            or difflib.SequenceMatcher(None, _norm_cmp(answer),
                                       _norm_cmp(st.last_answer)).ratio() >= 0.9):
        v.append("Your reply is (near-)identical to your previous reply. Say something NEW: "
                 "if they asked for more detail, go one level deeper (e.g. the six lifecycle "
                 "stages, one clause each); otherwise pivot to a different useful angle or "
                 "question. Do not restate the same sentences.")
    # (a2) hollow filler phrase survived cleaning (single-sentence filler replies)
    if _HOLLOW_PAT.search(answer):
        v.append("Your reply contains a banned filler phrase ('let me know' / 'feel free to ask' "
                 "or similar, rule 17). Replace the reply with something real: either one fresh, "
                 "specific follow-up question not yet asked, or a clean close with no question.")
    # (b) re-asked question
    q = _extract_question(answer)
    if q:
        qn = _norm_cmp(q)
        for prev in st.asked_questions:
            if _norm_cmp(prev) == qn:
                v.append(f"You already asked this exact question earlier: \"{prev}\". "
                         "Never repeat a question. Either ask for the missing detail in clearly "
                         "different words with an example, or drop the thread and pivot.")
                break
    # (c0) the only forbidden fee - £2.00 does not exist anywhere in Fine Flow
    if re.search(r"£2\.00\b", answer):
        v.append("You mentioned a £2.00 fee. There is NO £2.00 fee anywhere in Fine Flow (rule 14). "
                 "The real figures are £0.75 within allowance, £2.50 overage, £2.75 PAYG.")
    # (c) invented £ amounts (savings/prices not grounded in KB or customer context)
    allowed = set(m.replace(",", "") for m in _MONEY_RE.findall(
        (kb_ctx or "") + " " + st.context_block() + " 99 199 399 499 0.75 2.50 2.75 400 1200 1,200 4000 4,000 75"))
    allowed.discard("2.00")  # the one fee that does not exist
    for amt in _MONEY_RE.findall(answer):
        if amt.replace(",", "") not in allowed:
            v.append(f"£{amt} does not appear in the knowledge base excerpts — you invented or "
                     "calculated it, which is forbidden (rule 4). Remove the figure; describe the "
                     "benefit without a made-up number.")
    return v


def _regenerate(msgs: List[Dict], draft: str, violations: List[str]) -> Optional[str]:
    """One corrective, non-streaming, tool-free regeneration."""
    fix_msgs = list(msgs)
    fix_msgs.append({"role": "assistant", "content": draft})
    fix_msgs.append({"role": "user", "content":
        "SYSTEM CORRECTION — your draft above violates these rules:\n- "
        + "\n- ".join(violations)
        + "\nRewrite the reply now, fixing every violation. Keep it to 1-2 sentences, same tone, "
          "plain text, no apologies. Output only the corrected reply."})
    return _openai(fix_msgs, OPENAI_MODEL, LLM_MAX_TOKENS, 0.3)


def _hard_guard(answer: str, st: "State") -> str:
    """Last-resort deterministic gate: the £2.00 fee does not exist and can never
    reach the user. Everything else is handled by the verifier + regeneration."""
    out = answer
    if re.search(r"£2\.00\b", out):
        logger.error("HARD-GUARD £2.00 rewrite fired; original=%r", out[:200])
        sentences = re.split(r"(?<=[.!?])\s+", out)
        kept = [s for s in sentences if "£2.00" not in s]
        out = " ".join(kept).strip() or ("Fines are £0.75 each within your allowance, £2.50 "
                                          "per fine beyond it, and £2.75 on pay-as-you-go.")
    return out


_OUTAGE = ("I'm having trouble reaching my knowledge base right now. The Fine Flow team can help "
           "directly on +47 32 28 50 00 or ff.sales@fineflow.com.")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry — streaming generator + blocking wrapper
# ─────────────────────────────────────────────────────────────────────────────

def build_response_stream(query: str, session_id: str = "default", user_id: int = 0
                          ) -> Generator[Union[str, Dict[str, Any]], None, None]:
    """
    Yields str text chunks as they arrive, then ONE final dict:
    {"answer": str, "request_email": bool, "trigger_ticket_popup": False}
    NOTE: streamed chunks are raw model output; the final dict carries the cleaned
    answer (hollow filler stripped). The widget should replace the streamed text
    with the final answer when it arrives.
    """
    query = (query or "").strip()
    session_id = session_id or "default"
    if not query:
        a = "Ask me anything about Fine Flow — fines, pricing, appeals or setup. What's on your mind?"
        yield a
        yield {"answer": a, "request_email": False, "trigger_ticket_popup": False}
        return

    st = load_state(session_id)
    history = load_history(session_id, user_id)
    st.turns += 1

    # 1. Retrieval (history-aware)
    search_q = _standalone_query(query, history, st)
    kb_ctx, _ = retrieve_context(search_q, top_k=TOP_K)
    logger.info("session=%s q=%r → search=%r", session_id, query[:80], search_q[:80])

    # 2. Messages
    system = (
        _SYSTEM
        + "\n\nCUSTOMER CONTEXT (confirmed facts — never ask for these again):\n" + st.context_block()
        + ("\n\nCONVERSATION SUMMARY SO FAR:\n" + st.summary if st.summary else "")
    )
    reminder = ""
    if st.no_question_streak >= max(1, FOLLOWUP_EVERY - 1):
        reminder = ("\n\nREMINDER: your previous reply had no question. If there's a genuinely new, relevant "
                    "question left to ask (check the already-asked list above first), this reply should end "
                    "with it (rule 9). If every natural question has already been asked, close cleanly instead.")
    user_block = (
        f"KNOWLEDGE BASE EXCERPTS (retrieved for this message):\n{kb_ctx or '(nothing relevant found)'}"
        f"{reminder}\n\nUSER MESSAGE:\n{query}"
    )
    msgs: List[Dict[str, Any]] = [{"role": "system", "content": system}]
    msgs.extend(history)
    msgs.append({"role": "user", "content": user_block})

    # 3. Model + tool loop (streams text; assembles tool calls)
    answer_parts: List[str] = []
    failed = False
    for _round in range(4):
        result: Optional[Dict[str, Any]] = None
        for kind, payload in _openai_stream(msgs, _TOOLS):
            if kind == "text":
                c = _clean_chunk(payload)
                if c:
                    answer_parts.append(c)
                    yield c
            elif kind == "done":
                result = payload
            else:
                logger.error("OpenAI stream error: %s", payload)
        if result is None:
            failed = True
            break
        calls = result.get("tool_calls") or []
        if not calls:
            break
        msgs.append({"role": "assistant", "content": result.get("content") or None, "tool_calls": calls})
        for tc in calls:
            fn = tc["function"]
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:
                args = {}
            out = _run_tool(fn.get("name", ""), args, st, session_id, user_id)
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(out)})

    answer = _clean("".join(answer_parts))
    if failed and not answer:
        answer = _OUTAGE
        yield answer
    if not answer:
        answer = "What would you like to know about Fine Flow — pricing, appeals, or how fines are captured?"
        yield answer

    # v3.3: deterministic verifier — repetition and invented £ figures get one corrective pass.
    violations = _find_violations(answer, st, kb_ctx)
    if violations and answer not in (_OUTAGE,):
        logger.warning("VERIFIER session=%s violations=%s", session_id, violations)
        fixed = _regenerate(msgs, answer, violations)
        if fixed:
            fixed = _clean(fixed)
            if fixed and not _find_violations(fixed, st, kb_ctx):
                answer = fixed
            elif fixed:
                answer = fixed  # still better than a verbatim repeat; violations logged above

    # v3.5: final deterministic gate — worst failure classes cannot pass this line.
    answer = _hard_guard(answer, st)

    # v3.2 tripwire: removed plans/prices appearing in an answer = stale KB deployed.
    if _LOCKED_VIOLATION.search(answer):
        logger.error("LOCKED-FACT VIOLATION session=%s answer=%r — £2.00 fee mentioned; "
                     "this fee does not exist.", session_id, answer[:200])

    # 4. Persist (SQL only)
    asked_q = _extract_question(answer)
    st.no_question_streak = 0 if asked_q else st.no_question_streak + 1
    if asked_q and asked_q.lower() not in [q.lower() for q in st.asked_questions]:
        st.asked_questions.append(asked_q)
        st.asked_questions = st.asked_questions[-20:]
    st.last_answer = answer
    save_turn(session_id, user_id, "user", query)
    save_turn(session_id, user_id, "assistant", answer)
    if st.turns % SUMMARY_EVERY_TURNS == 0:
        _update_summary(st, history + [{"role": "user", "content": query},
                                       {"role": "assistant", "content": answer}])
    save_state(session_id, user_id, st)

    yield {"answer": answer, "request_email": bool(st.awaiting_email), "trigger_ticket_popup": False}


def build_response(query: str, session_id: str = "default", user_id: int = 0) -> Dict[str, Any]:
    final: Dict[str, Any] = {}
    for item in build_response_stream(query, session_id, user_id):
        if isinstance(item, dict):
            final = item
    return final or {"answer": _OUTAGE, "request_email": False, "trigger_ticket_popup": False}


def answer_sync(q: str, session_id: str = "default", user_id: int = 0) -> Dict[str, Any]:
    try:
        return build_response(q, session_id, user_id)
    except Exception:
        logger.exception("build_response crashed")
        return {"answer": "Something went wrong on my side — please try that again.",
                "request_email": False, "trigger_ticket_popup": False}