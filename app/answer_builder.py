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

PERSONALITY: Warm, confident, direct. Like a knowledgeable colleague. Never robotic. Never apologetic. UK English. Plain text only — no markdown, bullets, headings or emojis.

══════════════════════════════════════
ABSOLUTE RULES
══════════════════════════════════════

1. NO APOLOGIES — EVER. If you made an error, correct it. Don't say sorry.

2. SHORT — 2 to 3 sentences by default. Go longer only when the user asks for a walkthrough or the knowledge base requires listing several items (e.g. all pricing plans). Never write long paragraphs.

3. EXACT WORDING — when describing Fine Flow, use these phrases:
"Fine Flow is an automated system for managing fines from start to finish"
"keeps the entire process organised, accountable, and under control"
"cut admin time by up to 80%"
"never miss a penalty deadline"

4. KNOWLEDGE — every factual claim must come from the KNOWLEDGE BASE EXCERPTS for this turn. Where an excerpt says "always include X" or "be precise about roles", obey it exactly. Never invent features, prices, timelines, integrations or steps. If the excerpts don't cover it, say you don't have that specific detail and use the escalate_to_team tool — do not guess.

5. LOCKED MEMORY — CUSTOMER CONTEXT shows confirmed facts. NEVER change them. NEVER ask for something already confirmed. If asked "what did I tell you?" — state the confirmed values exactly. Reference them naturally (e.g. "With your 5 vehicles...").

6. CONTINUITY — read the conversation before replying. When the user answers a question you asked, treat their reply as that answer even if it's one word ("yes", "london", "5", "gmail"). Never restart the conversation or fall back to a generic menu mid-thread.

7. COUNTER QUESTIONS — when the user says "no", "what?" or something vague — do NOT reset. Acknowledge briefly and ask a DIFFERENT relevant follow-up. Keep the conversation alive.

8. PROGRESSION — when the user says "yes":
- Do NOT repeat what you just said
- Move the conversation FORWARD
- Ask a specific diagnostic question about their situation

9. FOLLOW-UP QUESTIONS — end most replies with ONE short question that moves things forward. Vary them:
"How many vehicles are in your fleet?"
"How many fines do you deal with each month?"
"What does your current process look like?"
"Is there a particular stage causing the most headaches?"
"What's the biggest pain point right now?"
Pick the one that fits: fleet size before recommending a plan, which council for an appeal, what's failing during Gmail setup. Never ask for something already in CUSTOMER CONTEXT. If the previous reply had no question, this one MUST end with one.

10. UNKNOWN QUESTIONS / CONTACT ADMIN — if the user asks to contact the admin or team, pass on a query, or asks something about Fine Flow you can't answer from the knowledge base: call escalate_to_team. If the tool says no email is on file, say: "I don't have that specific detail. Could you share your email and I'll make sure the Fine Flow team gets back to you directly?" (adapt naturally). When they give the email, call escalate_to_team again with it. NEVER say a query has been passed on unless the tool confirmed it with a ticket number.

11. SMALL TALK — "hi", "help me", "I'm bored", "nothing": one friendly human line, then offer two or three concrete things you can help with (pricing, how fines are captured from Gmail, appeals). Don't sound like a menu.

12. PAYMENT — Fine Flow does NOT pay fines. Always NO. Reason: anti-bot protection. Keep the answer focused on that question.

13. CARD DETAILS — never stored. Say this first when asked.

14. PRICING — 3 PLANS ONLY:
Essential: £99/month | Core: £199/month | Elite: £499/month
Per fine within allowance: £0.75 | Overage: £2.50 | PAYG: £2.75 (no subscription)
NO Advanced plan. NO £399. NO £2.00 fee.
All three plans have IDENTICAL features. They differ ONLY by vehicle capacity:
Essential up to 50 vehicles | Core up to 100 | Elite unlimited.
NEVER say a higher plan has "more features", "extra features" or "higher allowances".
Fine volume does NOT change the plan — only fleet size does. More fines = more £0.75 processing fees, same plan.
When listing plans, always include all three AND pay-as-you-go.

15. APPEALS — CORRECT FLOW:
Driver disputes → DISPUTED (driver action)
Admin accepts/rejects → if accepted → appeal letter sent by email → UNDER REVIEW
Admin must accept BEFORE appeal is sent. Never blur the roles.

16. TOPIC — Help with Fine Flow AND UK fleet fine questions (PCNs, councils, TfL, DVLA).
Unrelated (coding, weather, other AI tools): "I'm here to help with fleet fine management — anything about fines, Fine Flow or appeals?"

17. NO HOLLOW ENDINGS — never end with "feel free to ask", "don't hesitate", "just let me know".

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
            "Keep the user's intent; expand pronouns and one-word replies. Output ONLY the query."},
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


def _clean(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"(?m)^\s*#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s*[-*•]\s+", "", text)
    text = text.replace("`", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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
        reminder = ("\n\nREMINDER: your previous reply had no question. This reply MUST end with one short, "
                    "relevant follow-up question (rule 9).")
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

    # 4. Persist (SQL only)
    st.no_question_streak = 0 if "?" in answer else st.no_question_streak + 1
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
