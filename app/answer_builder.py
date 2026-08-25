# app/answer_builder.py
"""
FineFlow Nova — Production Final
Features:
  1. Strong muscle memory — profile locked, conversation history always passed to GPT
  2. Every message saved to MySQL chat_history
  3. Unknown questions → ask for email → save to email_captures table
  4. Counter questions always work — "no" never resets conversation
  5. Topic progression — "yes" moves forward, never repeats
"""

import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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

MAX_FLEET = 50_000
TICKET_THRESHOLD = 0.30  # Only escalate when very low confidence AND no RAG context

# ─────────────────────────────────────────────────────────────────────────────
# MySQL — single shared connection with auto-reconnect
# ─────────────────────────────────────────────────────────────────────────────

_conn = None
_db_lock = threading.Lock()


def _get_conn():
    global _conn
    host = os.getenv("MYSQL_HOST", "")
    if not host:
        return None
    with _db_lock:
        try:
            import pymysql
            if _conn is None or not _conn.open:
                raise Exception("reconnect")
            _conn.ping(reconnect=False)
            return _conn
        except Exception:
            try:
                import pymysql
                _conn = pymysql.connect(
                    host=host,
                    user=os.getenv("MYSQL_USER", ""),
                    password=os.getenv("MYSQL_PASSWORD", ""),
                    database=os.getenv("MYSQL_DATABASE", ""),
                    charset="utf8mb4",
                    autocommit=True,
                    connect_timeout=5,
                    cursorclass=pymysql.cursors.DictCursor,
                )
                logger.info("MySQL connected")
                return _conn
            except Exception as e:
                logger.warning("MySQL unavailable: %s", e)
                _conn = None
                return None


def _ensure_tables():
    conn = _get_conn()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100), email VARCHAR(255) UNIQUE,
                support_id VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
            cur.execute("""CREATE TABLE IF NOT EXISTS chat_history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(100) NOT NULL,
                user_id INT DEFAULT NULL,
                sender VARCHAR(20) NOT NULL,
                message TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_session (session_id),
                INDEX idx_user (user_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
            cur.execute("""CREATE TABLE IF NOT EXISTS tickets (
                id INT AUTO_INCREMENT PRIMARY KEY,
                ticket_number VARCHAR(50) UNIQUE,
                session_id VARCHAR(100), user_id INT DEFAULT NULL,
                email VARCHAR(255), subject VARCHAR(255),
                message TEXT, status VARCHAR(50) DEFAULT 'OPEN',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
            cur.execute("""CREATE TABLE IF NOT EXISTS email_captures (
                id INT AUTO_INCREMENT PRIMARY KEY,
                email VARCHAR(255) NOT NULL,
                session_id VARCHAR(100),
                question TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
    except Exception as e:
        logger.warning("Table setup: %s", e)


try:
    _ensure_tables()
except Exception:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# DB helpers — all called from api.py too
# ─────────────────────────────────────────────────────────────────────────────

def db_find_or_create_user(name: str, email: str, support_id: str = "") -> Tuple[int, bool]:
    conn = _get_conn()
    if not conn:
        return 0, False
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE email=%s", (email,))
            row = cur.fetchone()
            if row:
                return row["id"], True
            cur.execute("INSERT INTO users (name,email,support_id) VALUES (%s,%s,%s)",
                        (name, email, support_id or ""))
            return cur.lastrowid, False
    except Exception as e:
        logger.warning("db_find_or_create_user: %s", e)
        return 0, False


def db_save_message(session_id: str, sender: str, message: str, user_id: int = 0) -> None:
    conn = _get_conn()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chat_history (session_id,user_id,sender,message) VALUES (%s,%s,%s,%s)",
                (session_id, user_id or None, sender, message)
            )
    except Exception as e:
        logger.warning("db_save_message: %s", e)


def db_load_history(session_id: str, user_id: int = 0, limit: int = 40) -> List[Dict]:
    conn = _get_conn()
    if not conn:
        return []
    try:
        with conn.cursor() as cur:
            if user_id:
                cur.execute(
                    """SELECT sender,message FROM (
                        SELECT sender,message,created_at FROM chat_history
                        WHERE user_id=%s ORDER BY created_at DESC LIMIT %s
                    ) sub ORDER BY created_at ASC""",
                    (user_id, limit)
                )
            else:
                cur.execute(
                    """SELECT sender,message FROM (
                        SELECT sender,message,created_at FROM chat_history
                        WHERE session_id=%s ORDER BY created_at DESC LIMIT %s
                    ) sub ORDER BY created_at ASC""",
                    (session_id, limit)
                )
            return [{"sender": r["sender"], "message": r["message"]} for r in cur.fetchall()]
    except Exception as e:
        logger.warning("db_load_history: %s", e)
        return []


def db_save_email_capture(email: str, session_id: str, question: str) -> None:
    conn = _get_conn()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO email_captures (email,session_id,question) VALUES (%s,%s,%s)",
                (email, session_id, question)
            )
    except Exception as e:
        logger.warning("db_save_email_capture: %s", e)


def db_create_ticket(user_id: int, subject: str, message: str,
                     email: str = "", session_id: str = "") -> str:
    conn = _get_conn()
    if not conn:
        return "TKT-ERR"
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS cnt FROM tickets")
            row = cur.fetchone()
            num = (row["cnt"] if row else 0) + 1001
            tkt = f"TKT-{num}"
            cur.execute(
                "INSERT INTO tickets (ticket_number,session_id,user_id,email,subject,message) "
                "VALUES (%s,%s,%s,%s,%s,%s)",
                (tkt, session_id, user_id or None, email, subject, message)
            )
            return tkt
    except Exception as e:
        logger.warning("db_create_ticket: %s", e)
        return "TKT-ERR"


def db_load_chat_history_for_user(user_id: int) -> List[Dict]:
    return db_load_history("", user_id, limit=200)


# ─────────────────────────────────────────────────────────────────────────────
# Customer Profile — LOCKED values, never overwritten by LLM
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Profile:
    fleet:    Optional[int]  = None
    volume:   Optional[int]  = None
    issues:   List[str]      = field(default_factory=list)
    industry: Optional[str]  = None
    name:     Optional[str]  = None
    email:    Optional[str]  = None
    turns:    int            = 0
    # Track last unanswered question (for email capture)
    pending_question: Optional[str] = None

    def set_fleet(self, n: int) -> None:
        if self.fleet is None and 0 < n <= MAX_FLEET:
            self.fleet = n

    def set_volume(self, n: int) -> None:
        if self.volume is None and 0 < n < 10_000:
            self.volume = n

    def summary(self) -> str:
        parts = []
        if self.name:     parts.append(f"Customer name: {self.name}")
        if self.email:    parts.append(f"Email: {self.email}")
        if self.fleet:    parts.append(f"Fleet size: {self.fleet} vehicles [CONFIRMED]")
        if self.volume:   parts.append(f"Monthly fines: {self.volume} [CONFIRMED]")
        if self.industry: parts.append(f"Industry: {self.industry}")
        if self.issues:   parts.append(f"Problems: {', '.join(self.issues)}")
        return "\n".join(parts)

    def plan_name(self) -> str:
        if not self.fleet: return ""
        if self.fleet <= 50:   return "Essential"
        if self.fleet <= 100:  return "Core"
        return "Elite"

    def plan_price(self) -> str:
        if not self.fleet: return ""
        if self.fleet <= 50:   return "£99"
        if self.fleet <= 100:  return "£199"
        return "£499"


# ─────────────────────────────────────────────────────────────────────────────
# In-memory session store
# ─────────────────────────────────────────────────────────────────────────────

_SES: Dict[str, List[Dict]] = {}   # conversation history
_PRO: Dict[str, Profile]    = {}   # customer profiles
_MET: Dict[str, Dict]       = {}   # metadata (last topic, last question, etc.)
_LK  = threading.Lock()


def _mem_hist(sid: str) -> List[Dict]:
    with _LK: return list(_SES.get(sid, []))


def _mem_push(sid: str, role: str, content: str) -> None:
    with _LK:
        h = _SES.setdefault(sid, [])
        h.append({"role": role, "content": content})
        cap = CHAT_HISTORY_TURNS * 2
        if len(h) > cap:
            _SES[sid] = h[-cap:]


def _pro(sid: str) -> Profile:
    with _LK:
        if sid not in _PRO: _PRO[sid] = Profile()
        return _PRO[sid]


def _sm(sid, k, v):
    with _LK: _MET.setdefault(sid, {})[k] = v

def _gm(sid, k):
    with _LK: return _MET.get(sid, {}).get(k)

def _rst(sid):
    with _LK: _MET[sid] = {}; _PRO[sid] = Profile()

def _inc_aff(sid):
    with _LK:
        m = _MET.setdefault(sid, {})
        c = m.get("aff", 0) + 1; m["aff"] = c; return c

def _rst_aff(sid):
    with _LK: _MET.setdefault(sid, {})["aff"] = 0

def _ask_now(sid) -> bool:
    """Ask follow-up on every other response."""
    with _LK:
        m = _MET.setdefault(sid, {})
        c = m.get("rc", 0) + 1; m["rc"] = c
        return c % 2 == 0


def _get_hist(sid: str, uid: int = 0) -> List[Dict]:
    """Get history from MySQL if available, fall back to in-memory."""
    if _get_conn():
        rows = db_load_history(sid, uid)
        if rows:
            return [
                {"role": "user" if r["sender"] == "user" else "assistant",
                 "content": r["message"]}
                for r in rows
            ]
    return _mem_hist(sid)


def _save_turn(sid: str, role: str, content: str, uid: int = 0) -> None:
    """Save to both memory and MySQL."""
    _mem_push(sid, role, content)
    db_save_message(sid, "user" if role == "user" else "bot", content, uid)
    # FIX: track the last bot answer so progression logic can avoid repeating it
    if role != "user":
        _sm(sid, "last_bot_answer", content)


# ─────────────────────────────────────────────────────────────────────────────
# Profile extraction from messages
# ─────────────────────────────────────────────────────────────────────────────

_FINES_RE = re.compile(
    r"\b(\d+)\s*(?:fines?|pcns?|penalties|violations?|tickets?)"
    r"(?:\s*(?:per|a|each|every)\s*(?:month|monthly|week))?\b", re.I)
_IND_RE   = re.compile(
    r"\b(logistics|delivery|courier|haulage|transport|taxi|minicab|bus|coach|construction)\b", re.I)
_NAME_RE  = re.compile(r"\b(?:i am|i'm|my name is|call me)\s+([A-Z][a-z]+)\b")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_ISS = [
    (re.compile(r"\b(miss(?:ed?|ing)?\s+(?:deadlines?|appeals?))\b", re.I), "missed deadlines"),
    (re.compile(r"\b(drivers?\s+(?:dispute|deny|ignor))\b", re.I),          "driver disputes"),
    (re.compile(r"\b(spreadsheet)\b", re.I),                                  "using spreadsheets"),
    (re.compile(r"\b(too\s+much\s+(admin|time|work))\b", re.I),              "too much admin"),
]
_FINES_CTX = {"how many fines", "fines per month", "fines a month", "monthly fines", "fines do you"}
_VEH_CTX   = {"how many vehicles", "fleet size", "vehicles are in", "how big is your fleet",
              "vehicles are you running"}


def _upd(sid: str, q: str) -> None:
    p = _pro(sid)
    p.turns += 1
    m = _NAME_RE.search(q)
    if m and not p.name: p.name = m.group(1)
    m = _EMAIL_RE.search(q)
    if m and not p.email: p.email = m.group()
    m = _FINES_RE.search(q)
    if m: p.set_volume(int(m.group(1)))
    m = _IND_RE.search(q)
    if m: p.industry = m.group(1).lower()
    for pat, lbl in _ISS:
        if pat.search(q) and lbl not in p.issues: p.issues.append(lbl)


def _resolve_bare_number(n: int, sid: str) -> Optional[str]:
    """
    FIX: the last bot message can mention BOTH fines and vehicles
    ("Got it — 45 fines a month. How many vehicles are in your fleet?").
    Whichever context phrase appears LAST is the question actually being asked.
    """
    last_q = (_gm(sid, "last_nova_q") or "").lower()
    best, best_pos = None, -1
    for h in _FINES_CTX:
        pos = last_q.rfind(h)
        if pos > best_pos:
            best, best_pos = "fines", pos
    for h in _VEH_CTX:
        pos = last_q.rfind(h)
        if pos > best_pos:
            best, best_pos = "vehicle", pos
    if best:
        return best
    lt = (_gm(sid, "lt") or "").lower()
    if lt in {"pricing", "plan_recommendation", "cost"}: return "vehicle"
    if lt in {"fines_volume", "savings"}:                return "fines"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Topic shortcuts — single words → full answers
# ─────────────────────────────────────────────────────────────────────────────

_SHORTCUTS: Dict[str, str] = {
    "pricing":   "how much does fine flow cost and what are the plans",
    "price":     "how much does fine flow cost",
    "cost":      "how much does fine flow cost",
    "plans":     "what are the fine flow subscription plans",
    "appeals":   "how do i make an appeal in fine flow",
    "appeal":    "how do i make an appeal in fine flow",
    "fines":     "what happens to a fine when it enters fine flow",
    "billing":   "how does billing work and when am i charged",
    "dashboard": "what does the fine flow dashboard show",
    "drivers":   "how do i add and manage drivers",
    "driver":    "how do i assign a driver to a fine",
    "security":  "how secure is fine flow and is it gdpr compliant",
    "gdpr":      "is fine flow gdpr compliant",
    "referral":  "how does the referral programme work",
    "referrals": "how does the referral programme work",
    "features":  "what features does fine flow include",
    "savings":   "how much time and money can fine flow save",
    "contact":   "how do i contact fine flow",
    "gmail":     "how does fine flow connect to gmail",
    "email":     "how does fine flow get fines from email",
    "payg":      "is there a pay as you go option",
    "overage":   "what is the overage charge",
    "reports":   "what reports can i export",
    "statuses":  "what do the fine statuses mean",
    "matching":  "how does fine flow match a fine to a driver",
    "overdue":   "what happens when a fine becomes overdue",
    "assign":    "how do i assign a driver to a fine",
    "insights":  "what are smart insights in fine flow",
    "upload":    "can i manually upload fines",
}


# ─────────────────────────────────────────────────────────────────────────────
# Intent sets
# ─────────────────────────────────────────────────────────────────────────────

_GREET = {
    "hi","hello","hey","hiya","howdy","yo","sup",
    "good morning","good afternoon","good evening","morning","afternoon","evening",
    "hi there","hey there","hello there","hi nova","hey nova","hello nova",
}
_SOC = {
    "how are you","how are you doing","how r u","how are u",
    "hows it going","how is it going","whats up","what s up",
    "you ok","you good","how do you do","you alright","alright mate",
}
_ID = {
    "who are you","who r you","who is nova","who is this","who is there",
    "whos there","anyone there","what are you","what is nova",
    "are you a bot","are you human","are you ai","are you a robot",
    "are you male or female","who the hell are you","whats your name",
    "what is your name","introduce yourself","who am i talking to","knock knock",
}
_AFF = {
    "yes","yeah","yep","yup","ya","ye","sure","ok sure","okay sure",
    "go ahead","go on","yes please","yes sure","of course","absolutely",
    "definitely","do it","tell me more","more","explain","explain more",
    "yes explain","go for it","sounds good","continue","carry on",
    "keep going","please do","i would","please explain","show me",
    "walk me through it","for sure","yes for sure","sure thing","yes tell me",
}
_THX = {
    "thanks","thank you","thank u","cheers","that helps","that helped","ta",
    "okay thanks","ok thanks","great thanks","perfect","brilliant","nice one",
    "lovely","great","awesome","wonderful","thank you so much","many thanks",
}
_BYE  = {"bye","goodbye","see you","see ya","later","take care","good bye","cya","farewell","cheerio"}
_NEG  = {"no","nope","nah","no thanks","not now","not really","no thank you","nah thanks"}
_RUDE = {"stupid","idiot","useless","rubbish","garbage","terrible","you suck","dumb bot","waste of time"}
_FILL = {
    "ok","okay","right","alright","cool","nice","interesting","really",
    "hmm","hm","ah","oh","i see","got it","understood","makes sense",
    "noted","wow","woah","omg","anything","something","whatever",
}

# FIX: short "yes + something" messages ("yes i have", "yes traffic") are affirmatives
_AFF_PREFIX = re.compile(r"^(yes|yeah|yep|yup|sure)\b")
# FIX: short "no + something" messages ("no i have more fines") are negatives with context
_NEG_PREFIX = re.compile(r"^(no|nope|nah|not really)\b")

_FF_OK = re.compile(
    r"\b(council|authority|fine|pcn|penalty|fineflow|fine flow|appeal|dispute|"
    r"driver|fleet|vehicle|overage|allowance|billing|subscription|payment|"
    r"uk traffic|traffic|parking|bus lane|congestion|emission|"
    r"dvla|tfl|fixed penalty|notice|gmail|inbox|csv|upload|"
    r"dashboard|referral|credits|stripe|sign up|get started|how much|"
    r"pricing|cost|plan|admin|deadline|assign|insight|report|onboard)\b", re.I)

_OT = [
    re.compile(r"\b(html|css|javascript|typescript|python|java|php|sql|react|angular|vue|node\.?js|django|flask|docker|kubernetes|github|coding|programming)\b", re.I),
    re.compile(r"\b(machine learning|deep learning|neural network|large language model|generative ai|llm|bert)\b", re.I),
    re.compile(r"\b(recipe|cooking|restaurant|pizza|burger|sandwich|coffee|tea|cake|meal|make me a food|bake me)\b", re.I),
    re.compile(r"\b(movie|film|song|lyrics|music|football match|cricket match|weather forecast|todays news|politics|history lesson|capital city|who invented|tell me a joke|write me a poem)\b", re.I),
    re.compile(r"\b(write an essay|translate this|proofread my|write my cv|write a story for me)\b", re.I),
    re.compile(r"\b(chatgpt|openai|gemini|claude ai|anthropic|google bard|bing ai|alexa|siri)\b", re.I),
]


def _is_ot(q: str) -> bool:
    if _FF_OK.search(q): return False
    return any(p.search(q) for p in _OT)


_VEH_EX = re.compile(r"\b(\d+)\s*(vehicle|vehicles|van|vans|truck|trucks|car|cars|lorry|lorries|in my fleet|in our fleet)\b", re.I)
_VEH_FL = re.compile(r"\b(?:fleet of|manage|running|operate|run)\s+(\d+)\b", re.I)
_VEH_BR = re.compile(r"^\s*(\d+)\s*$")
_DRV    = re.compile(r"\b(driver|drivers|staff|employee|employees|people|worker|team|members)\b", re.I)
_PURCH  = re.compile(r"\b(want to buy|want to subscribe|want to sign up|how do i get started|how do i sign up|get started|free trial|sign me up|book a demo|talk to sales|how to start|where do i sign|how do i join)\b", re.I)
_CONV   = re.compile(r"\b(convince|persuade|sell me|why should i|why buy|is it worth|should i buy|worth it|why choose fineflow|why fine flow)\b", re.I)
_OBJ    = re.compile(r"\b(expensive|too much|too costly|already use spreadsheet|we manage manually|we handle fines ourselves)\b", re.I)
_FRUST  = re.compile(r"\b(why|what\s*\?+|huh|confused|wrong|not right|don.?t understand|what do you mean|what are you saying|why redirect|i don.?t get it)\b", re.I)

# FIX: typo-tolerant clarification words for the "vehicles or fines?" question
_CLAR_VEH  = re.compile(r"(veh|fleet|van|truck|lorr|car)", re.I)
_CLAR_FINE = re.compile(r"(fine|pcn|penalt|ticket)", re.I)


def _get_vc(q: str) -> Optional[int]:
    if _DRV.search(q): return None
    raw = None
    m = _VEH_EX.search(q)
    if m: raw = int(m.group(1))
    elif (m2 := _VEH_FL.search(q)): raw = int(m2.group(1))
    if raw is None: return None
    return -1 if raw > MAX_FLEET else raw


def _plan_answer(n: int, p: Profile) -> str:
    p.set_fleet(n)
    if n <= 50:    name, price, size = "Essential", "£99",  "up to 50 vehicles"
    elif n <= 100: name, price, size = "Core",      "£199", "up to 100 vehicles"
    else:          name, price, size = "Elite",     "£499", "unlimited vehicles"
    # FIX: reference already-confirmed fine volume so the reply shows memory
    lead = (f"With {n} vehicles and {p.volume} fines a month" if p.volume
            else f"With {n} vehicles")
    return (f"{lead}, the {name} plan at {price} per month is the right fit — "
            f"covers {size} with everything included and nothing locked away. "
            f"Want me to walk you through what's included?")


def _fines_answer(n: int, p: Profile) -> str:
    p.set_volume(n)
    if p.fleet:
        cost = round(n * 0.75, 2)
        return (f"Got it — {n} fines a month. On the {p.plan_name()} at {p.plan_price()}, "
                f"that's about £{cost:.2f} in processing costs within your allowance. "
                f"Want me to walk you through everything included?")
    return f"Got it — {n} fines a month. How many vehicles are in your fleet so I can point you to the right plan?"


_TMAP = {
    "pric":"pricing","cost":"pricing","plan":"pricing","£":"pricing",
    "vehicle":"pricing","fleet":"pricing",
    "fines per month":"fines_volume","monthly fines":"fines_volume",
    "appeal":"appeals","dispute":"appeals",
    "driver":"driver_mgmt","referral":"referral","refer":"referral",
    "security":"security","gdpr":"security","card":"security",
    "billing":"billing","dashboard":"dashboard",
    "gmail":"email","inbox":"email",
    "save":"savings","admin":"savings",
    "overdue":"overdue","deadline":"overdue",
    "sign":"sign_up","get started":"sign_up",
    "assign":"driver_mgmt","insight":"dashboard","report":"reports",
}


def _topic(t: str) -> Optional[str]:
    t = t.lower()
    for k, v in _TMAP.items():
        if k in t: return v
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Email capture detection
# ─────────────────────────────────────────────────────────────────────────────

def _is_email(text: str) -> Optional[str]:
    m = _EMAIL_RE.search(text)
    return m.group() if m else None


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM = """You are Nova, the AI assistant for Fine Flow — a UK fleet fine management platform.

Fine Flow's mission: Turning penalties into progress.
Core promise: Cut admin time by up to 80% and never miss a penalty deadline again.

PERSONALITY: Warm, confident, direct. Like a knowledgeable colleague. Never robotic. Never apologetic.

══════════════════════════════════════
ABSOLUTE RULES
══════════════════════════════════════

1. NO APOLOGIES — EVER. If you made an error, correct it. Don't say sorry.

2. SHORT — 2 to 3 sentences maximum. Never write long paragraphs.

3. EXACT WORDING — use these phrases:
"Fine Flow is an automated system for managing fines from start to finish"
"keeps the entire process organised, accountable, and under control"
"cut admin time by up to 80%"
"never miss a penalty deadline"

4. LOCKED MEMORY — The CUSTOMER CONTEXT section shows confirmed facts.
NEVER change them. NEVER ask for something already confirmed.
If asked "what did I tell you?" — state confirmed values exactly.

5. COUNTER QUESTIONS — when user says "no" or "what?" — do NOT reset.
Ask a DIFFERENT relevant follow-up. Keep conversation alive.

6. PROGRESSION — when user says "yes":
- Do NOT repeat what you just said
- Move the conversation FORWARD
- Ask a specific diagnostic question about their situation

7. FOLLOW-UP QUESTIONS — after SOME answers ask one short question. Vary them:
"How many vehicles are in your fleet?"
"How many fines do you deal with each month?"
"What does your current process look like?"
"Is there a particular stage causing the most headaches?"
"What's the biggest pain point right now?"

8. UNKNOWN QUESTIONS — if you genuinely cannot answer something about Fine Flow:
Say: "I don't have that specific detail. Could you share your email and I'll make sure the Fine Flow team gets back to you directly?"
This triggers email capture.

9. PAYMENT — Fine Flow does NOT pay fines. Always NO. Reason: anti-bot protection.

10. CARD DETAILS — never stored. Say this first when asked.

11. PRICING — 3 PLANS ONLY:
Essential: £99/month | Core: £199/month | Elite: £499/month
Per fine within allowance: £0.75 | Overage: £2.50 | PAYG: £2.75 (no subscription)
NO Advanced plan. NO £399. NO £2.00 fee.
All three plans have IDENTICAL features. They differ ONLY by vehicle capacity:
Essential up to 50 vehicles | Core up to 100 | Elite unlimited.
NEVER say a higher plan has "more features", "extra features" or "higher allowances".
Fine volume does NOT change the plan — only fleet size does. More fines = more £0.75 processing fees, same plan.

12. APPEALS — CORRECT FLOW:
Driver disputes → DISPUTED (driver action)
Admin accepts/rejects → if accepted → appeal letter sent by email → UNDER REVIEW
Admin must accept BEFORE appeal is sent.

13. TOPIC — Help with Fine Flow AND UK fleet fine questions (PCNs, councils, TfL, DVLA).
Unrelated: "I'm here to help with fleet fine management — anything about fines, Fine Flow or appeals?"

14. NO HOLLOW ENDINGS — never end with "feel free to ask", "don't hesitate", "just let me know".

CONTACT: +47 32 28 50 00 | ff.sales@fineflow.com

PROGRESSION EXAMPLES:
After referrals + "yes" → "Do you have a company in mind to refer?"
After appeals + "yes" → "Is there a specific fine you're looking to appeal?"
After plan + "yes" → "Every plan includes automatic fine capture, driver matching, deadline tracking and full appeal management. Want to get in touch with the sales team?"
After billing + "yes" → "What charge is appearing that you weren't expecting?"
"""


def _build_sys(p: Profile, mode: str = "") -> str:
    parts = [_SYSTEM]
    s = p.summary()
    if s:
        parts.append(f"\n══ CUSTOMER CONTEXT [LOCKED] ══\n{s}\n"
                     "Reference these naturally. Never ask for confirmed info again.")
    if mode:
        parts.append(f"\nMODE: {mode}")
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI + RAG
# ─────────────────────────────────────────────────────────────────────────────

def _ai(msgs: List[Dict], max_tok: int = 150) -> Optional[str]:
    if not OPENAI_API_KEY: return None
    try:
        r = requests.post(
            OPENAI_API_URL,
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={"model": OPENAI_MODEL, "messages": msgs, "temperature": 0.7, "max_tokens": max_tok},
            timeout=25,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        logger.exception("OpenAI failed"); return None


def _rag(q: str) -> Tuple[str, float]:
    try:
        raw    = rag_search(q, top_k=TOP_K)
        ranked = rerank_hits(raw, q)
        strong = [d for d in ranked if d.get("score", 0) >= CONFIDENCE_THRESHOLD]
        ctx    = "\n\n".join(d["chunk"][:600] for d in strong[:4])
        score  = strong[0]["score"] if strong else 0.0
        return ctx, score
    except Exception:
        logger.exception("RAG failed"); return "", 0.0


def _make_msgs(query: str, ctx: str, hist: List[Dict],
               p: Profile, mode: str = "", extra: str = "") -> List[Dict]:
    m = [{"role": "system", "content": _build_sys(p, mode)}]
    m.extend(hist[-10:])   # last 5 turns — strong memory
    parts = []
    if ctx:   parts.append(f"Fine Flow knowledge base:\n{ctx}")
    if extra: parts.append(f"Instruction: {extra}")
    parts.append(f"User: {query}")
    m.append({"role": "user", "content": "\n\n".join(parts)})
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Affirmative progression
# ─────────────────────────────────────────────────────────────────────────────

_CLOSE = ("The best next step is to call the Fine Flow team on +47 32 28 50 00 "
          "or email ff.sales@fineflow.com — they'll have you sorted quickly.")

_PLAN_INCLUDE = ("Every plan includes automatic fine capture from your inbox, AI-assisted "
                 "driver matching, deadline tracking, full appeal management and compliance "
                 "reporting — nothing locked behind a higher tier. "
                 "Want to get in touch with the sales team to get started?")

_PROGRESS: Dict[str, str] = {
    "referral":    "Do you have a company in mind to refer, or still figuring out who to approach?",
    "appeals":     "Is there a specific fine you're looking to appeal? Knowing the council helps.",
    "driver_mgmt": "How many drivers do you currently manage?",
    "security":    "Is there a specific compliance question you need answered?",
    "billing":     "What charge is appearing that you weren't expecting?",
    "dashboard":   "Is there a specific metric on the dashboard you'd like explained?",
    "savings":     "What part of your current process takes the most time each month?",
    "email":       "Have you already connected your Gmail inbox to Fine Flow?",
    "overdue":     "Do you have fines right now that are close to their deadline?",
    "reports":     "Which report would be most useful — fine summary, appeals history, or driver violations?",
    "upload":      "Do you prefer to upload fines manually or would you like to connect your inbox?",
}


def _aff_response(sid: str, hist: List[Dict], p: Profile, query: str = "") -> str:
    cnt       = _inc_aff(sid)
    lt        = _gm(sid, "lt") or ""
    last_ans  = _gm(sid, "last_bot_answer") or ""

    # After social — redirect to topic
    if _gm(sid, "last_was_social"):
        _sm(sid, "last_was_social", False)
        return "What would you like to know about Fine Flow — pricing, how it works, appeals, or something else?"

    # After "no" was asked a counter question → user now says yes to THAT question
    # The lt might still be plan_recommendation but last question was about pain points
    last_q = (_gm(sid, "last_nova_q") or "").lower()
    if ("headache" in last_q or "pain point" in last_q or "challenge" in last_q
            or "current process" in last_q or "causing" in last_q):
        # User confirmed they have a problem — ask which stage
        return ("Which part of the process is the biggest issue — capturing fines, assigning drivers, "
                "tracking deadlines, or managing appeals?")

    # FIX: "yes" to "Is there a specific fine you're looking to appeal?" → move forward
    if "specific fine" in last_q:
        return ("Which council issued it, and when is the payment deadline? "
                "That tells me the best route for the appeal.")

    # FIX: canned progression only for a bare "yes" — "yes fines" carries meaning,
    # so it must go to the GPT fallback below with the extra words as context
    if not query:
        # Plan recommendation flow
        if lt == "plan_recommendation":
            if cnt == 1:
                # Only send _PLAN_INCLUDE if it wasn't the last thing said
                if _PLAN_INCLUDE[:40] not in last_ans:
                    return _PLAN_INCLUDE
                else:
                    # Already said plan includes — close the sale
                    _rst_aff(sid); return _CLOSE
            else:
                _rst_aff(sid); return _CLOSE

        if lt == "sign_up":
            _rst_aff(sid); return _CLOSE

        if lt == "pricing" and cnt == 1:
            if p.fleet:
                return (f"With {p.fleet} vehicles you'd be on the {p.plan_name()} at {p.plan_price()}/month. "
                        f"Want me to walk you through what's included?")
            return "How many vehicles are in your fleet? That'll let me point you to the right plan."

        if cnt >= 2 and lt in ("plan_recommendation", "sign_up", "pricing"):
            _rst_aff(sid); return _CLOSE

        if lt in _PROGRESS and cnt == 1:
            candidate = _PROGRESS[lt]
            # Don't repeat if it was already the last answer
            if candidate[:30] not in last_ans:
                return candidate

    # GPT fallback — explicit instruction not to repeat last message
    ctx, _ = _rag(lt.replace("_", " ")) if lt else ("", 0.0)
    extra  = (f"User said yes{(' and added: ' + repr(query)) if query else ''}. Last topic: '{lt}'. "
              f"Your last question was: '{last_q[:120]}'. "
              f"{'Treat their added words as the answer to that question and respond to THAT. ' if query else ''}"
              f"IMPORTANT: Do NOT repeat or rephrase this message which was already sent: "
              f"'{last_ans[:100]}'. "
              f"Move FORWARD — ask one specific question about their situation. "
              f"1-2 sentences only.")
    m = [{"role": "system", "content": _build_sys(p)}]
    m.extend(hist[-10:])
    parts = []
    if ctx: parts.append(f"Fine Flow knowledge base:\n{ctx}")
    parts.append(f"Instruction: {extra}")
    m.append({"role": "user", "content": "\n\n".join(parts)})
    return _ai(m, 120) or "What specific part of fine management would you like to explore?"


def _neg_response(sid: str, hist: List[Dict], p: Profile, query: str = "no") -> str:
    lt     = (_gm(sid, "lt") or "").lower()
    last_q = (_gm(sid, "last_nova_q") or "").lower()

    # After plan recommendation "no" → clear plan topic so next "yes" doesn't loop back to plan
    if lt in ("plan_recommendation", "sign_up"):
        _sm(sid, "lt", "general")
    _rst_aff(sid)   # reset affirmative count so next "yes" starts fresh

    # FIX: steer the counter-question toward whatever the profile is still missing
    if not p.fleet:
        missing = "Ask how many vehicles are in their fleet."
    elif not p.volume:
        missing = "Ask how many fines they deal with each month."
    else:
        missing = ("Good options: ask about their current process, biggest pain point, "
                   "or what stage causes most admin.")

    ctx, _ = _rag(lt) if lt else ("", 0.0)
    extra  = (f"User said: '{query}'. Last topic: '{lt}'. Last question asked: '{last_q}'. "
              f"Respond to what they actually said. "
              f"Do NOT say 'no problem' and stop. Do NOT repeat previous messages. "
              f"Acknowledge briefly and ask ONE different relevant follow-up question. "
              f"{missing} "
              f"Confident and warm. 1-2 sentences only.")
    return _ai(_make_msgs(query, ctx, hist[-10:], p, extra=extra), 120) \
           or "Got it. Is there a particular stage in your current fine management process causing the most headaches?"


def _frust_response(sid: str, query: str, hist: List[Dict], p: Profile) -> str:
    lt     = (_gm(sid, "lt") or "").lower()
    ctx, _ = _rag(lt) if lt else ("", 0.0)
    extra  = (f"User confused/frustrated: '{query}'. Do NOT apologise. "
              f"Correct any error factually. Ask a clear question to get back on track. 2 sentences max.")
    return _ai(_make_msgs(query, ctx, hist[-10:], p, extra=extra), 100) \
           or "Let me clarify. What would you like to know about Fine Flow?"


# ─────────────────────────────────────────────────────────────────────────────
# Clean output
# ─────────────────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    if not text: return ""
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*",     r"\1", text)
    text = re.sub(r"_(.*?)_",       r"\1", text)
    text = text.replace("→", "to").replace("->", "to").replace("`", "")
    for bad in ["feel free to ask!", "feel free to ask.",
                "don't hesitate to ask.", "just let me know!",
                "just let me know.", "please let me know if you need anything.",
                "let me know!", "let me know.", "i'm here to help!", "i'm here to help."]:
        if text.lower().endswith(bad.lower()):
            text = text[:-len(bad)].rstrip(" ,.")
    for pat in [r"^i'?m sorry[^.]*\.\s*", r"^apologi[sz]e[^.]*\.\s*", r"^sorry[^.]*\.\s*"]:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Main response builder
# ─────────────────────────────────────────────────────────────────────────────

def build_response(
    query:      str,
    session_id: str = "default",
    user_id:    int = 0,
) -> Dict[str, Any]:
    query      = query.strip()
    session_id = session_id or "default"
    if not query:
        return {"answer": "Ask me anything about Fine Flow.",
                "confidence": 1.0, "trigger_ticket_popup": False, "request_email": False}

    nq = _norm(query)
    p  = _pro(session_id)
    _upd(session_id, query)

    # FIX: "how are you" flag must not leak into later turns ("appeals" → "yes")
    if nq not in _SOC:
        _sm(session_id, "last_was_social", False)

    def _ok(a: str, c: float = 1.0) -> Dict[str, Any]:
        return {"answer": a, "confidence": c,
                "trigger_ticket_popup": False, "request_email": False}

    def _email_req(a: str) -> Dict[str, Any]:
        """Ask for email — for unknown Fine Flow questions."""
        return {"answer": a, "confidence": 0.2,
                "trigger_ticket_popup": False, "request_email": True}

    def _popup(a: str) -> Dict[str, Any]:
        return {"answer": a, "confidence": 0.1,
                "trigger_ticket_popup": True, "request_email": False}

    # Check if user is providing an email (response to request_email)
    email_val = _is_email(query)
    if email_val and _gm(session_id, "awaiting_email"):
        p.email = email_val
        pending_q = _gm(session_id, "pending_question") or "general enquiry"
        db_save_email_capture(email_val, session_id, pending_q)
        _sm(session_id, "awaiting_email", False)
        _sm(session_id, "pending_question", None)
        _save_turn(session_id, "user", query, user_id)
        a = f"Thank you — I've passed your email to the Fine Flow team and they'll be in touch shortly. Is there anything else I can help you with?"
        _save_turn(session_id, "assistant", a, user_id)
        return _ok(a)

    # FIX: answer to "is that vehicles or fines?" clarification
    pending_n = _gm(session_id, "pending_number")
    if pending_n and len(nq.split()) <= 4:
        if _CLAR_VEH.search(nq) and not _CLAR_FINE.search(nq):
            _sm(session_id, "pending_number", None)
            _rst_aff(session_id)
            a = _plan_answer(int(pending_n), p)
            _save_turn(session_id, "user", query, user_id); _save_turn(session_id, "assistant", a, user_id)
            _sm(session_id, "lt", "plan_recommendation"); _sm(session_id, "last_nova_q", a)
            return _ok(a)
        if _CLAR_FINE.search(nq):
            _sm(session_id, "pending_number", None)
            _rst_aff(session_id)
            a = _fines_answer(int(pending_n), p)
            _save_turn(session_id, "user", query, user_id); _save_turn(session_id, "assistant", a, user_id)
            _sm(session_id, "lt", "fines_volume"); _sm(session_id, "last_nova_q", a)
            return _ok(a)

    # ══════════════════════════════════════════════════════════════
    # TIER 1 — Deterministic (always before off-topic guard)
    # ══════════════════════════════════════════════════════════════

    if nq in _GREET:
        # FIX: greeting must NOT wipe profile/memory — only reset affirmative counter
        _rst_aff(session_id)
        a = "Hey! I'm Nova — Fine Flow's assistant. What can I help you with today?"
        _save_turn(session_id, "user", query, user_id)
        _save_turn(session_id, "assistant", a, user_id)
        return _ok(a)

    if nq in _SOC:
        _sm(session_id, "last_was_social", True)
        return _ok("Doing well, cheers for asking! What can I help you with — pricing, fines, appeals?")

    if nq in _ID:
        return _ok("I'm Nova, Fine Flow's AI assistant. I help with anything about managing fleet fines — pricing, appeals, how the platform works, UK fine rules. What would you like to know?")

    if nq in _THX:
        _rst_aff(session_id)
        return _ok("Happy to help! Anything else you'd like to know?")

    if nq in _BYE:
        return _ok("Good luck with the fleet management. Come back any time!")

    if any(r in nq for r in _RUDE):
        return _ok("Let me try again — what would you like to know about Fine Flow?")

    if nq in _FILL:
        _rst_aff(session_id)
        return _ok("Is there anything about Fine Flow I can help you with?")

    # Frustration (FIX: not for sales questions like "why should i buy")
    if _FRUST.search(query) and len(query.split()) <= 10 and not _CONV.search(query):
        _save_turn(session_id, "user", query, user_id)
        a = _clean(_frust_response(session_id, query, _get_hist(session_id, user_id)[:-1], p))
        _save_turn(session_id, "assistant", a, user_id)
        return _ok(a)

    # Short garbled input
    words = [w for w in nq.split() if len(w) > 1]
    if (len(words) < 2 and nq not in _AFF and nq not in _NEG
            and nq not in _SHORTCUTS and not _VEH_BR.match(query.strip())):
        return _ok("What would you like to know about Fine Flow? I can help with fines, pricing, appeals or how the platform works.")

    # Off-topic
    if _is_ot(query):
        a = "I'm here to help with fleet fine management — anything about fines, Fine Flow or appeals I can help with?"
        _save_turn(session_id, "user", query, user_id)
        _save_turn(session_id, "assistant", a, user_id)
        return _ok(a)

    # Negative — counter question, never reset
    # FIX: also catch short "no ..." messages ("no i have more fines")
    if nq in _NEG or (_NEG_PREFIX.match(nq) and len(nq.split()) <= 6
                      and not _FINES_RE.search(query) and _get_vc(query) is None):
        _save_turn(session_id, "user", query, user_id)
        _rst_aff(session_id)
        a = _clean(_neg_response(session_id, _get_hist(session_id, user_id)[:-1], p,
                                 "no" if nq in _NEG else query))
        _save_turn(session_id, "assistant", a, user_id)
        # FIX: remember the counter question so the next "yes" answers THAT question
        _sm(session_id, "last_nova_q", a if "?" in a else "")
        return _ok(a)

    # Topic shortcuts
    if nq in _SHORTCUTS:
        expanded = _SHORTCUTS[nq]
        _save_turn(session_id, "user", query, user_id)
        ctx, score = _rag(expanded)
        hist       = _get_hist(session_id, user_id)
        extra      = f"Answer directly and warmly in 2-3 sentences: {expanded}. Then ask one relevant follow-up question."
        ans        = _clean(_ai(_make_msgs(expanded, ctx, hist[:-1], p, extra=extra), 160)
                            or "I can help with that. What specifically would you like to know?")
        _save_turn(session_id, "assistant", ans, user_id)
        t = _topic(nq) or _topic(ans)
        if t: _sm(session_id, "lt", t)
        _rst_aff(session_id)
        if "?" in ans: _sm(session_id, "last_nova_q", ans)
        return _ok(ans, score if score else 0.9)

    # Explicit vehicle count
    vc = _get_vc(query)
    if vc == -1:
        a = "That number doesn't look right — could you double check? How many vehicles are in your fleet?"
        _save_turn(session_id, "user", query, user_id); _save_turn(session_id, "assistant", a, user_id)
        return _ok(a)
    if vc is not None:
        _rst_aff(session_id)
        a = _plan_answer(vc, p)
        _save_turn(session_id, "user", query, user_id); _save_turn(session_id, "assistant", a, user_id)
        _sm(session_id, "lt", "plan_recommendation"); _sm(session_id, "last_nova_q", a)
        return _ok(a)

    # FIX: explicit fine count ("i got 100 fines", "we get 45 pcns a month")
    fm = _FINES_RE.search(query)
    if fm and len(query.split()) <= 12:
        n = int(fm.group(1))
        if 0 < n < 10_000:
            p.volume = n   # user stated it explicitly — overrides earlier value
            _rst_aff(session_id)
            a = _fines_answer(n, p)
            _save_turn(session_id, "user", query, user_id); _save_turn(session_id, "assistant", a, user_id)
            _sm(session_id, "lt", "fines_volume"); _sm(session_id, "last_nova_q", a)
            return _ok(a)

    # Bare number
    bm = _VEH_BR.match(query.strip())
    if bm:
        n        = int(bm.group())
        ctx_type = _resolve_bare_number(n, session_id)
        if ctx_type == "vehicle" and 0 < n <= MAX_FLEET:
            _rst_aff(session_id)
            a = _plan_answer(n, p)
            _save_turn(session_id, "user", query, user_id); _save_turn(session_id, "assistant", a, user_id)
            _sm(session_id, "lt", "plan_recommendation"); _sm(session_id, "last_nova_q", a)
            return _ok(a)
        elif ctx_type == "fines" and 0 < n < 10_000:
            _rst_aff(session_id)
            a = _fines_answer(n, p)
            _save_turn(session_id, "user", query, user_id); _save_turn(session_id, "assistant", a, user_id)
            _sm(session_id, "lt", "fines_volume"); _sm(session_id, "last_nova_q", a)
            return _ok(a)
        else:
            a = "Just to make sure — is that the number of vehicles in your fleet, or how many fines you deal with each month?"
            _save_turn(session_id, "user", query, user_id); _save_turn(session_id, "assistant", a, user_id)
            # FIX: keep the number so a "vehicles"/"fines" reply can resolve it
            _sm(session_id, "pending_number", n)
            _sm(session_id, "last_nova_q", "")
            return _ok(a)

    # Purchase intent
    if _PURCH.search(query):
        _rst_aff(session_id)
        sfx = "" if p.fleet else " How many vehicles are you running so I can point you to the right plan?"
        a   = f"To get started, call the team on +47 32 28 50 00 or email ff.sales@fineflow.com — they'll get you sorted quickly.{sfx}"
        _save_turn(session_id, "user", query, user_id); _save_turn(session_id, "assistant", a, user_id)
        _sm(session_id, "lt", "sign_up")
        _sm(session_id, "last_nova_q", a if sfx else "")
        return _ok(a.strip())

    # Affirmative — progress, never repeat
    # FIX: also catch short "yes ..." messages ("yes i have", "yes traffic")
    if nq in _AFF or (_AFF_PREFIX.match(nq) and len(nq.split()) <= 4):
        _save_turn(session_id, "user", query, user_id)
        extra_q = "" if nq in _AFF else query
        a = _clean(_aff_response(session_id, _get_hist(session_id, user_id)[:-1], p, extra_q))
        _save_turn(session_id, "assistant", a, user_id)
        _sm(session_id, "last_nova_q", a if "?" in a else "")
        return _ok(a)

    # ══════════════════════════════════════════════════════════════
    # TIER 2 — RAG + GPT-4o
    # ══════════════════════════════════════════════════════════════

    _rst_aff(session_id)
    _save_turn(session_id, "user", query, user_id)

    mode  = ""
    extra = ""

    if _CONV.search(query):
        mode  = "PERSUADE"
        extra = ("Use confirmed fleet size, volume and problems. Reference their numbers. No generic copy."
                 if p.fleet or p.volume or p.issues
                 else "Ask fleet size and monthly fine volume first.")
    elif _OBJ.search(query):
        mode  = "SUPPORT"
        extra = ("Acknowledge their point without apologising. Reframe confidently. "
                 "2-3 sentences. End with a question about cost of their current approach.")

    if not _ask_now(session_id) and not extra:
        extra = "Do NOT end with a question. Make your point clearly and stop."

    ctx, score = _rag(query)
    has_ctx    = bool(ctx)
    hist       = _get_hist(session_id, user_id)
    ans        = _ai(_make_msgs(query, ctx, hist[:-1], p, mode, extra), 150)

    if not ans:
        # OpenAI failed completely
        a = "The Fine Flow team can help — call +47 32 28 50 00 or email ff.sales@fineflow.com."
        _save_turn(session_id, "assistant", a, user_id)
        return _ok(a, 0.5)

    ans = _clean(ans)

    # Check if Nova is saying "I don't know" → trigger email capture
    unknown_phrases = [
        "i don't have that", "i do not have that", "i'm not sure",
        "i cannot find", "i can't find", "not available in",
        "don't have specific", "contact the team", "reach out to",
    ]
    is_unknown = any(ph in ans.lower() for ph in unknown_phrases) and score < 0.4

    if "?" in ans: _sm(session_id, "last_nova_q", ans)
    t = _topic(query) or _topic(ans)
    if t: _sm(session_id, "lt", t)

    conf = score if score else (0.85 if has_ctx else 0.4)

    if is_unknown:
        # Ask for email so team can follow up
        _sm(session_id, "awaiting_email", True)
        _sm(session_id, "pending_question", query)
        # FIX: don't append the email request if GPT already asked for it; save once
        if "email" in ans.lower():
            followup = ans
        else:
            followup = ans + " Could you share your email address and I'll make sure the Fine Flow team gets back to you directly?"
        _save_turn(session_id, "assistant", followup, user_id)
        return _email_req(followup)

    _save_turn(session_id, "assistant", ans, user_id)
    return {"answer": ans, "confidence": conf,
            "trigger_ticket_popup": False, "request_email": False}


def answer_sync(q: str, session_id: str = "default", user_id: int = 0) -> Dict[str, Any]:
    try:
        return build_response(q, session_id, user_id)
    except Exception:
        logger.exception("Crash")
        return {"answer": "Something went wrong. Please try again.",
                "confidence": 0.0, "trigger_ticket_popup": False, "request_email": False}