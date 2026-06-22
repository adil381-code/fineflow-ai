# app/answer_builder.py
"""
FineFlow Nova — Production Final (Clean)
=========================================
Fixes in this version:
1. trigger_ticket_popup ONLY fires when confidence < 0.35 AND no RAG context found
   — no longer fires on good answers that happen to mention contact details
2. Bare numbers (54, 43) with NO prior context → ask what they mean, don't guess
3. Counter-questions on "no" / "what?" / frustrated messages work correctly
4. Memory carries fleet size, volume, name, industry across the full conversation
5. Client's exact wording used throughout
6. Escalation message is clean — it does NOT append "redirecting" to good answers
"""

import json
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

# Only trigger ticket popup when RAG finds NOTHING and confidence is very low
TICKET_TRIGGER_THRESHOLD = 0.35


def _should_escalate(answer: str, confidence: float, has_rag_context: bool) -> bool:
    """
    Returns True ONLY when the bot genuinely has no knowledge to help.
    Does NOT trigger just because the answer mentions contact details.
    """
    # Good answers never trigger escalation
    if confidence >= TICKET_TRIGGER_THRESHOLD:
        return False
    if has_rag_context:
        return False
    # Only trigger when we have very low confidence AND no RAG context
    return confidence < TICKET_TRIGGER_THRESHOLD and not has_rag_context


# ─────────────────────────────────────────────────────────────────────────────
# MySQL — lazy, auto-reconnect, never crashes chatbot if DB is down
# ─────────────────────────────────────────────────────────────────────────────

_mysql_conn = None
_mysql_lock = threading.Lock()


def _get_conn():
    global _mysql_conn
    host = os.getenv("MYSQL_HOST", "")
    if not host:
        return None
    with _mysql_lock:
        try:
            import pymysql
            if _mysql_conn is None or not _mysql_conn.open:
                raise Exception("reconnect")
            _mysql_conn.ping(reconnect=False)
            return _mysql_conn
        except Exception:
            try:
                import pymysql
                _mysql_conn = pymysql.connect(
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
                return _mysql_conn
            except Exception as e:
                logger.warning("MySQL unavailable: %s", e)
                _mysql_conn = None
                return None


def _ensure_tables():
    conn = _get_conn()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(100),
                    email VARCHAR(255) UNIQUE,
                    support_id VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    sender VARCHAR(20) NOT NULL,
                    message TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    INDEX idx_user_id (user_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS tickets (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    ticket_number VARCHAR(50) UNIQUE,
                    user_id INT,
                    subject VARCHAR(255),
                    message TEXT,
                    status VARCHAR(50) DEFAULT 'OPEN',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)
    except Exception as e:
        logger.warning("Table setup: %s", e)


try:
    _ensure_tables()
except Exception:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# MySQL public helpers (called from api.py too)
# ─────────────────────────────────────────────────────────────────────────────

def db_find_or_create_user(name: str, email: str, support_id: str = "") -> Tuple[int, bool]:
    conn = _get_conn()
    if not conn:
        return 0, False
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE email = %s", (email,))
            row = cur.fetchone()
            if row:
                return row["id"], True
            cur.execute(
                "INSERT INTO users (name, email, support_id) VALUES (%s, %s, %s)",
                (name, email, support_id or ""),
            )
            return cur.lastrowid, False
    except Exception as e:
        logger.warning("db_find_or_create_user: %s", e)
        return 0, False


def db_save_message(user_id: int, sender: str, message: str) -> None:
    if not user_id:
        return
    conn = _get_conn()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chat_history (user_id, sender, message) VALUES (%s, %s, %s)",
                (user_id, sender, message),
            )
    except Exception as e:
        logger.warning("db_save_message: %s", e)


def db_load_history(user_id: int, limit: int = 40) -> List[Dict[str, str]]:
    if not user_id:
        return []
    conn = _get_conn()
    if not conn:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT sender, message FROM (
                    SELECT sender, message, created_at
                    FROM chat_history WHERE user_id = %s
                    ORDER BY created_at DESC LIMIT %s
                ) sub ORDER BY created_at ASC
                """,
                (user_id, limit),
            )
            return [{"sender": r["sender"], "message": r["message"]} for r in cur.fetchall()]
    except Exception as e:
        logger.warning("db_load_history: %s", e)
        return []


def db_create_ticket(user_id: int, subject: str, message: str) -> str:
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
                "INSERT INTO tickets (ticket_number, user_id, subject, message) VALUES (%s, %s, %s, %s)",
                (tkt, user_id or None, subject, message),
            )
            return tkt
    except Exception as e:
        logger.warning("db_create_ticket: %s", e)
        return "TKT-ERR"


# ─────────────────────────────────────────────────────────────────────────────
# Customer profile — persisted in-memory, synced to MySQL via chat history
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Profile:
    fleet:    Optional[int]  = None
    volume:   Optional[int]  = None
    issues:   List[str]      = field(default_factory=list)
    industry: Optional[str]  = None
    name:     Optional[str]  = None
    turns:    int            = 0

    def summary(self) -> str:
        parts = []
        if self.name:    parts.append(f"Customer name: {self.name}")
        if self.fleet and self.fleet <= MAX_FLEET:
            parts.append(f"Fleet size: {self.fleet} vehicles")
        if self.volume:  parts.append(f"Monthly fines: ~{self.volume}")
        if self.industry: parts.append(f"Industry: {self.industry}")
        if self.issues:  parts.append(f"Problems: {', '.join(self.issues)}")
        return "\n".join(parts)

    def plan_name(self) -> str:
        if not self.fleet or self.fleet > MAX_FLEET: return ""
        if self.fleet <= 50:   return "Essential"
        if self.fleet <= 100:  return "Core"
        if self.fleet <= 200:  return "Advanced"
        return "Elite"

    def plan_price(self) -> str:
        if not self.fleet or self.fleet > MAX_FLEET: return ""
        if self.fleet <= 50:   return "£99"
        if self.fleet <= 100:  return "£199"
        if self.fleet <= 200:  return "£399"
        return "£499"


# ─────────────────────────────────────────────────────────────────────────────
# Session memory
# ─────────────────────────────────────────────────────────────────────────────

_SES: Dict[str, List[Dict]] = {}
_PRO: Dict[str, Profile]    = {}
_MET: Dict[str, Dict]       = {}
_LK  = threading.Lock()


def _hist(sid: str, uid: int = 0) -> List[Dict[str, str]]:
    """Returns history as OpenAI message format [{role, content}]."""
    if uid:
        rows = db_load_history(uid)
        return [
            {"role": "user" if r["sender"] == "user" else "assistant", "content": r["message"]}
            for r in rows
        ]
    with _LK:
        return list(_SES.get(sid, []))


def _push(sid: str, role: str, content: str, uid: int = 0) -> None:
    if uid:
        db_save_message(uid, "user" if role == "user" else "bot", content)
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


def _save_pro(sid: str, p: Profile) -> None:
    with _LK: _PRO[sid] = p


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

def _ask_now(sid):
    """Ask a question on every other response — avoids interrogation fatigue."""
    with _LK:
        m = _MET.setdefault(sid, {})
        c = m.get("rc", 0) + 1; m["rc"] = c
        return c % 2 == 0


# ─────────────────────────────────────────────────────────────────────────────
# Profile extraction from user messages
# ─────────────────────────────────────────────────────────────────────────────

_FINES_RE = re.compile(
    r"\b(\d+)\s*(?:fines?|pcns?|penalties|violations?|tickets?)"
    r"(?:\s*(?:per|a|each|every)\s*(?:month|monthly|week))?\b", re.I)
_IND_RE  = re.compile(
    r"\b(logistics|delivery|courier|haulage|transport|taxi|minicab|bus|coach|construction)\b", re.I)
_NAME_RE = re.compile(r"\b(?:i am|i'm|my name is|call me)\s+([A-Z][a-z]+)\b")
_ISS = [
    (re.compile(r"\b(miss(?:ed?|ing)?\s+(?:deadlines?|appeals?|due\s*dates?))\b", re.I), "missed deadlines"),
    (re.compile(r"\b(drivers?\s+(?:dispute|deny|ignor))\b", re.I),                      "driver disputes"),
    (re.compile(r"\b(spreadsheet)\b", re.I),                                             "using spreadsheets"),
    (re.compile(r"\b(too\s+much\s+(admin|time|work))\b", re.I),                         "too much admin"),
    (re.compile(r"\b(out of (my )?time|no time|always busy)\b", re.I),                  "time pressure"),
]

# Words that mean the number is definitely a vehicle count
_VEH_WORDS = {
    "vehicle", "vehicles", "van", "vans", "truck", "trucks", "lorry",
    "lorries", "car", "cars", "fleet", "in my fleet", "in our fleet",
}
# Words that mean the number is definitely a fines volume
_FINE_WORDS = {
    "fines", "fine", "pcns", "pcn", "penalties", "violations", "tickets",
}
# Context hints — last question asked
_FINES_CTX = {
    "how many fines", "fines per month", "fines a month", "monthly fines",
    "deal with each month", "fines do you", "fine volume",
}
_VEH_CTX = {
    "how many vehicles", "fleet size", "vehicles do you", "vehicles are in",
    "how big is your fleet", "size of your fleet",
}


def _upd(sid: str, q: str) -> None:
    p = _pro(sid)
    p.turns += 1
    m = _NAME_RE.search(q)
    if m and not p.name: p.name = m.group(1)
    m = _FINES_RE.search(q)
    if m and p.volume is None:
        v = int(m.group(1))
        if v < 10_000: p.volume = v
    m = _IND_RE.search(q)
    if m: p.industry = m.group(1).lower()
    for pat, lbl in _ISS:
        if pat.search(q) and lbl not in p.issues: p.issues.append(lbl)
    _save_pro(sid, p)


def _resolve_bare_number(n: int, sid: str) -> Optional[str]:
    """
    When user sends only a number, figure out if it's vehicles or fines
    based on the last question Nova asked. If unclear, return None → ask.
    """
    last_q = (_gm(sid, "last_nova_q") or "").lower()
    lt     = (_gm(sid, "lt") or "").lower()

    # Check last question text for explicit hints
    for hint in _FINES_CTX:
        if hint in last_q: return "fines"
    for hint in _VEH_CTX:
        if hint in last_q: return "vehicle"

    # Check last topic
    if lt in {"pricing", "plan_recommendation", "cost"}: return "vehicle"
    if lt in {"fines_volume", "savings"}:                return "fines"

    # Cannot determine — ask the user
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Topic shortcuts — single words answered immediately without asking to clarify
# ─────────────────────────────────────────────────────────────────────────────

_TOPIC_SHORTCUTS: Dict[str, str] = {
    "pricing":   "how much does fine flow cost and what are the plans",
    "price":     "how much does fine flow cost and what are the plans",
    "cost":      "how much does fine flow cost and what are the plans",
    "plans":     "what are the fine flow subscription plans",
    "appeals":   "can fine flow help me appeal a fine and how does it work",
    "appeal":    "can fine flow help me appeal a fine and how does it work",
    "fines":     "what happens to a fine when it enters fine flow",
    "billing":   "how does billing work in fine flow",
    "dashboard": "what does the fine flow dashboard show",
    "drivers":   "how does fine flow manage drivers and assign fines",
    "driver":    "how does fine flow manage drivers and assign fines",
    "security":  "how secure is fine flow and is it gdpr compliant",
    "gdpr":      "how secure is fine flow and is it gdpr compliant",
    "referral":  "does fine flow have a referral programme",
    "referrals": "does fine flow have a referral programme",
    "features":  "what features does fine flow include in every plan",
    "savings":   "how much time and money can fine flow save me",
    "contact":   "how do i contact fine flow sales team",
    "email":     "how does fine flow connect to gmail to get fines",
    "gmail":     "how does fine flow connect to gmail to get fines",
    "payg":      "is there a pay as you go option in fine flow",
    "overage":   "what is the overage charge in fine flow",
    "reports":   "what reports can i export from fine flow",
    "statuses":  "what are the fine statuses in fine flow",
    "matching":  "how does fine flow match a fine to the correct driver",
    "overdue":   "what happens when a fine becomes overdue in fine flow",
}


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    if not text: return ""
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*",     r"\1", text)
    text = re.sub(r"_(.*?)_",       r"\1", text)
    text = text.replace("→", "to").replace("->", "to").replace("`", "")
    for bad in [
        "feel free to ask!", "feel free to ask.",
        "don't hesitate to ask.", "just let me know!",
        "just let me know.", "please let me know if you need anything.",
    ]:
        if text.lower().endswith(bad.lower()):
            text = text[:-len(bad)].rstrip(" ,.")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


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
    "yes please explain","yes walk me through",
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

# Frustration / confusion — user is pushing back, needs empathy + redirect
_FRUSTRATION_RE = re.compile(
    r"\b(why|what\??|what\s+\?|huh|confused|wrong|incorrect|not right|"
    r"that.?s wrong|don.?t understand|makes no sense|what do you mean|"
    r"what are you (talking|saying)|why redirect|why are you|stop redirect)\b",
    re.I,
)

_FF_OK = re.compile(
    r"\b(council|authority|fine|pcn|penalty|fineflow|fine flow|appeal|dispute|"
    r"driver|fleet|vehicle|overage|allowance|billing|subscription|payment|"
    r"uk traffic|traffic violation|parking|bus lane|congestion|emission|"
    r"dvla|tfl|fixed penalty|notice to owner|gmail|inbox|csv|upload|"
    r"dashboard|referral|credits|stripe|sign up|get started|how much|"
    r"pricing|cost|plan|time|admin|deadline|miss)\b", re.I)

_OT = [
    re.compile(r"\b(html|css|javascript|typescript|python|java|php|sql|react|angular|vue|node\.?js|django|flask|docker|kubernetes|github|coding|programming|teach me|how to code)\b", re.I),
    re.compile(r"\b(machine learning|deep learning|neural network|large language model|generative ai|llm|bert)\b", re.I),
    re.compile(r"\b(recipe|cooking|restaurant|pizza|burger|sandwich|coffee|tea|cake|meal|make me a food|bake me)\b", re.I),
    re.compile(r"\b(movie|film|song|lyrics|music|football match|cricket match|weather forecast|todays news|politics|history lesson|capital city|who invented|tell me a joke|write me a poem)\b", re.I),
    re.compile(r"\b(write an essay|translate this|proofread my|write my cv|write a story for me)\b", re.I),
    re.compile(r"\b(chatgpt|openai|gemini|claude ai|anthropic|google bard|bing ai|alexa|siri)\b", re.I),
]


def _is_ot(q: str) -> bool:
    if _FF_OK.search(q): return False
    return any(p.search(q) for p in _OT)


_VEH_EX = re.compile(
    r"\b(\d+)\s*(vehicle|vehicles|van|vans|truck|trucks|car|cars|lorry|lorries|in my fleet|in our fleet)\b", re.I)
_VEH_FL = re.compile(r"\b(?:fleet of|manage|running|operate|run)\s+(\d+)\b", re.I)
_VEH_BR = re.compile(r"^\s*(\d+)\s*$")
_DRV_CT = re.compile(r"\b(driver|drivers|staff|employee|employees|people|worker|team|members)\b", re.I)
_PURCH  = re.compile(
    r"\b(want to buy|want to subscribe|want to sign up|how do i get started|"
    r"how do i sign up|get started|free trial|sign me up|book a demo|"
    r"talk to sales|how to start|where do i sign|how do i join)\b", re.I)
_CONV   = re.compile(
    r"\b(convince|persuade|sell me|why should i|why buy|is it worth|"
    r"should i buy|worth it|why choose fineflow|why fine flow)\b", re.I)
_OBJ    = re.compile(
    r"\b(expensive|too much|too costly|already use spreadsheet|"
    r"we manage manually|we handle fines ourselves|manage fines manually)\b", re.I)


def _get_vc(q: str, sid: str) -> Optional[int]:
    if _DRV_CT.search(q): return None
    raw = None
    m = _VEH_EX.search(q)
    if m: raw = int(m.group(1))
    elif (m2 := _VEH_FL.search(q)): raw = int(m2.group(1))
    if raw is None: return None
    return -1 if raw > MAX_FLEET else raw


def _plan_answer(n: int, p: Profile) -> str:
    p.fleet = n
    if n <= 50:    name, price, lim = "Essential", "£99",  "50"
    elif n <= 100: name, price, lim = "Core",      "£199", "100"
    elif n <= 200: name, price, lim = "Advanced",  "£399", "200"
    else:          name, price, lim = "Elite",      "£499", "unlimited"
    return (
        f"With {n} vehicles, the {name} plan at {price} per month is the right fit — "
        f"covers up to {lim} vehicles with everything included and nothing locked away. "
        f"Want me to walk you through what's included?"
    )


_TMAP = {
    "pric":"pricing","cost":"pricing","plan":"pricing","£":"pricing",
    "vehicle":"pricing","fleet":"pricing","package":"pricing",
    "fines per month":"fines_volume","fines a month":"fines_volume","monthly fines":"fines_volume",
    "appeal":"appeals","dispute":"appeals","driver":"driver_mgmt",
    "referral":"referral","refer":"referral","discount":"referral",
    "security":"security","gdpr":"security","card":"security",
    "billing":"billing","stripe":"billing","dashboard":"dashboard",
    "gmail":"email","inbox":"email","save":"savings","admin":"savings",
    "overdue":"overdue","deadline":"overdue","sign":"sign_up","get started":"sign_up",
}


def _topic(t: str) -> Optional[str]:
    t = t.lower()
    for k, v in _TMAP.items():
        if k in t: return v
    return None


# ─────────────────────────────────────────────────────────────────────────────
# System prompt — client's exact wording, warm human tone
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM = """You are Nova, the AI assistant for Fine Flow — a UK fleet fine management platform.

Fine Flow's mission: Turning penalties into progress.
Core promise: Cut admin time by up to 80% and never miss a penalty deadline again.
Fine Flow provides 24/7 fleet fine tracking, management and compliance in one place.

YOUR PERSONALITY:
Warm, humble, direct — like a knowledgeable colleague who genuinely cares. Conversational. Never robotic. Never pushy. Think of a brilliant product expert who listens carefully and gives real, useful answers.

ABSOLUTE RULES:

1. SHORT AND PRECISE
2 to 3 sentences. Never write long paragraphs. If they want more they'll ask.

2. CLIENT'S EXACT WORDING — use these phrases when describing Fine Flow:
"Fine Flow is an automated system for managing fines from start to finish"
"keeps the entire process organised, accountable, and under control"
"cut admin time by up to 80%"
"never miss a penalty deadline"
"turning penalties into progress"

3. MEMORY — ALWAYS USE WHAT YOU KNOW
If you know their fleet size, volume, industry or problems — always reference it naturally. Never ask for something they already told you. Personalise every answer.

4. FOLLOW-UP QUESTIONS
After most answers, ask one short relevant question to continue the conversation:
"How many vehicles are in your fleet?"
"How many fines do you deal with each month?"
"What does your current process look like?"
"Is there a particular stage causing the most headaches?"
"Have you had fines escalate because of missed deadlines?"
"What's the biggest pain point right now?"
Vary them. Never repeat the same question twice in a row.

5. COUNTER QUESTIONS ON "NO" OR FRUSTRATION
When user says "no", "what?", "why?" or expresses confusion/frustration — DO NOT reset.
DO NOT say "no problem" and stop.
Acknowledge warmly and ask a DIFFERENT relevant follow-up that continues the conversation.
Example: if they say "why redirect?" → say "That's fair — I shouldn't have added that. What would you like to know more about?"

6. AFFIRMATIVE LOOP PREVENTION
First yes/sure → explain warmly what's included in 2-3 sentences.
Second yes/sure on same topic → give contact details and close.
NEVER dump the full pricing list again on a second yes.

7. PAYMENT
Fine Flow does NOT pay fines automatically. Always say NO clearly. The reason is anti-bot protection and card verification on authority payment portals.

8. CARD DETAILS
Fine Flow never stores card details. Say this first when asked.

9. TOPIC SCOPE
Help with Fine Flow AND general UK fleet fine questions (PCNs, FPNs, councils, TfL, DVLA, appeal rights, parking rules). For completely unrelated topics say:
"I'm here to help with fleet fine management — anything about fines, Fine Flow or appeals?"

10. NO HOLLOW ENDINGS
Never end with "feel free to ask", "don't hesitate", "just let me know", "let me know if you have more questions".

11. NEVER ADD "REDIRECTING TO SUPPORT" TO A GOOD ANSWER
Only mention support escalation when you genuinely cannot answer something at all. A good, confident answer should NEVER end with a support redirect.

PRICING (memorise exactly — never get this wrong):
Essential: £99/month — 5 to 50 vehicles
Core: £199/month — 51 to 100 vehicles
Advanced: £399/month — 101 to 200 vehicles
Elite: £499/month — 200+ vehicles (unlimited)
Per fine within allowance: £0.75
Overage beyond allowance: £2.50 per fine
Pay-as-you-go (no subscription): £2.75 per fine
All plans include identical features. No £2.00 fee exists anywhere.

REFERRAL PROGRAMME:
Credits earned when a referred company subscribes:
1-25 vehicles: 100 credits | 26-100: 250 | 101-500: 750 | 500+: 2,000 credits
Silver (3 referrals): 100 bonus credits
Gold (5): 10% off subscription for 12 months
Platinum (10): 15% off for 12 months
Titan (25): 20% off for life
New joiners with referral code: £75 credits

CONTACT: +47 32 28 50 00 | ff.sales@fineflow.com

EXAMPLE RESPONSES (copy this tone exactly):

User: fines
Nova: Fine Flow is an automated system for managing fines from start to finish. The moment one arrives in your inbox it's captured automatically, details extracted, matched to the right driver and tracked through to resolution — keeping everything organised, accountable and under control. Want me to walk you through how it all works?

User: i am always out of time
Nova: That's exactly what Fine Flow is built for. It cuts admin time by up to 80% by automating everything from capturing fines to managing appeals, so your team isn't chasing emails and spreadsheets all day. How many fines are you typically dealing with each month?

User: why redirect?
Nova: That's fair — I shouldn't have added that. What would you like to know more about?

User: yes (after social)
Nova: What would you like to know about Fine Flow — pricing, how it works, appeals, or something else?

User: no
Nova: Fair enough — what's the biggest challenge with your current fine management setup?

User: 43 vehicles
Nova: With 43 vehicles, the Essential plan at £99 per month is the right fit — covers up to 50 vehicles with everything included and nothing locked away. Want me to walk you through what's included?

User: 54 (with no context)
Nova: Just to make sure I point you in the right direction — is that the number of vehicles in your fleet, or how many fines you deal with each month?
"""


def _build_sys(p: Profile, mode: str = "") -> str:
    parts = [_SYSTEM]
    s = p.summary()
    if s:
        parts.append(
            f"\nWHAT I KNOW ABOUT THIS CUSTOMER:\n{s}\n"
            "Always use this naturally. Never ask for information you already have."
        )
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
    """Returns (context, best_score). Score 0.0 = nothing found."""
    try:
        raw    = rag_search(q, top_k=TOP_K)
        ranked = rerank_hits(raw, q)
        strong = [d for d in ranked if d.get("score", 0) >= CONFIDENCE_THRESHOLD]
        ctx    = "\n\n".join(d["chunk"][:600] for d in strong[:4])
        score  = strong[0]["score"] if strong else 0.0
        return ctx, score
    except Exception:
        logger.exception("RAG failed"); return "", 0.0


def _make_msgs(query: str, ctx: str, hist: List[Dict], p: Profile, mode: str = "", extra: str = "") -> List[Dict]:
    m = [{"role": "system", "content": _build_sys(p, mode)}]
    m.extend(hist[-8:])
    parts = []
    if ctx:   parts.append(f"Fine Flow knowledge base:\n{ctx}")
    if extra: parts.append(f"Instruction: {extra}")
    parts.append(f"User: {query}")
    m.append({"role": "user", "content": "\n\n".join(parts)})
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Affirmative + Negative + Frustration handlers
# ─────────────────────────────────────────────────────────────────────────────

_CLOSE = (
    "The best next step is to get in touch with the Fine Flow team directly — "
    "call +47 32 28 50 00 or email ff.sales@fineflow.com and they'll have you sorted quickly."
)

_EXPAND = {
    "pricing":             "Explain Fine Flow pricing warmly in 2-3 sentences. Ask how many vehicles if unknown.",
    "plan_recommendation": "Explain what every Fine Flow plan includes in 2-3 sentences. Do NOT list all plans again. Ask if they want to get in touch with sales.",
    "appeals":             "Explain the appeal process: driver disputes fine → DISPUTED → admin accepts/rejects → if accepted, appeal letter sent by email to authority, status UNDER REVIEW. 2-3 sentences.",
    "driver_mgmt":         "Explain how drivers are added (individually or CSV) and matched to fines using vehicle, date and time. 2-3 sentences.",
    "fines_volume":        "Based on their fine volume, recommend the right plan. Use their actual numbers.",
    "referral":            "Explain the referral programme: credits per referral by fleet size, tier discounts up to 20% for life. 2-3 sentences.",
    "security":            "Explain GDPR: JWT, bcrypt, AES-256-CBC, never stores card data, never shares data. 2-3 sentences.",
    "billing":             "Explain billing: monthly via Stripe, credits reset each cycle, overage collected end of cycle, cooldown on cancellation. 2-3 sentences.",
    "dashboard":           "Explain the company dashboard: urgency-based, shows outstanding fines, deadlines, credit balance, billing status. 2-3 sentences.",
    "savings":             "Give specific savings figures for their fleet size if known. If not, give general figures.",
    "email":               "Explain Gmail connection: OAuth 2.0 (recommended) or App Password IMAP. 2-3 sentences.",
    "overdue":             "Explain how Fine Flow checks for overdue fines at midnight every day and marks them automatically. 2-3 sentences.",
    "sign_up":             "Tell them to call +47 32 28 50 00 or email ff.sales@fineflow.com to get started.",
}


def _aff_response(sid: str, hist: List[Dict], p: Profile) -> str:
    cnt = _inc_aff(sid)
    lt  = _gm(sid, "lt") or ""

    # After social response — redirect to topic
    if _gm(sid, "last_was_social"):
        _sm(sid, "last_was_social", False)
        return "What would you like to know about Fine Flow — pricing, how it works, appeals, or something else?"

    # Second consecutive affirmative on plan/pricing → close the sale
    if cnt >= 2 and lt in ("plan_recommendation", "sign_up", "pricing"):
        _rst_aff(sid)
        return _CLOSE

    prompt = _EXPAND.get(lt, "Expand naturally on the most recent Fine Flow topic in 2-3 sentences. Use the customer's information if available.")
    ctx, _ = _rag(lt.replace("_", " ")) if lt else ("", 0.0)
    m = [{"role": "system", "content": _build_sys(p, "Warm, concise. 2-3 sentences. One natural follow-up question if it helps.")}]
    m.extend(hist[-8:])
    parts = []
    if ctx: parts.append(f"Fine Flow knowledge base:\n{ctx}")
    parts.append(f"Instruction: {prompt}")
    m.append({"role": "user", "content": "\n\n".join(parts)})
    return _ai(m, 180) or "What would you like to know more about?"


def _neg_response(sid: str, hist: List[Dict], p: Profile) -> str:
    lt     = (_gm(sid, "lt") or "").lower()
    last_q = (_gm(sid, "last_nova_q") or "").lower()
    ctx, _ = _rag(lt) if lt else ("", 0.0)
    extra  = (
        f"The user said 'no'. Last topic: '{lt}'. Last question asked: '{last_q}'. "
        f"Do NOT say 'no problem' and stop. Acknowledge briefly and ask a DIFFERENT "
        f"relevant follow-up that keeps the conversation going. Warm and natural."
    )
    m   = _make_msgs("no", ctx, hist[-8:], p, extra=extra)
    ans = _ai(m, 120)
    return ans or "Fair enough — what's the biggest challenge with your current fine management setup?"


def _frustration_response(sid: str, query: str, hist: List[Dict], p: Profile) -> str:
    lt    = (_gm(sid, "lt") or "").lower()
    ctx, _ = _rag(lt) if lt else ("", 0.0)
    extra = (
        f"The user is expressing confusion or frustration: '{query}'. "
        f"Acknowledge their feeling briefly and warmly. Apologise if Nova made an error. "
        f"Then ask a clear, helpful question to get the conversation back on track. "
        f"2-3 sentences max. Do not be defensive."
    )
    m   = _make_msgs(query, ctx, hist[-8:], p, extra=extra)
    ans = _ai(m, 120)
    return ans or "That's fair — let me start fresh. What would you like to know about Fine Flow?"


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
        return {"answer": "Ask me anything about Fine Flow.", "confidence": 1.0, "trigger_ticket_popup": False}

    nq = _norm(query)
    p  = _pro(session_id)
    _upd(session_id, query)

    # Helper to build final response dict
    def _ok(answer: str, conf: float = 1.0) -> Dict[str, Any]:
        return {"answer": answer, "confidence": conf, "trigger_ticket_popup": False}

    def _escalate(answer: str) -> Dict[str, Any]:
        return {"answer": answer, "confidence": 0.2, "trigger_ticket_popup": True}

    # ══════════════════════════════════════════════════════════════════
    # TIER 1 — Deterministic (greetings/identity BEFORE off-topic guard)
    # ══════════════════════════════════════════════════════════════════

    if nq in _GREET:
        _rst(session_id)
        a = "Hey! I'm Nova — Fine Flow's assistant. What can I help you with today?"
        _push(session_id, "user", query, user_id)
        _push(session_id, "assistant", a, user_id)
        return _ok(a)

    if nq in _SOC:
        _sm(session_id, "last_was_social", True)
        a = "Doing well, cheers for asking! What can I help you with — pricing, fines, appeals?"
        return _ok(a)

    if nq in _ID:
        a = "I'm Nova, Fine Flow's AI assistant. I can help with anything about managing fleet fines — pricing, appeals, how the platform works, UK fine rules. What would you like to know?"
        return _ok(a)

    if nq in _THX:
        _rst_aff(session_id)
        return _ok("Happy to help! Anything else you'd like to know?")

    if nq in _BYE:
        return _ok("Good luck with the fleet management. Come back any time!")

    if any(r in nq for r in _RUDE):
        return _ok("Let me try again — what would you like to know about Fine Flow?")

    if nq in _FILL:
        _rst_aff(session_id)
        lt = _gm(session_id, "lt")
        if lt:
            return _ok(f"Is there anything more you'd like to know about Fine Flow?")
        return _ok("Is there anything about Fine Flow I can help you with?")

    # Frustration / confusion — handle with empathy
    if _FRUSTRATION_RE.search(query) and len(query.split()) <= 8:
        _push(session_id, "user", query, user_id)
        a = _clean(_frustration_response(session_id, query, _hist(session_id, user_id)[:-1], p))
        _push(session_id, "assistant", a, user_id)
        return _ok(a)

    # Garbled / too short — handle gracefully
    words = [w for w in nq.split() if len(w) > 1]
    if (len(words) < 2
            and nq not in _AFF
            and nq not in _NEG
            and nq not in _TOPIC_SHORTCUTS
            and not _VEH_BR.match(query.strip())):
        return _ok("What would you like to know about Fine Flow? I can help with fines, pricing, appeals or how the platform works.")

    # Off-topic — AFTER all deterministic checks
    if _is_ot(query):
        a = "I'm here to help with fleet fine management — anything about fines, Fine Flow or appeals I can help with?"
        _push(session_id, "user", query, user_id)
        _push(session_id, "assistant", a, user_id)
        return _ok(a)

    # Negative / "no"
    if nq in _NEG:
        _push(session_id, "user", query, user_id)
        _rst_aff(session_id)
        a = _clean(_neg_response(session_id, _hist(session_id, user_id)[:-1], p))
        _push(session_id, "assistant", a, user_id)
        return _ok(a)

    # Topic shortcuts — single words like "appeals", "fines", "pricing"
    if nq in _TOPIC_SHORTCUTS:
        expanded = _TOPIC_SHORTCUTS[nq]
        _push(session_id, "user", query, user_id)
        ctx, score = _rag(expanded)
        hist       = _hist(session_id, user_id)
        extra      = f"Answer this directly and warmly in 2-3 sentences: {expanded}. Then ask one relevant follow-up question."
        msgs       = _make_msgs(expanded, ctx, hist[:-1], p, extra=extra)
        ans        = _clean(_ai(msgs, 160) or "I can help with that. What specifically would you like to know?")
        _push(session_id, "assistant", ans, user_id)
        t = _topic(nq) or _topic(ans)
        if t: _sm(session_id, "lt", t)
        _rst_aff(session_id)
        if "?" in ans: _sm(session_id, "last_nova_q", ans)
        return _ok(ans, score if score else 0.9)

    # Explicit vehicle count in message
    vc = _get_vc(query, session_id)
    if vc == -1:
        a = "That number doesn't look right — could you double check? How many vehicles are in your fleet?"
        _push(session_id, "user", query, user_id)
        _push(session_id, "assistant", a, user_id)
        return _ok(a)
    if vc is not None:
        _rst_aff(session_id)
        a = _plan_answer(vc, p)
        _save_pro(session_id, p)
        _push(session_id, "user", query, user_id)
        _push(session_id, "assistant", a, user_id)
        _sm(session_id, "lt", "plan_recommendation")
        _sm(session_id, "last_nova_q", a)
        return _ok(a)

    # Bare number — resolve by context or ask
    bm = _VEH_BR.match(query.strip())
    if bm:
        n        = int(bm.group())
        ctx_type = _resolve_bare_number(n, session_id)

        if ctx_type == "vehicle" and 0 < n <= MAX_FLEET:
            _rst_aff(session_id)
            a = _plan_answer(n, p)
            _save_pro(session_id, p)
            _push(session_id, "user", query, user_id)
            _push(session_id, "assistant", a, user_id)
            _sm(session_id, "lt", "plan_recommendation")
            _sm(session_id, "last_nova_q", a)
            return _ok(a)

        elif ctx_type == "fines" and 0 < n < 10_000:
            p.volume = n
            _rst_aff(session_id)
            _save_pro(session_id, p)
            if p.fleet:
                cost = round(n * 0.75, 2)
                a = (
                    f"Got it — {n} fines a month. On the {p.plan_name()} plan at {p.plan_price()}, "
                    f"that's about £{cost:.2f} in processing costs within your allowance. "
                    f"Want me to walk you through everything that's included?"
                )
            else:
                a = (
                    f"Got it — {n} fines a month. Fine Flow would handle that cleanly. "
                    f"How many vehicles are in your fleet so I can point you to the right plan?"
                )
            _push(session_id, "user", query, user_id)
            _push(session_id, "assistant", a, user_id)
            _sm(session_id, "lt", "fines_volume")
            _sm(session_id, "last_nova_q", a)
            return _ok(a)

        else:
            # No context — ask what the number means
            a = "Just to make sure I point you in the right direction — is that the number of vehicles in your fleet, or how many fines you deal with each month?"
            _push(session_id, "user", query, user_id)
            _push(session_id, "assistant", a, user_id)
            _sm(session_id, "last_nova_q", a)
            return _ok(a)

    # Purchase intent
    if _PURCH.search(query):
        _rst_aff(session_id)
        sfx = "" if p.fleet else " How many vehicles are you running so I can point you to the right plan?"
        a   = f"To get started, call the team on +47 32 28 50 00 or email ff.sales@fineflow.com — they'll get you sorted quickly.{sfx}"
        _push(session_id, "user", query, user_id)
        _push(session_id, "assistant", a, user_id)
        _sm(session_id, "lt", "sign_up")
        return _ok(a.strip())

    # Affirmative
    if nq in _AFF:
        _push(session_id, "user", query, user_id)
        a = _clean(_aff_response(session_id, _hist(session_id, user_id)[:-1], p))
        _push(session_id, "assistant", a, user_id)
        _sm(session_id, "last_nova_q", "")
        return _ok(a)

    # ══════════════════════════════════════════════════════════════════
    # TIER 2 — RAG + GPT-4o
    # ══════════════════════════════════════════════════════════════════

    _rst_aff(session_id)
    _push(session_id, "user", query, user_id)

    mode  = ""
    extra = ""

    if _CONV.search(query):
        mode  = "PERSUADE"
        extra = (
            "Use the customer's exact fleet size, volume and problems. Reference their numbers directly. No generic copy."
            if p.fleet or p.volume or p.issues
            else "Ask about their fleet size and monthly fine volume first — you need their situation to give a tailored answer."
        )
    elif _OBJ.search(query):
        mode  = "SUPPORT"
        extra = (
            "Acknowledge their point warmly first — do not dismiss it. "
            "Reframe using their specific situation. 2-3 sentences max. "
            "End with a question about the real cost of their current approach."
        )

    if not _ask_now(session_id) and not extra:
        extra = "Do NOT end with a question this time. Make your point clearly and naturally, then stop."

    ctx, score   = _rag(query)
    has_ctx      = bool(ctx)
    hist         = _hist(session_id, user_id)
    ans          = _ai(_make_msgs(query, ctx, hist[:-1], p, mode, extra), 150)

    # If OpenAI fails completely
    if not ans:
        return _escalate(
            "I couldn't find a confident answer for that. "
            "Our support team can help — I'm opening a support form for you."
        )

    ans = _clean(ans)
    _push(session_id, "assistant", ans, user_id)

    if "?" in ans:
        _sm(session_id, "last_nova_q", ans)

    t = _topic(query) or _topic(ans)
    if t: _sm(session_id, "lt", t)

    conf = score if score else (0.85 if has_ctx else 0.4)

    # Escalate ONLY when we have very low confidence AND no RAG context found
    if _should_escalate(ans, conf, has_ctx):
        return _escalate(
            ans + " For anything more specific our support team can help — I'm opening a support form for you."
        )

    return {"answer": ans, "confidence": conf, "trigger_ticket_popup": False}


def answer_sync(q: str, session_id: str = "default", user_id: int = 0) -> Dict[str, Any]:
    try:
        return build_response(q, session_id, user_id)
    except Exception:
        logger.exception("Crash in answer_sync")
        return {"answer": "Something went wrong. Please try again.", "confidence": 0.0, "trigger_ticket_popup": False}