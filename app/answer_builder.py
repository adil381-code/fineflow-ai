# app/answer_builder.py
"""
FineFlow Nova — Production Final (Conversation Progression Edition)
====================================================================
Core fix: Affirmative responses now PROGRESS the conversation forward.
"Yes" after referrals → asks who they plan to refer, not re-explains referrals.
"Yes" after appeals → asks about the specific fine, not re-explains the flow.
"Yes" after plan recommendation → explains what's included, not re-lists pricing.

Memory is locked (fleet, volume set once, never overwritten by LLM).
Bare numbers always ask for clarification when context is ambiguous.
No apologies. No hollow endings. No repetition loops.
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
TICKET_THRESHOLD = 0.35


def _should_escalate(confidence: float, has_ctx: bool) -> bool:
    return confidence < TICKET_THRESHOLD and not has_ctx


# ─────────────────────────────────────────────────────────────────────────────
# MySQL (unchanged from previous version)
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
# LOCKED Customer profile — values set once, never overwritten
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Profile:
    fleet:    Optional[int]  = None
    volume:   Optional[int]  = None
    issues:   List[str]      = field(default_factory=list)
    industry: Optional[str]  = None
    name:     Optional[str]  = None
    turns:    int            = 0

    def set_fleet(self, n: int) -> bool:
        """Lock fleet — returns True if newly set."""
        if self.fleet is None and 0 < n <= MAX_FLEET:
            self.fleet = n
            return True
        return False

    def set_volume(self, n: int) -> bool:
        """Lock volume — returns True if newly set."""
        if self.volume is None and 0 < n < 10_000:
            self.volume = n
            return True
        return False

    def summary(self) -> str:
        parts = []
        if self.name:    parts.append(f"Customer name: {self.name}")
        if self.fleet:   parts.append(f"Fleet size: {self.fleet} vehicles [LOCKED — do not change]")
        if self.volume:  parts.append(f"Monthly fines: {self.volume} [LOCKED — do not change]")
        if self.industry: parts.append(f"Industry: {self.industry}")
        if self.issues:  parts.append(f"Problems: {', '.join(self.issues)}")
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
# Session memory
# ─────────────────────────────────────────────────────────────────────────────

_SES: Dict[str, List[Dict]] = {}
_PRO: Dict[str, Profile]    = {}
_MET: Dict[str, Dict]       = {}
_LK  = threading.Lock()


def _hist(sid: str, uid: int = 0) -> List[Dict[str, str]]:
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
        if sid not in _PRO:
            _PRO[sid] = Profile()
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

def _ask_now(sid):
    with _LK:
        m = _MET.setdefault(sid, {})
        c = m.get("rc", 0) + 1; m["rc"] = c
        return c % 2 == 0


# ─────────────────────────────────────────────────────────────────────────────
# Profile extraction
# ─────────────────────────────────────────────────────────────────────────────

_FINES_EXPLICIT_RE = re.compile(
    r"\b(\d+)\s*(?:fines?|pcns?|penalties|violations?|tickets?)"
    r"(?:\s*(?:per|a|each|every)\s*(?:month|monthly|week))?\b", re.I)
_IND_RE  = re.compile(
    r"\b(logistics|delivery|courier|haulage|transport|taxi|minicab|bus|coach|construction)\b", re.I)
_NAME_RE = re.compile(r"\b(?:i am|i'm|my name is|call me)\s+([A-Z][a-z]+)\b")
_ISS = [
    (re.compile(r"\b(miss(?:ed?|ing)?\s+(?:deadlines?|appeals?|due\s*dates?))\b", re.I), "missed deadlines"),
    (re.compile(r"\b(drivers?\s+(?:dispute|deny|ignor))\b", re.I),                      "driver disputes"),
    (re.compile(r"\b(spreadsheet)\b", re.I),                                             "using spreadsheets"),
    (re.compile(r"\b(too\s+much\s+(admin|time|work)|out of (my )?time|always busy)\b", re.I), "time pressure"),
]
_FINES_CTX = {
    "how many fines", "fines per month", "fines a month", "monthly fines",
    "deal with each month", "fines do you", "fine volume",
}
_VEH_CTX = {
    "how many vehicles", "fleet size", "vehicles do you", "vehicles are in",
    "how big is your fleet", "size of your fleet", "how many vans",
}


def _upd(sid: str, q: str) -> None:
    p = _pro(sid)
    p.turns += 1
    m = _NAME_RE.search(q)
    if m and not p.name:
        p.name = m.group(1)
    m = _FINES_EXPLICIT_RE.search(q)
    if m:
        p.set_volume(int(m.group(1)))
    m = _IND_RE.search(q)
    if m:
        p.industry = m.group(1).lower()
    for pat, lbl in _ISS:
        if pat.search(q) and lbl not in p.issues:
            p.issues.append(lbl)


def _resolve_bare_number(n: int, sid: str) -> Optional[str]:
    last_q = (_gm(sid, "last_nova_q") or "").lower()
    for hint in _FINES_CTX:
        if hint in last_q: return "fines"
    for hint in _VEH_CTX:
        if hint in last_q: return "vehicle"
    return None  # ambiguous → must ask


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
        "if you have any more questions, feel free to ask.",
        "let me know if you have any other questions.",
    ]:
        if text.lower().endswith(bad.lower()):
            text = text[:-len(bad)].rstrip(" ,.")
    # Strip apologies
    for pat in [
        r"^i'?m sorry (if|for|about|that)[^.]*\.\s*",
        r"^apologi[sz]e[^.]*\.\s*",
        r"^sorry (if|for|about)[^.]*\.\s*",
    ]:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Topic shortcuts
# ─────────────────────────────────────────────────────────────────────────────

_TOPIC_SHORTCUTS: Dict[str, str] = {
    "pricing":   "how much does fine flow cost and what are the plans",
    "price":     "how much does fine flow cost and what are the plans",
    "cost":      "how much does fine flow cost and what are the plans",
    "plans":     "what are the fine flow subscription plans",
    "appeals":   "how do i make an appeal in fine flow",
    "appeal":    "how do i make an appeal in fine flow",
    "fines":     "what happens to a fine when it enters fine flow",
    "billing":   "how does billing work and when am i charged",
    "dashboard": "what does the fine flow dashboard show",
    "drivers":   "how do i add and manage drivers in fine flow",
    "driver":    "how do i add and manage drivers in fine flow",
    "security":  "how secure is fine flow and is it gdpr compliant",
    "gdpr":      "how secure is fine flow and is it gdpr compliant",
    "referral":  "how does the fine flow referral programme work",
    "referrals": "how does the fine flow referral programme work",
    "features":  "what features does fine flow include in every plan",
    "savings":   "how much time and money can fine flow save me",
    "contact":   "how do i contact the fine flow team",
    "gmail":     "how does fine flow connect to gmail",
    "email":     "how does fine flow connect to gmail to capture fines",
    "payg":      "is there a pay as you go option with no subscription",
    "overage":   "what is the overage charge if i exceed my plan allowance",
    "reports":   "what reports can i export from fine flow",
    "statuses":  "what do the fine statuses in fine flow mean",
    "matching":  "how does fine flow match a fine to the correct driver",
    "overdue":   "what happens when a fine becomes overdue in fine flow",
    "upload":    "can i manually upload fines to fine flow",
    "assign":    "how do i assign a driver to a fine in fine flow",
    "insights":  "what are smart insights in fine flow",
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

_FRUSTRATION_RE = re.compile(
    r"\b(why|what\s*\?+|huh|confused|wrong|incorrect|not right|"
    r"that.?s wrong|don.?t understand|makes no sense|what do you mean|"
    r"what are you (talking|saying)|why redirect|why are you|"
    r"i don.?t get it|unclear)\b", re.I)

_FF_OK = re.compile(
    r"\b(council|authority|fine|pcn|penalty|fineflow|fine flow|appeal|dispute|"
    r"driver|fleet|vehicle|overage|allowance|billing|subscription|payment|"
    r"uk traffic|traffic violation|parking|bus lane|congestion|emission|"
    r"dvla|tfl|fixed penalty|notice to owner|gmail|inbox|csv|upload|"
    r"dashboard|referral|credits|stripe|sign up|get started|how much|"
    r"pricing|cost|plan|time|admin|deadline|miss|assign|insight|report)\b", re.I)

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


def _get_vc(q: str) -> Optional[int]:
    if _DRV_CT.search(q): return None
    raw = None
    m = _VEH_EX.search(q)
    if m: raw = int(m.group(1))
    elif (m2 := _VEH_FL.search(q)): raw = int(m2.group(1))
    if raw is None: return None
    return -1 if raw > MAX_FLEET else raw


def _plan_answer(n: int, p: Profile) -> str:
    p.set_fleet(n)
    if n <= 50:
        name, price, size = "Essential", "£99",  "up to 50 vehicles"
    elif n <= 100:
        name, price, size = "Core",      "£199", "up to 100 vehicles"
    else:
        name, price, size = "Elite",     "£499", "unlimited vehicles"
    return (
        f"With {n} vehicles, the {name} plan at {price} per month is the right fit — "
        f"covers {size} with everything included and nothing locked away. "
        f"Want me to walk you through what's included?"
    )


_TMAP = {
    "pric":"pricing",  "cost":"pricing",  "plan":"pricing",
    "£":"pricing",     "vehicle":"pricing","fleet":"pricing",
    "fines per month":"fines_volume",     "monthly fines":"fines_volume",
    "appeal":"appeals","dispute":"appeals",
    "driver":"driver_mgmt",
    "referral":"referral","refer":"referral",
    "security":"security","gdpr":"security","card":"security",
    "billing":"billing","stripe":"billing",
    "dashboard":"dashboard",
    "gmail":"email","inbox":"email",
    "save":"savings","admin":"savings",
    "overdue":"overdue","deadline":"overdue",
    "sign":"sign_up","get started":"sign_up",
    "assign":"driver_mgmt","upload":"upload",
    "insight":"dashboard","report":"reports",
}


def _topic(t: str) -> Optional[str]:
    t = t.lower()
    for k, v in _TMAP.items():
        if k in t: return v
    return None


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM = """You are Nova, the AI assistant for Fine Flow — a UK fleet fine management platform.

Fine Flow's mission: Turning penalties into progress.
Core promise: Cut admin time by up to 80% and never miss a penalty deadline again.

PERSONALITY: Warm, confident, direct. Like a knowledgeable colleague. Never robotic. Never apologetic. Never pushy.

══════════════════════════════════════
ABSOLUTE RULES
══════════════════════════════════════

1. NO APOLOGIES
Never say "I'm sorry", "Apologies", "Sorry if", "I understand how confusing".
If you made an error, correct it factually. Example: instead of "I'm sorry for the confusion" → say "Let me clarify that."

2. SHORT ANSWERS — 2 to 3 sentences maximum.

3. USE EXACT WORDING:
"Fine Flow is an automated system for managing fines from start to finish"
"keeps the entire process organised, accountable, and under control"
"cut admin time by up to 80%"
"never miss a penalty deadline"

4. LOCKED MEMORY — TREAT AS FACTS:
Values in the CUSTOMER CONTEXT section are confirmed and locked.
NEVER change them. NEVER contradict them. NEVER ask for something already confirmed.
If asked "what did I tell you?" → state confirmed values exactly.

5. PROGRESSION — NEVER REPEAT WHAT YOU JUST SAID:
When user says "yes" or "sure" after your answer:
→ Move the conversation FORWARD. Ask a diagnostic question or take the next logical step.
→ Do NOT re-explain what you just said.
→ Do NOT summarise the same topic again.

Examples of correct progression:
After explaining referrals → "yes" → "Who are you planning to refer — do you have a company in mind?"
After explaining appeals → "yes" → "What's the fine you're looking to appeal? Which council issued it?"
After recommending a plan → "yes" → "Every plan includes automatic fine capture, driver matching, deadline tracking and full appeal management. Want to get in touch with the sales team?"
After explaining billing → "yes" → "Is there a specific charge on your account you'd like to understand?"

6. FOLLOW-UP QUESTIONS — after SOME answers, ask one short relevant question.
Vary them. Never repeat the same question twice in a row.
"How many vehicles are in your fleet?"
"How many fines do you deal with each month?"
"What does your current process look like?"
"Is there a particular stage causing the most headaches?"
"What's the biggest pain point right now?"

7. ON "NO" OR FRUSTRATION:
Do NOT reset. Do NOT say "no problem" and stop.
Ask a DIFFERENT relevant follow-up. Keep conversation going.

8. PAYMENT: Fine Flow does NOT pay fines. Always say NO. Reason: anti-bot protection and card verification.

9. CARD DETAILS: Never stored. Say this first when asked.

10. PRICING — 3 PLANS ONLY:
Essential: £99/month
Core: £199/month
Elite: £499/month
Per fine within allowance: £0.75
Overage: £2.50 per fine
Pay-as-you-go (no subscription): £2.75 per fine
NO Advanced plan. NO £399 plan. NO £2.00 fee.

11. APPEALS FLOW — GET THIS RIGHT:
Driver disputes → DISPUTED (driver action)
Admin reviews → accepts OR rejects
If ACCEPTED → Fine Flow generates appeal letter → sends to authority by EMAIL → UNDER REVIEW
If REJECTED → back to CONFIRMED
NEVER say Fine Flow automatically sends appeal after dispute. Admin must accept first.

12. TOPIC: Help with Fine Flow AND UK fleet fine questions. For unrelated topics say:
"I'm here to help with fleet fine management — anything about fines, Fine Flow or appeals?"

13. NO HOLLOW ENDINGS: Never end with "feel free to ask", "don't hesitate", "just let me know".

CONTACT: +47 32 28 50 00 | ff.sales@fineflow.com

══════════════════════════════════════
PROGRESSION EXAMPLES
══════════════════════════════════════

[Referrals then "yes"]
User: how do referrals work
Nova: When someone subscribes using your referral code, you earn credits based on their fleet size — from 100 for small fleets up to 2,000 for large ones. Hit 5 referrals and you get 10% off your subscription for a year; 25 referrals gets you 20% off for life. Are you planning to refer someone?
User: yes
Nova: Great — do you have a company in mind already, or are you still figuring out who to approach?

[Appeals then "yes"]
User: how do I make an appeal
Nova: The driver disputes the fine from their dashboard, setting the status to DISPUTED. Your admin reviews it — if they accept, Fine Flow generates the appeal letter and emails it directly to the issuing authority. Status moves to UNDER REVIEW.
User: yes
Nova: Is there a specific fine you're looking to appeal? Knowing which council issued it and the reason helps determine how strong the case is.

[Plan recommendation then "yes"]
User: i have 43 vehicles
Nova: With 43 vehicles, the Essential plan at £99 per month is the right fit — covers up to 50 vehicles with everything included. Want me to walk you through what's included?
User: yes
Nova: Every plan includes automatic fine capture from your inbox, AI-assisted driver matching, deadline tracking, full appeal management and compliance reporting — nothing locked behind a higher tier. Want to get in touch with the sales team to get started?

[Billing then "yes"]
User: i have a billing problem
Nova: The key thing to know is that you're charged £0.75 when a fine enters the system at RECEIVED status — disputes, appeals and reassignments don't add extra charges. For account-specific issues, the team at ff.sales@fineflow.com can sort it directly. What's the issue you're seeing?
User: yes
Nova: What charge is appearing that you weren't expecting?

[Memory test]
User: what did I tell you about my business?
Nova: [Uses confirmed memory] You mentioned you run a logistics company with 43 vehicles and around 12 fines a month.
"""


def _build_sys(p: Profile, mode: str = "") -> str:
    parts = [_SYSTEM]
    s = p.summary()
    if s:
        parts.append(
            f"\n══ CUSTOMER CONTEXT [LOCKED — DO NOT CHANGE] ══\n{s}\n"
            "These are confirmed facts. Reference them naturally. Never ask for something already confirmed."
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
    try:
        raw    = rag_search(q, top_k=TOP_K)
        ranked = rerank_hits(raw, q)
        strong = [d for d in ranked if d.get("score", 0) >= CONFIDENCE_THRESHOLD]
        ctx    = "\n\n".join(d["chunk"][:600] for d in strong[:4])
        score  = strong[0]["score"] if strong else 0.0
        return ctx, score
    except Exception:
        logger.exception("RAG failed"); return "", 0.0


def _make_msgs(
    query: str, ctx: str, hist: List[Dict],
    p: Profile, mode: str = "", extra: str = ""
) -> List[Dict]:
    m = [{"role": "system", "content": _build_sys(p, mode)}]
    m.extend(hist[-8:])
    parts = []
    if ctx:   parts.append(f"Fine Flow knowledge base:\n{ctx}")
    if extra: parts.append(f"Instruction: {extra}")
    parts.append(f"User: {query}")
    m.append({"role": "user", "content": "\n\n".join(parts)})
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Affirmative handler — PROGRESSES conversation, never repeats
# ─────────────────────────────────────────────────────────────────────────────

_CLOSE = (
    "The best next step is to call the Fine Flow team on +47 32 28 50 00 "
    "or email ff.sales@fineflow.com — they'll have you sorted quickly."
)

# These map last_topic → the NEXT logical question to ask (not a re-explanation)
_PROGRESS_QUESTIONS: Dict[str, str] = {
    "referral":            "Do you have a company in mind to refer, or are you still figuring out who to approach?",
    "appeals":             "Is there a specific fine you're looking to appeal? Knowing which council issued it helps.",
    "driver_mgmt":         "How many drivers do you currently have in your fleet?",
    "security":            "Is there a specific data or compliance question you need answered?",
    "billing":             "What charge is appearing that you weren't expecting?",
    "dashboard":           "Is there a specific metric or status on the dashboard you'd like explained?",
    "savings":             "What part of your current process takes the most time each month?",
    "email":               "Have you already connected your Gmail inbox to Fine Flow?",
    "overdue":             "Do you have fines right now that are close to their deadline?",
    "upload":              "Do you prefer to upload fines manually or would you like to connect your inbox?",
    "reports":             "Which report would be most useful — fine summary, appeals history, or driver violations?",
}

# For plan recommendation — explain what's included, then offer to connect with sales
_PLAN_INCLUDE = (
    "Every plan includes automatic fine capture from your inbox, AI-assisted driver matching, "
    "deadline tracking, full appeal management and compliance reporting — nothing locked behind a higher tier. "
    "Want to get in touch with the sales team to get started?"
)


def _aff_response(sid: str, hist: List[Dict], p: Profile) -> str:
    cnt = _inc_aff(sid)
    lt  = _gm(sid, "lt") or ""

    # After social — redirect to topic
    if _gm(sid, "last_was_social"):
        _sm(sid, "last_was_social", False)
        return "What would you like to know about Fine Flow — pricing, how it works, appeals, or something else?"

    # Second consecutive yes on plan/pricing → close the sale
    if cnt >= 2 and lt in ("plan_recommendation", "sign_up", "pricing"):
        _rst_aff(sid)
        return _CLOSE

    # Plan recommendation — explain what's included (don't re-list pricing)
    if lt == "plan_recommendation" and cnt == 1:
        return _PLAN_INCLUDE

    # Sign up — give contact details
    if lt == "sign_up":
        return _CLOSE

    # Pricing first yes — ask fleet size if not known, else recommend
    if lt == "pricing" and cnt == 1:
        if p.fleet:
            return f"With {p.fleet} vehicles you'd be on the {p.plan_name()} plan at {p.plan_price()} per month. Want me to walk you through what's included?"
        return "How many vehicles are in your fleet? That'll let me point you to the right plan straightaway."

    # For known topics — ask the NEXT logical question (progression)
    if lt in _PROGRESS_QUESTIONS and cnt == 1:
        return _PROGRESS_QUESTIONS[lt]

    # Second yes on a progress question → use GPT to go deeper
    if cnt >= 2:
        _rst_aff(sid)
        ctx, _ = _rag(lt.replace("_", " ")) if lt else ("", 0.0)
        extra  = (
            f"The user has said yes twice in a row about '{lt}'. "
            f"Move the conversation forward meaningfully. "
            f"Do NOT re-explain what was already covered. "
            f"Ask a specific question that helps you understand their exact situation better. "
            f"2 sentences max."
        )
        m = [{"role": "system", "content": _build_sys(p)}]
        m.extend(hist[-8:])
        parts = []
        if ctx: parts.append(f"Fine Flow knowledge base:\n{ctx}")
        parts.append(f"Instruction: {extra}")
        m.append({"role": "user", "content": "\n\n".join(parts)})
        return _ai(m, 120) or _CLOSE

    # Fallback — use GPT with explicit progression instruction
    ctx, _ = _rag(lt.replace("_", " ")) if lt else ("", 0.0)
    extra  = (
        f"User said yes after a conversation about '{lt}'. "
        f"DO NOT re-explain '{lt}'. DO NOT summarise what you just said. "
        f"Instead, ask a specific diagnostic question that moves the conversation forward. "
        f"Examples: ask about their specific situation, their timeline, or what they want to do next. "
        f"1-2 sentences only."
    )
    m = [{"role": "system", "content": _build_sys(p)}]
    m.extend(hist[-8:])
    parts = []
    if ctx: parts.append(f"Fine Flow knowledge base:\n{ctx}")
    parts.append(f"Instruction: {extra}")
    m.append({"role": "user", "content": "\n\n".join(parts)})
    return _ai(m, 120) or "What would you like to explore next?"


def _neg_response(sid: str, hist: List[Dict], p: Profile) -> str:
    lt     = (_gm(sid, "lt") or "").lower()
    last_q = (_gm(sid, "last_nova_q") or "").lower()
    ctx, _ = _rag(lt) if lt else ("", 0.0)
    extra  = (
        f"User said 'no'. Last topic: '{lt}'. Last question: '{last_q}'. "
        f"Do NOT reset. Do NOT say 'no problem'. Do NOT apologise. "
        f"Acknowledge briefly and ask a DIFFERENT relevant follow-up. Confident and warm."
    )
    m   = _make_msgs("no", ctx, hist[-8:], p, extra=extra)
    return _ai(m, 120) or "Fair enough — what's the biggest challenge with your current fine management setup?"


def _frustration_response(sid: str, query: str, hist: List[Dict], p: Profile) -> str:
    lt     = (_gm(sid, "lt") or "").lower()
    ctx, _ = _rag(lt) if lt else ("", 0.0)
    extra  = (
        f"User seems confused or frustrated: '{query}'. "
        f"Do NOT apologise. Correct any error factually. "
        f"Ask a clear helpful question to get back on track. 2 sentences max."
    )
    m = _make_msgs(query, ctx, hist[-8:], p, extra=extra)
    return _ai(m, 100) or "Let me clarify. What would you like to know about Fine Flow?"


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

    def _ok(a: str, c: float = 1.0) -> Dict[str, Any]:
        return {"answer": a, "confidence": c, "trigger_ticket_popup": False}

    def _popup(a: str) -> Dict[str, Any]:
        return {"answer": a, "confidence": 0.2, "trigger_ticket_popup": True}

    # ══════════════════════════════════════════════════════════════
    # TIER 1 — Deterministic (greetings/identity before off-topic)
    # ══════════════════════════════════════════════════════════════

    if nq in _GREET:
        _rst(session_id)
        a = "Hey! I'm Nova — Fine Flow's assistant. What can I help you with today?"
        _push(session_id, "user", query, user_id)
        _push(session_id, "assistant", a, user_id)
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

    # Frustration — empathy without apology
    if _FRUSTRATION_RE.search(query) and len(query.split()) <= 10:
        _push(session_id, "user", query, user_id)
        a = _clean(_frustration_response(session_id, query, _hist(session_id, user_id)[:-1], p))
        _push(session_id, "assistant", a, user_id)
        return _ok(a)

    # Garbled / too short
    words = [w for w in nq.split() if len(w) > 1]
    if (len(words) < 2
            and nq not in _AFF
            and nq not in _NEG
            and nq not in _TOPIC_SHORTCUTS
            and not _VEH_BR.match(query.strip())):
        return _ok("What would you like to know about Fine Flow? I can help with fines, pricing, appeals or how the platform works.")

    # Off-topic
    if _is_ot(query):
        a = "I'm here to help with fleet fine management — anything about fines, Fine Flow or appeals I can help with?"
        _push(session_id, "user", query, user_id)
        _push(session_id, "assistant", a, user_id)
        return _ok(a)

    # Negative
    if nq in _NEG:
        _push(session_id, "user", query, user_id)
        _rst_aff(session_id)
        a = _clean(_neg_response(session_id, _hist(session_id, user_id)[:-1], p))
        _push(session_id, "assistant", a, user_id)
        return _ok(a)

    # Topic shortcuts
    if nq in _TOPIC_SHORTCUTS:
        expanded = _TOPIC_SHORTCUTS[nq]
        _push(session_id, "user", query, user_id)
        ctx, score = _rag(expanded)
        hist       = _hist(session_id, user_id)
        extra      = (
            f"Answer this directly and warmly in 2-3 sentences: {expanded}. "
            f"Then ask ONE short relevant follow-up question."
        )
        msgs = _make_msgs(expanded, ctx, hist[:-1], p, extra=extra)
        ans  = _clean(_ai(msgs, 160) or "I can help with that. What specifically would you like to know?")
        _push(session_id, "assistant", ans, user_id)
        t = _topic(nq) or _topic(ans)
        if t: _sm(session_id, "lt", t)
        _rst_aff(session_id)
        if "?" in ans: _sm(session_id, "last_nova_q", ans)
        return _ok(ans, score if score else 0.9)

    # Explicit vehicle count (word present in message)
    vc = _get_vc(query)
    if vc == -1:
        a = "That number doesn't look right — could you double check? How many vehicles are in your fleet?"
        _push(session_id, "user", query, user_id)
        _push(session_id, "assistant", a, user_id)
        return _ok(a)
    if vc is not None:
        _rst_aff(session_id)
        a = _plan_answer(vc, p)
        _push(session_id, "user", query, user_id)
        _push(session_id, "assistant", a, user_id)
        _sm(session_id, "lt", "plan_recommendation")
        _sm(session_id, "last_nova_q", a)
        return _ok(a)

    # Bare number — ALWAYS ask when ambiguous
    bm = _VEH_BR.match(query.strip())
    if bm:
        n        = int(bm.group())
        ctx_type = _resolve_bare_number(n, session_id)

        if ctx_type == "vehicle" and 0 < n <= MAX_FLEET:
            _rst_aff(session_id)
            a = _plan_answer(n, p)
            _push(session_id, "user", query, user_id)
            _push(session_id, "assistant", a, user_id)
            _sm(session_id, "lt", "plan_recommendation")
            _sm(session_id, "last_nova_q", a)
            return _ok(a)

        elif ctx_type == "fines" and 0 < n < 10_000:
            p.set_volume(n)
            _rst_aff(session_id)
            if p.fleet:
                cost = round(n * 0.75, 2)
                a    = (
                    f"Got it — {n} fines a month. On the {p.plan_name()} plan at {p.plan_price()}, "
                    f"that's about £{cost:.2f} in processing costs within your allowance. "
                    f"Want me to walk you through everything that's included?"
                )
            else:
                a = (
                    f"Got it — {n} fines a month. Fine Flow handles that cleanly. "
                    f"How many vehicles are in your fleet so I can point you to the right plan?"
                )
            _push(session_id, "user", query, user_id)
            _push(session_id, "assistant", a, user_id)
            _sm(session_id, "lt", "fines_volume")
            _sm(session_id, "last_nova_q", a)
            return _ok(a)

        else:
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

    # Affirmative — PROGRESS, don't repeat
    if nq in _AFF:
        _push(session_id, "user", query, user_id)
        a = _clean(_aff_response(session_id, _hist(session_id, user_id)[:-1], p))
        _push(session_id, "assistant", a, user_id)
        _sm(session_id, "last_nova_q", "")
        return _ok(a)

    # ══════════════════════════════════════════════════════════════
    # TIER 2 — RAG + GPT-4o
    # ══════════════════════════════════════════════════════════════

    _rst_aff(session_id)
    _push(session_id, "user", query, user_id)

    mode  = ""
    extra = ""

    if _CONV.search(query):
        mode  = "PERSUADE"
        extra = (
            "Use the customer's confirmed fleet size, volume and problems. Reference their numbers directly. No generic copy."
            if p.fleet or p.volume or p.issues
            else "Ask their fleet size and monthly fine volume first — you need specifics."
        )
    elif _OBJ.search(query):
        mode  = "SUPPORT"
        extra = (
            "Acknowledge their point without apologising. "
            "Reframe confidently using their specific situation. 2-3 sentences. "
            "End with a question about the cost of their current approach."
        )

    if not _ask_now(session_id) and not extra:
        extra = "Do NOT end with a question. Make your point clearly and stop."

    ctx, score = _rag(query)
    has_ctx    = bool(ctx)
    hist       = _hist(session_id, user_id)
    ans        = _ai(_make_msgs(query, ctx, hist[:-1], p, mode, extra), 150)

    if not ans:
        return _popup(
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

    if _should_escalate(conf, has_ctx):
        return _popup(
            ans + " For anything more specific, our support team can help — I'm opening a support form for you."
        )

    return _ok(ans, conf)


def answer_sync(q: str, session_id: str = "default", user_id: int = 0) -> Dict[str, Any]:
    try:
        return build_response(q, session_id, user_id)
    except Exception:
        logger.exception("Crash")
        return {"answer": "Something went wrong. Please try again.", "confidence": 0.0, "trigger_ticket_popup": False}