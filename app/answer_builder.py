# app/answer_builder.py
"""
FineFlow Nova — Production Final with MongoDB Memory
=====================================================
Fixes:
1. Single-word topics ("appeals", "fines", "pricing") now route to full answers
2. "yes" after social response no longer triggers pricing
3. MongoDB persistent memory for logged-in users
4. Session-only memory for guests
5. Counter-questions on "no" working properly
6. Affirmative loop prevention (second "yes/sure" → close sale)
"""

import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
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

MAX_FLEET = 50_000

# ─────────────────────────────────────────────────────────────────────────────
# MongoDB — optional, only used when MONGO_URI is set
# ─────────────────────────────────────────────────────────────────────────────

_mongo_client = None
_mongo_db     = None

def _init_mongo():
    global _mongo_client, _mongo_db
    uri = os.getenv("MONGO_URI", "")
    if not uri:
        logger.info("MONGO_URI not set — using in-memory sessions only")
        return
    try:
        from pymongo import MongoClient
        _mongo_client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        _mongo_client.server_info()
        _mongo_db = _mongo_client[os.getenv("MONGO_DB", "fineflow_nova")]
        logger.info("MongoDB connected: %s", os.getenv("MONGO_DB", "fineflow_nova"))
    except Exception as e:
        logger.warning("MongoDB unavailable (%s) — using in-memory sessions", e)
        _mongo_client = None
        _mongo_db     = None

_init_mongo()


def _db_save_turn(user_id: str, role: str, content: str) -> None:
    """Save one turn to MongoDB for a logged-in user."""
    if not _mongo_db or not user_id:
        return
    try:
        _mongo_db.chat_history.insert_one({
            "user_id":   user_id,
            "role":      role,
            "content":   content,
            "timestamp": datetime.now(timezone.utc),
        })
    except Exception:
        logger.exception("MongoDB save failed")


def _db_load_history(user_id: str, limit: int = 20) -> List[Dict[str, str]]:
    """Load recent conversation history for a logged-in user."""
    if not _mongo_db or not user_id:
        return []
    try:
        docs = list(
            _mongo_db.chat_history
            .find({"user_id": user_id}, {"_id": 0, "role": 1, "content": 1})
            .sort("timestamp", -1)
            .limit(limit)
        )
        docs.reverse()
        return [{"role": d["role"], "content": d["content"]} for d in docs]
    except Exception:
        logger.exception("MongoDB load failed")
        return []


def _db_save_profile(user_id: str, profile_data: dict) -> None:
    """Save customer profile to MongoDB."""
    if not _mongo_db or not user_id:
        return
    try:
        _mongo_db.user_profiles.update_one(
            {"user_id": user_id},
            {"$set": {**profile_data, "updated_at": datetime.now(timezone.utc)}},
            upsert=True,
        )
    except Exception:
        logger.exception("MongoDB profile save failed")


def _db_load_profile(user_id: str) -> Optional[dict]:
    """Load customer profile from MongoDB."""
    if not _mongo_db or not user_id:
        return None
    try:
        return _mongo_db.user_profiles.find_one({"user_id": user_id}, {"_id": 0})
    except Exception:
        logger.exception("MongoDB profile load failed")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Customer profile
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
        if self.issues:  parts.append(f"Problems mentioned: {', '.join(self.issues)}")
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

    def to_dict(self) -> dict:
        return {
            "fleet": self.fleet, "volume": self.volume,
            "issues": self.issues, "industry": self.industry,
            "name": self.name, "turns": self.turns,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Profile":
        return cls(
            fleet=d.get("fleet"), volume=d.get("volume"),
            issues=d.get("issues", []), industry=d.get("industry"),
            name=d.get("name"), turns=d.get("turns", 0),
        )


# ─────────────────────────────────────────────────────────────────────────────
# In-memory session store (for guests without MongoDB)
# ─────────────────────────────────────────────────────────────────────────────

_SES: Dict[str, List[Dict]] = {}
_PRO: Dict[str, Profile]    = {}
_MET: Dict[str, Dict]       = {}
_LK  = threading.Lock()


def _hist(sid: str, user_id: str = "") -> List[Dict[str, str]]:
    """Get history — from MongoDB if user_id set, else in-memory."""
    if user_id and _mongo_db:
        return _db_load_history(user_id)
    with _LK:
        return list(_SES.get(sid, []))


def _push(sid: str, role: str, content: str, user_id: str = "") -> None:
    """Save turn — to MongoDB if user_id set, always to in-memory."""
    if user_id and _mongo_db:
        _db_save_turn(user_id, role, content)
    with _LK:
        h = _SES.setdefault(sid, [])
        h.append({"role": role, "content": content})
        cap = CHAT_HISTORY_TURNS * 2
        if len(h) > cap: _SES[sid] = h[-cap:]


def _pro(sid: str, user_id: str = "") -> Profile:
    """Get profile — from MongoDB if user_id set, else in-memory."""
    if user_id and _mongo_db:
        data = _db_load_profile(user_id)
        if data:
            return Profile.from_dict(data)
    with _LK:
        if sid not in _PRO: _PRO[sid] = Profile()
        return _PRO[sid]


def _save_pro(sid: str, p: Profile, user_id: str = "") -> None:
    """Save profile — to MongoDB if user_id set, always to in-memory."""
    if user_id and _mongo_db:
        _db_save_profile(user_id, p.to_dict())
    with _LK:
        _PRO[sid] = p


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
# Profile extraction
# ─────────────────────────────────────────────────────────────────────────────

_FINES_RE = re.compile(
    r"\b(\d+)\s*(?:fines?|pcns?|penalties|violations?|tickets?)"
    r"(?:\s*(?:per|a|each|every)\s*(?:month|monthly|week))?\b", re.I)
_IND_RE = re.compile(
    r"\b(logistics|delivery|courier|haulage|transport|taxi|minicab|bus|coach|construction)\b", re.I)
_NAME_RE = re.compile(r"\b(?:i am|i'm|my name is|call me)\s+([A-Z][a-z]+)\b")
_ISS = [
    (re.compile(r"\b(miss(?:ed?|ing)?\s+(?:deadlines?|appeals?|due\s*dates?))\b", re.I), "missed deadlines"),
    (re.compile(r"\b(drivers?\s+(?:dispute|deny|ignor))\b", re.I),                      "driver disputes"),
    (re.compile(r"\b(spreadsheet)\b", re.I),                                             "using spreadsheets"),
    (re.compile(r"\b(too\s+much\s+admin)\b", re.I),                                     "too much admin"),
]
_FINES_CTX = {"how many fines", "fines per month", "fines a month", "monthly fines", "deal with each month", "fines do you"}
_VEH_CTX   = {"how many vehicles", "fleet size", "vehicles do you", "vehicles are in", "how big is your fleet", "size of your fleet"}


def _upd(sid, q, uid=""):
    p = _pro(sid, uid)
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
    _save_pro(sid, p, uid)


def _resolve_bare_number(n: int, sid: str) -> Optional[str]:
    last_q = (_gm(sid, "last_nova_q") or "").lower()
    lt     = (_gm(sid, "lt") or "").lower()
    for hint in _FINES_CTX:
        if hint in last_q: return "fines"
    for hint in _VEH_CTX:
        if hint in last_q: return "vehicle"
    if lt in {"pricing", "plan_recommendation", "vehicles", "cost"}: return "vehicle"
    if lt in {"fines_volume", "savings"}: return "fines"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# TOPIC SHORTCUTS — single words that should get full answers immediately
# ─────────────────────────────────────────────────────────────────────────────

_TOPIC_SHORTCUTS: Dict[str, str] = {
    "pricing":    "how much does fine flow cost and what are the plans",
    "price":      "how much does fine flow cost and what are the plans",
    "cost":       "how much does fine flow cost and what are the plans",
    "plans":      "what are the fine flow subscription plans",
    "appeals":    "can fine flow help me appeal a fine and how does it work",
    "appeal":     "can fine flow help me appeal a fine and how does it work",
    "fines":      "what happens to a fine when it enters fine flow",
    "billing":    "how does billing work in fine flow",
    "dashboard":  "what does the fine flow dashboard show",
    "drivers":    "how does fine flow manage drivers and assign fines",
    "driver":     "how does fine flow manage drivers and assign fines",
    "security":   "how secure is fine flow and is it gdpr compliant",
    "gdpr":       "how secure is fine flow and is it gdpr compliant",
    "referral":   "does fine flow have a referral programme",
    "referrals":  "does fine flow have a referral programme",
    "features":   "what features does fine flow include in every plan",
    "savings":    "how much time and money can fine flow save me",
    "contact":    "how do i contact fine flow sales team",
    "email":      "how does fine flow connect to gmail to get fines",
    "gmail":      "how does fine flow connect to gmail to get fines",
    "payg":       "is there a pay as you go option in fine flow",
    "overage":    "what is the overage charge in fine flow",
    "reports":    "what reports can i export from fine flow",
    "statuses":   "what are the fine statuses in fine flow",
    "matching":   "how does fine flow match a fine to the correct driver",
    "overdue":    "what happens when a fine becomes overdue in fine flow",
}


# ─────────────────────────────────────────────────────────────────────────────
# Intent sets
# ─────────────────────────────────────────────────────────────────────────────

_GREET = {
    "hi", "hello", "hey", "hiya", "howdy", "yo", "sup",
    "good morning", "good afternoon", "good evening", "morning", "afternoon", "evening",
    "hi there", "hey there", "hello there", "hi nova", "hey nova", "hello nova",
}
_SOC = {
    "how are you", "how are you doing", "how r u", "how are u",
    "hows it going", "how is it going", "whats up", "what s up",
    "you ok", "you good", "how do you do", "you alright", "alright mate",
}
_ID = {
    "who are you", "who r you", "who is nova", "who is this", "who is there",
    "whos there", "anyone there", "what are you", "what is nova",
    "are you a bot", "are you human", "are you ai", "are you a robot",
    "are you male or female", "who the hell are you", "whats your name",
    "what is your name", "introduce yourself", "who am i talking to", "knock knock",
}
_AFF = {
    "yes", "yeah", "yep", "yup", "ya", "ye", "sure", "ok sure", "okay sure",
    "go ahead", "go on", "yes please", "yes sure", "of course", "absolutely",
    "definitely", "do it", "tell me more", "more", "explain", "explain more",
    "yes explain", "go for it", "sounds good", "continue", "carry on",
    "keep going", "please do", "i would", "please explain", "show me",
    "walk me through it", "for sure", "yes for sure", "sure thing", "yes tell me",
}
_THX = {
    "thanks", "thank you", "thank u", "cheers", "that helps", "that helped", "ta",
    "okay thanks", "ok thanks", "great thanks", "perfect", "brilliant", "nice one",
    "lovely", "great", "awesome", "wonderful", "thank you so much", "many thanks",
}
_BYE  = {"bye", "goodbye", "see you", "see ya", "later", "take care", "good bye", "cya", "farewell", "cheerio"}
_NEG  = {"no", "nope", "nah", "no thanks", "not now", "not really", "no thank you", "nah thanks", "not sure"}
_RUDE = {"stupid", "idiot", "useless", "rubbish", "garbage", "terrible", "you suck", "dumb bot", "waste of time"}
_FILL = {
    "ok", "okay", "right", "alright", "cool", "nice", "interesting", "really",
    "hmm", "hm", "ah", "oh", "i see", "got it", "understood", "makes sense",
    "noted", "wow", "woah", "omg", "anything", "something", "whatever",
}

_FF_OK = re.compile(
    r"\b(council|authority|fine|pcn|penalty|fineflow|fine flow|appeal|dispute|"
    r"driver|fleet|vehicle|overage|allowance|billing|subscription|payment|"
    r"uk traffic|traffic violation|parking|bus lane|congestion|emission|"
    r"dvla|tfl|fixed penalty|notice to owner|gmail|inbox|csv|upload|"
    r"dashboard|referral|credits|stripe|sign up|get started|how much|pricing|cost|plan)\b", re.I)

_OT = [
    re.compile(r"\b(html|css|javascript|typescript|python|java|php|sql|react|angular|vue|node\.?js|django|flask|docker|kubernetes|github|coding|programming|teach me|how to code)\b", re.I),
    re.compile(r"\b(machine learning|deep learning|neural network|large language model|generative ai|llm|bert)\b", re.I),
    re.compile(r"\b(recipe|cooking|restaurant|pizza|burger|sandwich|coffee|tea|cake|meal|make me a food|bake me)\b", re.I),
    re.compile(r"\b(movie|film|song|lyrics|music|football match|cricket match|weather forecast|todays news|politics|history lesson|capital city|who invented|tell me a joke|write me a poem)\b", re.I),
    re.compile(r"\b(write an essay|translate this|proofread my|write my cv|write a story for me)\b", re.I),
    re.compile(r"\b(chatgpt|openai|gemini|claude ai|anthropic|google bard|bing ai|alexa|siri)\b", re.I),
]


def _is_ot(q):
    if _FF_OK.search(q): return False
    return any(p.search(q) for p in _OT)


_VEH_EX = re.compile(r"\b(\d+)\s*(vehicle|vehicles|van|vans|truck|trucks|car|cars|lorry|lorries|in my fleet|in our fleet)\b", re.I)
_VEH_FL = re.compile(r"\b(?:fleet of|manage|running|operate|run)\s+(\d+)\b", re.I)
_VEH_BR = re.compile(r"^\s*(\d+)\s*$")
_DRV_CT = re.compile(r"\b(driver|drivers|staff|employee|employees|people|worker|team|members)\b", re.I)
_PURCH  = re.compile(r"\b(want to buy|want to subscribe|want to sign up|how do i get started|how do i sign up|get started|free trial|sign me up|book a demo|talk to sales|how to start|where do i sign|how do i join)\b", re.I)
_CONV   = re.compile(r"\b(convince|persuade|sell me|why should i|why buy|is it worth|should i buy|worth it|why choose fineflow|why fine flow)\b", re.I)
_OBJ    = re.compile(r"\b(expensive|too much|too costly|already use spreadsheet|we manage manually|we handle fines ourselves|manage fines manually)\b", re.I)


def _get_vc(q, sid):
    if _DRV_CT.search(q): return None
    raw = None
    m = _VEH_EX.search(q)
    if m: raw = int(m.group(1))
    elif (m2 := _VEH_FL.search(q)): raw = int(m2.group(1))
    if raw is None: return None
    return -1 if raw > MAX_FLEET else raw


def _plan_answer(n, p):
    p.fleet = n
    if n <= 50:    name, price, lim = "Essential", "£99",  "50"
    elif n <= 100: name, price, lim = "Core",      "£199", "100"
    elif n <= 200: name, price, lim = "Advanced",  "£399", "200"
    else:          name, price, lim = "Elite",      "£499", "unlimited"
    return (f"With {n} vehicles, the {name} plan at {price} per month is the right fit — "
            f"covers up to {lim} vehicles with everything included and nothing locked away. "
            f"Want me to walk you through what's included?")


_TMAP = {
    "pric":"pricing","cost":"pricing","plan":"pricing","£":"pricing",
    "vehicle":"pricing","fleet":"pricing","package":"pricing",
    "fines per month":"fines_volume","fines a month":"fines_volume","monthly fines":"fines_volume",
    "appeal":"appeals","dispute":"appeals",
    "driver":"driver_mgmt","referral":"referral","refer":"referral","discount":"referral",
    "security":"security","gdpr":"security","card":"security",
    "billing":"billing","stripe":"billing","dashboard":"dashboard",
    "gmail":"email","inbox":"email","save":"savings","admin time":"savings",
    "overdue":"overdue","deadline":"overdue","sign":"sign_up","get started":"sign_up",
}


def _topic(t):
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
Fine Flow provides 24/7 fleet fine tracking, management and compliance in one place.

YOUR PERSONALITY:
Warm, humble, direct — like a knowledgeable colleague who genuinely cares. Conversational and concise. Never robotic. Never pushy. Think of a great sales rep who listens first and helps second.

CRITICAL RULES:

1. SHORT ANSWERS
2 to 3 sentences. Never write long paragraphs. If they want more they will ask.

2. CLIENT'S EXACT WORDING
When describing Fine Flow use these exact phrases:
"Fine Flow is an automated system for managing fines from start to finish"
"keeps the entire process organised, accountable, and under control"
"cut admin time by up to 80%"
"never miss a penalty deadline"

3. MEMORY — USE EVERYTHING YOU KNOW
If you know their fleet size, volume, industry or problems — always use it. Never ask for something they already told you.

4. COUNTER QUESTIONS ON "NO"
When user says "no" — do NOT reset. Do NOT say "no problem" and stop. Ask a DIFFERENT relevant follow-up that continues the conversation. Keep them engaged.

5. FOLLOW-UP QUESTIONS
After most answers ask one short relevant question. Vary them. Good examples:
"How many vehicles are in your fleet?"
"How many fines do you deal with each month?"
"What does your current process look like?"
"Is there a particular stage causing the most headaches?"
"Have you had fines escalate because of missed deadlines?"
"What's the biggest pain point right now?"

6. AFFIRMATIVE LOOP
First "yes/sure" → explain warmly what's included (2-3 sentences, no lists)
Second "yes/sure" → push to action: give contact details
Never dump full pricing list again on second yes.

7. TOPIC SHORTCUTS
When user sends a single word like "appeals", "fines", "pricing" — answer it directly. Do not ask them to clarify. Give a proper answer.

8. PAYMENT
Fine Flow does NOT pay fines automatically. Payment is always by the user on the authority's site.

9. CARD DETAILS
Fine Flow never stores card details. Say this first when asked.

10. TOPIC
Help with Fine Flow AND general UK fleet fine questions (PCNs, FPNs, councils, TfL, DVLA, appeal rights). For unrelated topics: "I'm here to help with fleet fine management — anything about fines, Fine Flow or appeals?"

11. NO HOLLOW ENDINGS
Never end with "feel free to ask", "don't hesitate", "just let me know".

PRICING:
Essential: £99/month — 5 to 50 vehicles
Core: £199/month — 51 to 100 vehicles
Advanced: £399/month — 101 to 200 vehicles
Elite: £499/month — 200+ vehicles
Per fine within allowance: £0.75 | Overage: £2.50 | PAYG: £2.75 (no subscription)
All plans identical features. No £2.00 fee.

REFERRAL:
Credits: 100 (1-25v) | 250 (26-100v) | 750 (101-500v) | 2000 (500+v)
Silver 3=100 credits | Gold 5=10% off 12mo | Platinum 10=15% off 12mo | Titan 25=20% for life
New joiners with code: £75 credits

CONTACT: +47 32 28 50 00 | ff.sales@fineflow.com

EXAMPLES:

User: appeals
Nova: Yes — Fine Flow doesn't just manage appeals, it learns from them. When a driver disputes a fine, the admin reviews it and if accepted, Fine Flow generates an appeal letter and sends it directly to the issuing authority. Over time it refines its approach based on wins and losses, giving you a win probability before you submit. Have you had fines you felt could have been challenged?

User: pricing
Nova: Fine Flow's plans are based on fleet size. Essential is £99/month for up to 50 vehicles, Core is £199 for up to 100, Advanced is £399 for up to 200, and Elite is £499 for 200+. Every fine within your allowance costs £0.75, and there's also a pay-as-you-go option at £2.75. How many vehicles are you running?

User: fines
Nova: Fine Flow is an automated system for managing fines from start to finish. The moment a fine arrives in your inbox it's captured automatically, the details extracted, matched to the right driver and tracked through to resolution — keeping everything organised, accountable and under control. Want me to walk you through how it all works?

User: yes (after social response)
Nova: What would you like to know about Fine Flow — pricing, how it works, appeals, or something else?

User: no (after follow-up question)
Nova: Fair enough — what's the biggest challenge with your current fine management setup? That'll help me point you to what's most useful.
"""


def _build_sys(p: Profile, mode: str = "") -> str:
    parts = [_SYSTEM]
    s = p.summary()
    if s:
        parts.append(
            f"\nWHAT I KNOW ABOUT THIS CUSTOMER:\n{s}\n"
            "Always use this. Never ask for information you already have."
        )
    if mode:
        parts.append(f"\nMODE: {mode}")
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI + RAG
# ─────────────────────────────────────────────────────────────────────────────

def _ai(msgs, max_tok=150):
    if not OPENAI_API_KEY: return None
    try:
        r = requests.post(
            OPENAI_API_URL,
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={"model": OPENAI_MODEL, "messages": msgs, "temperature": 0.7, "max_tokens": max_tok},
            timeout=25)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        logger.exception("OpenAI failed"); return None


def _rag(q):
    try:
        raw    = rag_search(q, top_k=TOP_K)
        ranked = rerank_hits(raw, q)
        strong = [d for d in ranked if d.get("score", 0) >= CONFIDENCE_THRESHOLD]
        return "\n\n".join(d["chunk"][:600] for d in strong[:4])
    except Exception:
        logger.exception("RAG failed"); return ""


def _make_msgs(query, ctx, hist, p, mode="", extra=""):
    m = [{"role": "system", "content": _build_sys(p, mode)}]
    m.extend(hist[-8:])
    parts = []
    if ctx:   parts.append(f"Fine Flow knowledge base:\n{ctx}")
    if extra: parts.append(f"Instruction: {extra}")
    parts.append(f"User: {query}")
    m.append({"role": "user", "content": "\n\n".join(parts)})
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Affirmative + Negative handlers
# ─────────────────────────────────────────────────────────────────────────────

_CLOSE = "The best next step is to call the team on +47 32 28 50 00 or email ff.sales@fineflow.com — they'll have you sorted quickly."

_EXPAND = {
    "pricing":             "Explain Fine Flow pricing warmly in 2-3 sentences. Ask how many vehicles if unknown.",
    "plan_recommendation": "Explain warmly what is included in every Fine Flow plan in 2-3 sentences. Do NOT list plans again. Ask if they want to get in touch with sales.",
    "appeals":             "Explain the Fine Flow appeal process: driver disputes → DISPUTED → admin accepts/rejects → if accepted, appeal letter sent by email, status UNDER REVIEW. 2-3 sentences.",
    "driver_mgmt":         "Explain how drivers are added (individually or CSV) and matched to fines using vehicle, date and time. 2-3 sentences.",
    "fines_volume":        "Based on their fine volume, recommend the right plan. Be specific and use their numbers.",
    "referral":            "Explain the referral programme: credits per referral based on fleet size, plus tier discounts up to 20% for life. 2-3 sentences.",
    "security":            "Explain GDPR compliance and security: JWT, bcrypt, AES-256, never stores card data, never shares data. 2-3 sentences.",
    "billing":             "Explain billing: monthly Stripe, credits reset, overage end of cycle, cooldown on cancel. 2-3 sentences.",
    "dashboard":           "Explain what the company dashboard shows: urgency-based, outstanding fines, deadlines, credits, billing. 2-3 sentences.",
    "savings":             "Give specific savings figures for their fleet size if known.",
    "email":               "Explain Gmail connection: OAuth (recommended) or App Password. 2-3 sentences.",
    "overdue":             "Explain how Fine Flow tracks deadlines and marks fines OVERDUE at midnight automatically. 2-3 sentences.",
    "sign_up":             "Tell them warmly to call +47 32 28 50 00 or email ff.sales@fineflow.com.",
}


def _aff_response(sid, hist, p):
    cnt = _inc_aff(sid)
    lt  = _gm(sid, "lt") or ""

    # Was the last exchange just social? If so, redirect to topic selection
    last_was_social = _gm(sid, "last_was_social") or False
    if last_was_social:
        _sm(sid, "last_was_social", False)
        return "What would you like to know about Fine Flow — pricing, how it works, appeals, or something else?"

    if cnt >= 2 and lt in ("plan_recommendation", "sign_up", "pricing"):
        _rst_aff(sid); return _CLOSE

    prompt = _EXPAND.get(lt, "Expand naturally on the most recent Fine Flow topic in 2-3 sentences. Use the customer's information if available.")
    ctx    = _rag(lt.replace("_", " ")) if lt else ""
    m      = [{"role": "system", "content": _build_sys(p, "Warm, concise. 2-3 sentences. Natural question at end if it helps.")}]
    m.extend(hist[-8:])
    parts  = []
    if ctx: parts.append(f"Fine Flow knowledge base:\n{ctx}")
    parts.append(f"Instruction: {prompt}")
    m.append({"role": "user", "content": "\n\n".join(parts)})
    return _ai(m, 180) or "What would you like to know more about?"


def _neg_response(sid, hist, p):
    lt     = (_gm(sid, "lt") or "").lower()
    last_q = (_gm(sid, "last_nova_q") or "").lower()
    ctx    = _rag(lt) if lt else ""
    extra  = (
        f"The user said 'no'. Last topic: '{lt}'. Last question asked: '{last_q}'. "
        f"Do NOT say 'no problem' and stop. Do NOT reset. "
        f"Acknowledge briefly and ask a DIFFERENT relevant follow-up that keeps the conversation going. "
        f"Be warm and natural."
    )
    m   = _make_msgs("no", ctx, hist[-8:], p, extra=extra)
    ans = _ai(m, 120)
    return ans or "Fair enough — what's the biggest challenge with your current fine management setup?"


# ─────────────────────────────────────────────────────────────────────────────
# Main response builder
# ─────────────────────────────────────────────────────────────────────────────

def build_response(
    query: str,
    session_id: str = "default",
    user_id: str = "",          # set when user is logged in
) -> Dict[str, Any]:
    query      = query.strip()
    session_id = session_id or "default"
    if not query:
        return {"answer": "Ask me anything about Fine Flow.", "confidence": 1.0}

    nq = _norm(query)
    p  = _pro(session_id, user_id)
    _upd(session_id, query, user_id)

    # ══════════════════════════════════════════════════════════
    # TIER 1: Deterministic — greetings BEFORE off-topic guard
    # ══════════════════════════════════════════════════════════

    if nq in _GREET:
        _rst(session_id)
        a = "Hey! I'm Nova — Fine Flow's assistant. What can I help you with today?"
        return {"answer": a, "confidence": 1.0}

    if nq in _SOC:
        _sm(session_id, "last_was_social", True)
        a = "Doing well, cheers for asking! What can I help you with — pricing, fines, appeals?"
        return {"answer": a, "confidence": 1.0}

    if nq in _ID:
        return {"answer": "I'm Nova, Fine Flow's AI assistant. I can help with anything about managing fleet fines — pricing, appeals, how the platform works, UK fine rules. What would you like to know?", "confidence": 1.0}

    if nq in _THX:
        _rst_aff(session_id)
        return {"answer": "Happy to help! Anything else you'd like to know?", "confidence": 1.0}

    if nq in _BYE:
        return {"answer": "Good luck with the fleet management. Come back any time!", "confidence": 1.0}

    if any(r in nq for r in _RUDE):
        return {"answer": "Let me try again — what would you like to know about Fine Flow?", "confidence": 1.0}

    if nq in _FILL:
        _rst_aff(session_id)
        lt = _gm(session_id, "lt")
        if lt:
            return {"answer": f"Is there anything more you'd like to know about Fine Flow?", "confidence": 1.0}
        return {"answer": "Is there anything about Fine Flow I can help you with?", "confidence": 1.0}

    # Garbled / too short
    words = [w for w in nq.split() if len(w) > 1]
    if len(words) < 2 and nq not in _AFF and nq not in _NEG and not _VEH_BR.match(query.strip()) and nq not in _TOPIC_SHORTCUTS:
        return {"answer": "What would you like to know about Fine Flow? I can help with fines, pricing, appeals or how the platform works.", "confidence": 1.0}

    # Off-topic — AFTER all deterministic checks
    if _is_ot(query):
        a = "I'm here to help with fleet fine management — anything about fines, Fine Flow or appeals I can help with?"
        _push(session_id, "user", query, user_id)
        _push(session_id, "assistant", a, user_id)
        return {"answer": a, "confidence": 1.0}

    # Negative / "no" — counter-question
    if nq in _NEG:
        _push(session_id, "user", query, user_id)
        _rst_aff(session_id)
        a = _clean(_neg_response(session_id, _hist(session_id, user_id)[:-1], p))
        _push(session_id, "assistant", a, user_id)
        return {"answer": a, "confidence": 1.0}

    # ── TOPIC SHORTCUTS — single word like "appeals", "pricing", "fines" ──
    if nq in _TOPIC_SHORTCUTS:
        expanded_query = _TOPIC_SHORTCUTS[nq]
        _push(session_id, "user", query, user_id)
        ctx  = _rag(expanded_query)
        hist = _hist(session_id, user_id)
        extra = f"Answer this directly and warmly in 2-3 sentences: {expanded_query}. Then ask one relevant follow-up question."
        msgs = _make_msgs(expanded_query, ctx, hist[:-1], p, extra=extra)
        ans  = _clean(_ai(msgs, 160) or "")
        if not ans:
            ans = "I can help with that. What specifically would you like to know?"
        _push(session_id, "assistant", ans, user_id)
        t = _topic(nq) or _topic(ans)
        if t: _sm(session_id, "lt", t)
        _rst_aff(session_id)
        return {"answer": ans, "confidence": 1.0}

    # Explicit vehicle count
    vc = _get_vc(query, session_id)
    if vc == -1:
        a = "That number doesn't look right — could you double check? How many vehicles are in your fleet?"
        _push(session_id, "user", query, user_id)
        _push(session_id, "assistant", a, user_id)
        return {"answer": a, "confidence": 1.0}
    if vc is not None:
        _rst_aff(session_id)
        a = _plan_answer(vc, p)
        _save_pro(session_id, p, user_id)
        _push(session_id, "user", query, user_id)
        _push(session_id, "assistant", a, user_id)
        _sm(session_id, "lt", "plan_recommendation")
        return {"answer": a, "confidence": 1.0}

    # Bare number — resolve by context
    m = _VEH_BR.match(query.strip())
    if m:
        n        = int(m.group())
        ctx_type = _resolve_bare_number(n, session_id)
        if ctx_type == "vehicle" and 0 < n <= MAX_FLEET:
            _rst_aff(session_id)
            a = _plan_answer(n, p)
            _save_pro(session_id, p, user_id)
            _push(session_id, "user", query, user_id)
            _push(session_id, "assistant", a, user_id)
            _sm(session_id, "lt", "plan_recommendation")
            return {"answer": a, "confidence": 1.0}
        elif ctx_type == "fines" and 0 < n < 10_000:
            p.volume = n
            _rst_aff(session_id)
            _save_pro(session_id, p, user_id)
            if p.fleet:
                cost = round(n * 0.75, 2)
                a = (f"Got it — {n} fines a month. On the {p.plan_name()} plan at {p.plan_price()}, "
                     f"that's about £{cost:.2f} in processing costs within your allowance. "
                     f"Want me to walk you through everything that's included?")
            else:
                a = (f"Got it — {n} fines a month. Fine Flow would handle that cleanly. "
                     f"How many vehicles are in your fleet so I can point you to the right plan?")
            _push(session_id, "user", query, user_id)
            _push(session_id, "assistant", a, user_id)
            _sm(session_id, "lt", "fines_volume")
            return {"answer": a, "confidence": 1.0}
        else:
            a = "Just to make sure I give you the right info — is that the number of vehicles in your fleet, or your monthly fine volume?"
            _push(session_id, "user", query, user_id)
            _push(session_id, "assistant", a, user_id)
            return {"answer": a, "confidence": 1.0}

    # Purchase intent
    if _PURCH.search(query):
        _rst_aff(session_id)
        sfx = "" if p.fleet else " How many vehicles are you running so I can point you to the right plan?"
        a   = f"To get started, call the team on +47 32 28 50 00 or email ff.sales@fineflow.com — they'll get you sorted quickly.{sfx}"
        _push(session_id, "user", query, user_id)
        _push(session_id, "assistant", a, user_id)
        _sm(session_id, "lt", "sign_up")
        return {"answer": a.strip(), "confidence": 1.0}

    # Affirmative
    if nq in _AFF:
        _push(session_id, "user", query, user_id)
        a = _clean(_aff_response(session_id, _hist(session_id, user_id)[:-1], p))
        _push(session_id, "assistant", a, user_id)
        _sm(session_id, "last_nova_q", "")
        return {"answer": a, "confidence": 1.0}

    # ══════════════════════════════════════════════════════════
    # TIER 2: RAG + GPT-4o
    # ══════════════════════════════════════════════════════════

    _rst_aff(session_id)
    _push(session_id, "user", query, user_id)

    mode  = ""
    extra = ""

    if _CONV.search(query):
        mode  = "PERSUADE"
        extra = (
            "Use the customer's exact fleet size, volume and problems. Reference their numbers. No generic copy."
            if p.fleet or p.volume or p.issues
            else "Ask about fleet size and monthly fine volume first."
        )
    elif _OBJ.search(query):
        mode  = "SUPPORT"
        extra = "Acknowledge their point warmly first. Reframe using their specific situation. 2-3 sentences. End with a question about the real cost of their current approach."

    if not _ask_now(session_id) and not extra:
        extra = "Do NOT end with a question this time. Make your point clearly and naturally, then stop."

    ctx  = _rag(query)
    hist = _hist(session_id, user_id)
    ans  = _ai(_make_msgs(query, ctx, hist[:-1], p, mode, extra), 150)

    if not ans:
        ans = "The Fine Flow team can help with that directly — call +47 32 28 50 00 or email ff.sales@fineflow.com."

    ans = _clean(ans)
    _push(session_id, "assistant", ans, user_id)

    if "?" in ans:
        _sm(session_id, "last_nova_q", ans)

    t = _topic(query) or _topic(ans)
    if t: _sm(session_id, "lt", t)

    return {"answer": ans, "confidence": 0.9 if ctx else 0.5}


# ─────────────────────────────────────────────────────────────────────────────
# Public wrappers
# ─────────────────────────────────────────────────────────────────────────────

def answer_sync(q: str, session_id: str = "default", user_id: str = "") -> Dict[str, Any]:
    try:
        return build_response(q, session_id, user_id)
    except Exception:
        logger.exception("Crash")
        return {"answer": "Something went wrong. Please try again.", "confidence": 0.0} 