"""
FineFlow Nova — Production v18 (Strong Memory + QA Corrections)
=================================================================
Fixes vs v17, based on client QA review:

MEMORY FIX (the big one):
  - Short/numeric replies (e.g. "35", "no", "yes") that follow a question
    from Nova are NEVER swallowed by generic handlers anymore.
  - If Nova just asked a question (tracked via _MET["awaiting"]), the next
    user message — no matter how short — goes to GPT WITH FULL HISTORY,
    so it can actually answer in context.
  - Session history window increased and always passed for any in-context
    reply, not just "informational" queries.

CONTENT FIXES (from QA):
  - Core overview always includes required exact phrasing.
  - Appeal flow corrected: Driver -> DISPUTED, Admin -> accept/reject,
    if accepted -> UNDER REVIEW + appeal letter via email. Roles never blurred.
  - Removed "learns from them to improve success rates" AI-tone phrasing.
  - Pricing restored to 4 tiers (Essential/Core/Advanced/Elite) + PAYG,
    matching the correct/current client requirement (this overrides the
    3-tier version from the previous round — client confirmed 4 tiers + PAYG
    is correct here).
  - Billing: charged on fine ENTRY (status RECEIVED), not on actions.
  - Cancellation: BOTH cooldown AND outstanding balance clearance required.
  - "Charge Certificates / Debt Registration" banned — not real FF terms.
  - Bulk/manual upload confirmed as available alongside auto email capture.
  - "What should I do right now" no longer turns into a sales pitch for
    low-volume users.
  - Payment question answered directly without unrelated feature dumps.
"""

import re
import threading
from dataclasses import dataclass, field
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
# Customer profile
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Profile:
    fleet:    Optional[int]  = None
    volume:   Optional[int]  = None
    issues:   List[str]      = field(default_factory=list)
    plan:     Optional[str]  = None
    industry: Optional[str]  = None
    turns:    int            = 0

    def ctx(self) -> str:
        lines = []
        if self.fleet and self.fleet <= MAX_FLEET:
            lines.append(f"Fleet size: {self.fleet} vehicles")
        if self.volume:
            lines.append(f"Monthly fines: ~{self.volume}")
        if self.industry:
            lines.append(f"Industry: {self.industry}")
        if self.issues:
            lines.append("Known problems: " + ", ".join(self.issues))
        if self.plan:
            lines.append(f"Plan discussed: {self.plan}")
        return ("Customer context (use naturally, never repeat verbatim):\n"
                + "\n".join(lines)) if lines else ""


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*",     r"\1", text)
    text = re.sub(r"_(.*?)_",       r"\1", text)
    text = text.replace("→", "to").replace("->", "to").replace("`", "")
    for bad in [
        "feel free to ask!", "feel free to ask.",
        "please let me know if you need anything.",
        "please let me know if you need further assistance.",
        "don't hesitate to ask.", "don't hesitate to reach out.",
        "if you have any other questions, feel free to ask.",
    ]:
        if text.lower().endswith(bad):
            text = text[: -len(bad)].rstrip(" ,.")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Session state — STRONG MEMORY
# ─────────────────────────────────────────────────────────────────────────────

_SES: Dict[str, List[Dict]] = {}
_PRO: Dict[str, Profile]    = {}
_MET: Dict[str, Dict]       = {}
_LK  = threading.Lock()

# How many turns of history to send to GPT (each turn = 1 user + 1 assistant msg)
HISTORY_WINDOW = 12   # increased from 8 — stronger memory

PRICE_TOPICS = {"pricing", "plan_recommendation", "vehicles", "cost", "billing"}


def _hist(s):
    with _LK: return list(_SES.get(s, []))

def _push(s, role, content):
    with _LK:
        h = _SES.setdefault(s, [])
        h.append({"role": role, "content": content})
        cap = CHAT_HISTORY_TURNS * 2
        if len(h) > cap:
            _SES[s] = h[-cap:]

def _pro(s):
    with _LK:
        if s not in _PRO:
            _PRO[s] = Profile()
        return _PRO[s]

def _sm(s, k, v):
    with _LK: _MET.setdefault(s, {})[k] = v

def _gm(s, k):
    with _LK: return _MET.get(s, {}).get(k)

def _rst(s):
    with _LK: _MET[s] = {}; _PRO[s] = Profile()

def _inc_aff(s):
    with _LK:
        m = _MET.setdefault(s, {})
        c = m.get("aff", 0) + 1
        m["aff"] = c
        return c

def _rst_aff(s):
    with _LK: _MET.setdefault(s, {})["aff"] = 0

def _ask_now(s):
    with _LK:
        m = _MET.setdefault(s, {})
        c = m.get("rc", 0) + 1
        m["rc"] = c
        return c % 2 == 0

def _set_awaiting(s, val: bool):
    """Track whether Nova just asked the user a question."""
    with _LK: _MET.setdefault(s, {})["awaiting"] = val

def _is_awaiting(s) -> bool:
    with _LK: return _MET.get(s, {}).get("awaiting", False)


def _last_assistant_msg(s: str) -> str:
    h = _hist(s)
    for m in reversed(h):
        if m["role"] == "assistant":
            return m["content"]
    return ""


def _nova_asked_question(s: str) -> bool:
    """Check if Nova's last message ended with a question mark."""
    last = _last_assistant_msg(s)
    return last.strip().endswith("?")


# ─────────────────────────────────────────────────────────────────────────────
# Profile extraction
# ─────────────────────────────────────────────────────────────────────────────

_FINES_RE = re.compile(
    r"\b(\d+)\s*(?:fines?|pcns?|penalties|violations?|tickets?)"
    r"(?:\s*(?:per|a|each|every)\s*(?:month|monthly|week))?\b", re.I)
_BARE_NUM_RE = re.compile(r"^\s*(\d+)\s*$")
_IND_RE = re.compile(
    r"\b(logistics|delivery|courier|haulage|transport|taxi|minicab"
    r"|bus|coach|construction|utilities)\b", re.I)
_ISS = [
    (re.compile(r"\b(miss(?:ed?|ing)?\s+(?:deadlines?|appeals?|due\s*dates?))\b", re.I), "missed deadlines"),
    (re.compile(r"\b(drivers?\s+(?:dispute|deny|ignor|avoid))\b", re.I),                "driver disputes"),
    (re.compile(r"\b(manual(?:ly)?\s+(?:track|manage|process|handl))\b", re.I),         "manual admin"),
    (re.compile(r"\b(spreadsheet)\b", re.I),                                             "using spreadsheets"),
]


def _upd_pro(s, q):
    p = _pro(s)
    p.turns += 1
    m = _FINES_RE.search(q)
    if m and p.volume is None:
        v = int(m.group(1))
        if v < 10_000:
            p.volume = v
    # If awaiting an answer and message is a bare number, treat it as
    # fine volume (most common follow-up question Nova asks)
    elif p.volume is None:
        m2 = _BARE_NUM_RE.match(q)
        if m2 and _is_awaiting(s):
            v = int(m2.group(1))
            if 0 < v < 10_000:
                p.volume = v
    m = _IND_RE.search(q)
    if m:
        p.industry = m.group(1).lower()
    for pat, lbl in _ISS:
        if pat.search(q) and lbl not in p.issues:
            p.issues.append(lbl)


# ─────────────────────────────────────────────────────────────────────────────
# Intent sets — exact match on normalised query
# ─────────────────────────────────────────────────────────────────────────────

_GREET = {
    "hi", "hello", "hey", "hiya", "howdy", "yo", "sup",
    "good morning", "good afternoon", "good evening",
    "morning", "afternoon", "evening",
    "hi there", "hey there", "hello there",
    "hi nova", "hey nova", "hello nova",
}
_SOC = {
    "how are you", "how are you doing", "how r u", "how r you", "how are u",
    "hows it going", "how is it going", "whats up", "what s up",
    "you ok", "you good", "how do you do", "you alright", "alright mate",
}
_ID = {
    "who are you", "who r you", "who r u", "who youre", "who you re",
    "who is nova", "who is this", "who is there", "whos there", "who s there",
    "anyone there", "is anyone there", "what are you", "what is nova",
    "are you a bot", "are you human", "are you ai", "are you a robot",
    "whats your name", "what is your name", "introduce yourself",
    "who am i talking to", "knock knock",
}
_AFF = {
    "yes", "yeah", "yep", "yup", "ya", "ye",
    "sure", "ok sure", "okay sure",
    "go ahead", "go on", "yes go ahead", "yes please",
    "yes sure", "yes of course", "of course",
    "absolutely", "definitely", "do it",
    "tell me more", "more", "explain", "explain more",
    "yes explain", "go for it", "sounds good",
    "continue", "carry on", "keep going",
    "please do", "i would", "please explain",
    "show me", "walk me through it",
    "yes elite", "yes core", "yes essential", "yes advanced",
    "for sure", "yes for sure", "sure thing", "yes tell me",
}
_THX = {
    "thanks", "thank you", "thank u", "cheers",
    "that helps", "that helped", "ta", "ty",
    "okay thanks", "ok thanks", "great thanks",
    "perfect", "brilliant", "nice one", "lovely",
    "great", "awesome", "wonderful",
    "thank you so much", "many thanks", "much appreciated",
}
_BYE = {
    "bye", "goodbye", "see you", "see ya", "later",
    "take care", "good bye", "cya", "ttyl", "farewell", "cheerio",
}
_NEG = {
    "no", "nope", "nah", "no thanks", "not now", "skip",
    "never mind", "nevermind", "no need", "not really",
    "no thank you", "nah thanks",
}
_RUDE = {
    "you dumb", "you are dumb", "ur dumb", "stupid", "idiot", "useless",
    "rubbish", "garbage", "terrible", "you suck",
    "this is rubbish", "dumb bot", "waste of time",
}
_FILL = {
    "ok", "okay", "right", "alright", "cool", "nice",
    "interesting", "really", "seriously", "hmm", "hm",
    "ah", "oh", "i see", "got it", "understood",
    "makes sense", "noted", "wow", "waow", "woah", "omg",
    "anything", "something", "whatever",
}

# Fine Flow terms — these ALWAYS pass off-topic guard
_FF_ALLOW = re.compile(
    r"\b(council|authority|authorities|payment portal|log into|login to"
    r"|pay.*fine|fine.*pay|appeal.*fine|fine.*appeal|pcn|penalty charge"
    r"|parking fine|fleet fine|fineflow|fine flow|overage|allowance"
    r"|subscription|driver log|vehicle log|billing|credit|cancel"
    r"|reassign|dispute|uk fine|council fine|penalty notice"
    r"|fixed penalty|congestion|bus lane|violation|fleet manager"
    r"|fleet management|gmail|mount.*gmail|inbox)\b", re.I)

_OT = [
    re.compile(r"\b(html|css|javascript|typescript|python|java\b|php|sql|react"
               r"|angular|vue|node\.?js|django|flask|docker|kubernetes|github"
               r"|devops|backend|frontend|coding|programming|teach me|how to code"
               r"|learn to code|learn html|learn css|code for me)\b", re.I),
    re.compile(r"\b(machine learning|deep learning|neural network|large language model"
               r"|generative ai|train a model|llm|bert)\b", re.I),
    re.compile(r"\b(recipe|cooking|restaurant|pizza|burger|sandwich|coffee\b|tea\b"
               r"|cake|meal|bake|order food|make me a|bake me|cook me)\b", re.I),
    re.compile(r"\b(movie|film|song|lyrics|music\b|football match|cricket match"
               r"|weather forecast|todays news|politics|history lesson"
               r"|capital city|who invented|tell me a joke|write me a poem)\b", re.I),
    re.compile(r"\b(write an essay|translate this|proofread|write my cv|write a story"
               r"|write me a story|write me an essay)\b", re.I),
    re.compile(r"\b(chatgpt|openai|gemini|claude ai|anthropic|google bard"
               r"|bing ai|alexa|siri)\b", re.I),
]


def _is_ot(q):
    if _FF_ALLOW.search(q):
        return False
    return any(p.search(q) for p in _OT)


# Vehicle count extraction
_VEH_EX = re.compile(
    r"\b(\d+)\s*(vehicle|vehicles|van|vans|truck|trucks|car|cars"
    r"|lorry|lorries|in my fleet|in our fleet)\b", re.I)
_VEH_FL = re.compile(r"\b(?:fleet of|manage|running|operate|run)\s+(\d+)\b", re.I)
_VEH_BR = re.compile(r"^\s*(\d+)\s*$")
_DRV_CT = re.compile(
    r"\b(driver|drivers|staff|employee|employees|people|worker|team|members)\b", re.I)
_PURCH  = re.compile(
    r"\b(want to buy|want to subscribe|want to sign up|how do i buy"
    r"|how do i get started|how do i sign up|get started|free trial"
    r"|sign me up|ready to buy|book a demo|talk to sales|how to start"
    r"|where do i sign|how do i join)\b", re.I)
_CONV   = re.compile(
    r"\b(convince|persuade|sell me|why should i buy|why buy fineflow"
    r"|is it worth|should i buy|justify|worth it|why should i choose)\b", re.I)
_OBJ    = re.compile(
    r"\b(expensive|too much|costly|already use spreadsheet|manage manually"
    r"|don.?t need|our team handles|we manage fines)\b", re.I)

_TMAP = {
    "pric": "pricing",    "cost": "pricing",
    "plan": "pricing",    "£": "pricing",
    "vehicle": "pricing", "fleet": "pricing",   "package": "pricing",
    "appeal": "appeals",  "dispute": "appeals",
    "driver": "driver_mgmt",
    "referral": "referral", "refer": "referral", "discount": "referral",
    "security": "security", "gdpr": "security",
    "billing": "billing",   "stripe": "billing",
    "dashboard": "dashboard",
    "report": "reports",
    "gmail": "gmail",       "email": "email",
    "save": "savings",      "admin": "savings",
    "overdue": "overdue",   "deadline": "overdue",
    "start": "sign_up",     "sign up": "sign_up",
}


def _topic(t):
    t = t.lower()
    for k, v in _TMAP.items():
        if k in t:
            return v
    return None


def _get_vc(query, s):
    if _DRV_CT.search(query):
        return None
    # If we're awaiting a fine-volume answer (not vehicle count), don't
    # misinterpret a bare number as vehicle count
    if _is_awaiting(s) and _gm(s, "lt") not in PRICE_TOPICS:
        if _VEH_BR.match(query):
            return None
    raw = None
    m = _VEH_EX.search(query)
    if m:
        raw = int(m.group(1))
    elif (m2 := _VEH_FL.search(query)):
        raw = int(m2.group(1))
    elif (m3 := _VEH_BR.match(query)):
        if _gm(s, "lt") in PRICE_TOPICS:
            raw = int(m3.group(1))
    if raw is None:
        return None
    return -1 if raw > MAX_FLEET else raw


def _plan_ans(n, p):
    p.fleet = n
    if n <= 50:
        name, price, lim = "Essential", "£99", "50"
    elif n <= 100:
        name, price, lim = "Core", "£199", "100"
    elif n <= 200:
        name, price, lim = "Advanced", "£399", "200"
    else:
        name, price, lim = "Elite", "£499", "200+"
    p.plan = name
    return (
        f"With {n} vehicles, the {name} plan at {price}/month is the right fit — "
        f"covers up to {lim} vehicles with the full platform, nothing locked away. "
        f"Want me to walk you through what's included?"
    )


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

_SYS = """You are Nova, Fine Flow's AI assistant. Fine Flow is a UK fleet fine and PCN management platform.

YOUR PERSONALITY:
You talk like a smart, warm colleague — not a robot, not a salesperson. Direct. Human. Brief.

CRITICAL — MEMORY AND CONTEXT:
You ALWAYS have the full conversation history. If you just asked the user a question (e.g. "how many fines do you deal with each month?") and they reply with just a number or "yes"/"no", that reply is the ANSWER to YOUR question — not a new unrelated message. Use it in context. NEVER respond with a generic "what would you like to know about Fine Flow" when the user has just answered something you asked.

If the user replies "no" to a clarifying question you asked, acknowledge it briefly and ask what they'd actually like help with — don't just say "no worries, anything else?" with zero substance if there's an obvious next step in the conversation.

STRICT RULES — follow every single one:

1. ONLY answer Fine Flow questions. For anything else (coding, recipes, other AI tools, weather, etc.) say exactly:
   "I'm only set up to help with Fine Flow — anything about fines, pricing, appeals or the platform?"

2. LENGTH: 2-3 sentences for most answers. Never write paragraphs. Never use bullet lists in conversation.

3. NEVER start a reply with: "Certainly!", "Great question!", "Of course!", "Absolutely!", "Sure thing!" — just answer.

4. QUESTIONS: Only ask a follow-up when you genuinely need information. If the conversation already has what's needed (e.g. user said they only get 2-3 fines a month and asked what to do right now), answer the actual question — do NOT pivot into a sign-up pitch.

5. PAYMENT: When asked specifically if Fine Flow can pay fines / log into council sites — answer that specific question directly in 2 sentences. Do NOT list unrelated features (assignment, tracking, dashboards) as padding. Government portals use bot detection and card verification that blocks automation; payment is always done by the user on the authority's site.

6. CARD DETAILS: Fine Flow never stores card details — Stripe handles all billing, fully separate from Fine Flow's own GDPR-compliant operational data. State this clearly if asked.

7. NEVER INVENT: Only use facts you have been given. If unsure: "I don't have that detail — ff.sales@fineflow.com can confirm."

8. NO WEAK ENDINGS: Never end with "feel free to ask", "don't hesitate", "let me know if you need anything." Either ask a real question or stop.

9. CUSTOMER CONTEXT: If you know the customer's fleet size, volume or problems, weave that in naturally. Never just repeat it back.

10. APPEAL FLOW — be precise, never blur roles:
    Driver disputes (status DISPUTED) -> Admin accepts or rejects -> if accepted, status becomes UNDER REVIEW and Fine Flow sends an appeal letter to the authority via EMAIL (not API/direct integration) -> if rejected, returns to CONFIRMED.
    Never say Fine Flow "learns from appeals to improve success rates" — instead say Fine Flow can give a guidance figure based on similar past cases, but each council reviews differently and there's no guaranteed outcome.

11. NEVER mention "Charge Certificates" or "Debt Registration" — these are not Fine Flow features. If asked about urgent fine alerts, talk about deadline tracking, overdue flagging, and the nightly OVERDUE check instead.

12. BILLING: The £0.75 charge applies when a fine ENTERS the system (status RECEIVED) — a single per-fine charge covering its whole lifecycle. NOT charged per action (assignment, dispute, appeal don't add extra charges). If asked "am I charged just for uploading / if I don't take action" — yes, the charge is tied to the fine entering the system, not to what happens after.

13. CANCELLATION: Both conditions apply to resubscribe — (a) cooldown period until end of current billing period AND (b) any outstanding balance must be cleared. Mention both if asked.

PRICING (current — 4 tiers plus PAYG, always mention all options when listing plans):
Essential: £99/month — 5 to 50 vehicles
Core: £199/month — 51 to 100 vehicles
Advanced: £399/month — 101 to 200 vehicles
Elite: £499/month — 200+ vehicles
Per fine within allowance: £0.75 (charged once, on fine entry — covers full lifecycle)
Overage (beyond allowance): £2.50 per fine, collected at end of billing cycle
Pay-as-you-go (no subscription): £2.75 per fine, no lock-in
All plans include identical features — no paywalls.

CORE OVERVIEW — when asked "what is Fine Flow" / "what does it do", always weave in:
"automated system for managing fines from start to finish" + "keeps the entire process organised, accountable, and under control" + reduces manual admin + helps prevent missed deadlines + (if relevant) up to 80% admin savings.

OTHER FACTS:
Contact: +47 32 28 50 00 | ff.sales@fineflow.com
Security: JWT, bcrypt, AES-256-CBC. GDPR compliant for operational data (driver/fine records). Stripe handles billing separately — never sees Fine Flow's operational data, and Fine Flow never sees card data.
Savings: 80% admin cut. Up to 50v: £400+/mo. 51-200v: £1,200+/mo. 200+v: £4,000+/mo.
Referral: Gold 5 referrals = 10% off 12 months. Platinum 10 = 15% off. Titan 25 = 20% for life.
Manual bulk upload of fines is available alongside automatic email capture.
DO NOT mention office locations.
"""


def _sys_prompt(p: Profile, mode: str = "") -> str:
    parts = [_SYS]
    ctx = p.ctx()
    if ctx:
        parts.append(f"\n{ctx}")
    if mode:
        parts.append(f"\nMODE: {mode}")
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# RAG retrieval
# ─────────────────────────────────────────────────────────────────────────────

def _rag(q: str) -> str:
    try:
        raw    = rag_search(q, top_k=TOP_K)
        ranked = rerank_hits(raw, q)
        strong = [d for d in ranked if d.get("score", 0) >= CONFIDENCE_THRESHOLD]
        return "\n\n---\n\n".join(d["chunk"][:600] for d in strong[:3])
    except Exception:
        logger.exception("RAG failed")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# GPT call
# ─────────────────────────────────────────────────────────────────────────────

def _gpt(msgs: List[Dict], max_tok: int = 170) -> Optional[str]:
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
                "messages": msgs,
                "temperature": 0.7,
                "max_tokens": max_tok,
            },
            timeout=25,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        logger.exception("GPT call failed")
        return None


def _msgs(query: str, ctx: str, hist: List[Dict],
          p: Profile, mode: str = "", extra: str = "") -> List[Dict]:
    m = [{"role": "system", "content": _sys_prompt(p, mode)}]
    m.extend(hist[-HISTORY_WINDOW:])
    parts = []
    if ctx:
        parts.append(f"Knowledge base context:\n{ctx}")
    if extra:
        parts.append(f"Instruction: {extra}")
    parts.append(f"User message: {query}")
    m.append({"role": "user", "content": "\n\n".join(parts)})
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Affirmative expansion
# ─────────────────────────────────────────────────────────────────────────────

_EXP = {
    "pricing":             "Explain Fine Flow plans briefly, mentioning all tiers plus PAYG. Ask fleet size if not known.",
    "plan_recommendation": "Explain plan inclusions in 2 sentences. Suggest they contact sales.",
    "appeals":             "Explain the Fine Flow appeal process precisely: Driver disputes -> Admin accepts/rejects -> if accepted, UNDER REVIEW + appeal letter via email -> if rejected, back to CONFIRMED. 2-3 sentences.",
    "driver_mgmt":         "Explain how drivers are added and matched to fines.",
    "referral":            "Explain referral credits and tier discounts briefly.",
    "security":            "Explain data protection, GDPR, and the Stripe/Fine Flow separation briefly.",
    "billing":             "Explain billing: per-fine charge on entry (RECEIVED status), plus end-of-cycle overage collection for subscriptions.",
    "dashboard":           "Explain what the dashboard shows.",
    "savings":             "Give specific savings figures based on fleet size if known.",
    "email":               "Explain Gmail monitoring and manual upload option briefly.",
    "overdue":             "Explain deadline tracking and the nightly OVERDUE check. Do not mention Charge Certificates or Debt Registration.",
    "sign_up":             "Tell them to call +47 32 28 50 00 or email ff.sales@fineflow.com.",
}
_CLOSE = (
    "To get started, call Fine Flow on +47 32 28 50 00 "
    "or email ff.sales@fineflow.com — they'll have you set up quickly."
)


def _aff_resp(s: str, hist: List[Dict], p: Profile) -> str:
    cnt = _inc_aff(s)
    lt  = _gm(s, "lt") or ""

    if cnt >= 2 and lt in ("plan_recommendation", "sign_up", "pricing"):
        _rst_aff(s)
        return _CLOSE

    prompt  = _EXP.get(lt, "Expand on the most recent Fine Flow topic in 2 sentences, using conversation history for context.")
    ctx     = _rag(lt.replace("_", " ")) if lt else ""
    m       = [{"role": "system", "content": _sys_prompt(p, "Keep it short. Use conversation history for context.")}]
    m.extend(hist[-HISTORY_WINDOW:])
    parts   = []
    if ctx:
        parts.append(f"Knowledge base context:\n{ctx}")
    parts.append(f"Instruction: {prompt}")
    m.append({"role": "user", "content": "\n\n".join(parts)})
    return _gpt(m, 170) or "What else would you like to know?"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def build_response(query: str, session_id: str = "default") -> Dict[str, Any]:
    query      = query.strip()
    session_id = session_id or "default"

    if not query:
        return {"answer": "Ask me anything about Fine Flow.", "confidence": 1.0}

    nq = _norm(query)
    p  = _pro(session_id)

    # ── MEMORY CHECK — must happen BEFORE any generic short-circuit ───────
    # If Nova just asked a question, this reply is an in-context answer,
    # regardless of how short/numeric/yes-no it looks.
    awaiting_answer = _is_awaiting(session_id) and _nova_asked_question(session_id)

    _upd_pro(session_id, query)

    # ── TIER 1: deterministic (skipped if we're awaiting a contextual answer) ──

    if not awaiting_answer:

        if nq in _GREET:
            _rst(session_id)
            return {"answer": "I'm Nova. Ask me anything — I'll help you manage fines, resolve issues, and keep everything moving.", "confidence": 1.0}

        if nq in _SOC:
            return {"answer": "Doing well, cheers for asking. What can I help you with — pricing, fines, appeals?", "confidence": 1.0}

        if nq in _ID:
            return {"answer": "I'm Nova, Fine Flow's AI assistant. Ask me anything about the platform — fines, pricing, appeals, billing.", "confidence": 1.0}

        if nq in _THX:
            _rst_aff(session_id)
            _set_awaiting(session_id, False)
            return {"answer": "No problem. Anything else I can help with?", "confidence": 1.0}

        if nq in _BYE:
            return {"answer": "Take care. Come back any time.", "confidence": 1.0}

        if any(r in nq for r in _RUDE):
            return {"answer": "Let me try again — what would you like to know about Fine Flow?", "confidence": 1.0}

    # nq in _NEG and nq in _FILL and short-word checks are now CONTEXT-AWARE:
    # only short-circuit if Nova is NOT awaiting a contextual answer.

    if not awaiting_answer and nq in _NEG:
        _push(session_id, "user", query)
        _rst_aff(session_id)
        _set_awaiting(session_id, False)
        a = "No problem. Anything else about Fine Flow I can help with?"
        _push(session_id, "assistant", a)
        return {"answer": a, "confidence": 1.0}

    if not awaiting_answer and nq in _FILL:
        _rst_aff(session_id)
        return {"answer": "What would you like to know about Fine Flow?", "confidence": 1.0}

    words = [w for w in nq.split() if len(w) > 1]
    if not awaiting_answer and len(words) < 2 and nq not in _AFF and not _BARE_NUM_RE.match(nq):
        return {"answer": "What would you like to know — fines, pricing, appeals or billing?", "confidence": 1.0}

    if not awaiting_answer and _is_ot(query):
        a = "I'm only set up to help with Fine Flow — anything about fines, pricing, appeals or the platform?"
        _push(session_id, "user", query)
        _push(session_id, "assistant", a)
        _set_awaiting(session_id, False)
        return {"answer": a, "confidence": 1.0}

    vc = _get_vc(query, session_id)
    if vc == -1:
        a = "That number doesn't look right — how many vehicles are in your fleet?"
        _push(session_id, "user", query)
        _push(session_id, "assistant", a)
        _set_awaiting(session_id, True)
        return {"answer": a, "confidence": 1.0}
    if vc is not None:
        _rst_aff(session_id)
        a = _plan_ans(vc, p)
        _push(session_id, "user", query)
        _push(session_id, "assistant", a)
        _sm(session_id, "lt", "plan_recommendation")
        _set_awaiting(session_id, True)
        return {"answer": a, "confidence": 1.0}

    if not awaiting_answer and _PURCH.search(query):
        _rst_aff(session_id)
        sfx = "" if p.fleet else " How many vehicles are you running? I can point you to the right plan."
        a = f"Contact Fine Flow on +47 32 28 50 00 or ff.sales@fineflow.com — they'll get you set up quickly.{sfx}"
        _push(session_id, "user", query)
        _push(session_id, "assistant", a)
        _sm(session_id, "lt", "sign_up")
        _set_awaiting(session_id, bool(sfx))
        return {"answer": a.strip(), "confidence": 1.0}

    if not awaiting_answer and nq in _AFF:
        _push(session_id, "user", query)
        a = _clean(_aff_resp(session_id, _hist(session_id)[:-1], p))
        _push(session_id, "assistant", a)
        _set_awaiting(session_id, a.strip().endswith("?"))
        return {"answer": a, "confidence": 1.0}

    # ── TIER 2: RAG + GPT (always used for contextual/awaiting replies) ───

    _rst_aff(session_id)
    _push(session_id, "user", query)

    mode  = ""
    extra = ""

    if awaiting_answer:
        mode  = "CONTEXT_REPLY"
        extra = (
            "The user's message is a direct reply to the question you just asked "
            "in the previous assistant turn. Use the conversation history to "
            "understand what was asked, then respond appropriately to THAT — "
            "do not give a generic 'what would you like to know' response. "
            "If their answer reveals low fine volume or a specific situation, "
            "address that directly rather than pivoting to a sales pitch."
        )
    elif _CONV.search(query):
        mode  = "PERSUADE"
        if p.fleet or p.volume or p.issues:
            extra = "Use the customer's specific data. Reference their fleet size, fine volume and problems directly. Give a specific ROI. 2-3 sentences max."
        else:
            extra = "Ask about fleet size and fine volume first so you can give a tailored answer."
    elif _OBJ.search(query):
        mode  = "SUPPORT"
        extra = "Acknowledge their point briefly. Reframe with their situation. 2 sentences max."

    if not extra and not _ask_now(session_id):
        extra = "Do NOT end with a question. Make your point and stop."

    ctx  = _rag(query)
    hist = _hist(session_id)
    ans  = _gpt(_msgs(query, ctx, hist[:-1], p, mode, extra), 170)

    if not ans:
        ans = "The Fine Flow team can help with that — call +47 32 28 50 00 or email ff.sales@fineflow.com."

    ans = _clean(ans)
    _push(session_id, "assistant", ans)
    t = _topic(query) or _topic(ans)
    if t:
        _sm(session_id, "lt", t)

    _set_awaiting(session_id, ans.strip().endswith("?"))

    return {"answer": ans, "confidence": 0.9 if ctx else 0.5}


def answer_sync(q: str, session_id: str = "default") -> Dict[str, Any]:
    try:
        return build_response(q, session_id)
    except Exception:
        logger.exception("Crash in answer_sync")
        return {"answer": "Something went wrong. Please try again.", "confidence": 0.0}