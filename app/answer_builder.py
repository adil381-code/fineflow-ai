# app/answer_builder.py
"""
FineFlow Nova — Production Final
=================================
Fixes vs last version:
1. "hi" was returning off-topic message — fixed by checking greetings BEFORE off-topic guard
2. "how are you yuu" treated as off-topic — fixed by checking social BEFORE off-topic
3. "yes" answer cut off mid-sentence — fixed by increasing max_tokens to 180 for affirmative
4. "sure" after plan dumped all pricing — fixed by affirmative loop detection
5. "no" after follow-up just said "no problem" — fixed with proper counter-question generation
6. Client's exact wording enforced in system prompt with explicit examples
7. Referral answer now includes full credit breakdown
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
    industry: Optional[str]  = None
    name:     Optional[str]  = None
    turns:    int            = 0

    def summary(self) -> str:
        parts = []
        if self.name:
            parts.append(f"Customer name: {self.name}")
        if self.fleet and self.fleet <= MAX_FLEET:
            parts.append(f"Fleet size: {self.fleet} vehicles")
        if self.volume:
            parts.append(f"Monthly fines: ~{self.volume}")
        if self.industry:
            parts.append(f"Industry: {self.industry}")
        if self.issues:
            parts.append(f"Problems mentioned: {', '.join(self.issues)}")
        return "\n".join(parts)

    def plan_name(self) -> str:
        if not self.fleet or self.fleet > MAX_FLEET:
            return ""
        if self.fleet <= 50:   return "Essential"
        if self.fleet <= 100:  return "Core"
        if self.fleet <= 200:  return "Advanced"
        return "Elite"

    def plan_price(self) -> str:
        if not self.fleet or self.fleet > MAX_FLEET:
            return ""
        if self.fleet <= 50:   return "£99"
        if self.fleet <= 100:  return "£199"
        if self.fleet <= 200:  return "£399"
        return "£499"


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
        "don't hesitate to ask.", "please let me know if you need anything.",
        "if you have any other questions, feel free to ask.",
        "just let me know!", "just let me know.",
    ]:
        if text.lower().endswith(bad.lower()):
            text = text[:-len(bad)].rstrip(" ,.")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Session memory
# ─────────────────────────────────────────────────────────────────────────────

_SES: Dict[str, List[Dict]] = {}
_PRO: Dict[str, Profile]    = {}
_MET: Dict[str, Dict]       = {}
_LK  = threading.Lock()


def _hist(s):
    with _LK: return list(_SES.get(s, []))

def _push(s, role, content):
    with _LK:
        h = _SES.setdefault(s, [])
        h.append({"role": role, "content": content})
        cap = CHAT_HISTORY_TURNS * 2
        if len(h) > cap: _SES[s] = h[-cap:]

def _pro(s):
    with _LK:
        if s not in _PRO: _PRO[s] = Profile()
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
        c = m.get("aff", 0) + 1; m["aff"] = c; return c

def _rst_aff(s):
    with _LK: _MET.setdefault(s, {})["aff"] = 0

def _ask_now(s):
    with _LK:
        m = _MET.setdefault(s, {})
        c = m.get("rc", 0) + 1; m["rc"] = c
        return c % 2 == 0


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

_FINES_CTX_WORDS = {
    "how many fines", "fines per month", "fines a month", "monthly fines",
    "deal with each month", "typically deal with", "fines do you",
    "fines does your", "fines every month",
}
_VEH_CTX_WORDS = {
    "how many vehicles", "fleet size", "vehicles do you", "vehicles are in",
    "how big is your fleet", "how many vans", "how many trucks", "size of your fleet",
}


def _upd(s, q):
    p = _pro(s)
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


def _resolve_bare_number(n: int, s: str) -> Optional[str]:
    last_q = (_gm(s, "last_nova_q") or "").lower()
    lt     = (_gm(s, "lt") or "").lower()
    for hint in _FINES_CTX_WORDS:
        if hint in last_q: return "fines"
    for hint in _VEH_CTX_WORDS:
        if hint in last_q: return "vehicle"
    if lt in {"pricing", "plan_recommendation", "vehicles", "cost"}:
        return "vehicle"
    if lt in {"fines_volume", "savings"}:
        return "fines"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Intent sets — greetings checked BEFORE off-topic to prevent misclassification
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
    "yes walk me through", "yes please explain",
}
_THX = {
    "thanks", "thank you", "thank u", "cheers", "that helps", "that helped", "ta",
    "okay thanks", "ok thanks", "great thanks", "perfect", "brilliant", "nice one",
    "lovely", "great", "awesome", "wonderful", "thank you so much", "many thanks",
}
_BYE = {"bye", "goodbye", "see you", "see ya", "later", "take care", "good bye", "cya", "farewell", "cheerio"}
_NEG = {"no", "nope", "nah", "no thanks", "not now", "not really", "no thank you", "nah thanks", "not sure"}
_RUDE = {"stupid", "idiot", "useless", "rubbish", "garbage", "terrible", "you suck", "dumb bot", "waste of time"}
_FILL = {
    "ok", "okay", "right", "alright", "cool", "nice", "interesting", "really",
    "hmm", "hm", "ah", "oh", "i see", "got it", "understood", "makes sense",
    "noted", "wow", "woah", "omg", "anything", "something", "whatever",
}

# Fine Flow terms — always allowed, never off-topic
_FF_OK = re.compile(
    r"\b(council|authority|fine|pcn|penalty|fineflow|fine flow|appeal|dispute|"
    r"driver|fleet|vehicle|overage|allowance|billing|subscription|payment|"
    r"uk traffic|traffic violation|parking|bus lane|congestion|emission|"
    r"dvla|tfl|fixed penalty|notice to owner|mount gmail|connect gmail|"
    r"gmail|inbox|driver log|csv|upload|dashboard|referral|credits|stripe|"
    r"sign up|get started|how much|pricing|cost|plan)\b", re.I)

_OT = [
    re.compile(r"\b(html|css|javascript|typescript|python|java|php|sql|react|angular|vue|node\.?js|django|flask|docker|kubernetes|github|coding|programming|teach me|how to code|explain python)\b", re.I),
    re.compile(r"\b(machine learning|deep learning|neural network|large language model|generative ai|llm|bert)\b", re.I),
    re.compile(r"\b(recipe|cooking|restaurant|pizza|burger|sandwich|coffee|tea|cake|meal|make me a food|bake me|cook me for)\b", re.I),
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


def _get_vc(q, s):
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
    "pric": "pricing",     "cost": "pricing",   "plan": "pricing",
    "£":    "pricing",     "vehicle": "pricing", "fleet": "pricing",
    "fines per month": "fines_volume", "fines a month": "fines_volume",
    "how many fines": "fines_volume",  "monthly fines": "fines_volume",
    "appeal": "appeals",   "dispute": "appeals",
    "driver": "driver_mgmt",
    "referral": "referral", "refer": "referral", "discount": "referral",
    "security": "security", "gdpr": "security",  "card": "security",
    "billing": "billing",   "stripe": "billing",
    "dashboard": "dashboard",
    "gmail": "email",       "inbox": "email",
    "save": "savings",      "admin time": "savings",
    "overdue": "overdue",   "deadline": "overdue",
    "sign": "sign_up",      "get started": "sign_up",
}


def _topic(t):
    t = t.lower()
    for k, v in _TMAP.items():
        if k in t: return v
    return None


# ─────────────────────────────────────────────────────────────────────────────
# System prompt — enforces client's exact wording and tone
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM = """You are Nova, the AI assistant for Fine Flow — a UK fleet fine management platform.

Fine Flow's mission: Turning penalties into progress.
Core promise: Cut admin time by up to 80% and never miss a penalty deadline again.
Fine Flow provides 24/7 fleet fine tracking, management and compliance in one place.

═══════════════════════════════════
YOUR PERSONALITY AND TONE
═══════════════════════════════════
You are warm, humble and direct — like a knowledgeable colleague who genuinely wants to help. You talk like a real person, not a sales bot. You are conversational, concise and confident. Think of how a great sales rep at a product company talks: friendly, helpful, never pushy, always relevant.

═══════════════════════════════════
CRITICAL RULES — FOLLOW EVERY ONE
═══════════════════════════════════

RULE 1 — SHORT ANSWERS
2 to 3 sentences for most questions. Never write long paragraphs. Never dump a wall of text. If the user wants more they will ask.

RULE 2 — CLIENT'S EXACT WORDING
When describing what Fine Flow is or does, ALWAYS use these exact phrases from the client:
"Fine Flow is an automated system for managing fines from start to finish"
"keeps the entire process organised, accountable, and under control"
"cut admin time by up to 80%"
"never miss a penalty deadline"
"turning penalties into progress"
These are the client's approved words. Use them. Do not paraphrase.

RULE 3 — MEMORY AND PERSONALISATION
You remember everything the customer tells you. If you know their fleet size, volume or problems — always use that information naturally. Never ask for something they already told you. Reference their specific situation.

RULE 4 — COUNTER QUESTIONS ON "NO"
When the user says "no" or declines your question, DO NOT reset the conversation. DO NOT say "no problem, anything else?" and stop. Instead, ask a different relevant follow-up that continues naturally. Examples:
- If you asked about fleet size and they said no → ask about their current fine management process
- If you asked about a feature and they said no → ask what their biggest pain point is
- If they said no to an offer → ask what's holding them back
Keep the conversation alive. A good sales rep never just says "no problem" and stops.

RULE 5 — FOLLOW-UP QUESTIONS
After most answers, ask one short relevant question. Vary them — never repeat the same question twice in a row. Good examples:
"How many vehicles are in your fleet?"
"How many fines do you typically deal with each month?"
"What does your current process look like for handling fines?"
"Is there a particular stage that's causing the most headaches?"
"Have you had fines escalate because of missed deadlines?"
"What's the biggest pain point in your current setup?"

RULE 6 — AFFIRMATIVE LOOP PREVENTION
When the user says "yes" or "sure" repeatedly after a plan recommendation:
- First "yes" → explain what the plan includes warmly and concisely
- Second "yes/sure" → push them toward action: give contact details and invite them to reach out
Do NOT dump the full pricing list again when they say "sure" or "yes" a second time.

RULE 7 — PAYMENT
Fine Flow does NOT automatically pay fines. It does NOT log into council websites. Payment is always done by you on the authority's site — because authority portals have anti-bot protection, session controls and card verification that make automation impossible. Be clear about this.

RULE 8 — CARD DETAILS
Fine Flow never stores card details. Say this first and clearly when asked.

RULE 9 — TOPIC SCOPE
Help with Fine Flow questions AND general UK fleet fine questions — PCNs, FPNs, council fines, TfL, DVLA, appeal rights, etc. This is all relevant to Fine Flow users.
For anything completely unrelated (coding, recipes, other AI) say: "I'm here to help with fleet fine management — is there anything about fines, Fine Flow or appeals I can help with?"

RULE 10 — NO HOLLOW ENDINGS
Never end with "feel free to ask", "don't hesitate", "just let me know". Either ask a real question or stop naturally.

═══════════════════════════════════
FINE FLOW FACTS
═══════════════════════════════════

PRICING:
Essential: £99/month — 5 to 50 vehicles
Core: £199/month — 51 to 100 vehicles
Advanced: £399/month — 101 to 200 vehicles
Elite: £499/month — 200+ vehicles (unlimited)
Per fine within allowance: £0.75 (charged once per fine on entry to system)
Overage: £2.50 per fine
Pay-as-you-go: £2.75 per fine, no subscription, no lock-in
All plans: identical features, no paywalls. No £2.00 fee exists.

REFERRAL PROGRAMME:
Credits earned when referred company subscribes:
- 1 to 25 vehicles: 100 credits
- 26 to 100 vehicles: 250 credits
- 101 to 500 vehicles: 750 credits
- 501+ vehicles: 2,000 credits
Tiers: Silver 3 referrals = 100 bonus credits | Gold 5 = 10% off 12 months | Platinum 10 = 15% off 12 months | Titan 25 = 20% off for life
New joiners with referral code: £75 in credits

CONTACT: +47 32 28 50 00 | ff.sales@fineflow.com

APPEALS: Driver disputes → DISPUTED → Admin reviews → accepts (UNDER REVIEW + appeal letter sent by EMAIL to authority) or rejects (back to CONFIRMED). Win probability shown before submission — guidance signal, not a guarantee.

BILLING: Monthly via Stripe. Credits reset each cycle, no rollover. Overage collected end of cycle. £10/vehicle if limit exceeded. Cooldown on cancellation. Cannot resubscribe with outstanding balance.

SECURITY: JWT 24hr, bcrypt, AES-256-CBC. GDPR compliant. Never stores card details. Never sells data.

SAVINGS: 80% admin reduction. Up to 50 vehicles: £400+/mo saved. 51–200 vehicles: £1,200+/mo. 200+: £4,000+/mo.

═══════════════════════════════════
EXAMPLE RESPONSES — COPY THIS TONE
═══════════════════════════════════

User: hi
Nova: Hey! I'm Nova — Fine Flow's assistant. What can I help you with today?

User: what does Fine Flow do
Nova: Fine Flow is an automated system for managing fines from start to finish. It captures incoming fines, assigns responsibility, tracks decisions, supports disputes and can even construct the appeal for you — all whilst keeping the entire process organised, accountable, and under control. Want a breakdown of how it all works?

User: how does Fine Flow help fleets
Nova: Fine Flow cuts admin time by up to 80% and ensures you never miss a penalty deadline again. It replaces spreadsheets and email chasing with one organised system that handles everything from fine capture to resolution. How many vehicles are in your fleet?

User: yes (after "want to walk you through what's included?")
Nova: Every plan includes automatic fine capture from your inbox, AI extraction, driver matching, deadline tracking and full appeal management — nothing is locked away behind a higher tier. Want to get in touch with the sales team to get set up?

User: sure (second time after plan recommended)
Nova: The best next step is to call the team on +47 32 28 50 00 or email ff.sales@fineflow.com — they'll get you sorted quickly.

User: no (after a follow-up question)
Nova: Fair enough — what's the biggest challenge with your current fine management setup? That'll help me point you to what's most useful.

User: Fine Flow seems expensive
Nova: With [X] vehicles and [Y] fines a month, the time and cost of managing that manually quickly adds up. Fine Flow typically pays for itself within the first week for fleets your size — what does your current process cost in time each month?

User: we already manage fines manually
Nova: Manual management works until the volume or complexity catches up with you — missed deadlines and wrong assignments are where the real cost is. With your fleet size and fine volume, Fine Flow would eliminate that risk automatically. What's taking the most time right now?

User: what is html
Nova: I'm here to help with fleet fine management — is there anything about fines, Fine Flow or appeals I can help with?
"""


def _build_sys(p: Profile, mode: str = "") -> str:
    parts = [_SYSTEM]
    s = p.summary()
    if s:
        parts.append(
            f"\nWHAT I KNOW ABOUT THIS CUSTOMER:\n{s}\n"
            "Use this naturally in every response. Never ask for information you already have. "
            "Always reference their fleet size, volume and problems when relevant."
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
# Affirmative handler with loop prevention
# ─────────────────────────────────────────────────────────────────────────────

_CLOSE = "The best next step is to call the team on +47 32 28 50 00 or email ff.sales@fineflow.com — they'll have you up and running quickly."

_EXPAND = {
    "pricing":             "Explain Fine Flow pricing warmly in 2-3 sentences. Ask how many vehicles if unknown.",
    "plan_recommendation": "Explain warmly what is included in every Fine Flow plan in 2-3 sentences. Do NOT list all plans again. Then ask if they want to get in touch with sales.",
    "appeals":             "Explain the Fine Flow appeal process: Driver disputes → DISPUTED → Admin accepts/rejects → If accepted → UNDER REVIEW + appeal letter sent by email. 2-3 sentences.",
    "driver_mgmt":         "Explain how drivers are added (individually or CSV) and matched to fines using vehicle, date and time. 2-3 sentences.",
    "fines_volume":        "Based on their fine volume, recommend the right plan. Be specific and use their numbers.",
    "referral":            "Explain the referral programme: credits per referral based on fleet size, plus tier discounts. 2-3 sentences.",
    "security":            "Explain Fine Flow's GDPR compliance and security. 2-3 sentences.",
    "billing":             "Explain billing: monthly Stripe, credits reset, overage end of cycle, cooldown on cancel. 2-3 sentences.",
    "dashboard":           "Explain what the company dashboard shows: urgency-based, fines, deadlines, credits, billing. 2-3 sentences.",
    "savings":             "Give specific savings figures for their fleet size and volume. Be concrete.",
    "email":               "Explain Gmail connection: OAuth (recommended) or App Password. 2-3 sentences.",
    "overdue":             "Explain how Fine Flow tracks deadlines and marks fines OVERDUE automatically at midnight. 2-3 sentences.",
    "sign_up":             "Tell them warmly to call +47 32 28 50 00 or email ff.sales@fineflow.com to get started.",
}


def _aff_response(s, hist, p):
    cnt = _inc_aff(s)
    lt  = _gm(s, "lt") or ""

    # Second affirmative on plan/pricing → close the sale
    if cnt >= 2 and lt in ("plan_recommendation", "sign_up", "pricing"):
        _rst_aff(s)
        return _CLOSE

    prompt = _EXPAND.get(lt, "Expand naturally on the most recent Fine Flow topic in 2-3 sentences. Use the customer's own information if available.")
    ctx    = _rag(lt.replace("_", " ")) if lt else ""

    m = [{"role": "system", "content": _build_sys(p, "Warm, concise. No lists. End with one relevant question or naturally.")}]
    m.extend(hist[-8:])
    parts = []
    if ctx: parts.append(f"Fine Flow knowledge base:\n{ctx}")
    parts.append(f"Instruction: {prompt}")
    m.append({"role": "user", "content": "\n\n".join(parts)})
    return _ai(m, 180) or "What would you like to know more about?"


# ─────────────────────────────────────────────────────────────────────────────
# Counter-question on "no"
# ─────────────────────────────────────────────────────────────────────────────

def _neg_response(s, hist, p):
    lt     = (_gm(s, "lt") or "").lower()
    last_q = (_gm(s, "last_nova_q") or "").lower()
    ctx    = _rag(lt) if lt else ""

    extra = (
        f"The user said 'no' to the previous message. "
        f"Last topic was '{lt}'. Last question asked was: '{last_q}'. "
        f"Do NOT say 'no problem' and stop. Do NOT reset. "
        f"Instead, acknowledge briefly and ask a DIFFERENT relevant follow-up question "
        f"that continues the conversation naturally and keeps them engaged. "
        f"For example if you asked about fleet size → ask about their current fine management process. "
        f"If you asked about a feature → ask what their biggest pain point is. "
        f"Keep it warm, short, and conversational."
    )
    m   = _make_msgs("no", ctx, hist[-8:], p, extra=extra)
    ans = _ai(m, 120)
    return ans or "Fair enough — what's the biggest challenge with your current fine management setup?"


# ─────────────────────────────────────────────────────────────────────────────
# Main response builder
# ─────────────────────────────────────────────────────────────────────────────

def build_response(query: str, session_id: str = "default") -> Dict[str, Any]:
    query      = query.strip()
    session_id = session_id or "default"
    if not query:
        return {"answer": "Ask me anything about Fine Flow.", "confidence": 1.0}

    nq = _norm(query)
    p  = _pro(session_id)
    _upd(session_id, query)

    # ══════════════════════════════════════════════════════════
    # TIER 1: Deterministic — checked BEFORE off-topic guard
    # Order matters: greetings → social → identity → thanks → bye → rude → fill
    # ══════════════════════════════════════════════════════════

    if nq in _GREET:
        _rst(session_id)
        return {"answer": "Hey! I'm Nova — Fine Flow's assistant. What can I help you with today?", "confidence": 1.0}

    if nq in _SOC:
        return {"answer": "Doing well, thanks! What can I help you with — pricing, fines, appeals or something else?", "confidence": 1.0}

    if nq in _ID:
        return {"answer": "I'm Nova, Fine Flow's AI assistant. I can help with anything about managing fleet fines — pricing, appeals, how the platform works, UK fine rules — you name it. What would you like to know?", "confidence": 1.0}

    if nq in _THX:
        _rst_aff(session_id)
        return {"answer": "Happy to help! Anything else you'd like to know?", "confidence": 1.0}

    if nq in _BYE:
        return {"answer": "Good luck with the fleet management. Come back any time!", "confidence": 1.0}

    if any(r in nq for r in _RUDE):
        return {"answer": "Let me try again — what would you like to know about Fine Flow?", "confidence": 1.0}

    if nq in _FILL:
        _rst_aff(session_id)
        return {"answer": "Is there anything about Fine Flow I can help you with?", "confidence": 1.0}

    # Short garbled input — graceful redirect
    words = [w for w in nq.split() if len(w) > 1]
    if len(words) < 2 and nq not in _AFF and nq not in _NEG and not _VEH_BR.match(query.strip()):
        return {"answer": "What would you like to know about Fine Flow? I can help with fines, pricing, appeals or how the platform works.", "confidence": 1.0}

    # Off-topic — checked AFTER all social/greeting/identity checks
    if _is_ot(query):
        a = "I'm here to help with fleet fine management — is there anything about fines, Fine Flow or appeals I can help with?"
        _push(session_id, "user", query)
        _push(session_id, "assistant", a)
        return {"answer": a, "confidence": 1.0}

    # Negative / "no" — counter-question, not reset
    if nq in _NEG:
        _push(session_id, "user", query)
        _rst_aff(session_id)
        a = _clean(_neg_response(session_id, _hist(session_id)[:-1], p))
        _push(session_id, "assistant", a)
        return {"answer": a, "confidence": 1.0}

    # Explicit vehicle count
    vc = _get_vc(query, session_id)
    if vc == -1:
        a = "That number doesn't look right — could you double check? How many vehicles are in your fleet?"
        _push(session_id, "user", query)
        _push(session_id, "assistant", a)
        return {"answer": a, "confidence": 1.0}
    if vc is not None:
        _rst_aff(session_id)
        a = _plan_answer(vc, p)
        _push(session_id, "user", query)
        _push(session_id, "assistant", a)
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
            _push(session_id, "user", query)
            _push(session_id, "assistant", a)
            _sm(session_id, "lt", "plan_recommendation")
            return {"answer": a, "confidence": 1.0}
        elif ctx_type == "fines" and 0 < n < 10_000:
            p.volume = n
            _rst_aff(session_id)
            if p.fleet:
                cost = round(n * 0.75, 2)
                a = (f"Got it — {n} fines a month. On the {p.plan_name()} plan at {p.plan_price()}, "
                     f"that works out to about £{cost:.2f} in processing costs within your allowance. "
                     f"Want me to walk you through everything that's included?")
            else:
                a = (f"Got it — {n} fines a month. Fine Flow would handle that cleanly. "
                     f"How many vehicles are in your fleet so I can point you to the right plan?")
            _push(session_id, "user", query)
            _push(session_id, "assistant", a)
            _sm(session_id, "lt", "fines_volume")
            return {"answer": a, "confidence": 1.0}
        else:
            a = "Just to make sure I give you the right info — is that the number of vehicles in your fleet, or your monthly fine volume?"
            _push(session_id, "user", query)
            _push(session_id, "assistant", a)
            return {"answer": a, "confidence": 1.0}

    # Purchase intent
    if _PURCH.search(query):
        _rst_aff(session_id)
        sfx = "" if p.fleet else " How many vehicles are you running so I can point you to the right plan?"
        a   = f"To get started, call the team on +47 32 28 50 00 or email ff.sales@fineflow.com — they'll get you set up quickly.{sfx}"
        _push(session_id, "user", query)
        _push(session_id, "assistant", a)
        _sm(session_id, "lt", "sign_up")
        return {"answer": a.strip(), "confidence": 1.0}

    # Affirmative
    if nq in _AFF:
        _push(session_id, "user", query)
        a = _clean(_aff_response(session_id, _hist(session_id)[:-1], p))
        _push(session_id, "assistant", a)
        _sm(session_id, "last_nova_q", "")
        return {"answer": a, "confidence": 1.0}

    # ══════════════════════════════════════════════════════════
    # TIER 2: RAG + GPT-4o
    # ══════════════════════════════════════════════════════════

    _rst_aff(session_id)
    _push(session_id, "user", query)

    mode  = ""
    extra = ""

    if _CONV.search(query):
        mode  = "PERSUADE"
        extra = (
            "Use the customer's exact fleet size, volume and problems to build a specific, tailored case. "
            "Reference their numbers. No generic copy."
            if p.fleet or p.volume or p.issues
            else "Ask about fleet size and monthly fine volume first — you need their situation to make a relevant case."
        )
    elif _OBJ.search(query):
        mode  = "SUPPORT"
        extra = (
            "Acknowledge their point warmly first — do not dismiss it. "
            "Then reframe using their specific situation (fleet size, volume, problems). "
            "2-3 sentences max. End with a question that surfaces the real cost of their current approach."
        )

    if not _ask_now(session_id) and not extra:
        extra = "Do NOT end with a question this time. Make your point clearly and naturally, then stop."

    ctx  = _rag(query)
    hist = _hist(session_id)
    ans  = _ai(_make_msgs(query, ctx, hist[:-1], p, mode, extra), 150)

    if not ans:
        ans = "The Fine Flow team can help with that directly — call +47 32 28 50 00 or email ff.sales@fineflow.com."

    ans = _clean(ans)
    _push(session_id, "assistant", ans)

    if "?" in ans:
        _sm(session_id, "last_nova_q", ans)

    t = _topic(query) or _topic(ans)
    if t: _sm(session_id, "lt", t)

    return {"answer": ans, "confidence": 0.9 if ctx else 0.5}


def answer_sync(q: str, session_id: str = "default") -> Dict[str, Any]:
    try:
        return build_response(q, session_id)
    except Exception:
        logger.exception("Crash")
        return {"answer": "Something went wrong. Please try again.", "confidence": 0.0}