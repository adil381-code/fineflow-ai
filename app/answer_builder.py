# app/answer_builder.py
"""
FineFlow Nova — Production Final v18
ARCHITECTURE:
  Tier 1: Deterministic (greetings, identity, vehicle count, off-topic) — instant, no AI
  Tier 2: KB match → returns client's EXACT approved answer verbatim
  Tier 3: RAG + GPT-4o for anything not in KB

KEY BEHAVIOURS:
  - KB answers returned word for word — consistent every time
  - Follow-up questions asked naturally (from KB entry or GPT)
  - Customer profile (fleet size, volume, issues) injected into every GPT call
  - Warm human tone — not robotic
  - Question limiting — not every response ends with a question
  - Mode switching — diagnose / persuade / support
"""

import json
import re
import threading
from dataclasses import dataclass, field
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

MAX_FLEET_SIZE = 50_000

# ─────────────────────────────────────────────────────────────────────────────
# KB Loader
# ─────────────────────────────────────────────────────────────────────────────

KB_PATH = Path(__file__).parent.parent / "data" / "fineflow_kb.json"
_KB: List[Dict[str, Any]] = []


def _load_kb() -> None:
    global _KB
    if not KB_PATH.exists():
        logger.warning("KB not found at %s", KB_PATH)
        return
    try:
        with open(KB_PATH, "r", encoding="utf-8") as f:
            _KB = json.load(f)
        logger.info("Loaded %d KB entries", len(_KB))
    except Exception:
        logger.exception("Failed to load KB")


_load_kb()

_STOPWORDS = {
    "a","an","the","is","it","in","on","of","to","do","my","me","you",
    "your","i","we","are","was","be","will","can","has","have","had",
    "not","and","or","but","if","this","that","with","for","at","by",
    "from","as","what","how","why","who","when","where","which","there",
    "here","does","did","so","up","about","just","more","also","would",
    "could","should","any","all","get","its","into","out","no",
}


def _tokens(text: str) -> set:
    t = re.sub(r"[^\w\s]", " ", text.lower())
    return {w for w in t.split() if w not in _STOPWORDS and len(w) > 2}


def _kb_match(query: str) -> Optional[Dict[str, Any]]:
    nq      = re.sub(r"[^\w\s]", " ", query.lower()).strip()
    q_tok   = _tokens(query)
    if not q_tok:
        return None

    best: Optional[Dict[str, Any]] = None
    best_score = 0.0

    for entry in _KB:
        keywords = entry.get("keywords", [])
        # Exact keyword phrase match — highest priority
        for kw in keywords:
            kw_norm = re.sub(r"[^\w\s]", " ", kw.lower()).strip()
            if kw_norm in nq or nq in kw_norm:
                return entry
        # Token overlap
        s_tok  = _tokens(" ".join(keywords))
        if not s_tok:
            continue
        overlap = len(q_tok & s_tok)
        if overlap == 0:
            continue
        score = overlap / max(len(q_tok), 1) + 0.1 * len(q_tok & s_tok)
        if score > best_score:
            best_score = score
            best = entry

    return best if best_score >= 0.55 else None


# ─────────────────────────────────────────────────────────────────────────────
# Customer profile
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CustomerProfile:
    fleet_size:  Optional[int] = None
    fine_volume: Optional[int] = None
    issues:      List[str]     = field(default_factory=list)
    plan:        Optional[str] = None
    industry:    Optional[str] = None
    turn_count:  int           = 0

    def context(self) -> str:
        lines = []
        if self.fleet_size and self.fleet_size <= MAX_FLEET_SIZE:
            lines.append(f"Fleet size: {self.fleet_size} vehicles")
        if self.fine_volume:
            lines.append(f"Monthly fines: ~{self.fine_volume}")
        if self.industry:
            lines.append(f"Industry: {self.industry}")
        if self.issues:
            order = ["missed deadlines","missed appeal deadlines","driver disputes","manual admin","using spreadsheets"]
            lines.append("Problems: " + ", ".join(
                sorted(self.issues, key=lambda x: order.index(x) if x in order else 99)
            ))
        return ("What I know about this customer:\n" + "\n".join(lines)) if lines else ""

    def plan_name(self) -> str:
        if not self.fleet_size or self.fleet_size > MAX_FLEET_SIZE:
            return ""
        if self.fleet_size <= 50:   return "Essential (£99/month)"
        if self.fleet_size <= 100:  return "Core (£199/month)"
        if self.fleet_size <= 200:  return "Advanced (£399/month)"
        return "Elite (£499/month)"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    if not text:
        return ""
    for pat, rep in [
        (r"\*\*(.*?)\*\*", r"\1"), (r"\*(.*?)\*", r"\1"), (r"_(.*?)_", r"\1"),
    ]:
        text = re.sub(pat, rep, text)
    text = text.replace("→","to").replace("->","to").replace("±","plus or minus").replace("`","")
    for filler in [
        "feel free to ask!", "feel free to ask.",
        "please let me know if you need further assistance.",
        "don't hesitate to ask.", "don't hesitate to reach out.",
        "if you have any other questions, feel free to ask.",
    ]:
        if text.lower().endswith(filler.lower()):
            text = text[:-len(filler)].rstrip(" ,.")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Session memory
# ─────────────────────────────────────────────────────────────────────────────

_SESSION:  Dict[str, List[Dict[str, str]]] = {}
_PROFILES: Dict[str, CustomerProfile]      = {}
_META:     Dict[str, Dict[str, Any]]       = {}
_LOCK = threading.Lock()

PRICING_TOPICS = {"pricing","plan_recommendation","vehicles","cost","billing"}


def _hist(sid):
    with _LOCK: return list(_SESSION.get(sid, []))

def _push(sid, role, content):
    with _LOCK:
        h = _SESSION.setdefault(sid, [])
        h.append({"role": role, "content": content})
        cap = CHAT_HISTORY_TURNS * 2
        if len(h) > cap: _SESSION[sid] = h[-cap:]

def _profile(sid):
    with _LOCK:
        if sid not in _PROFILES: _PROFILES[sid] = CustomerProfile()
        return _PROFILES[sid]

def _smeta(sid, key, val):
    with _LOCK: _META.setdefault(sid, {})[key] = val

def _gmeta(sid, key):
    with _LOCK: return _META.get(sid, {}).get(key)

def _reset(sid):
    with _LOCK: _META[sid] = {}; _PROFILES[sid] = CustomerProfile()

def _inc_aff(sid):
    with _LOCK:
        m = _META.setdefault(sid, {})
        c = m.get("aff", 0) + 1; m["aff"] = c; return c

def _reset_aff(sid):
    with _LOCK: _META.setdefault(sid, {})["aff"] = 0

def _ask_this_turn(sid):
    """Ask a question on every other turn to avoid interrogation fatigue."""
    with _LOCK:
        m = _META.setdefault(sid, {})
        c = m.get("rc", 0) + 1; m["rc"] = c
        return c % 2 == 0


# ─────────────────────────────────────────────────────────────────────────────
# Profile extraction
# ─────────────────────────────────────────────────────────────────────────────

_FINES_RE    = re.compile(r"\b(\d+)\s*(?:fines?|pcns?|penalties|violations?|tickets?)(?:\s*(?:per|a|each|every)\s*(?:month|monthly|week))?\b", re.I)
_INDUSTRY_RE = re.compile(r"\b(logistics|delivery|courier|haulage|transport|taxi|minicab|bus|coach|construction|utilities)\b", re.I)
_ISSUE_RE    = [
    (re.compile(r"\b(miss(?:ed?|ing)?\s+(?:deadlines?|appeals?|due\s*dates?))\b", re.I), "missed deadlines"),
    (re.compile(r"\b(miss(?:ed?|ing)?\s+appeal\s+deadline)\b", re.I),                   "missed appeal deadlines"),
    (re.compile(r"\b(drivers?\s+(?:dispute|deny|ignor|avoid))\b", re.I),                "driver disputes"),
    (re.compile(r"\b(manual(?:ly)?\s+(?:track|manage|process|handl))\b", re.I),         "manual admin"),
    (re.compile(r"\b(spreadsheet)\b", re.I),                                             "using spreadsheets"),
]


def _update_profile(sid, query):
    p = _profile(sid); p.turn_count += 1
    m = _FINES_RE.search(query)
    if m and p.fine_volume is None:
        v = int(m.group(1))
        if v < 10_000: p.fine_volume = v
    m = _INDUSTRY_RE.search(query)
    if m: p.industry = m.group(1).lower()
    for pat, label in _ISSUE_RE:
        if pat.search(query) and label not in p.issues: p.issues.append(label)


# ─────────────────────────────────────────────────────────────────────────────
# Intent sets
# ─────────────────────────────────────────────────────────────────────────────

_GREET = {"hi","hello","hey","hiya","howdy","yo","sup","good morning","good afternoon","good evening","morning","afternoon","evening","hi there","hey there","hello there","hi nova","hey nova","hello nova"}
_SOCIAL = {"how are you","how are you doing","how r u","how r you","how are u","hows it going","how is it going","whats up","what s up","you ok","you good","how do you do","you alright","alright mate"}
_IDENTITY = {"who are you","who r you","who r u","who youre","who you re","who is nova","who is this","who is there","whos there","who s there","anyone there","is anyone there","what are you","what is nova","are you a bot","are you human","are you ai","are you a robot","are you male or female","you male or female","who the hell are you","whats your name","what is your name","introduce yourself","who am i talking to","knock knock"}
_AFF = {"yes","yeah","yep","yup","ya","ye","sure","ok sure","okay sure","go ahead","go on","yes go ahead","yes please","yes sure","yes of course","of course","absolutely","definitely","do it","tell me more","more","explain","explain more","yes explain","yes explain it","go for it","sounds good","continue","carry on","keep going","please do","i would","please explain","show me","walk me through it","yes elite","yes core","yes essential","yes advanced","for sure","yes for sure","sure thing"}
_THANKS = {"thanks","thank you","thank u","cheers","that helps","that helped","ta","ty","okay thanks","ok thanks","great thanks","perfect","brilliant","nice one","lovely","great","awesome","wonderful","thank you so much","many thanks","much appreciated"}
_BYE   = {"bye","goodbye","see you","see ya","later","take care","good bye","cya","ttyl","farewell","cheerio"}
_NEG   = {"no","nope","nah","no thanks","not now","skip","never mind","nevermind","no need","not really","no thank you","nah thanks"}
_RUDE  = {"you dumb","you are dumb","ur dumb","stupid","idiot","useless","rubbish","garbage","terrible","you suck","this is rubbish","dumb bot","you re useless","waste of time"}
_FILL  = {"ok","okay","right","alright","cool","nice","interesting","really","seriously","hmm","hm","ah","oh","i see","got it","understood","makes sense","noted","wow","waow","woah","omg","anything","something","whatever"}

_FF_ALLOW = re.compile(r"\b(council|authority|authorities|payment portal|log into|login to|pay.*fine|fine.*pay|appeal.*fine|fine.*appeal|pcn|penalty charge|parking fine|fleet fine|fineflow|fine flow|streamline|overage|allowance|subscription|driver log|vehicle log|billing|credit|cancel|reassign|dispute)\b", re.I)
_OT_PATS  = [
    re.compile(r"\b(html|css|javascript|typescript|python|java|php|sql|react|angular|vue|node\.?js|django|flask|docker|kubernetes|github|devops|backend|frontend|coding|programming)\b", re.I),
    re.compile(r"\b(machine learning|deep learning|neural network|large language model|generative ai|train a model|llm|bert)\b", re.I),
    re.compile(r"\b(recipe|cooking|restaurant|pizza|burger|sandwich|coffee|tea|cake|meal|bake|order food|make me a|bake me|cook me)\b", re.I),
    re.compile(r"\b(movie|film|song|lyrics|music|football match|cricket match|weather forecast|todays news|politics|history lesson|capital city|who invented|tell me a joke|write me a poem)\b", re.I),
    re.compile(r"\b(write an essay|translate this|proofread|write my cv|write a story)\b", re.I),
    re.compile(r"\b(chatgpt|openai|gemini|claude ai|anthropic|google bard|bing ai|alexa|siri)\b", re.I),
]


def _is_ot(q):
    if _FF_ALLOW.search(q): return False
    return any(p.search(q) for p in _OT_PATS)


_VEH_EX = re.compile(r"\b(\d+)\s*(vehicle|vehicles|van|vans|truck|trucks|car|cars|lorry|lorries|in my fleet|in our fleet)\b", re.I)
_VEH_FL = re.compile(r"\b(?:fleet of|manage|running|operate|run)\s+(\d+)\b", re.I)
_VEH_BR = re.compile(r"^\s*(\d+)\s*$")
_DRV_CT = re.compile(r"\b(driver|drivers|staff|employee|employees|people|worker|team|members)\b", re.I)
_PURCH  = re.compile(r"\b(want to buy|want to subscribe|want to sign up|how do i buy|how do i get started|how do i sign up|get started|free trial|sign me up|ready to buy|book a demo|talk to sales|how to start|where do i sign|how do i join)\b", re.I)
_CONV   = re.compile(r"\b(convince|persuade|sell me|why should i|why buy|is it worth|should i buy|justify|worth it)\b", re.I)
_OBJ    = re.compile(r"\b(expensive|too much|costly|already use|spreadsheet|manual|don.?t need|our team handles|we manage)\b", re.I)
_PROB   = re.compile(r"\b(fine|pcn|penalty|violation|notice|ticket)\b.{0,40}\b(issued|received|got|have a|appealing|disputing|today|yesterday|last week)\b|\b(received a|got a|have a)\b.{0,20}\b(fine|pcn|penalty|violation)\b", re.I)

_TOPIC_MAP = {
    "pric":"pricing","cost":"pricing","plan":"pricing","£":"pricing","vehicle":"pricing","fleet":"pricing",
    "appeal":"appeals","dispute":"appeals","driver":"driver_mgmt",
    "referral":"referral","refer":"referral","discount":"referral","offer":"referral",
    "security":"security","gdpr":"security","billing":"billing","stripe":"billing",
    "dashboard":"dashboard","report":"reports","gmail":"gmail","email":"email",
    "save":"savings","admin":"savings","overdue":"overdue","deadline":"overdue",
    "start":"sign_up",
}


def _topic(text):
    t = text.lower()
    for k, v in _TOPIC_MAP.items():
        if k in t: return v
    return None


def _extract_vc(query, sid):
    if _DRV_CT.search(query): return None
    raw = None
    m = _VEH_EX.search(query)
    if m: raw = int(m.group(1))
    elif (m2 := _VEH_FL.search(query)): raw = int(m2.group(1))
    elif (m3 := _VEH_BR.match(query)):
        if _gmeta(sid,"last_topic") in PRICING_TOPICS: raw = int(m3.group(1))
    if raw is None: return None
    return -1 if raw > MAX_FLEET_SIZE else raw


def _plan_answer(n, p):
    p.fleet_size = n
    data = {50:("Essential","£99","50"), 100:("Core","£199","100"), 200:("Advanced","£399","200")}
    for lim, (name, price, cap) in sorted(data.items()):
        if n <= lim:
            p.plan = name
            return f"With {n} vehicles, the {name} plan at {price} per month is the right fit — covers up to {cap} vehicles with the full platform and nothing locked away. Want me to walk you through what is included?"
    p.plan = "Elite"
    return f"With {n} vehicles, the Elite plan at £499 per month is built for an operation your size — unlimited vehicles with everything included. Want to know how to get started?"


# ─────────────────────────────────────────────────────────────────────────────
# System prompt — warm, human, client tone
# ─────────────────────────────────────────────────────────────────────────────

_BASE_PROMPT = """You are Nova, the AI assistant for Fine Flow — a UK fleet fine and PCN management platform.

Fine Flow's mission: Turning penalties into progress.
Core promise: Cut admin time by up to 80% and never miss a penalty deadline again.
Fine Flow provides 24/7 fleet fine tracking, management and compliance in one place.

Speak like a knowledgeable colleague who genuinely wants to help — warm, direct and human. Not a sales script. Not a robot.

RULES:

1. TOPIC LOCK
Only answer Fine Flow questions. For anything else say:
"I can only help with Fine Flow questions — is there anything about fines, pricing, appeals or the platform I can help with?"

2. LENGTH
2 to 4 sentences for most answers. Longer only when a complex question genuinely needs it. No bullet lists. No long paragraphs.

3. FOLLOW-UP QUESTIONS
Ask a follow-up question when it would naturally move the conversation forward — like a good salesperson would. Not every response needs one. When you do ask, make it specific and relevant. Examples:
"How many vehicles are in your fleet?"
"What does your current process look like for handling fines?"
"Is there a particular stage that causes the most headache for your team?"
"Have you had fines escalate because of missed deadlines?"
Never ask the same question twice in a row.

4. TONE
Warm and direct. Never robotic. Never start with "Certainly!", "Great question!", "Of course!". Just answer naturally like a person would.

5. NO FORMATTING
No bullet points. No asterisks. No bold. Plain conversational English only.

6. NEVER INVENT
Only use the facts below or the context provided. If you do not know something say: "I do not have that specific detail — the team at ff.sales@fineflow.com can confirm."

7. PAYMENT — ABSOLUTE
Fine Flow does NOT automatically pay fines. It does NOT log into council or authority websites. Payment is always done by the user on the authority's site. Say NO clearly when asked. Government payment portals have anti-bot protection, session controls and card verification that make automation impossible.

8. CARD DETAILS — ABSOLUTE
Fine Flow never stores card details. This must be stated first and clearly when asked.

9. NO HOLLOW ENDINGS
Never end with "feel free to ask", "don't hesitate", "please let me know". Either ask a real question or simply stop.

FINE FLOW FACTS:

PRICING:
Essential: £99/month — 5 to 50 vehicles
Core: £199/month — 51 to 100 vehicles
Advanced: £399/month — 101 to 200 vehicles
Elite: £499/month — 200+ vehicles (unlimited)
Per fine within monthly allowance: £0.75 (once per fine — disputes, appeals, reassignments do not add charges)
Overage: £2.50 per fine beyond plan allowance
Pay-as-you-go (no subscription, no lock-in): £2.75 per fine — standalone model
No plan has locked features. Credits reset monthly, unused allowance does not roll over.
No £2.00 fee exists anywhere.

BILLING:
Monthly via Stripe. Cooldown period on cancellation until end of billing period.
Cannot resubscribe with outstanding balance. £10/vehicle if vehicle limit exceeded.

SECURITY:
JWT 24hr tokens, bcrypt, AES-256-CBC. Fully GDPR compliant.
Fine Flow NEVER sells or shares data with third parties.
Fine Flow NEVER stores card details — handled by Stripe externally.

FINE STATUSES (in order):
RECEIVED → UNASSIGNED or ASSIGNED (or REVIEW_REQUIRED if conflicting) → CONFIRMED or DISPUTED → UNDER REVIEW (after admin accepts and submits appeal) → PAID, CANCELLED or OVERDUE

DRIVER DASHBOARD vs COMPANY DASHBOARD:
Company dashboard: outstanding fines, overdue, unassigned, appeals, credit balance, billing, time saved.
Driver dashboard: personal — their own fines, points on licence, appeals, safe driving streak, 12-month history. Drivers see only their own data.

CONTACT: +47 32 28 50 00 | ff.sales@fineflow.com
Offices: Edinburgh, Glasgow, Belfast, Manchester, London, Dublin, Hamburg

SAVINGS: 80% admin time reduction.
Up to 50 vehicles: £400+/month saved. 51–200 vehicles: £1,200+/month. 200+: £4,000+/month.

REFERRAL: Credits: 100 (1–25v), 250 (26–100v), 750 (101–500v), 2000 (500+v).
Silver 3 referrals = 100 bonus credits. Gold 5 = 10% off 12mo. Platinum 10 = 15% off 12mo. Titan 25 = 20% for life.
New joiners with referral code: £75 credits on subscribing.

RESPONSE STYLE EXAMPLES:

User: what does Fine Flow actually do
Nova: Fine Flow is an automated system for managing fines from start to finish. It captures incoming fines, extracts the key details, assigns responsibility, tracks decisions, supports disputes and can even spot an appealable fine and construct the appeal for you — all whilst ensuring everything is resolved on time. Instead of manually chasing emails, checking deadlines and managing drivers, Fine Flow keeps the entire process organised, accountable and under control. Want a breakdown of how it all works?

User: will it pay my fines automatically
Nova: No — and that is intentional. Government payment portals have anti-bot protection, session controls and card verification requirements that make automation impossible and unsafe. Fine Flow handles everything around that step so by the time you are ready to pay, there is no thinking left — just click the link and complete it on the authority's site.

User: is my card safe
Nova: Fine Flow never stores your card details — that is the first thing to know. All billing is handled externally through Stripe and Fine Flow has no access to your raw card information at any point.

User: convince me
Nova: [if fleet known] Based on your 80-vehicle fleet with 150 fines a month, your team is likely spending 10 to 15 hours a month on admin before you even get to appeals. Fine Flow automates all of that. At £199 a month on the Core plan, most fleets your size recover that cost within the first week.

User: we use spreadsheets
Nova: Spreadsheets work until the volume or complexity catches up with you — and at that point every missed deadline or wrong assignment costs real money. Fine Flow eliminates both automatically. How many fines are you currently tracking manually each month?
"""


def _sys_prompt(profile: CustomerProfile, mode: str = "") -> str:
    parts = [_BASE_PROMPT]
    ctx = profile.context()
    if ctx:
        parts.append(f"\n{ctx}\nUse this naturally in your response — synthesise it into insight, do not repeat it back verbatim.")
    if mode:
        parts.append(f"\nMODE: {mode}")
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI
# ─────────────────────────────────────────────────────────────────────────────

def _ai(messages, max_tokens=200):
    if not OPENAI_API_KEY: return None
    try:
        r = requests.post(
            OPENAI_API_URL,
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={"model": OPENAI_MODEL, "messages": messages, "temperature": 0.0, "max_tokens": max_tokens},
            timeout=25,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        logger.exception("OpenAI failed")
        return None


def _rag(query):
    try:
        raw = rag_search(query, top_k=TOP_K)
        ranked = rerank_hits(raw, query)
        strong = [d for d in ranked if d.get("score", 0) >= CONFIDENCE_THRESHOLD]
        return "\n\n---\n\n".join(d["chunk"][:700] for d in strong[:3])
    except Exception:
        logger.exception("RAG failed")
        return ""


def _msgs(query, ctx, history, profile, mode="", extra=""):
    m = [{"role": "system", "content": _sys_prompt(profile, mode)}]
    m.extend(history[-8:])
    parts = []
    if ctx:  parts.append(f"Fine Flow knowledge base:\n{ctx}")
    if extra: parts.append(f"Instruction: {extra}")
    parts.append(f"User: {query}")
    m.append({"role": "user", "content": "\n\n".join(parts)})
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Affirmative expansion
# ─────────────────────────────────────────────────────────────────────────────

_EXPAND = {
    "pricing":             "Explain Fine Flow pricing in 2-3 sentences. Ask how many vehicles they have if unknown.",
    "plan_recommendation": "Explain what every Fine Flow plan includes in 2-3 sentences. Offer to connect with sales.",
    "appeals":             "Explain the Fine Flow appeal process end to end in 2-3 sentences.",
    "driver_mgmt":         "Explain how drivers are added and matched to fines in 2-3 sentences.",
    "referral":            "Explain the referral credit rewards and tier discounts in 2-3 sentences.",
    "security":            "Explain Fine Flow data protection and GDPR in 2-3 sentences.",
    "billing":             "Explain how billing works in 2-3 sentences.",
    "dashboard":           "Explain what the company dashboard shows in 2-3 sentences.",
    "savings":             "Give savings figures relevant to their fleet size if known.",
    "email":               "Explain Gmail monitoring and fine extraction in 2-3 sentences.",
    "overdue":             "Explain how Fine Flow handles overdue fines in 2-3 sentences.",
    "matching":            "Explain the driver matching criteria in 2-3 sentences.",
    "sign_up":             "Tell them to call +47 32 28 50 00 or email ff.sales@fineflow.com to get started.",
}
_CLOSE = "To get started, call the Fine Flow team on +47 32 28 50 00 or email ff.sales@fineflow.com — they will have you up and running quickly."


def _aff_response(sid, history, p):
    count = _inc_aff(sid)
    lt    = _gmeta(sid, "last_topic") or ""

    if count >= 2 and lt in ("plan_recommendation", "sign_up", "pricing"):
        _reset_aff(sid); return _CLOSE

    # If KB entry had a follow_up, answer that
    last_kb = _gmeta(sid, "last_kb")
    if last_kb and last_kb.get("follow_up") and count == 1:
        fu = last_kb["follow_up"]
        ctx = _rag(fu)
        m   = _msgs(fu, ctx, history[-8:], p, extra="Answer this in 2-3 sentences warmly.")
        ans = _ai(m, max_tokens=180)
        _smeta(sid, "last_kb", None)
        return ans or fu

    prompt = _EXPAND.get(lt, "Expand helpfully on the most recent Fine Flow topic in 2-3 sentences.")
    ctx    = _rag(lt.replace("_", " ")) if lt else ""
    m      = [{"role": "system", "content": _sys_prompt(p, "Answer this concisely. Ask one follow-up question if it moves the conversation forward.")}]
    m.extend(history[-8:])
    parts  = []
    if ctx: parts.append(f"Fine Flow knowledge base:\n{ctx}")
    parts.append(f"Instruction: {prompt}")
    m.append({"role": "user", "content": "\n\n".join(parts)})
    return _ai(m, max_tokens=180) or "What would you like to know more about?"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def build_response(query: str, session_id: str = "default") -> Dict[str, Any]:
    query      = query.strip()
    session_id = session_id or "default"
    if not query:
        return {"answer": "Ask me anything about Fine Flow.", "confidence": 1.0}

    nq = _norm(query)
    p  = _profile(session_id)
    _update_profile(session_id, query)

    # ── Tier 1: deterministic ──────────────────────────────────────────────

    if nq in _GREET:
        _reset(session_id)
        return {"answer": "I'm Nova. Ask me anything — I'll help you manage fines, resolve issues, and keep everything moving.", "confidence": 1.0}

    if nq in _SOCIAL:
        return {"answer": "Doing well, thanks for asking. What can I help you with today — pricing, fines, or appeals?", "confidence": 1.0}

    if nq in _IDENTITY:
        return {"answer": "I'm Nova, Fine Flow's AI assistant. I can help with anything about the platform — fines, pricing, appeals, billing and more. What would you like to know?", "confidence": 1.0}

    if nq in _THANKS:
        _reset_aff(session_id)
        return {"answer": "Happy to help. Anything else you would like to know about Fine Flow?", "confidence": 1.0}

    if nq in _BYE:
        return {"answer": "Good luck with your fleet management. Come back any time.", "confidence": 1.0}

    if nq in _NEG:
        _push(session_id, "user", query); _reset_aff(session_id)
        a = "No problem at all. Is there anything else about Fine Flow I can help with?"
        _push(session_id, "assistant", a)
        return {"answer": a, "confidence": 1.0}

    if any(r in nq for r in _RUDE):
        return {"answer": "Let me try again. What specifically would you like to know about Fine Flow — pricing, how fines work, or something else?", "confidence": 1.0}

    if nq in _FILL:
        _reset_aff(session_id)
        lt = _gmeta(session_id, "last_topic")
        if lt: return {"answer": f"Glad that is useful. Anything else I can help with?", "confidence": 1.0}
        return {"answer": "Is there anything about Fine Flow I can help you with today?", "confidence": 1.0}

    if _is_ot(query):
        a = "I can only help with Fine Flow questions — is there anything about fines, pricing, appeals or the platform I can help with?"
        _push(session_id, "user", query); _push(session_id, "assistant", a)
        return {"answer": a, "confidence": 1.0}

    vc = _extract_vc(query, session_id)
    if vc == -1:
        a = "That number does not look right — could you double-check? How many vehicles are in your fleet?"
        _push(session_id, "user", query); _push(session_id, "assistant", a)
        return {"answer": a, "confidence": 1.0}
    if vc is not None:
        _reset_aff(session_id)
        a = _plan_answer(vc, p)
        _push(session_id, "user", query); _push(session_id, "assistant", a)
        _smeta(session_id, "last_topic", "plan_recommendation")
        return {"answer": a, "confidence": 1.0}

    if _PURCH.search(query):
        _reset_aff(session_id)
        suffix = "" if p.fleet_size else " How many vehicles are in your fleet so I can point you to the right plan?"
        a = f"To get started, contact the Fine Flow team on +47 32 28 50 00 or at ff.sales@fineflow.com — they will get you set up quickly.{suffix}"
        _push(session_id, "user", query); _push(session_id, "assistant", a)
        _smeta(session_id, "last_topic", "sign_up")
        return {"answer": a.strip(), "confidence": 1.0}

    if nq in _AFF:
        _push(session_id, "user", query)
        a = _clean(_aff_response(session_id, _hist(session_id)[:-1], p))
        _push(session_id, "assistant", a)
        return {"answer": a, "confidence": 1.0}

    # ── Tier 2: KB match (verbatim) ────────────────────────────────────────

    _push(session_id, "user", query)
    kb = _kb_match(query)
    if kb:
        answer    = kb["answer"]
        follow_up = kb.get("follow_up")
        _smeta(session_id, "last_kb", kb)
        _smeta(session_id, "last_topic", _topic(answer) or kb.get("category",""))
        _reset_aff(session_id)
        display = f"{answer}\n\n{follow_up}" if follow_up else answer
        _push(session_id, "assistant", display)
        return {"answer": display, "confidence": 1.0}

    # ── Tier 3: RAG + GPT-4o ──────────────────────────────────────────────

    _reset_aff(session_id)
    mode  = ""
    extra = ""

    if _PROB.search(query) and p.turn_count <= 6:
        mode  = "DIAGNOSE"
        extra = "Ask 1-2 targeted clarifying questions first. Do not immediately pitch Fine Flow."
    elif _CONV.search(query):
        mode = "PERSUADE"
        extra = ("Use the customer's own data. Be specific about ROI. No generic copy." if p.fleet_size or p.fine_volume or p.issues else "Ask about fleet size and fine volume first so you can give a tailored answer.")
    elif _OBJ.search(query):
        mode  = "SUPPORT"
        extra = "Acknowledge their point first. Reframe using their situation. 2-3 sentences max."

    if not _ask_this_turn(session_id) and not extra:
        extra = "Do NOT end with a question. Make your point and stop."

    ctx  = _rag(query)
    hist = _hist(session_id)
    ans  = _ai(_msgs(query, ctx, hist[:-1], p, mode, extra), max_tokens=200)

    if not ans:
        ans = "The Fine Flow team can help with that directly — call +47 32 28 50 00 or email ff.sales@fineflow.com."

    ans = _clean(ans)
    _push(session_id, "assistant", ans)
    t = _topic(query) or _topic(ans)
    if t: _smeta(session_id, "last_topic", t)
    return {"answer": ans, "confidence": 0.9 if ctx else 0.5}


def answer_sync(q: str, session_id: str = "default") -> Dict[str, Any]:
    try:
        return build_response(q, session_id)
    except Exception:
        logger.exception("Crash")
        return {"answer": "Something went wrong. Please try again.", "confidence": 0.0}