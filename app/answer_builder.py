# app/answer_builder.py
"""
FineFlow Nova — Final Production Version
=========================================
Three-tier architecture:
  1. Deterministic handlers  — greetings, identity, vehicle count, off-topic
  2. KB match               — client's exact approved answers, verbatim
  3. RAG + GPT-4o           — anything not in KB

Key behaviours:
  - Garbled / short / nonsense input handled gracefully (not as off-topic)
  - KB keywords are long phrases so wrong entries never win
  - Customer profile stored and used in every GPT call
  - Follow-up questions from KB entries are delivered when user says yes
  - Question limiting — not every response ends with a question
  - Warm, human UK tone
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

MAX_FLEET = 50_000
MIN_QUERY_TOKENS = 2   # queries shorter than this skip KB and RAG

# ─────────────────────────────────────────────────────────────────────────────
# KB Loader
# ─────────────────────────────────────────────────────────────────────────────

KB_PATH = Path(__file__).parent.parent / "data" / "fineflow_kb.json"
_KB: List[Dict] = []


def _load_kb():
    global _KB
    if not KB_PATH.exists():
        logger.warning("KB not found: %s", KB_PATH)
        return
    try:
        with open(KB_PATH, encoding="utf-8") as f:
            _KB = json.load(f)
        logger.info("Loaded %d KB entries", len(_KB))
    except Exception:
        logger.exception("KB load failed")


_load_kb()

_STOP = {
    "a","an","the","is","it","in","on","of","to","do","my","me","you","your",
    "i","we","are","was","be","will","can","has","have","had","not","and","or",
    "but","if","this","that","with","for","at","by","from","as","what","how",
    "why","who","when","where","which","there","here","does","did","so","up",
    "about","just","more","also","would","could","should","any","all","get",
    "its","into","out","no","yes","please","want","need",
}


def _tok(text: str) -> set:
    return {w for w in re.sub(r"[^\w\s]", " ", text.lower()).split()
            if w not in _STOP and len(w) > 2}


def _kb_match(query: str) -> Optional[Dict]:
    nq      = re.sub(r"[^\w\s]", " ", query.lower()).strip()
    q_tok   = _tok(query)
    if len(q_tok) < 1:
        return None

    best      = None
    best_score = 0.0

    for entry in _KB:
        kws = entry.get("keywords", [])
        # Exact phrase match — highest priority
        for kw in kws:
            kw_n = re.sub(r"[^\w\s]", " ", kw.lower()).strip()
            if kw_n in nq or nq in kw_n:
                return entry
        # Token overlap
        s_tok = _tok(" ".join(kws))
        if not s_tok:
            continue
        overlap = len(q_tok & s_tok)
        if overlap == 0:
            continue
        score = overlap / max(len(q_tok), 1) + 0.05 * len(q_tok & s_tok)
        if score > best_score:
            best_score = score
            best = entry

    return best if best_score >= 0.6 else None


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
            order = ["missed deadlines","missed appeal deadlines",
                     "driver disputes","manual admin","using spreadsheets"]
            lines.append("Problems: " + ", ".join(
                sorted(self.issues, key=lambda x: order.index(x) if x in order else 99)))
        return ("What I know about this customer:\n" + "\n".join(lines)) if lines else ""

    def plan_name(self) -> str:
        if not self.fleet or self.fleet > MAX_FLEET: return ""
        if self.fleet <= 50:  return "Essential (£99/month)"
        if self.fleet <= 100: return "Core (£199/month)"
        if self.fleet <= 200: return "Advanced (£399/month)"
        return "Elite (£499/month)"


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    if not text: return ""
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*",     r"\1", text)
    text = re.sub(r"_(.*?)_",       r"\1", text)
    text = text.replace("→","to").replace("->","to").replace("`","")
    for bad in ["feel free to ask!","feel free to ask.",
                "please let me know if you need anything.",
                "don't hesitate to ask.","don't hesitate to reach out."]:
        if text.lower().endswith(bad): text = text[:-len(bad)].rstrip(" ,.")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Session memory
# ─────────────────────────────────────────────────────────────────────────────

_SES: Dict[str, List[Dict]]  = {}
_PRO: Dict[str, Profile]     = {}
_MET: Dict[str, Dict]        = {}
_LK  = threading.Lock()

PRICE_TOPICS = {"pricing","plan_recommendation","vehicles","cost","billing"}


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
    r"\b(logistics|delivery|courier|haulage|transport|taxi|minicab|bus|coach|construction|utilities)\b", re.I)
_ISS = [
    (re.compile(r"\b(miss(?:ed?|ing)?\s+(?:deadlines?|appeals?|due\s*dates?))\b", re.I), "missed deadlines"),
    (re.compile(r"\b(miss(?:ed?|ing)?\s+appeal\s+deadline)\b", re.I),                   "missed appeal deadlines"),
    (re.compile(r"\b(drivers?\s+(?:dispute|deny|ignor|avoid))\b", re.I),                "driver disputes"),
    (re.compile(r"\b(manual(?:ly)?\s+(?:track|manage|process|handl))\b", re.I),         "manual admin"),
    (re.compile(r"\b(spreadsheet)\b", re.I),                                             "using spreadsheets"),
]


def _upd_pro(s, q):
    p = _pro(s); p.turns += 1
    m = _FINES_RE.search(q)
    if m and p.volume is None:
        v = int(m.group(1))
        if v < 10_000: p.volume = v
    m = _IND_RE.search(q)
    if m: p.industry = m.group(1).lower()
    for pat, lbl in _ISS:
        if pat.search(q) and lbl not in p.issues: p.issues.append(lbl)


# ─────────────────────────────────────────────────────────────────────────────
# Intent sets
# ─────────────────────────────────────────────────────────────────────────────

_GREET = {
    "hi","hello","hey","hiya","howdy","yo","sup",
    "good morning","good afternoon","good evening","morning","afternoon","evening",
    "hi there","hey there","hello there","hi nova","hey nova","hello nova",
}
_SOC = {
    "how are you","how are you doing","how r u","how r you","how are u",
    "hows it going","how is it going","whats up","what s up",
    "you ok","you good","how do you do","you alright","alright mate",
}
_ID = {
    "who are you","who r you","who r u","who youre","who you re",
    "who is nova","who is this","who is there","whos there","who s there",
    "anyone there","is anyone there","what are you","what is nova",
    "are you a bot","are you human","are you ai","are you a robot",
    "are you male or female","you male or female","who the hell are you",
    "whats your name","what is your name","introduce yourself",
    "who am i talking to","knock knock",
}
_AFF = {
    "yes","yeah","yep","yup","ya","ye","sure","ok sure","okay sure",
    "go ahead","go on","yes go ahead","yes please","yes sure","yes of course",
    "of course","absolutely","definitely","do it","tell me more","more",
    "explain","explain more","yes explain","go for it","sounds good",
    "continue","carry on","keep going","please do","i would","please explain",
    "show me","walk me through it","yes elite","yes core","yes essential",
    "yes advanced","for sure","yes for sure","sure thing","yes tell me",
}
_THX = {
    "thanks","thank you","thank u","cheers","that helps","that helped","ta","ty",
    "okay thanks","ok thanks","great thanks","perfect","brilliant","nice one",
    "lovely","great","awesome","wonderful","thank you so much","many thanks","much appreciated",
}
_BYE = {"bye","goodbye","see you","see ya","later","take care","good bye","cya","ttyl","farewell","cheerio"}
_NEG = {"no","nope","nah","no thanks","not now","skip","never mind","nevermind","no need","not really","no thank you","nah thanks"}
_RUDE = {"you dumb","you are dumb","ur dumb","stupid","idiot","useless","rubbish",
         "garbage","terrible","you suck","this is rubbish","dumb bot","waste of time"}
_FILL = {"ok","okay","right","alright","cool","nice","interesting","really",
         "seriously","hmm","hm","ah","oh","i see","got it","understood",
         "makes sense","noted","wow","waow","woah","omg","anything","something","whatever"}

# Fine Flow related terms — NEVER treated as off-topic
_FF_ALLOW = re.compile(
    r"\b(council|authority|authorities|payment portal|log into|login to|pay.*fine|"
    r"fine.*pay|appeal.*fine|fine.*appeal|pcn|penalty charge|parking fine|fleet fine|"
    r"fineflow|fine flow|streamline|overage|allowance|subscription|driver log|"
    r"vehicle log|billing|credit|cancel|reassign|dispute|uk fine|council fine|"
    r"penalty notice|fixed penalty|congestion|bus lane|violation)\b", re.I)

_OT = [
    re.compile(r"\b(html|css|javascript|typescript|python|java|php|sql|react|angular|vue|node\.?js|django|flask|docker|kubernetes|github|devops|backend|frontend|coding|programming)\b", re.I),
    re.compile(r"\b(machine learning|deep learning|neural network|large language model|generative ai|train a model|llm|bert)\b", re.I),
    re.compile(r"\b(recipe|cooking|restaurant|pizza|burger|sandwich|coffee|tea|cake|meal|bake|order food|make me a|bake me|cook me)\b", re.I),
    re.compile(r"\b(movie|film|song|lyrics|music|football match|cricket match|weather forecast|todays news|politics|history lesson|capital city|who invented|tell me a joke|write me a poem)\b", re.I),
    re.compile(r"\b(write an essay|translate this|proofread|write my cv|write a story)\b", re.I),
    re.compile(r"\b(chatgpt|openai|gemini|claude ai|anthropic|google bard|bing ai|alexa|siri)\b", re.I),
]


def _is_ot(q): return not _FF_ALLOW.search(q) and any(p.search(q) for p in _OT)


_VEH_EX = re.compile(r"\b(\d+)\s*(vehicle|vehicles|van|vans|truck|trucks|car|cars|lorry|lorries|in my fleet|in our fleet)\b", re.I)
_VEH_FL = re.compile(r"\b(?:fleet of|manage|running|operate|run)\s+(\d+)\b", re.I)
_VEH_BR = re.compile(r"^\s*(\d+)\s*$")
_DRV_CT = re.compile(r"\b(driver|drivers|staff|employee|employees|people|worker|team|members)\b", re.I)
_PURCH  = re.compile(r"\b(want to buy|want to subscribe|want to sign up|how do i buy|how do i get started|how do i sign up|get started|free trial|sign me up|ready to buy|book a demo|talk to sales|how to start|where do i sign|how do i join)\b", re.I)
_CONV   = re.compile(r"\b(convince|persuade|sell me|why should i buy|why buy fineflow|is it worth|should i buy|justify|worth it|why should i choose)\b", re.I)
_OBJ    = re.compile(r"\b(expensive|too much|costly|already use spreadsheet|manage manually|don.?t need|our team handles|we manage fines)\b", re.I)
_PROB   = re.compile(
    r"\b(fine|pcn|penalty|violation|notice|ticket)\b.{0,40}\b(issued|received|got|have a|appealing|disputing|today|yesterday|last week)\b"
    r"|\b(received a|got a|have a)\b.{0,20}\b(fine|pcn|penalty|violation)\b", re.I)

_TMAP = {
    "pric":"pricing","cost":"pricing","plan":"pricing","£":"pricing",
    "vehicle":"pricing","fleet":"pricing","package":"pricing",
    "appeal":"appeals","dispute":"appeals",
    "driver":"driver_mgmt","referral":"referral","refer":"referral",
    "discount":"referral","security":"security","gdpr":"security",
    "billing":"billing","stripe":"billing","dashboard":"dashboard",
    "report":"reports","gmail":"gmail","email":"email",
    "save":"savings","admin":"savings","overdue":"overdue","deadline":"overdue",
    "start":"sign_up","sign up":"sign_up",
}


def _topic(t):
    t = t.lower()
    for k, v in _TMAP.items():
        if k in t: return v
    return None


def _get_vc(query, s):
    if _DRV_CT.search(query): return None
    raw = None
    m = _VEH_EX.search(query)
    if m: raw = int(m.group(1))
    elif (m2 := _VEH_FL.search(query)): raw = int(m2.group(1))
    elif (m3 := _VEH_BR.match(query)):
        if _gm(s, "lt") in PRICE_TOPICS: raw = int(m3.group(1))
    if raw is None: return None
    return -1 if raw > MAX_FLEET else raw


def _plan_ans(n, p):
    p.fleet = n
    if n <= 50:   name, price, lim = "Essential", "£99",  "50"
    elif n <= 100: name, price, lim = "Core",     "£199", "100"
    elif n <= 200: name, price, lim = "Advanced", "£399", "200"
    else:          name, price, lim = "Elite",    "£499", "unlimited"
    p.plan = name
    return (f"With {n} vehicles, the {name} plan at {price} per month is the right fit — "
            f"covers up to {lim} vehicles with the full platform and nothing locked away. "
            f"Want me to walk you through what is included?")


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

_SYS = """You are Nova, the AI assistant for Fine Flow — a UK fleet fine and PCN management platform.

Fine Flow's mission: Turning penalties into progress.
Fine Flow provides 24/7 fleet fine tracking, management and compliance in one place.
Core promise: Cut admin time by up to 80% and never miss a penalty deadline again.

Speak warmly and directly — like a knowledgeable colleague, not a robot or a sales script.

RULES:

1. TOPIC: Only answer Fine Flow questions. For anything unrelated say: "I can only help with Fine Flow questions — is there anything about fines, pricing, appeals or the platform I can help with?"

2. LENGTH: 2 to 4 sentences for most answers. Plain English. No bullet lists. No asterisks.

3. QUESTIONS: Only ask a follow-up when it genuinely moves the conversation forward. Not every response needs one. Never repeat the same question twice. Never end with hollow phrases like "feel free to ask" or "don't hesitate to reach out."

4. TONE: Direct and warm. Never start with "Certainly!", "Great question!", "Of course!". Just answer.

5. PAYMENT: Fine Flow does NOT pay fines automatically. Payment is always done by the user on the authority's site. Say NO clearly. Government portals have anti-bot protection, session controls and card verification that make automation impossible.

6. CARD DETAILS: Fine Flow NEVER stores card details. State this first and clearly.

7. NEVER INVENT: Only use context provided or facts below. If you do not know: "I don't have that specific detail — the team at ff.sales@fineflow.com can confirm."

FACTS:
Pricing: Essential £99 (5–50v), Core £199 (51–100v), Advanced £399 (101–200v), Elite £499 (200+v)
Per fine within allowance: £0.75 (once per fine — no extra for disputes/appeals/reassignments)
Overage: £2.50 per fine | PAYG no subscription: £2.75 per fine
All plans identical features — no paywalls. No £2.00 fee.
Contact: +47 32 28 50 00 | ff.sales@fineflow.com
Offices: Edinburgh, Glasgow, Belfast, Manchester, London, Dublin, Hamburg
Billing: Monthly via Stripe. No rollover. Cooldown on cancellation. £10/vehicle over limit.
Security: JWT 24hr, bcrypt, AES-256-CBC. GDPR. Never stores card data. Never sells data.
Savings: 80% admin cut. Up to 50v: £400+/mo. 51–200v: £1,200+/mo. 200+v: £4,000+/mo.
Referral: 100/250/750/2000 credits by fleet size. Gold 5=10% off. Platinum 10=15%. Titan 25=20% for life.
"""


def _sys(p: Profile, mode=""):
    parts = [_SYS]
    ctx = p.ctx()
    if ctx:
        parts.append(f"\n{ctx}\nUse this naturally — synthesise into insight, do not repeat it back.")
    if mode:
        parts.append(f"\nMODE: {mode}")
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI + RAG
# ─────────────────────────────────────────────────────────────────────────────

def _ai(msgs, max_tok=200):
    if not OPENAI_API_KEY: return None
    try:
        r = requests.post(
            OPENAI_API_URL,
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={"model": OPENAI_MODEL, "messages": msgs, "temperature": 0.0, "max_tokens": max_tok},
            timeout=25)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        logger.exception("OpenAI failed"); return None


def _rag(q):
    try:
        raw = rag_search(q, top_k=TOP_K)
        ranked = rerank_hits(raw, q)
        strong = [d for d in ranked if d.get("score", 0) >= CONFIDENCE_THRESHOLD]
        return "\n\n---\n\n".join(d["chunk"][:700] for d in strong[:3])
    except Exception:
        logger.exception("RAG failed"); return ""


def _make(query, ctx, hist, p, mode="", extra=""):
    m = [{"role": "system", "content": _sys(p, mode)}]
    m.extend(hist[-8:])
    parts = []
    if ctx:   parts.append(f"Fine Flow knowledge base:\n{ctx}")
    if extra: parts.append(f"Instruction: {extra}")
    parts.append(f"User: {query}")
    m.append({"role": "user", "content": "\n\n".join(parts)})
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Affirmative expansion
# ─────────────────────────────────────────────────────────────────────────────

_EXP = {
    "pricing":             "Explain Fine Flow pricing concisely. Ask how many vehicles if unknown.",
    "plan_recommendation": "Explain what every plan includes in 2-3 sentences. Offer to connect with sales.",
    "appeals":             "Explain the Fine Flow appeal process in 2-3 sentences.",
    "driver_mgmt":         "Explain how drivers are added and matched to fines.",
    "referral":            "Explain referral credit rewards and tier discounts.",
    "security":            "Explain data protection and GDPR approach.",
    "billing":             "Explain how billing works.",
    "dashboard":           "Explain what the company dashboard shows.",
    "savings":             "Give savings figures relevant to their fleet size if known.",
    "email":               "Explain Gmail monitoring and fine extraction.",
    "overdue":             "Explain how Fine Flow handles overdue fines.",
    "matching":            "Explain the three driver matching criteria.",
    "sign_up":             "Tell them to call +47 32 28 50 00 or email ff.sales@fineflow.com.",
}
_CLOSE = "To get started, call the Fine Flow team on +47 32 28 50 00 or email ff.sales@fineflow.com — they will have you up and running quickly."


def _aff(s, hist, p):
    cnt = _inc_aff(s)
    lt  = _gm(s, "lt") or ""

    if cnt >= 2 and lt in ("plan_recommendation", "sign_up", "pricing"):
        _rst_aff(s); return _CLOSE

    # Answer the KB follow-up if available
    last_kb = _gm(s, "lkb")
    if last_kb and last_kb.get("follow_up") and cnt == 1:
        fu  = last_kb["follow_up"]
        ctx = _rag(fu)
        m   = _make(fu, ctx, hist[-8:], p, extra="Answer in 2-3 sentences warmly and naturally.")
        ans = _ai(m, 180)
        _sm(s, "lkb", None)
        return ans or fu

    prompt = _EXP.get(lt, "Expand helpfully on the most recent Fine Flow topic in 2-3 sentences.")
    ctx    = _rag(lt.replace("_", " ")) if lt else ""
    m      = [{"role": "system", "content": _sys(p, "Answer concisely. Ask one follow-up only if it genuinely helps.")}]
    m.extend(hist[-8:])
    parts  = []
    if ctx: parts.append(f"Fine Flow knowledge base:\n{ctx}")
    parts.append(f"Instruction: {prompt}")
    m.append({"role": "user", "content": "\n\n".join(parts)})
    return _ai(m, 180) or "What would you like to know more about?"


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
    _upd_pro(session_id, query)

    # ── TIER 1: Deterministic ──────────────────────────────────────────────

    if nq in _GREET:
        _rst(session_id)
        return {"answer": "I'm Nova. Ask me anything — I'll help you manage fines, resolve issues, and keep everything moving.", "confidence": 1.0}

    if nq in _SOC:
        return {"answer": "Doing well, thanks for asking. What can I help you with today — pricing, fines, or appeals?", "confidence": 1.0}

    if nq in _ID:
        return {"answer": "I'm Nova, Fine Flow's AI assistant. I can help with anything about the platform — fines, pricing, appeals, billing and more. What would you like to know?", "confidence": 1.0}

    if nq in _THX:
        _rst_aff(session_id)
        return {"answer": "Happy to help. Anything else you would like to know about Fine Flow?", "confidence": 1.0}

    if nq in _BYE:
        return {"answer": "Good luck with your fleet management. Come back any time.", "confidence": 1.0}

    if nq in _NEG:
        _push(session_id, "user", query); _rst_aff(session_id)
        a = "No problem at all. Anything else about Fine Flow I can help with?"
        _push(session_id, "assistant", a)
        return {"answer": a, "confidence": 1.0}

    if any(r in nq for r in _RUDE):
        return {"answer": "Let me try again. What specifically would you like to know about Fine Flow — pricing, how fines work, or something else?", "confidence": 1.0}

    if nq in _FILL:
        _rst_aff(session_id)
        lt = _gm(session_id, "lt")
        msg = f"Glad that is useful. Is there anything else I can help you with?" if lt else "Is there anything about Fine Flow I can help you with?"
        return {"answer": msg, "confidence": 1.0}

    # Garbled / very short input — ask what they meant rather than off-topic guard
    words = [w for w in nq.split() if len(w) > 1]
    if len(words) < MIN_QUERY_TOKENS and nq not in _AFF:
        return {"answer": "I did not quite catch that. What would you like to know about Fine Flow — fines, pricing, appeals or billing?", "confidence": 1.0}

    if _is_ot(query):
        a = "I can only help with Fine Flow questions — is there anything about fines, pricing, appeals or the platform I can help with?"
        _push(session_id, "user", query); _push(session_id, "assistant", a)
        return {"answer": a, "confidence": 1.0}

    vc = _get_vc(query, session_id)
    if vc == -1:
        a = "That number does not look right — could you double check? How many vehicles are in your fleet?"
        _push(session_id, "user", query); _push(session_id, "assistant", a)
        return {"answer": a, "confidence": 1.0}
    if vc is not None:
        _rst_aff(session_id)
        a = _plan_ans(vc, p)
        _push(session_id, "user", query); _push(session_id, "assistant", a)
        _sm(session_id, "lt", "plan_recommendation")
        return {"answer": a, "confidence": 1.0}

    if _PURCH.search(query):
        _rst_aff(session_id)
        sfx = "" if p.fleet else " How many vehicles are in your fleet so I can point you to the right plan?"
        a = f"To get started, contact the Fine Flow team on +47 32 28 50 00 or at ff.sales@fineflow.com — they will get you set up quickly.{sfx}"
        _push(session_id, "user", query); _push(session_id, "assistant", a)
        _sm(session_id, "lt", "sign_up")
        return {"answer": a.strip(), "confidence": 1.0}

    if nq in _AFF:
        _push(session_id, "user", query)
        a = _clean(_aff(session_id, _hist(session_id)[:-1], p))
        _push(session_id, "assistant", a)
        return {"answer": a, "confidence": 1.0}

    # ── TIER 2: KB match (verbatim) ───────────────────────────────────────

    _push(session_id, "user", query)
    kb = _kb_match(query)
    if kb:
        answer = kb["answer"]
        fu     = kb.get("follow_up")
        _sm(session_id, "lkb", kb)
        _sm(session_id, "lt",  _topic(answer) or kb.get("category", ""))
        _rst_aff(session_id)
        display = f"{answer}\n\n{fu}" if fu else answer
        _push(session_id, "assistant", display)
        return {"answer": display, "confidence": 1.0}

    # ── TIER 3: RAG + GPT-4o ─────────────────────────────────────────────

    _rst_aff(session_id)
    mode  = ""
    extra = ""

    if _PROB.search(query) and p.turns <= 6:
        mode  = "DIAGNOSE"
        extra = "Ask 1-2 targeted clarifying questions first. Do not immediately pitch Fine Flow."
    elif _CONV.search(query):
        mode  = "PERSUADE"
        extra = ("Use the customer's own data. Be specific about ROI. No generic copy."
                 if p.fleet or p.volume or p.issues
                 else "Ask about fleet size and fine volume first so you can give a tailored answer.")
    elif _OBJ.search(query):
        mode  = "SUPPORT"
        extra = "Acknowledge their point first. Reframe using their situation. 2-3 sentences."

    if not _ask_now(session_id) and not extra:
        extra = "Do NOT end with a question. Make your point clearly and stop."

    ctx  = _rag(query)
    hist = _hist(session_id)
    ans  = _ai(_make(query, ctx, hist[:-1], p, mode, extra), 200)

    if not ans:
        ans = "The Fine Flow team can help with that directly — call +47 32 28 50 00 or email ff.sales@fineflow.com."

    ans = _clean(ans)
    _push(session_id, "assistant", ans)
    t = _topic(query) or _topic(ans)
    if t: _sm(session_id, "lt", t)
    return {"answer": ans, "confidence": 0.9 if ctx else 0.5}


def answer_sync(q: str, session_id: str = "default") -> Dict[str, Any]:
    try:
        return build_response(q, session_id)
    except Exception:
        logger.exception("Crash")
        return {"answer": "Something went wrong. Please try again.", "confidence": 0.0}