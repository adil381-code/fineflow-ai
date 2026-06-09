# app/answer_builder.py
"""
FineFlow Nova — Final Production Version
=========================================
Architecture:
  1. Deterministic layer  — greetings, identity, off-topic, vehicle count
  2. RAG retrieval        — semantic search over fineflow_knowledge.txt
  3. GPT-4o generation    — writes answer in client's approved tone

No JSON KB. One TXT file. One source of truth.
Temperature 0.7 — warm and natural.
Max 150 tokens — short and precise.
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
# Customer profile — stored per session
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Profile:
    fleet:    Optional[int]  = None
    volume:   Optional[int]  = None
    issues:   List[str]      = field(default_factory=list)
    industry: Optional[str]  = None
    turns:    int            = 0

    def summary(self) -> str:
        lines = []
        if self.fleet and self.fleet <= MAX_FLEET:
            lines.append(f"Fleet size: {self.fleet} vehicles")
        if self.volume:
            lines.append(f"Monthly fines: ~{self.volume}")
        if self.industry:
            lines.append(f"Industry: {self.industry}")
        if self.issues:
            lines.append(f"Mentioned problems: {', '.join(self.issues)}")
        return "\n".join(lines)


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
    # Remove hollow filler endings
    for bad in [
        "feel free to ask!", "feel free to ask.",
        "don't hesitate to ask.", "don't hesitate to reach out.",
        "please let me know if you need anything.",
        "if you have any other questions, feel free to ask.",
    ]:
        if text.lower().endswith(bad.lower()):
            text = text[: -len(bad)].rstrip(" ,.")
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
    """Alternate asking questions — not every response needs one."""
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
_ISS = [
    (re.compile(r"\b(miss(?:ed?|ing)?\s+(?:deadlines?|appeals?|due\s*dates?))\b", re.I), "missed deadlines"),
    (re.compile(r"\b(drivers?\s+(?:dispute|deny|ignor))\b", re.I),                      "driver disputes"),
    (re.compile(r"\b(spreadsheet)\b", re.I),                                             "using spreadsheets"),
    (re.compile(r"\b(too\s+much\s+admin|manual\s+(?:track|process))\b", re.I),          "too much admin"),
]


def _upd(s, q):
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
_BYE = {"bye", "goodbye", "see you", "see ya", "later", "take care", "good bye", "cya", "farewell", "cheerio"}
_NEG = {"no", "nope", "nah", "no thanks", "not now", "skip", "never mind", "nevermind", "no need", "not really"}
_RUDE = {"stupid", "idiot", "useless", "rubbish", "garbage", "terrible", "you suck", "dumb bot", "waste of time"}
_FILL = {
    "ok", "okay", "right", "alright", "cool", "nice", "interesting", "really",
    "hmm", "hm", "ah", "oh", "i see", "got it", "understood", "makes sense",
    "noted", "wow", "woah", "omg", "anything", "something", "whatever",
}

# Fine Flow terms — never blocked as off-topic
_FF_OK = re.compile(
    r"\b(council|authority|fine|pcn|penalty|fineflow|fine flow|appeal|dispute|"
    r"driver|fleet|vehicle|overage|allowance|billing|subscription|payment|"
    r"uk traffic|traffic violation|parking fine|bus lane|congestion|emission|"
    r"dvla|tfl|transport for london|fixed penalty|notice to owner)\b", re.I)

_OT = [
    re.compile(r"\b(html|css|javascript|typescript|python|java|php|sql|react|angular|vue|node\.?js|django|flask|docker|kubernetes|github|coding|programming)\b", re.I),
    re.compile(r"\b(machine learning|deep learning|neural network|large language model|generative ai|llm|bert|teach me|explain python|how to code)\b", re.I),
    re.compile(r"\b(recipe|cooking|restaurant|pizza|burger|sandwich|coffee|tea|cake|meal|make me a food|bake me|cook me)\b", re.I),
    re.compile(r"\b(movie|film|song|lyrics|music|football match|cricket match|weather forecast|todays news|politics|history lesson|capital city|who invented|tell me a joke|write me a poem)\b", re.I),
    re.compile(r"\b(write an essay|translate this|proofread|write my cv|write a story|write my resume)\b", re.I),
    re.compile(r"\b(chatgpt|openai|gemini|claude ai|anthropic|google bard|bing ai|alexa|siri)\b", re.I),
]


def _is_ot(q): return not _FF_OK.search(q) and any(p.search(q) for p in _OT)


_VEH_EX = re.compile(r"\b(\d+)\s*(vehicle|vehicles|van|vans|truck|trucks|car|cars|lorry|lorries|in my fleet|in our fleet)\b", re.I)
_VEH_FL = re.compile(r"\b(?:fleet of|manage|running|operate|run)\s+(\d+)\b", re.I)
_VEH_BR = re.compile(r"^\s*(\d+)\s*$")
_DRV_CT = re.compile(r"\b(driver|drivers|staff|employee|employees|people|worker|team|members)\b", re.I)
_PURCH  = re.compile(r"\b(want to buy|want to subscribe|want to sign up|how do i get started|how do i sign up|get started|free trial|sign me up|book a demo|talk to sales|how to start|where do i sign|how do i join)\b", re.I)
_CONV   = re.compile(r"\b(convince|persuade|sell me|why should i|why buy|is it worth|should i buy|worth it|why choose fineflow|why fine flow)\b", re.I)
_OBJ    = re.compile(r"\b(expensive|too much|too costly|already use spreadsheet|we manage manually|don.?t need it|we handle fines ourselves)\b", re.I)


def _get_vc(q, s):
    if _DRV_CT.search(q): return None
    raw = None
    m = _VEH_EX.search(q)
    if m: raw = int(m.group(1))
    elif (m2 := _VEH_FL.search(q)): raw = int(m2.group(1))
    elif (m3 := _VEH_BR.match(q)):
        lt = _gm(s, "lt") or ""
        if lt in {"pricing", "plan_recommendation", "vehicles", "cost"}:
            raw = int(m3.group(1))
    if raw is None: return None
    return -1 if raw > MAX_FLEET else raw


def _plan_answer(n, p):
    p.fleet = n
    if n <= 50:
        name, price, lim = "Essential", "£99",  "50"
    elif n <= 100:
        name, price, lim = "Core",     "£199", "100"
    else:
        name, price, lim = "Elite",    "£499", "unlimited"
    return (f"With {n} vehicles, the {name} plan at {price} per month is the right fit — "
            f"covers up to {lim} vehicles with everything included and nothing locked away. "
            f"Want me to walk you through what is included?")


# ─────────────────────────────────────────────────────────────────────────────
# System prompt — client's tone, warm, short, precise
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM = """You are Nova, the AI assistant for Fine Flow — a UK fleet fine management platform.

Fine Flow's mission: Turning penalties into progress.
Core promise: Cut admin time by up to 80% and never miss a penalty deadline again.

YOUR PERSONALITY:
You are warm, direct and knowledgeable — like a helpful colleague who genuinely cares. Not a robot. Not a sales script. You talk like a person.

RULES YOU MUST FOLLOW:

1. SHORT ANSWERS ONLY
2 to 3 sentences maximum for most questions. Never write long paragraphs or lists. If the user wants more detail they will ask.

2. WARM AND NATURAL TONE
Talk like a person. Never start with "Certainly!", "Great question!", "Of course!", "Absolutely!". Never sound robotic. Use natural conversational English.

3. ASK FOLLOW-UP QUESTIONS
After answering, ask one short relevant question to keep the conversation going — but only when it makes sense. Not every response needs a question. Good examples:
"How many vehicles are in your fleet?"
"What does your current process look like?"
"Is there a particular stage causing the most headaches?"
"Have you had fines escalate because of missed deadlines?"

4. TOPIC
You can answer questions about Fine Flow AND general UK fleet fine questions — types of UK fines, how PCNs work, appeal rights, UK traffic authorities, DVLA, TfL, etc. This is all relevant to Fine Flow's users.
For anything completely unrelated (coding, recipes, general AI questions) say: "I'm here to help with fleet fine management — is there anything about fines, Fine Flow or appeals I can help with?"

5. PAYMENT — ABSOLUTE RULE
Fine Flow does NOT automatically pay fines. It does NOT log into council websites. Say NO clearly when asked. Payment is always done by the user on the authority's site.

6. CARD DETAILS
Fine Flow never stores card details. Say this first when asked.

7. USE CONTEXT
Use the Fine Flow knowledge provided to answer accurately. If you genuinely don't know something specific about Fine Flow say: "I don't have that specific detail — the team at ff.sales@fineflow.com can help."

8. NO HOLLOW ENDINGS
Never end with "feel free to ask", "don't hesitate", "please let me know". Either ask a real relevant question or simply stop.

9. USE THE CUSTOMER'S OWN INFORMATION
If you know their fleet size, volume or problems — use it naturally in your answer. Make them feel heard and understood.

FINE FLOW PRICING (memorise this — never get it wrong):
Essential: £99/month — up to 50 vehicles
Core: £199/month — up to 100 vehicles
Elite: £499/month — 100+ vehicles, unlimited
Per fine within allowance: £0.75
Overage: £2.50 per fine
Pay-as-you-go (no subscription): £2.75 per fine
All plans have identical features. No £2.00 fee exists.

CONTACT:
Phone: +47 32 28 50 00
Email: ff.sales@fineflow.com

EXAMPLE RESPONSES (copy this tone exactly):

User: what does Fine Flow do
Nova: Fine Flow takes the entire headache of managing fleet fines off your plate. It automatically picks up fines from your inbox, figures out which driver was responsible, and keeps everything tracked and organised right through to resolution — whether that's payment, dispute or appeal. Want me to walk you through how it works step by step?

User: will it pay my fines for me
Nova: No — and that's intentional. Authority payment portals have bot detection and card verification that make automation impossible. What Fine Flow does is organise everything so when you do pay, it takes seconds rather than hours of chasing. Is that the kind of process pain you're currently dealing with?

User: how much does it cost
Nova: It's based on fleet size. Essential is £99/month for up to 50 vehicles, Core is £199 for up to 100, and Elite is £499 for anything bigger. Every fine within your allowance is £0.75, and there's also a pay-as-you-go option at £2.75 per fine if you don't want a subscription. How many vehicles are you running?

User: we already use spreadsheets
Nova: Spreadsheets get painful pretty quickly when the volume picks up — missed deadlines, wrong assignments, and disputes that fall through the cracks are where the real cost is. Fine Flow eliminates all of that automatically. How many fines are you currently tracking each month?

User: what is html
Nova: I'm here to help with fleet fine management — is there anything about fines, Fine Flow or appeals I can help with?
"""


def _build_sys(p: Profile, mode: str = "") -> str:
    parts = [_SYSTEM]
    summary = p.summary()
    if summary:
        parts.append(f"\nWHAT I KNOW ABOUT THIS CUSTOMER:\n{summary}\nUse this naturally — make them feel understood, don't just repeat it back.")
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
            json={
                "model": OPENAI_MODEL,
                "messages": msgs,
                "temperature": 0.7,      # warm and natural
                "max_tokens": max_tok,
            },
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
    if extra: parts.append(f"Additional instruction: {extra}")
    parts.append(f"User: {query}")
    m.append({"role": "user", "content": "\n\n".join(parts)})
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Affirmative handler
# ─────────────────────────────────────────────────────────────────────────────

_CLOSE = "To get started, give the Fine Flow team a call on +47 32 28 50 00 or drop them an email at ff.sales@fineflow.com — they'll have you set up quickly."

_EXPAND = {
    "pricing":             "Explain Fine Flow pricing warmly in 2-3 sentences. Ask how many vehicles if unknown.",
    "plan_recommendation": "Explain what every Fine Flow plan includes in 2-3 sentences. Offer to connect with the sales team.",
    "appeals":             "Explain the Fine Flow appeal process in 2-3 sentences.",
    "driver_mgmt":         "Explain how drivers are added and matched to fines in 2-3 sentences.",
    "referral":            "Explain the referral programme in 2-3 sentences.",
    "security":            "Explain Fine Flow's security and GDPR approach in 2-3 sentences.",
    "billing":             "Explain how billing works in 2-3 sentences.",
    "dashboard":           "Explain what the company dashboard shows in 2-3 sentences.",
    "savings":             "Give savings figures relevant to their fleet size if known, in 2-3 sentences.",
    "email":               "Explain Gmail monitoring and fine capture in 2-3 sentences.",
    "overdue":             "Explain how Fine Flow handles overdue fines in 2-3 sentences.",
    "matching":            "Explain how fines are matched to drivers in 2-3 sentences.",
    "sign_up":             "Tell them warmly to call +47 32 28 50 00 or email ff.sales@fineflow.com.",
}


def _aff_response(s, hist, p):
    cnt = _inc_aff(s)
    lt  = _gm(s, "lt") or ""

    if cnt >= 2 and lt in ("plan_recommendation", "sign_up", "pricing"):
        _rst_aff(s); return _CLOSE

    prompt = _EXPAND.get(lt, "Expand naturally on the most recent Fine Flow topic in 2-3 sentences. Ask a follow-up if it makes sense.")
    ctx    = _rag(lt.replace("_", " ")) if lt else ""
    m      = [{"role": "system", "content": _build_sys(p)}]
    m.extend(hist[-8:])
    parts  = []
    if ctx: parts.append(f"Fine Flow knowledge base:\n{ctx}")
    parts.append(f"Instruction: {prompt}")
    m.append({"role": "user", "content": "\n\n".join(parts)})
    return _ai(m, 150) or "What would you like to know more about?"


# ─────────────────────────────────────────────────────────────────────────────
# Topic detection
# ─────────────────────────────────────────────────────────────────────────────

_TMAP = {
    "pric": "pricing",     "cost": "pricing",     "plan": "pricing",
    "£":    "pricing",     "vehicle": "pricing",  "fleet": "pricing",
    "appeal": "appeals",   "dispute": "appeals",
    "driver": "driver_mgmt",
    "referral": "referral","refer": "referral",   "discount": "referral",
    "security": "security","gdpr": "security",    "card": "security",
    "billing": "billing",  "stripe": "billing",
    "dashboard": "dashboard",
    "gmail": "email",      "email": "email",      "inbox": "email",
    "save": "savings",     "admin time": "savings",
    "overdue": "overdue",  "deadline": "overdue",
    "match": "matching",   "assign": "matching",
    "sign": "sign_up",     "start": "sign_up",
}


def _topic(t):
    t = t.lower()
    for k, v in _TMAP.items():
        if k in t: return v
    return None


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

    # ── Tier 1: Deterministic ──────────────────────────────────────────────

    if nq in _GREET:
        _rst(session_id)
        return {"answer": "I'm Nova. Ask me anything — I'll help you manage fines, resolve issues, and keep everything moving.", "confidence": 1.0}

    if nq in _SOC:
        return {"answer": "Doing well, thanks for asking! What can I help you with today — pricing, fines, appeals?", "confidence": 1.0}

    if nq in _ID:
        return {"answer": "I'm Nova, Fine Flow's AI assistant. I can help you with anything about managing fleet fines — pricing, appeals, how the platform works, you name it. What would you like to know?", "confidence": 1.0}

    if nq in _THX:
        _rst_aff(session_id)
        return {"answer": "Happy to help! Anything else you'd like to know about Fine Flow?", "confidence": 1.0}

    if nq in _BYE:
        return {"answer": "Good luck with your fleet management. Come back any time!", "confidence": 1.0}

    if nq in _NEG:
        _push(session_id, "user", query); _rst_aff(session_id)
        a = "No worries at all. Anything else I can help with?"
        _push(session_id, "assistant", a)
        return {"answer": a, "confidence": 1.0}

    if any(r in nq for r in _RUDE):
        return {"answer": "Let me try again — what would you like to know about Fine Flow?", "confidence": 1.0}

    if nq in _FILL:
        _rst_aff(session_id)
        return {"answer": "Is there anything about Fine Flow I can help you with today?", "confidence": 1.0}

    # Garbled / too short — handle gracefully
    words = [w for w in nq.split() if len(w) > 1]
    if len(words) < 2 and nq not in _AFF:
        return {"answer": "What would you like to know about Fine Flow? I can help with fines, pricing, appeals or how the platform works.", "confidence": 1.0}

    # Off-topic — but allow UK traffic / fine general knowledge
    if _is_ot(query):
        a = "I'm here to help with fleet fine management — is there anything about fines, Fine Flow or appeals I can help with?"
        _push(session_id, "user", query); _push(session_id, "assistant", a)
        return {"answer": a, "confidence": 1.0}

    # Vehicle count
    vc = _get_vc(query, session_id)
    if vc == -1:
        a = "That number doesn't look right — could you double check? How many vehicles are in your fleet?"
        _push(session_id, "user", query); _push(session_id, "assistant", a)
        return {"answer": a, "confidence": 1.0}
    if vc is not None:
        _rst_aff(session_id)
        a = _plan_answer(vc, p)
        _push(session_id, "user", query); _push(session_id, "assistant", a)
        _sm(session_id, "lt", "plan_recommendation")
        return {"answer": a, "confidence": 1.0}

    # Purchase intent
    if _PURCH.search(query):
        _rst_aff(session_id)
        sfx = "" if p.fleet else " How many vehicles are you running so I can point you to the right plan?"
        a   = f"To get started, just give the team a call on +47 32 28 50 00 or email ff.sales@fineflow.com — they'll get you set up quickly.{sfx}"
        _push(session_id, "user", query); _push(session_id, "assistant", a)
        _sm(session_id, "lt", "sign_up")
        return {"answer": a.strip(), "confidence": 1.0}

    # Affirmative
    if nq in _AFF:
        _push(session_id, "user", query)
        a = _clean(_aff_response(session_id, _hist(session_id)[:-1], p))
        _push(session_id, "assistant", a)
        return {"answer": a, "confidence": 1.0}

    # ── Tier 2: RAG + GPT-4o ──────────────────────────────────────────────

    _rst_aff(session_id)
    _push(session_id, "user", query)

    mode  = ""
    extra = ""

    if _CONV.search(query):
        mode  = "PERSUADE"
        extra = ("Use their specific fleet data and problems to make a tailored, specific case. No generic marketing. Be honest and direct."
                 if p.fleet or p.volume or p.issues
                 else "Ask about their fleet size and fine volume first so you can give a tailored answer.")
    elif _OBJ.search(query):
        mode  = "SUPPORT"
        extra = "Acknowledge their point warmly first, then reframe using their situation. 2-3 sentences."

    if not _ask_now(session_id) and not extra:
        extra = "Do NOT end with a question this time. Make your point and stop naturally."

    ctx  = _rag(query)
    hist = _hist(session_id)
    ans  = _ai(_make_msgs(query, ctx, hist[:-1], p, mode, extra), 150)

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