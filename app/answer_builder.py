# app/answer_builder.py
"""
FineFlow Nova — Final Production Version
=========================================
Architecture: Deterministic layer → RAG → GPT-4o
Single TXT knowledge file. No JSON KB.
Temperature 0.7. Max 150 tokens.

Memory fixes:
- Bare numbers (35, 50) are recognised in context and stored in profile
- Counter-questions (user says "no" to a question) ask a clarifying follow-up
- Profile is injected into every GPT call so answers are always personalised
- Last question asked is tracked so "no" triggers a relevant follow-up
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
    fleet:    Optional[int]  = None   # vehicle count
    volume:   Optional[int]  = None   # monthly fines
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

    def plan(self) -> str:
        if not self.fleet or self.fleet > MAX_FLEET:
            return ""
        if self.fleet <= 50:   return "Essential (£99/month)"
        if self.fleet <= 100:  return "Core (£199/month)"
        if self.fleet <= 200:  return "Advanced (£399/month)"
        return "Elite (£499/month)"


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
    """Ask a question on every other response to avoid interrogation fatigue."""
    with _LK:
        m = _MET.setdefault(s, {})
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
        "don't hesitate to ask.", "please let me know if you need anything.",
        "if you have any other questions, feel free to ask.",
    ]:
        if text.lower().endswith(bad.lower()):
            text = text[:-len(bad)].rstrip(" ,.")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Profile extraction from every user message
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

# Context hints — what kind of number makes sense right now
_FINES_CONTEXT = {
    "how many fines", "fines per month", "fines a month",
    "monthly fines", "how many fines do you", "how many fines a",
    "deal with each month", "typically deal with",
}
_VEHICLE_CONTEXT = {
    "how many vehicles", "fleet size", "vehicles do you", "vehicles are in",
    "how big is your fleet", "how many vans", "how many trucks",
}


def _upd(s, q):
    p = _pro(s)
    p.turns += 1

    # Name
    m = _NAME_RE.search(q)
    if m and not p.name:
        p.name = m.group(1)

    # Fine volume from explicit mention
    m = _FINES_RE.search(q)
    if m and p.volume is None:
        v = int(m.group(1))
        if v < 10_000: p.volume = v

    # Industry
    m = _IND_RE.search(q)
    if m: p.industry = m.group(1).lower()

    # Issues
    for pat, lbl in _ISS:
        if pat.search(q) and lbl not in p.issues: p.issues.append(lbl)


def _resolve_bare_number(n: int, s: str) -> Optional[str]:
    """
    When user sends a bare number, figure out what it means from context.
    Returns 'vehicle', 'fines', or None.
    """
    last_q = (_gm(s, "last_nova_question") or "").lower()
    last_t = (_gm(s, "lt") or "").lower()

    # Check last question asked
    for hint in _FINES_CONTEXT:
        if hint in last_q: return "fines"
    for hint in _VEHICLE_CONTEXT:
        if hint in last_q: return "vehicle"

    # Check last topic
    if last_t in {"pricing", "plan_recommendation", "vehicles", "cost"}:
        return "vehicle"
    if last_t in {"fines_volume", "savings"}:
        return "fines"

    # Default: if small number and no context, ask (return None)
    return None


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
    "yes walk me through","yes please explain",
}
_THX = {
    "thanks","thank you","thank u","cheers","that helps","that helped","ta",
    "okay thanks","ok thanks","great thanks","perfect","brilliant","nice one",
    "lovely","great","awesome","wonderful","thank you so much","many thanks",
}
_BYE = {"bye","goodbye","see you","see ya","later","take care","good bye","cya","farewell","cheerio"}
_RUDE = {"stupid","idiot","useless","rubbish","garbage","terrible","you suck","dumb bot","waste of time"}
_FILL = {
    "ok","okay","right","alright","cool","nice","interesting","really",
    "hmm","hm","ah","oh","i see","got it","understood","makes sense",
    "noted","wow","woah","omg","anything","something","whatever",
}

# Fine Flow related — never treated as off-topic
_FF_OK = re.compile(
    r"\b(council|authority|fine|pcn|penalty|fineflow|fine flow|appeal|dispute|"
    r"driver|fleet|vehicle|overage|allowance|billing|subscription|payment|"
    r"uk traffic|traffic violation|parking|bus lane|congestion|emission|"
    r"dvla|tfl|fixed penalty|notice to owner|mount gmail|connect gmail|"
    r"gmail|inbox|driver log|csv|upload|dashboard|referral|credits|stripe)\b", re.I)

_OT = [
    re.compile(r"\b(html|css|javascript|typescript|python|java|php|sql|react|angular|vue|node\.?js|django|flask|docker|kubernetes|github|coding|programming|teach me|how to code|explain python)\b", re.I),
    re.compile(r"\b(machine learning|deep learning|neural network|large language model|generative ai|llm|bert)\b", re.I),
    re.compile(r"\b(recipe|cooking|restaurant|pizza|burger|sandwich|coffee|tea|cake|meal|make me a food|bake me|cook me for)\b", re.I),
    re.compile(r"\b(movie|film|song|lyrics|music|football match|cricket match|weather forecast|todays news|politics|history lesson|capital city|who invented|tell me a joke|write me a poem)\b", re.I),
    re.compile(r"\b(write an essay|translate this|proofread my|write my cv|write a story for me)\b", re.I),
    re.compile(r"\b(chatgpt|openai|gemini|claude ai|anthropic|google bard|bing ai|alexa|siri)\b", re.I),
]


def _is_ot(q): return not _FF_OK.search(q) and any(p.search(q) for p in _OT)


_VEH_EX = re.compile(r"\b(\d+)\s*(vehicle|vehicles|van|vans|truck|trucks|car|cars|lorry|lorries|in my fleet|in our fleet)\b", re.I)
_VEH_FL = re.compile(r"\b(?:fleet of|manage|running|operate|run)\s+(\d+)\b", re.I)
_VEH_BR = re.compile(r"^\s*(\d+)\s*$")
_DRV_CT = re.compile(r"\b(driver|drivers|staff|employee|employees|people|worker|team|members)\b", re.I)
_PURCH  = re.compile(r"\b(want to buy|want to subscribe|want to sign up|how do i get started|how do i sign up|get started|free trial|sign me up|book a demo|talk to sales|how to start|where do i sign|how do i join)\b", re.I)
_CONV   = re.compile(r"\b(convince|persuade|sell me|why should i|why buy|is it worth|should i buy|worth it|why choose fineflow|why fine flow)\b", re.I)
_OBJ    = re.compile(r"\b(expensive|too much|too costly|already use spreadsheet|we manage manually|we handle fines ourselves)\b", re.I)
_NEG_RE = re.compile(r"^(no|nope|nah|no thanks|not now|not really|no thank you|nah thanks|not sure)$")


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
    if n <= 50:   name, price, lim = "Essential", "£99",  "50"
    elif n <= 100: name, price, lim = "Core",     "£199", "100"
    elif n <= 200: name, price, lim = "Advanced", "£399", "200"
    else:          name, price, lim = "Elite",    "£499", "unlimited"
    return (f"With {n} vehicles, the {name} plan at {price} per month is the right fit — "
            f"covers up to {lim} vehicles with everything included and nothing locked away. "
            f"Want me to walk you through what is included?")


# ─────────────────────────────────────────────────────────────────────────────
# Topic map
# ─────────────────────────────────────────────────────────────────────────────

_TMAP = {
    "pric":"pricing","cost":"pricing","plan":"pricing","£":"pricing",
    "vehicle":"pricing","fleet":"pricing","package":"pricing",
    "fines per month":"fines_volume","fines a month":"fines_volume",
    "how many fines":"fines_volume","monthly fines":"fines_volume",
    "appeal":"appeals","dispute":"appeals","driver":"driver_mgmt",
    "referral":"referral","refer":"referral","discount":"referral",
    "security":"security","gdpr":"security","card":"security",
    "billing":"billing","stripe":"billing","when am i billed":"billing",
    "dashboard":"dashboard","gmail":"email","inbox":"email",
    "save":"savings","admin time":"savings","overdue":"overdue","deadline":"overdue",
    "sign":"sign_up","get started":"sign_up",
}


def _topic(t):
    t = t.lower()
    for k, v in _TMAP.items():
        if k in t: return v
    return None


# ─────────────────────────────────────────────────────────────────────────────
# System prompt — enforces client's exact wording
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM = """You are Nova, the AI assistant for Fine Flow — a UK fleet fine management platform.

Fine Flow's mission: Turning penalties into progress.
Core promise: Cut admin time by up to 80% and never miss a penalty deadline again.
Fine Flow provides 24/7 fleet fine tracking, management and compliance in one place.

PERSONALITY:
You are warm, helpful and direct — like a colleague who genuinely cares. Not a robot. Not a sales pitch machine. You talk like a real person.

CRITICAL RULES:

1. SHORT ANSWERS
2 to 3 sentences maximum. No bullet lists unless explaining a multi-step process the user asked about. No long paragraphs.

2. CLIENT'S EXACT WORDING
When answering about what Fine Flow is, ALWAYS include these phrases:
- "Fine Flow is an automated system for managing fines from start to finish"
- "keeps the entire process organised, accountable, and under control"
- "cut admin time by up to 80%"
- "never miss a penalty deadline"
Use the client's exact approved language. Do not paraphrase or modernise it.

3. MEMORY — USE WHAT YOU KNOW
You have the customer's profile. Always use it. If you know their fleet size, volume or problems — reference it naturally. If they told you they have 35 fines a month, remember that and use it. Never ask for information you already have.

4. COUNTER QUESTIONS
When the user says "no" or challenges your answer, do NOT reset. Instead ask a clarifying follow-up that continues the conversation naturally. For example:
- If you asked "how many fines do you deal with?" and they said "no", ask "What part of the fine management process is causing the most headache for you right now?"
- If they dispute something, acknowledge it and explore what they actually need.

5. ASK FOLLOW-UP QUESTIONS
After most answers, ask one short relevant question to keep the conversation going. Vary the questions — don't repeat the same one. Good examples:
"How many vehicles are in your fleet?"
"How many fines do you typically deal with each month?"
"What does your current process look like for handling fines?"
"Is there a particular stage causing the most issues?"

6. PAYMENT — ABSOLUTE RULE
Fine Flow does NOT automatically pay fines. It does NOT log into council websites. Payment is always done by you on the authority's site. The reason is anti-bot protection, session controls and card verification requirements.

7. CARD DETAILS — SAY THIS FIRST
Fine Flow never stores card details. State this clearly and first when asked about card security.

8. PRICING — MEMORISE THIS EXACTLY
Essential: £99/month — 5 to 50 vehicles
Core: £199/month — 51 to 100 vehicles
Advanced: £399/month — 101 to 200 vehicles
Elite: £499/month — 200+ vehicles (unlimited)
Per fine within allowance: £0.75 (charged on entry to system, not per action)
Overage: £2.50 per fine
Pay-as-you-go: £2.75 per fine, no subscription, no lock-in
All plans identical features. No £2.00 fee exists anywhere.

9. APPEALS — SAY THIS CORRECTLY
The appeal flow is: Driver → DISPUTED → Admin accepts/rejects → If accepted → UNDER REVIEW + appeal letter sent by email to authority. Do NOT say Fine Flow sends appeals via direct API or system integration. Appeals are sent by email.

10. TOPIC SCOPE
You help with Fine Flow questions AND general UK fleet fine questions — PCNs, FPNs, council fines, TfL, DVLA, appeal rights, etc. For anything completely unrelated (coding, recipes, unrelated AI products) say: "I'm here to help with fleet fine management — is there anything about fines, Fine Flow or appeals I can help with?"

11. NO HOLLOW ENDINGS
Never end with "feel free to ask", "don't hesitate", "let me know if you need anything". Ask a real question or stop naturally.

CONTACT:
Phone: +47 32 28 50 00
Email: ff.sales@fineflow.com

EXAMPLE RESPONSES (follow this tone exactly):

User: What is Fine Flow?
Nova: Fine Flow is an automated system for managing fines from start to finish. It captures incoming fines, extracts the key details, assigns responsibility, tracks decisions, supports disputes and can even construct the appeal for you — keeping the entire process organised, accountable, and under control. Want a breakdown of how it all works?

User: How does Fine Flow help fleets?
Nova: Fine Flow cuts admin time by up to 80% and ensures you never miss a penalty deadline again. It replaces the chaos of chasing emails, spreadsheets and manual tracking with a single organised system that handles everything from fine capture to resolution. How many vehicles are in your fleet?

User: Will it pay my fines?
Nova: No — and that's intentional. Government payment portals have bot detection and card verification that make automation impossible, so payment is always done by you on the authority's site. What Fine Flow does is organise everything so by the time you pay, there's no thinking left — just click and complete it.

User: 35 (after being asked how many fines per month)
Nova: Got it — 35 fines a month. The Essential plan at £99 per month would cover you well, and at £0.75 per fine within your allowance that's under £27 in processing costs on top. Want me to run through what's included?

User: no (after being asked about fleet size)
Nova: No worries — what part of fine management is giving you the most trouble right now? I can focus on whatever's most relevant for you.
"""


def _build_sys(p: Profile, mode: str = "") -> str:
    parts = [_SYSTEM]
    s = p.summary()
    if s:
        parts.append(f"\nWHAT I KNOW ABOUT THIS CUSTOMER:\n{s}\nAlways use this information naturally. Never ask for information you already have.")
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
    if extra: parts.append(f"Additional instruction: {extra}")
    parts.append(f"User: {query}")
    m.append({"role": "user", "content": "\n\n".join(parts)})
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Affirmative handler
# ─────────────────────────────────────────────────────────────────────────────

_EXPAND = {
    "pricing":             "Explain Fine Flow pricing in 2-3 sentences using exact pricing figures. Ask how many vehicles if unknown.",
    "plan_recommendation": "Explain what is included in their recommended plan in 2-3 sentences. Ask if they want to get in touch with sales.",
    "appeals":             "Explain the Fine Flow appeal process step by step: Driver disputes → DISPUTED, Admin reviews → accepts/rejects, If accepted → UNDER REVIEW + appeal letter sent by email.",
    "driver_mgmt":         "Explain how drivers are added and matched to fines in 2-3 sentences.",
    "fines_volume":        "Based on their fine volume, recommend the right plan and explain why it fits.",
    "referral":            "Explain the referral programme in 2-3 sentences.",
    "security":            "Explain Fine Flow's GDPR compliance and security in 2-3 sentences.",
    "billing":             "Explain billing timing: monthly via Stripe, overage collected end of cycle, credits reset, cooldown on cancellation.",
    "dashboard":           "Explain what the company dashboard shows in 2-3 sentences.",
    "savings":             "Give specific savings figures relevant to their fleet size and volume if known.",
    "email":               "Explain how to connect Gmail using OAuth or App Password in 2-3 sentences.",
    "overdue":             "Explain how Fine Flow tracks deadlines and marks fines overdue automatically.",
    "sign_up":             "Tell them to call +47 32 28 50 00 or email ff.sales@fineflow.com to get started.",
}
_CLOSE = "The best next step is to get in touch directly — call +47 32 28 50 00 or email ff.sales@fineflow.com and they'll have you up and running quickly."


def _aff_response(s, hist, p):
    cnt = _inc_aff(s)
    lt  = _gm(s, "lt") or ""

    if cnt >= 2 and lt in ("plan_recommendation", "sign_up", "pricing"):
        _rst_aff(s); return _CLOSE

    prompt = _EXPAND.get(lt, "Expand naturally on the most recent Fine Flow topic in 2-3 sentences. Use the customer's profile information if available.")
    ctx    = _rag(lt.replace("_", " ")) if lt else ""
    m      = [{"role": "system", "content": _build_sys(p)}]
    m.extend(hist[-8:])
    parts  = []
    if ctx: parts.append(f"Fine Flow knowledge base:\n{ctx}")
    parts.append(f"Instruction: {prompt}")
    m.append({"role": "user", "content": "\n\n".join(parts)})
    return _ai(m, 150) or "What would you like to know more about?"


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
        return {"answer": "Doing well, thanks! What can I help you with today — pricing, fines, appeals, or something else?", "confidence": 1.0}

    if nq in _ID:
        return {"answer": "I'm Nova, Fine Flow's AI assistant. I can help with anything about managing fleet fines — pricing, appeals, how the platform works, UK fine rules — you name it. What would you like to know?", "confidence": 1.0}

    if nq in _THX:
        _rst_aff(session_id)
        return {"answer": "Happy to help! Anything else you'd like to know?", "confidence": 1.0}

    if nq in _BYE:
        return {"answer": "Good luck with your fleet management. Come back any time!", "confidence": 1.0}

    if any(r in nq for r in _RUDE):
        return {"answer": "Let me try again — what would you like to know about Fine Flow?", "confidence": 1.0}

    if nq in _FILL:
        _rst_aff(session_id)
        return {"answer": "Is there anything about Fine Flow I can help you with?", "confidence": 1.0}

    # Garbled / too short
    words = [w for w in nq.split() if len(w) > 1]
    if len(words) < 2 and nq not in _AFF and not _VEH_BR.match(query.strip()):
        return {"answer": "What would you like to know about Fine Flow? I can help with fines, pricing, appeals or how the platform works.", "confidence": 1.0}

    # Off-topic
    if _is_ot(query):
        a = "I'm here to help with fleet fine management — is there anything about fines, Fine Flow or appeals I can help with?"
        _push(session_id, "user", query); _push(session_id, "assistant", a)
        return {"answer": a, "confidence": 1.0}

    # Negative / "no" — counter-question not reset
    if _NEG_RE.match(nq):
        _push(session_id, "user", query)
        _rst_aff(session_id)
        last_q = (_gm(session_id, "last_nova_question") or "").lower()
        lt     = (_gm(session_id, "lt") or "").lower()

        # Generate a relevant follow-up based on what was last discussed
        ctx  = _rag(lt) if lt else ""
        hist = _hist(session_id)
        extra = (
            f"The user said 'no' to the previous question. "
            f"The last topic was '{lt}'. "
            f"Do NOT reset the conversation. Ask a different, relevant follow-up question "
            f"that continues exploring what they actually need. "
            f"For example if you asked about fleet size and they said no, ask about their "
            f"current fine management process instead."
        )
        msgs = _make_msgs("no", ctx, hist[:-1], p, extra=extra)
        a    = _clean(_ai(msgs, 120) or "No worries — what part of fine management is causing the most headache for you right now?")
        _push(session_id, "assistant", a)
        return {"answer": a, "confidence": 1.0}

    # Explicit vehicle count mention
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

    # Bare number — resolve by context
    m = _VEH_BR.match(query.strip())
    if m:
        n   = int(m.group())
        ctx_type = _resolve_bare_number(n, session_id)
        if ctx_type == "vehicle" and 0 < n <= MAX_FLEET:
            _rst_aff(session_id)
            a = _plan_answer(n, p)
            _push(session_id, "user", query); _push(session_id, "assistant", a)
            _sm(session_id, "lt", "plan_recommendation")
            return {"answer": a, "confidence": 1.0}
        elif ctx_type == "fines" and n < 10_000:
            p.volume = n
            _rst_aff(session_id)
            # Recommend based on both fleet and volume if available
            if p.fleet:
                plan = p.plan()
                a = (f"Got it — {n} fines a month. With your fleet of {p.fleet} vehicles on the {plan}, "
                     f"that works out to £{round(n * 0.75, 2):.2f} in processing costs within your allowance. "
                     f"Want me to walk you through everything that's included?")
            else:
                a = (f"Got it — {n} fines a month. That's a manageable volume and Fine Flow would handle it cleanly. "
                     f"How many vehicles are in your fleet so I can point you to the right plan?")
            _push(session_id, "user", query); _push(session_id, "assistant", a)
            _sm(session_id, "lt", "fines_volume")
            return {"answer": a, "confidence": 1.0}
        else:
            # Can't resolve — ask what they meant
            a = "Just to make sure I give you the right info — is that the number of vehicles in your fleet, or your monthly fine volume?"
            _push(session_id, "user", query); _push(session_id, "assistant", a)
            return {"answer": a, "confidence": 1.0}

    # Purchase intent
    if _PURCH.search(query):
        _rst_aff(session_id)
        sfx = "" if p.fleet else " How many vehicles are you running so I can point you to the right plan?"
        a   = f"To get started, just call the team on +47 32 28 50 00 or email ff.sales@fineflow.com — they'll get you set up quickly.{sfx}"
        _push(session_id, "user", query); _push(session_id, "assistant", a)
        _sm(session_id, "lt", "sign_up")
        return {"answer": a.strip(), "confidence": 1.0}

    # Affirmative
    if nq in _AFF:
        _push(session_id, "user", query)
        a = _clean(_aff_response(session_id, _hist(session_id)[:-1], p))
        _push(session_id, "assistant", a)
        _sm(session_id, "last_nova_question", "")
        return {"answer": a, "confidence": 1.0}

    # ── Tier 2: RAG + GPT-4o ──────────────────────────────────────────────

    _rst_aff(session_id)
    _push(session_id, "user", query)

    mode  = ""
    extra = ""

    if _CONV.search(query):
        mode  = "PERSUADE"
        extra = ("Use the customer's exact fleet size, volume and problems to make a specific case. No generic copy. Reference their numbers directly."
                 if p.fleet or p.volume or p.issues
                 else "Ask about their fleet size and monthly fine volume first — you need their situation to make a relevant case.")
    elif _OBJ.search(query):
        mode  = "SUPPORT"
        extra = "Acknowledge their point warmly first, then reframe with their specific situation. 2-3 sentences max."

    if not _ask_now(session_id) and not extra:
        extra = "Do NOT end with a question this time. Make your point clearly and stop."

    ctx  = _rag(query)
    hist = _hist(session_id)
    ans  = _ai(_make_msgs(query, ctx, hist[:-1], p, mode, extra), 150)

    if not ans:
        ans = "The Fine Flow team can help with that directly — call +47 32 28 50 00 or email ff.sales@fineflow.com."

    ans = _clean(ans)
    _push(session_id, "assistant", ans)

    # Store last question asked (for counter-question handling on "no")
    if "?" in ans:
        _sm(session_id, "last_nova_question", ans)

    t = _topic(query) or _topic(ans)
    if t: _sm(session_id, "lt", t)

    return {"answer": ans, "confidence": 0.9 if ctx else 0.5}


def answer_sync(q: str, session_id: str = "default") -> Dict[str, Any]:
    try:
        return build_response(q, session_id)
    except Exception:
        logger.exception("Crash")
        return {"answer": "Something went wrong. Please try again.", "confidence": 0.0}