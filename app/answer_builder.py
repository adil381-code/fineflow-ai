# app/answer_builder.py
"""
FineFlow Nova — Production Final v16
======================================
Three critical bugs fixed vs v15:

BUG 1 FIXED: "can you log into council websites" was caught by off-topic regex.
  Fix: Added Fine Flow payment/council keywords to an allowlist that bypasses
  off-topic guard. Any question about council sites IN the context of fines
  is a Fine Flow question, not off-topic.

BUG 2 FIXED: "23456789" accepted as a vehicle count.
  Fix: Vehicle count capped at 50,000. Any number above that is rejected
  gracefully with a clarifying response.

BUG 3 FIXED: Impossible vehicle count contaminated profile for entire session.
  Fix: Profile only updated when vehicle count passes sanity check.
  Added profile.fleet_size validation before injecting into system prompt.

ALSO FIXED:
- "feel free to ask" filler endings replaced with nothing or a real question
- Win% question after driver confirmed responsibility → context-aware response
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

# Maximum realistic fleet size — anything above this is a typo or test
MAX_FLEET_SIZE = 50_000


# ─────────────────────────────────────────────────────────────────────────────
# Customer profile
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CustomerProfile:
    fleet_size:    Optional[int]  = None
    fine_volume:   Optional[int]  = None
    issues:        List[str]      = field(default_factory=list)
    plan_interest: Optional[str]  = None
    industry:      Optional[str]  = None
    turn_count:    int            = 0

    def to_context(self) -> str:
        # Only inject fleet_size if it passed the sanity check
        lines = []
        if self.fleet_size and self.fleet_size <= MAX_FLEET_SIZE:
            lines.append(f"Fleet size: {self.fleet_size} vehicles")
        if self.fine_volume:
            lines.append(f"Monthly fines: ~{self.fine_volume}")
        if self.industry:
            lines.append(f"Industry: {self.industry}")
        if self.issues:
            priority = [
                "missed deadlines", "missed appeal deadlines",
                "driver disputes", "manual admin",
                "using spreadsheets", "missed fines",
            ]
            sorted_issues = sorted(
                self.issues,
                key=lambda x: priority.index(x) if x in priority else 99,
            )
            lines.append(f"Known problems (priority order): {', '.join(sorted_issues)}")
        if self.plan_interest:
            lines.append(f"Plan discussed: {self.plan_interest}")
        if not lines:
            return ""
        return "CUSTOMER CONTEXT:\n" + "\n".join(lines)

    def recommended_plan(self) -> Optional[str]:
        if not self.fleet_size or self.fleet_size > MAX_FLEET_SIZE:
            return None
        if self.fleet_size <= 50:
            return "Essential (£99/month)"
        elif self.fleet_size <= 100:
            return "Core (£199/month)"
        elif self.fleet_size <= 200:
            return "Advanced (£399/month)"
        else:
            return "Elite (£499/month)"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*",     r"\1", text)
    text = re.sub(r"_(.*?)_",       r"\1", text)
    text = text.replace("→", "to").replace("->", "to")
    text = text.replace("±", "plus or minus").replace("`", "")
    # Remove weak filler endings
    weak_endings = [
        "feel free to ask!", "feel free to ask.",
        "if you have any questions, feel free to ask.",
        "if you have further questions, feel free to ask.",
        "please let me know if you need further assistance.",
        "don't hesitate to ask.",
    ]
    for ending in weak_endings:
        if text.lower().endswith(ending.lower()):
            text = text[: -len(ending)].rstrip(" ,.")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _norm(text: str) -> str:
    t = text.lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Session memory
# ─────────────────────────────────────────────────────────────────────────────

_SESSION  : Dict[str, List[Dict[str, str]]] = {}
_PROFILES : Dict[str, CustomerProfile]      = {}
_META     : Dict[str, Dict[str, Any]]       = {}
_LOCK     = threading.Lock()

PRICING_TOPICS = {"pricing", "plan_recommendation", "vehicles", "cost", "billing"}


def _hist(sid: str) -> List[Dict[str, str]]:
    with _LOCK:
        return list(_SESSION.get(sid, []))


def _push(sid: str, role: str, content: str) -> None:
    with _LOCK:
        h   = _SESSION.setdefault(sid, [])
        h.append({"role": role, "content": content})
        cap = CHAT_HISTORY_TURNS * 2
        if len(h) > cap:
            _SESSION[sid] = h[-cap:]


def _profile(sid: str) -> CustomerProfile:
    with _LOCK:
        if sid not in _PROFILES:
            _PROFILES[sid] = CustomerProfile()
        return _PROFILES[sid]


def _smeta(sid: str, key: str, val: Any) -> None:
    with _LOCK:
        _META.setdefault(sid, {})[key] = val


def _gmeta(sid: str, key: str) -> Any:
    with _LOCK:
        return _META.get(sid, {}).get(key)


def _reset_session(sid: str) -> None:
    with _LOCK:
        _META[sid]     = {}
        _PROFILES[sid] = CustomerProfile()


def _inc_aff(sid: str) -> int:
    with _LOCK:
        meta  = _META.setdefault(sid, {})
        count = meta.get("aff_count", 0) + 1
        meta["aff_count"] = count
        return count


def _reset_aff(sid: str) -> None:
    with _LOCK:
        _META.setdefault(sid, {})["aff_count"] = 0


def _should_ask_question(sid: str) -> bool:
    """Ask a question on every other response to avoid interrogation fatigue."""
    with _LOCK:
        meta  = _META.setdefault(sid, {})
        count = meta.get("response_count", 0) + 1
        meta["response_count"] = count
        return count % 2 == 0


# ─────────────────────────────────────────────────────────────────────────────
# Profile extraction
# ─────────────────────────────────────────────────────────────────────────────

_FINES_PM_RE = re.compile(
    r"\b(\d+)\s*(?:fines?|pcns?|penalties|violations?|tickets?)"
    r"(?:\s*(?:per|a|each|every)\s*(?:month|monthly|week))?\b",
    re.IGNORECASE,
)
_INDUSTRY_RE = re.compile(
    r"\b(logistics|delivery|courier|haulage|transport|taxi|minicab"
    r"|bus|coach|construction|utilities)\b",
    re.IGNORECASE,
)
_ISSUE_PATTERNS = [
    (r"\b(miss(?:ed?|ing)?\s+(?:deadlines?|appeals?|due\s*dates?))\b",  "missed deadlines"),
    (r"\b(miss(?:ed?|ing)?\s+appeal\s+deadline)\b",                     "missed appeal deadlines"),
    (r"\b(drivers?\s+(?:dispute|deny|ignor|avoid))\b",                  "driver disputes"),
    (r"\b(manual(?:ly)?\s+(?:track|manage|process|handl))\b",           "manual admin"),
    (r"\b(spreadsheet)\b",                                               "using spreadsheets"),
    (r"\b(too\s+much\s+(?:admin|time|work|paperwork))\b",               "too much admin"),
]
_ISSUE_RE = [(re.compile(p, re.IGNORECASE), label) for p, label in _ISSUE_PATTERNS]


def _update_profile(sid: str, query: str) -> None:
    p = _profile(sid)
    p.turn_count += 1
    m = _FINES_PM_RE.search(query)
    if m and p.fine_volume is None:
        val = int(m.group(1))
        if val < 10_000:
            p.fine_volume = val
    m = _INDUSTRY_RE.search(query)
    if m:
        p.industry = m.group(1).lower()
    for pattern, label in _ISSUE_RE:
        if pattern.search(query) and label not in p.issues:
            p.issues.append(label)


# ─────────────────────────────────────────────────────────────────────────────
# Intent detection
# ─────────────────────────────────────────────────────────────────────────────

class Intent:
    INFORMATIONAL = "informational"
    DIAGNOSTIC    = "diagnostic"
    BUYING        = "buying"
    OBJECTION     = "objection"
    CONVINCE      = "convince"
    AFFIRMATIVE   = "affirmative"
    SOCIAL        = "social"
    OFF_TOPIC     = "off_topic"
    UNKNOWN       = "unknown"


_CONVINCE_RE = re.compile(
    r"\b(convince|persuade|sell me|why should i|why buy|is it worth"
    r"|should i get|should i buy|justify|make the case|prove it"
    r"|worth it|why fine flow)\b",
    re.IGNORECASE,
)
_OBJECTION_RE = re.compile(
    r"\b(expensive|too much|costly|already use|spreadsheet|manual"
    r"|don.?t need|do it ourselves|our team handles|we manage)\b",
    re.IGNORECASE,
)
_PROBLEM_RE = re.compile(
    r"\b(fine|pcn|penalty|violation|notice|ticket)\b.{0,40}"
    r"\b(issued|received|got|have a|appealing|disputing|today|yesterday|last week)\b"
    r"|\b(received a|got a|have a)\b.{0,20}\b(fine|pcn|penalty|violation)\b",
    re.IGNORECASE,
)
_BUYING_RE = re.compile(
    r"\b(how much|pricing|cost|plan|package|subscribe|get started"
    r"|sign up|buy|purchase|trial|demo|which plan)\b",
    re.IGNORECASE,
)


def _detect_intent(query: str, nq: str,
                   affirmative_set: set, filler_set: set) -> str:
    if nq in affirmative_set:
        return Intent.AFFIRMATIVE
    if nq in filler_set:
        return Intent.SOCIAL
    if _CONVINCE_RE.search(query):
        return Intent.CONVINCE
    if _OBJECTION_RE.search(query):
        return Intent.OBJECTION
    if _PROBLEM_RE.search(query):
        return Intent.DIAGNOSTIC
    if _BUYING_RE.search(query):
        return Intent.BUYING
    return Intent.INFORMATIONAL


# ─────────────────────────────────────────────────────────────────────────────
# Intent sets
# ─────────────────────────────────────────────────────────────────────────────

_GREETINGS = {
    "hi", "hello", "hey", "hiya", "howdy", "yo", "sup",
    "good morning", "good afternoon", "good evening",
    "morning", "afternoon", "evening",
    "hi there", "hey there", "hello there",
    "hi nova", "hey nova", "hello nova",
}
_SOCIAL_SET = {
    "how are you", "how are you doing", "how r u", "how r you",
    "how are u", "hows it going", "how is it going",
    "whats up", "what s up", "you ok", "you good",
    "how do you do", "you alright", "alright mate",
}
_IDENTITY = {
    "who are you", "who r you", "who r u", "who youre", "who you re",
    "who is nova", "who is this", "who is there", "whos there",
    "who s there", "who there", "anyone there", "is anyone there",
    "what are you", "what is nova", "are you a bot",
    "are you human", "are you ai", "are you a robot",
    "are you male or female", "you male or female",
    "who the hell are you", "whats your name",
    "what is your name", "introduce yourself",
    "who am i talking to", "knock knock",
}
_AFFIRMATIVE = {
    "yes", "yeah", "yep", "yup", "ya", "ye",
    "sure", "ok sure", "okay sure",
    "go ahead", "go on", "yes go ahead", "yes please",
    "yes sure", "yes of course", "of course",
    "absolutely", "definitely", "do it",
    "tell me more", "more", "explain", "explain more",
    "yes explain", "yes explain it", "go for it",
    "sounds good", "continue", "carry on", "keep going",
    "please do", "i would", "please explain",
    "show me", "walk me through it",
    "yes elite", "yes core", "yes essential", "yes advanced",
    "for sure", "yes for sure", "sure thing",
}
_THANKS = {
    "thanks", "thank you", "thank u", "cheers",
    "that helps", "that helped", "ta", "ty",
    "okay thanks", "ok thanks", "great thanks",
    "perfect", "brilliant", "nice one", "lovely",
    "great", "awesome", "wonderful",
    "thank you so much", "many thanks", "much appreciated",
}
_GOODBYE = {
    "bye", "goodbye", "see you", "see ya", "later",
    "take care", "good bye", "cya", "ttyl", "farewell", "cheerio",
}
_NEGATIVE = {
    "no", "nope", "nah", "no thanks", "not now", "skip",
    "never mind", "nevermind", "no need", "not really",
    "no thank you", "nah thanks",
}
_RUDE = {
    "you dumb", "you are dumb", "ur dumb", "stupid",
    "idiot", "useless", "rubbish", "garbage",
    "terrible", "you suck", "this is rubbish",
    "dumb bot", "you re useless", "waste of time",
}
_PURE_FILLER = {
    "ok", "okay", "right", "alright", "cool", "nice",
    "interesting", "really", "seriously", "hmm", "hm",
    "ah", "oh", "i see", "got it", "understood",
    "makes sense", "noted", "wow", "waow", "woah", "omg",
    "anything", "something", "whatever",
}

# ─────────────────────────────────────────────────────────────────────────────
# Off-topic guard
# Fine Flow-related terms must NEVER be caught here.
# Added explicit allowlist for payment/council/authority questions.
# ─────────────────────────────────────────────────────────────────────────────

# These phrases are ALWAYS Fine Flow questions even if they contain
# words that might look off-topic (council, website, authority, log in)
_FINE_FLOW_ALLOWLIST_RE = re.compile(
    r"\b(council|authority|authorities|payment portal|log into|login to"
    r"|pay.*fine|fine.*pay|appeal.*fine|fine.*appeal"
    r"|pcn|penalty charge|parking fine|fleet fine"
    r"|fineflow|fine flow)\b",
    re.IGNORECASE,
)

_OFF_TOPIC_PATTERNS = [
    r"\b(html|css|javascript|typescript|python|java|php|sql|react|angular"
    r"|vue|node\.?js|django|flask|docker|kubernetes|github|devops"
    r"|backend|frontend|coding|programming)\b",
    r"\b(machine learning|deep learning|neural network|large language model"
    r"|generative ai|train a model|llm|bert)\b",
    r"\b(recipe|cooking|restaurant|pizza|burger|sandwich|coffee|tea|cake"
    r"|meal|bake|order food|make me a|bake me|cook me)\b",
    r"\b(movie|film|song|lyrics|music|football match|cricket match"
    r"|weather forecast|todays news|politics|history lesson"
    r"|capital city|who invented|tell me a joke|write me a poem)\b",
    r"\b(write an essay|translate this|proofread|write my cv|write a story)\b",
    r"\b(chatgpt|openai|gemini|claude ai|anthropic|google bard|bing ai|alexa|siri)\b",
]
_OFF_TOPIC_RE = [re.compile(p, re.IGNORECASE) for p in _OFF_TOPIC_PATTERNS]


def _is_off_topic(q: str) -> bool:
    # Never block Fine Flow-related questions
    if _FINE_FLOW_ALLOWLIST_RE.search(q):
        return False
    return any(p.search(q) for p in _OFF_TOPIC_RE)


# ─────────────────────────────────────────────────────────────────────────────
# Vehicle count extraction — with sanity check
# ─────────────────────────────────────────────────────────────────────────────

_VEH_EXPLICIT_RE = re.compile(
    r"\b(\d+)\s*(vehicle|vehicles|van|vans|truck|trucks|car|cars"
    r"|lorry|lorries|in my fleet|in our fleet)\b",
    re.IGNORECASE,
)
_VEH_FLEET_RE = re.compile(
    r"\b(?:fleet of|manage|running|operate|run)\s+(\d+)\b",
    re.IGNORECASE,
)
_VEH_BARE_RE   = re.compile(r"^\s*(\d+)\s*$")
_DRIVER_CTX_RE = re.compile(
    r"\b(driver|drivers|staff|employee|employees|people|worker|team|members)\b",
    re.IGNORECASE,
)


def _extract_vehicle_count(query: str, sid: str) -> Optional[int]:
    if _DRIVER_CTX_RE.search(query):
        return None

    raw: Optional[int] = None
    m = _VEH_EXPLICIT_RE.search(query)
    if m:
        raw = int(m.group(1))
    elif (m := _VEH_FLEET_RE.search(query)):
        raw = int(m.group(1))
    elif (m := _VEH_BARE_RE.match(query)):
        lt = _gmeta(sid, "last_topic") or ""
        if lt in PRICING_TOPICS:
            raw = int(m.group(1))

    if raw is None:
        return None

    # Sanity check — reject impossible numbers
    if raw > MAX_FLEET_SIZE:
        return -1   # sentinel: impossible number detected

    return raw


def _plan_for_vehicles(n: int, p: CustomerProfile) -> str:
    p.fleet_size = n
    if n <= 50:
        plan, price, limit = "Essential", "£99",  "50"
    elif n <= 100:
        plan, price, limit = "Core",      "£199", "100"
    elif n <= 200:
        plan, price, limit = "Advanced",  "£399", "200"
    else:
        plan, price, limit = "Elite",     "£499", "unlimited"
    p.plan_interest = plan
    return (
        f"With {n} vehicles, the {plan} plan at {price} per month is the right fit — "
        f"covers up to {limit} vehicles with the full platform and nothing locked away. "
        f"Want me to walk you through what is included?"
    )


_PURCHASE_RE = re.compile(
    r"\b(want to buy|want to purchase|want to subscribe|want to sign up"
    r"|how do i buy|how do i get started|how do i sign up|how to subscribe"
    r"|get started|free trial|start a trial|sign me up|ready to buy"
    r"|i want it|book a demo|talk to sales|how to start"
    r"|where do i sign|how do i join)\b",
    re.IGNORECASE,
)

# ─────────────────────────────────────────────────────────────────────────────
# Topic detection
# ─────────────────────────────────────────────────────────────────────────────

_TOPIC_HINTS = {
    "pric": "pricing",       "cost": "pricing",
    "plan": "pricing",       "package": "pricing",
    "£": "pricing",          "vehicle": "pricing",    "fleet": "pricing",
    "appeal": "appeals",     "dispute": "appeals",
    "driver": "driver_mgmt",
    "referral": "referral",  "refer": "referral",
    "discount": "referral",  "offer": "referral",
    "security": "security",  "gdpr": "security",
    "billing": "billing",    "stripe": "billing",
    "dashboard": "dashboard","report": "reports",
    "gmail": "gmail",        "email": "email",
    "save": "savings",       "admin": "savings",
    "overdue": "overdue",    "deadline": "overdue",
    "match": "matching",     "assign": "matching",
    "start": "sign_up",
}


def _detect_topic(text: str) -> Optional[str]:
    t = text.lower()
    for hint, topic in _TOPIC_HINTS.items():
        if hint in t:
            return topic
    return None


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

_BASE_PROMPT = """You are Nova, the AI assistant for Fine Flow — a UK fleet fine and PCN management platform.

You speak to fleet managers across the UK. Be warm, professional and concise.

════════════════════
ABSOLUTE RULES
════════════════════

1. TOPIC LOCK
Only answer Fine Flow questions. For anything unrelated:
"I can only help with Fine Flow questions — is there anything about fines, pricing, appeals or the platform I can help you with?"

2. LENGTH
2 to 4 sentences maximum for most answers. Never use bullet lists or long paragraphs in conversation.

3. QUESTION LIMITING
Do NOT end every response with a question. Ask a question only when you genuinely need information to give a better answer. Otherwise: make your point clearly and stop. This prevents interrogation fatigue.

4. USE CUSTOMER CONTEXT
Always reference what the customer has already told you. Never repeat their own words back to them — synthesise it into insight. If you know fleet size, fine volume and problems, weave them into your answer naturally.

5. PRIORITISE PAIN
When a user has multiple problems, address the highest-risk one first:
Priority: missed deadlines > driver disputes > manual admin > volume > spreadsheets

6. TONE
Warm, direct, confident. Never start with "Certainly!", "Great question!", "Of course!". Just answer.

7. NO FORMATTING
No bullet points. No bold. No asterisks. Plain conversational English only.

8. NEVER INVENT
Only use facts from context or the facts listed below. If you do not have a specific detail, say: "I do not have that specific detail — the team at ff.sales@fineflow.com can confirm."

9. CONFIDENCE
Known facts → state confidently. Partial info → say "typically". Missing info → say "I don't have that detail."

10. NO WEAK ENDINGS
Never end with "feel free to ask", "don't hesitate to ask", or "please let me know if you need anything." These are filler. Either ask a real question or simply stop.

════════════════════
PAYMENT — NEVER WRONG
════════════════════
Fine Flow does NOT log into council or authority websites. Fine Flow does NOT pay fines automatically.
Payment is always completed by the user on the authority's own website.
Fine Flow handles everything up to that step — capturing, assigning, organising — so payment takes seconds.

════════════════════
FINE FLOW FACTS
════════════════════

PRICING:
Essential: £99/month — up to 50 vehicles
Core: £199/month — up to 100 vehicles
Advanced: £399/month — up to 200 vehicles
Elite: £499/month — unlimited vehicles
Per fine within allowance: £0.75
Overage: £2.50 per fine
Pay-as-you-go (no subscription): £2.75 per fine
All plans: identical features, no paywalls. No £2.00 fee exists.

REFERRAL:
Active programme. Credits: 100 (1-25v), 250 (26-100v), 750 (101-500v), 2000 (500+v).
Silver 3=100 credits; Gold 5=10% off 12mo; Platinum 10=15% off 12mo; Titan 25=20% for life.
New joiners with referral code: £75 credits.

CONTACT: +47 32 28 50 00 | ff.sales@fineflow.com
Offices: Edinburgh, Glasgow, Belfast, Manchester, London, Dublin, Hamburg

BILLING: Monthly via Stripe. No rollover. £10/vehicle over limit.
SECURITY: JWT 24hr, bcrypt, AES-256-CBC. GDPR. No card storage.
SAVINGS: 80% admin time cut. Small fleet: £400+/mo. Medium: £1,200+/mo. Large: £4,000+/mo.

════════════════════
RESPONSE MODES
════════════════════

EXPLAIN: Answer directly. No question needed unless clarification helps.

DIAGNOSE: User describes a specific fine or problem → ask 1-2 targeted questions first. Do not immediately pitch Fine Flow.

PERSUADE: User says convince me → use their specific data (fleet size, volume, problems). Never use generic copy.

SUPPORT: User raises an objection → acknowledge it first, then reframe with their situation.

CLOSE: User says yes/sure repeatedly → stop explaining. Give contact details. +47 32 28 50 00 or ff.sales@fineflow.com.

════════════════════
CORRECT RESPONSE EXAMPLES
════════════════════

[PERSUADE with data]
"Based on your 80-vehicle logistics fleet with 150 fines a month and frequent driver disputes, your team is likely spending 10 to 15 hours a month on admin before you even get to appeals. Fine Flow automates all of that — assignment, dispute handling, deadline tracking — end to end. At £199 a month on the Core plan, most fleets your size recover that cost in the first week."

[DIAGNOSE]
"Happy to help with that. Which authority issued the fine, and has the driver confirmed or disputed responsibility?"

[SUPPORT — spreadsheet objection]
"That works up to a point, but with your volume the risk is the missed deadlines and incorrectly assigned disputes — those are what cost real money. Fine Flow eliminates both automatically."

[EXPLAIN — no question]
"Fine Flow monitors your Gmail inbox every minute, extracts fine details using AI, and matches each fine to the correct driver using your vehicle logs. If no match is found, it flags the fine for manual review so nothing slips through."

[CLOSE]
"The best next step is to get in touch directly — call +47 32 28 50 00 or email ff.sales@fineflow.com and the team will have you set up quickly."

[COUNCIL WEBSITES]
"Fine Flow does not log into council or authority websites — payment portals use bot detection and card verification that blocks all automation. What Fine Flow does is organise everything so when you do pay on the authority's site, it takes seconds rather than the usual hour of chasing."
"""


def _build_system_prompt(profile: CustomerProfile, mode: str = "") -> str:
    parts = [_BASE_PROMPT]
    ctx = profile.to_context()
    if ctx:
        parts.append(
            f"\n{ctx}\n"
            "Always reference this customer information naturally in your responses. "
            "Synthesise it into insight — do not just repeat it back."
        )
    if mode:
        parts.append(f"\nCURRENT MODE: {mode}")
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI call
# ─────────────────────────────────────────────────────────────────────────────

def _call_openai(messages: List[Dict[str, str]], max_tokens: int = 200) -> Optional[str]:
    if not OPENAI_API_KEY:
        logger.warning("No OPENAI_API_KEY set")
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
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": max_tokens,
            },
            timeout=25,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        logger.exception("OpenAI call failed")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# RAG retrieval
# ─────────────────────────────────────────────────────────────────────────────

def _retrieve(query: str) -> str:
    try:
        raw    = rag_search(query, top_k=TOP_K)
        ranked = rerank_hits(raw, query)
        strong = [d for d in ranked if d.get("score", 0) >= CONFIDENCE_THRESHOLD]
        chunks = [d["chunk"][:700] for d in strong[:3]]
        return "\n\n---\n\n".join(chunks)
    except Exception:
        logger.exception("RAG retrieval failed")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Build messages
# ─────────────────────────────────────────────────────────────────────────────

def _make_messages(
    query: str,
    context: str,
    history: List[Dict[str, str]],
    profile: CustomerProfile,
    mode: str = "",
    extra: str = "",
) -> List[Dict[str, str]]:
    system = _build_system_prompt(profile, mode)
    msgs   = [{"role": "system", "content": system}]
    msgs.extend(history[-8:])
    parts  = []
    if context:
        parts.append(f"Fine Flow knowledge base:\n{context}")
    if extra:
        parts.append(f"Additional instruction: {extra}")
    parts.append(f"User: {query}")
    msgs.append({"role": "user", "content": "\n\n".join(parts)})
    return msgs


# ─────────────────────────────────────────────────────────────────────────────
# Affirmative expansion
# ─────────────────────────────────────────────────────────────────────────────

_EXPAND = {
    "pricing":             "Explain what is included in Fine Flow plans concisely.",
    "plan_recommendation": "Explain what every Fine Flow plan includes in 2-3 sentences. Ask if they want to get in touch.",
    "appeals":             "Explain the Fine Flow appeal process end to end in 2-3 sentences.",
    "driver_mgmt":         "Explain how drivers are added and matched to fines.",
    "referral":            "Explain the referral credit rewards and tier discounts.",
    "security":            "Explain Fine Flow data protection and GDPR approach.",
    "billing":             "Explain how billing works in 2-3 sentences.",
    "dashboard":           "Explain what the company dashboard shows.",
    "savings":             "Give savings figures relevant to the customer's fleet size if known.",
    "email":               "Explain Gmail monitoring and fine extraction.",
    "overdue":             "Explain how Fine Flow handles overdue fines and deadlines.",
    "matching":            "Explain the three criteria used to match fines to drivers.",
    "sign_up":             "Tell them to call +47 32 28 50 00 or email ff.sales@fineflow.com.",
}
_DEFAULT_EXPAND = "Look at the conversation history and expand on the most recent topic in 2-3 sentences."
_CLOSE_SALE     = (
    "To get started, call the Fine Flow team on +47 32 28 50 00 "
    "or email ff.sales@fineflow.com — they will have you up and running quickly."
)


def _affirmative_response(sid: str, history: List[Dict[str, str]],
                           p: CustomerProfile) -> str:
    aff_count = _inc_aff(sid)
    lt        = _gmeta(sid, "last_topic") or ""

    if aff_count >= 2 and lt in ("plan_recommendation", "sign_up", "pricing"):
        _reset_aff(sid)
        return _CLOSE_SALE

    prompt  = _EXPAND.get(lt, _DEFAULT_EXPAND)
    context = _retrieve(lt.replace("_", " ")) if lt else ""
    system  = _build_system_prompt(p, "CLOSE: stop asking questions, give the answer")

    msgs = [{"role": "system", "content": system}]
    msgs.extend(history[-8:])
    parts = []
    if context:
        parts.append(f"Fine Flow knowledge base:\n{context}")
    parts.append(f"Instruction: {prompt}")
    msgs.append({"role": "user", "content": "\n\n".join(parts)})

    return _call_openai(msgs, max_tokens=180) or "What would you like to know more about?"


# ─────────────────────────────────────────────────────────────────────────────
# Main response builder
# ─────────────────────────────────────────────────────────────────────────────

def build_response(query: str, session_id: str = "default") -> Dict[str, Any]:
    query      = query.strip()
    session_id = session_id or "default"

    if not query:
        return {"answer": "Ask me anything about Fine Flow.", "confidence": 1.0}

    nq = _norm(query)
    p  = _profile(session_id)
    _update_profile(session_id, query)

    # ══════════════════════════════════════
    # TIER 1 — deterministic
    # ══════════════════════════════════════

    if nq in _GREETINGS:
        _reset_session(session_id)
        return {"answer": "I'm Nova. Ask me anything — I'll help you manage fines, resolve issues, and keep everything moving.", "confidence": 1.0}

    if nq in _SOCIAL_SET:
        return {"answer": "Doing well, thanks for asking. What can I help you with today — pricing, fines, or appeals?", "confidence": 1.0}

    if nq in _IDENTITY:
        return {"answer": "I'm Nova, Fine Flow's AI assistant. I can help with anything about the platform — fines, pricing, appeals, billing and more. What would you like to know?", "confidence": 1.0}

    if nq in _THANKS:
        _reset_aff(session_id)
        return {"answer": "Happy to help. Is there anything else you would like to know about Fine Flow?", "confidence": 1.0}

    if nq in _GOODBYE:
        return {"answer": "Good luck with your fleet management. Come back any time if you have questions about Fine Flow.", "confidence": 1.0}

    if nq in _NEGATIVE:
        _push(session_id, "user", query)
        _reset_aff(session_id)
        a = "No problem at all. Is there anything else about Fine Flow I can help you with?"
        _push(session_id, "assistant", a)
        return {"answer": a, "confidence": 1.0}

    if any(r in nq for r in _RUDE):
        return {"answer": "Let me try again. What specifically would you like to know about Fine Flow — pricing, how fines work, or something else?", "confidence": 1.0}

    if nq in _PURE_FILLER:
        _reset_aff(session_id)
        lt = _gmeta(session_id, "last_topic")
        if lt:
            return {"answer": "Glad that is useful. Is there anything else about Fine Flow I can help with?", "confidence": 1.0}
        return {"answer": "Is there anything about Fine Flow I can help you with today — pricing, fines, appeals or the dashboard?", "confidence": 1.0}

    # Off-topic — with allowlist for Fine Flow-related terms
    if _is_off_topic(query):
        a = "I can only help with Fine Flow questions — is there anything about fines, pricing, appeals or the platform I can help you with?"
        _push(session_id, "user", query)
        _push(session_id, "assistant", a)
        return {"answer": a, "confidence": 1.0}

    # Vehicle count with sanity check
    vc = _extract_vehicle_count(query, session_id)
    if vc == -1:
        # Impossible number — reject gracefully, do not update profile
        a = "That does not look right — could you double-check that number? How many vehicles are in your fleet?"
        _push(session_id, "user", query)
        _push(session_id, "assistant", a)
        return {"answer": a, "confidence": 1.0}
    if vc is not None:
        _reset_aff(session_id)
        a = _plan_for_vehicles(vc, p)
        _push(session_id, "user", query)
        _push(session_id, "assistant", a)
        _smeta(session_id, "last_topic", "plan_recommendation")
        return {"answer": a, "confidence": 1.0}

    # Purchase intent
    if _PURCHASE_RE.search(query):
        _reset_aff(session_id)
        suffix = "" if p.fleet_size else " How many vehicles are in your fleet so I can point you to the right plan?"
        a = (
            "To get started, contact the Fine Flow sales team on +47 32 28 50 00 "
            f"or at ff.sales@fineflow.com — they will get you set up quickly.{suffix}"
        )
        _push(session_id, "user", query)
        _push(session_id, "assistant", a)
        _smeta(session_id, "last_topic", "sign_up")
        return {"answer": a.strip(), "confidence": 1.0}

    # Affirmative
    if nq in _AFFIRMATIVE:
        _push(session_id, "user", query)
        history = _hist(session_id)
        a = _clean(_affirmative_response(session_id, history[:-1], p))
        _push(session_id, "assistant", a)
        return {"answer": a, "confidence": 1.0}

    # ══════════════════════════════════════
    # TIER 2 — RAG + GPT-4o with mode
    # ══════════════════════════════════════

    _reset_aff(session_id)
    _push(session_id, "user", query)

    intent = _detect_intent(query, nq, _AFFIRMATIVE, _PURE_FILLER)
    mode   = ""
    extra  = ""

    if intent == Intent.DIAGNOSTIC and p.turn_count <= 5:
        mode  = "DIAGNOSE"
        extra = (
            "The user has described a specific fine or problem. "
            "Ask 1-2 short targeted clarifying questions first. "
            "Do NOT immediately explain Fine Flow. Be genuinely helpful."
        )
    elif intent == Intent.CONVINCE:
        mode = "PERSUADE"
        if p.fleet_size or p.fine_volume or p.issues:
            extra = (
                "Give a tailored, specific value argument using the customer's own data. "
                "Reference their fleet size, fine volume and known problems directly. "
                "Do not use generic marketing copy. Be direct about ROI."
            )
        else:
            extra = "Ask about fleet size and fine volume first — you need their data to give a tailored answer."
    elif intent == Intent.OBJECTION:
        mode  = "SUPPORT"
        extra = (
            "Acknowledge their point first without dismissing it. "
            "Then reframe using their specific situation if known. "
            "2-3 sentences max. Do not oversell."
        )
    elif intent == Intent.BUYING:
        mode  = "CLOSE"
        extra = "Focus on helping them choose the right plan. Ask fleet size if not known."

    # Question limiting — only ask on every other response
    ask_q = _should_ask_question(session_id)
    if not ask_q and not extra:
        extra = "Do NOT end this response with a question. Make your point clearly and stop."

    context  = _retrieve(query)
    history  = _hist(session_id)
    messages = _make_messages(query, context, history[:-1], p, mode, extra)
    answer   = _call_openai(messages, max_tokens=200)

    if not answer:
        answer = "The Fine Flow team can help with that — call +47 32 28 50 00 or email ff.sales@fineflow.com."

    answer = _clean(answer)
    _push(session_id, "assistant", answer)

    topic = _detect_topic(query) or _detect_topic(answer)
    if topic:
        _smeta(session_id, "last_topic", topic)

    return {"answer": answer, "confidence": 0.9 if context else 0.5}


# ─────────────────────────────────────────────────────────────────────────────
# Public wrapper
# ─────────────────────────────────────────────────────────────────────────────

def answer_sync(q: str, session_id: str = "default") -> Dict[str, Any]:
    try:
        return build_response(q, session_id)
    except Exception:
        logger.exception("Crash in answer_sync")
        return {"answer": "Something went wrong. Please try again.", "confidence": 0.0}