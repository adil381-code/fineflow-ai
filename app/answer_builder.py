# app/answer_builder.py
"""
FineFlow Nova — Production Final v14
======================================
Major upgrade: customer profile system.
Every fact the user reveals (fleet size, fine volume, problems, plan interest)
is stored in a profile and injected into every GPT-4o call.
This makes responses feel genuinely intelligent and tailored.

Changes vs v13:
  - CustomerProfile class stores fleet_size, fine_volume, issues, plan_interest
  - Profile injected into every system message so GPT-4o always knows the context
  - Varied follow-up questions — no more repeating "would you like to know more"
  - Diagnostic mode: when user describes a problem, ask clarifying questions
  - Tailored sales: "convince me" uses their actual fleet size and volume
  - Problem-solving mode: lane violation, driver dispute etc triggers questions
  - Ending rotation prevents repetitive closings
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


# ─────────────────────────────────────────────────────────────────────────────
# Customer profile — stored per session, injected into every GPT-4o call
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CustomerProfile:
    fleet_size:    Optional[int]       = None
    fine_volume:   Optional[int]       = None   # fines per month
    issues:        List[str]           = field(default_factory=list)
    plan_interest: Optional[str]       = None
    industry:      Optional[str]       = None
    turn_count:    int                 = 0

    def to_prompt(self) -> str:
        """Return a context block to inject into every system message."""
        lines = []
        if self.fleet_size:
            lines.append(f"Fleet size: {self.fleet_size} vehicles")
        if self.fine_volume:
            lines.append(f"Monthly fine volume: ~{self.fine_volume} fines")
        if self.industry:
            lines.append(f"Industry: {self.industry}")
        if self.issues:
            lines.append(f"Known problems: {', '.join(self.issues)}")
        if self.plan_interest:
            lines.append(f"Plan discussed: {self.plan_interest}")
        if not lines:
            return ""
        return "WHAT WE KNOW ABOUT THIS CUSTOMER:\n" + "\n".join(lines)

    def recommended_plan(self) -> Optional[str]:
        if self.fleet_size is None:
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


# ─────────────────────────────────────────────────────────────────────────────
# Profile extraction from user messages
# ─────────────────────────────────────────────────────────────────────────────

_FINES_PER_MONTH_RE = re.compile(
    r"\b(\d+)\s*(?:fines?|pcns?|penalties|violations?|tickets?)"
    r"\s*(?:per|a|each|every)?\s*(?:month|monthly|week)?\b",
    re.IGNORECASE,
)
_INDUSTRY_RE = re.compile(
    r"\b(logistics|delivery|courier|haulage|transport|taxi|minicab"
    r"|bus|coach|construction|utilities|local authority|council)\b",
    re.IGNORECASE,
)
_ISSUE_PATTERNS = [
    (r"\b(miss(?:ed?|ing)? (?:deadlines?|appeals?|due dates?))\b",   "missed deadlines"),
    (r"\b(drivers? (?:dispute|deny|ignor|avoid))\b",                  "driver disputes"),
    (r"\b(manual(?:ly)? (?:track|manage|process|handl))\b",           "manual admin"),
    (r"\b(spreadsheet)\b",                                             "using spreadsheets"),
    (r"\b(too much (?:admin|time|work|paperwork))\b",                 "too much admin"),
    (r"\b(miss(?:ed?|ing)? (?:fines?|payments?))\b",                  "missed fines"),
    (r"\b(appeal deadline)\b",                                         "missed appeal deadlines"),
]
_ISSUE_RE = [(re.compile(p, re.IGNORECASE), label) for p, label in _ISSUE_PATTERNS]


def _update_profile(sid: str, query: str) -> None:
    """Extract any new customer facts from this message and store them."""
    p = _profile(sid)
    p.turn_count += 1

    # Fine volume
    m = _FINES_PER_MONTH_RE.search(query)
    if m and p.fine_volume is None:
        p.fine_volume = int(m.group(1))

    # Industry
    m = _INDUSTRY_RE.search(query)
    if m:
        p.industry = m.group(1).lower()

    # Issues
    for pattern, label in _ISSUE_RE:
        if pattern.search(query) and label not in p.issues:
            p.issues.append(label)


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

_SOCIAL = {
    "how are you", "how are you doing", "how r u", "how r you",
    "how are u", "hows it going", "how is it going",
    "whats up", "what s up", "you ok", "you good",
    "how do you do", "you alright", "alright mate", "you well",
}

_IDENTITY = {
    "who are you", "who r you", "who r u", "who youre", "who you re",
    "who is nova", "who is this", "who is there", "whos there",
    "who s there", "who there", "anyone there", "is anyone there",
    "what are you", "what is nova", "what is this",
    "are you a bot", "are you human", "are you ai",
    "are you a robot", "are you real", "are you a person",
    "are you male or female", "you male or female",
    "who the hell are you", "who are u",
    "whats your name", "what is your name", "your name",
    "introduce yourself", "tell me about yourself",
    "who am i talking to", "who am i speaking to",
    "knock knock",
}

_AFFIRMATIVE = {
    "yes", "yeah", "yep", "yup", "ya", "ye",
    "sure", "ok sure", "okay sure",
    "go ahead", "go on",
    "yes go ahead", "yes go on", "yes please",
    "yes sure", "yes of course", "of course",
    "absolutely", "definitely", "do it",
    "tell me more", "more", "explain", "explain more",
    "yes explain", "yes explain it",
    "go for it", "sounds good", "continue",
    "carry on", "keep going", "please do",
    "i would", "i d like that", "i would like that",
    "please explain", "show me", "walk me through it",
    "yes elite", "yes core", "yes essential", "yes advanced",
    "for sure", "yes for sure", "yes sure thing", "sure thing",
}

_THANKS = {
    "thanks", "thank you", "thank u", "cheers",
    "that helps", "that helped", "ta", "ty",
    "okay thanks", "ok thanks", "great thanks",
    "perfect", "brilliant", "nice one", "lovely",
    "great", "awesome", "wonderful",
    "thank you so much", "many thanks", "much appreciated",
    "thats helpful", "that s helpful",
}

_GOODBYE = {
    "bye", "goodbye", "see you", "see ya", "later",
    "take care", "good bye", "cya", "ttyl",
    "talk later", "farewell", "cheerio",
}

_NEGATIVE = {
    "no", "nope", "nah", "no thanks", "not now", "skip",
    "never mind", "nevermind", "no need", "not really",
    "no thank you", "nah thanks",
}

_RUDE = {
    "you dumb", "you are dumb", "ur dumb",
    "stupid", "idiot", "useless", "rubbish",
    "garbage", "terrible", "you suck",
    "this is rubbish", "this is garbage",
    "dumb bot", "rubbish bot",
    "you re useless", "ur useless",
    "waste of time", "this is useless",
}

_PURE_FILLER = {
    "ok", "okay", "right", "alright", "cool", "nice",
    "interesting", "really", "seriously",
    "hmm", "hm", "ah", "oh", "i see",
    "got it", "understood", "makes sense", "noted",
    "wow", "waow", "woah", "whoa", "omg",
    "anything", "something", "whatever",
}

# Problem description patterns — trigger diagnostic questions
_PROBLEM_RE = re.compile(
    r"\b(fine|pcn|penalty|violation|notice|ticket)\b.*"
    r"\b(issued|received|got|have a|appealing|disputing|today|yesterday|last week|this week)\b"
    r"|\b(received a|got a|have a)\b.*\b(fine|pcn|penalty|violation|notice)\b",
    re.IGNORECASE,
)

# Convince / sales pitch request
_CONVINCE_RE = re.compile(
    r"\b(convince|persuade|sell me|why should i|why buy|is it worth"
    r"|should i get|should i buy|justify|make the case|prove it)\b",
    re.IGNORECASE,
)

# Off-topic
_OFF_TOPIC_PATTERNS = [
    r"\b(html|css|javascript|typescript|python|java|php|sql|react|angular"
    r"|vue|node\.?js|django|flask|docker|kubernetes|github|devops"
    r"|backend|frontend|fullstack|coding|programming)\b",
    r"\b(machine learning|deep learning|neural network|large language model"
    r"|generative ai|train a model|llm|gpt|bert)\b",
    r"\b(recipe|cooking|restaurant|pizza|burger|sandwich|coffee|tea|cake"
    r"|meal|bake|order food|make me a|bake me|cook me|cook for me)\b",
    r"\b(movie|film|song|lyrics|music|football match|cricket match"
    r"|weather forecast|todays news|politics|history lesson"
    r"|capital city|who invented|tell me a joke|write me a poem)\b",
    r"\b(write an essay|translate this|proofread|write my cv|write a story)\b",
    r"\b(chatgpt|openai|gemini|claude ai|anthropic|google bard"
    r"|bing ai|alexa|siri|cortana)\b",
]
_OFF_TOPIC_RE = [re.compile(p, re.IGNORECASE) for p in _OFF_TOPIC_PATTERNS]


def _is_off_topic(q: str) -> bool:
    return any(p.search(q) for p in _OFF_TOPIC_RE)


# ─────────────────────────────────────────────────────────────────────────────
# Vehicle count extraction
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
_VEH_BARE_RE    = re.compile(r"^\s*(\d+)\s*$")
_DRIVER_CTX_RE  = re.compile(
    r"\b(driver|drivers|staff|employee|employees|people|worker|workers|team|members)\b",
    re.IGNORECASE,
)


def _extract_vehicle_count(query: str, sid: str) -> Optional[int]:
    if _DRIVER_CTX_RE.search(query):
        return None
    m = _VEH_EXPLICIT_RE.search(query)
    if m:
        return int(m.group(1))
    m = _VEH_FLEET_RE.search(query)
    if m:
        return int(m.group(1))
    m = _VEH_BARE_RE.match(query)
    if m:
        lt = _gmeta(sid, "last_topic") or ""
        if lt in PRICING_TOPICS:
            return int(m.group(1))
    return None


def _plan_for_vehicles(n: int, p: CustomerProfile) -> str:
    # Store fleet size in profile
    p.fleet_size = n
    p.plan_interest = (
        "Essential" if n <= 50 else
        "Core"      if n <= 100 else
        "Advanced"  if n <= 200 else
        "Elite"
    )
    prices = {"Essential": "£99", "Core": "£199", "Advanced": "£399", "Elite": "£499"}
    limits = {"Essential": "50", "Core": "100",   "Advanced": "200",  "Elite": "unlimited"}
    plan   = p.plan_interest
    price  = prices[plan]
    limit  = limits[plan]
    return (
        f"With {n} vehicles, the {plan} plan at {price} per month is the right fit — "
        f"covers up to {limit} vehicles with the full platform included and nothing locked away. "
        f"Want me to walk you through what is included?"
    )


_PURCHASE_RE = re.compile(
    r"\b(want to buy|want to purchase|want to subscribe|want to sign up"
    r"|how do i buy|how do i get started|how do i sign up|how to subscribe"
    r"|get started|free trial|start a trial|sign me up|ready to buy"
    r"|i want it|book a demo|talk to sales|how to start"
    r"|where do i sign|how do i join|i d like to join)\b",
    re.IGNORECASE,
)

# ─────────────────────────────────────────────────────────────────────────────
# Topic detection
# ─────────────────────────────────────────────────────────────────────────────

_TOPIC_HINTS = {
    "pric": "pricing",        "cost": "pricing",
    "plan": "pricing",        "package": "pricing",
    "subscription": "pricing","£": "pricing",
    "vehicle": "pricing",     "fleet": "pricing",
    "appeal": "appeals",      "dispute": "appeals",
    "challenge": "appeals",
    "driver": "driver_mgmt",
    "referral": "referral",   "refer": "referral",
    "discount": "referral",   "offer": "referral",
    "security": "security",   "gdpr": "security",
    "safe": "security",       "card": "security",
    "billing": "billing",     "invoice": "billing",
    "stripe": "billing",
    "dashboard": "dashboard", "report": "reports",
    "gmail": "gmail",         "email": "email",
    "save": "savings",        "time": "savings",
    "money": "savings",       "admin": "savings",
    "overdue": "overdue",     "deadline": "overdue",
    "match": "matching",      "assign": "matching",
    "start": "sign_up",       "sign up": "sign_up",
}


def _detect_topic(text: str) -> Optional[str]:
    t = text.lower()
    for hint, topic in _TOPIC_HINTS.items():
        if hint in t:
            return topic
    return None


# ─────────────────────────────────────────────────────────────────────────────
# System prompt builder — injects customer profile dynamically
# ─────────────────────────────────────────────────────────────────────────────

_BASE_SYSTEM_PROMPT = """You are Nova, the AI assistant for Fine Flow — a UK fleet fine and PCN management platform.

You speak to fleet managers across the UK. Be warm, professional and concise. You are a knowledgeable product expert who genuinely wants to help solve real problems.

════════════════════════════════
ABSOLUTE RULES
════════════════════════════════

1. TOPIC LOCK
Only answer Fine Flow questions. For anything unrelated:
"I can only help with Fine Flow questions — is there anything about fines, pricing, appeals or the platform I can help you with?"

2. LENGTH
Maximum 3 sentences for simple questions. For complex problems (multiple issues described), you may use up to 5 sentences to address each point specifically. No bullet lists. No long paragraphs.

3. USE CUSTOMER CONTEXT
If you know the customer's fleet size, fine volume, industry or problems — USE THEM in your response. Never give a generic answer when you have their specific details. Reference what they told you. This makes you feel intelligent.

4. END EACH RESPONSE WITH A VARIED QUESTION
Rotate through different questions. Never repeat the same closing question twice in a row. Good options:
"How many vehicles are in your fleet?" (if unknown)
"What does your current process look like?"
"How long does it currently take your team to process each fine?"
"Is there a particular part of the process causing the most pain?"
"What would be most useful to know next?"
"Have you had to pay a fine you felt should have been challenged?"
"Shall I explain how [specific feature] works?"

5. TONE
Warm, direct, confident. Like a knowledgeable colleague. Never start with "Certainly!", "Great question!", "Of course!", "Absolutely!". Never sound robotic.

6. DIAGNOSTIC MODE
When a user describes a specific fine or problem, ask clarifying questions before explaining Fine Flow. Example: "Which authority issued it? When was it issued? And has the driver confirmed or denied responsibility?" This feels helpful, not salesy.

7. TAILORED SALES
When asked to convince or persuade, use the customer's own data. Reference their fleet size, fine volume, and problems in your answer. Never give generic sales copy.

8. NO FORMATTING
No bullet points. No bold. No asterisks. No numbered lists. Plain conversational English only.

9. NEVER INVENT
Only use facts from the context provided or the facts listed below. If you genuinely do not have information, say: "I do not have that specific detail — the Fine Flow team at ff.sales@fineflow.com can confirm."

════════════════════════════════
PAYMENT — NEVER GET THIS WRONG
════════════════════════════════
Fine Flow does NOT automatically pay fines. Payment is always done by you on the authority's website. Say NO clearly if asked.

════════════════════════════════
FINE FLOW FACTS
════════════════════════════════

PRICING:
Essential:  £99/month  — up to 50 vehicles
Core:       £199/month — up to 100 vehicles
Advanced:   £399/month — up to 200 vehicles
Elite:      £499/month — unlimited vehicles
Per fine within allowance: £0.75
Overage: £2.50 per fine
Pay-as-you-go (no subscription): £2.75 per fine
All plans: identical features, no paywalls.
NO £2.00 fee exists anywhere.

REFERRAL PROGRAMME:
Active referral programme with credit rewards and subscription discounts.
Credits: 100 (1-25 vehicles), 250 (26-100), 750 (101-500), 2000 (500+).
Tiers: Silver 3 referrals = 100 credits; Gold 5 = 10% off 12mo; Platinum 10 = 15% off 12mo; Titan 25 = 20% off for life.
New joiners with referral code: £75 credits.

CONTACT:
Phone: +47 32 28 50 00 | Email: ff.sales@fineflow.com
Offices: Edinburgh, Glasgow, Belfast, Manchester, London, Dublin, Hamburg

BILLING: Monthly via Stripe. No credit rollover. £10/vehicle if limit exceeded.

SECURITY: JWT 24hr tokens, bcrypt, AES-256-CBC. GDPR compliant. No card details stored.

SAVINGS: Up to 80% admin time reduction.
Small fleet (up to 50): £400+/month saved.
Medium (51-200): £1,200+/month saved.
Large (200+): £4,000+/month saved.

════════════════════════════════
EXAMPLE RESPONSES SHOWING CORRECT STYLE
════════════════════════════════

User: I manage a logistics fleet with 80 vehicles. We receive 150 fines a month. Drivers dispute responsibility. We miss appeal deadlines. How could Fine Flow help?
Nova: Based on what you've told me — 80 vehicles, 150 fines a month, frequent driver disputes and missed appeal deadlines — Fine Flow would address each of those directly. It automatically assigns fines to drivers using your vehicle logs, manages the full dispute and appeal process, and tracks every deadline so nothing is missed. For a fleet your size processing 150 fines a month, you would likely save over £1,200 a month compared to managing it manually. Which of those problems is causing the most damage right now?

User: Convince me to buy Fine Flow.
Nova: [If fleet size and volume known] Based on your fleet of 80 vehicles and 150 fines a month, your team is likely spending 10-15 admin hours monthly just on logging, chasing and tracking — before any appeals. Fine Flow automates all of that, assigns responsibility automatically, and handles appeals end to end. At £199 a month on the Core plan, most fleets your size recover that cost in the first week. Want to know how quickly it can be set up?

User: I have a lane violation fine.
Nova: Happy to help with that. Which authority issued it, and when was it issued? Also, has the driver confirmed or denied they were responsible?

User: Will it pay my fines automatically?
Nova: No — and that is intentional. Authority payment portals block all automation with bot detection and card verification, so payment is always done by you on their site. Fine Flow handles everything around that step so when you do pay, it takes seconds rather than the usual hour of chasing and organising. What does your current payment process look like?

User: We already use spreadsheets.
Nova: Spreadsheets work until the volume or complexity gets too high — and at that point every missed deadline or wrong assignment costs real money. Fine Flow automates the parts that spreadsheets make tedious: capturing fines automatically from email, matching them to drivers, tracking deadlines, and managing disputes. How many fines are you currently tracking manually each month?
"""


def _build_system_prompt(profile: CustomerProfile) -> str:
    """Inject customer profile into system prompt."""
    profile_block = profile.to_prompt()
    if not profile_block:
        return _BASE_SYSTEM_PROMPT
    return _BASE_SYSTEM_PROMPT + f"\n\n{profile_block}\n\nAlways reference this customer information in your responses where relevant."


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
# Build GPT-4o messages
# ─────────────────────────────────────────────────────────────────────────────

def _make_messages(
    query: str,
    context: str,
    history: List[Dict[str, str]],
    profile: CustomerProfile,
    extra: str = "",
) -> List[Dict[str, str]]:
    system = _build_system_prompt(profile)
    msgs: List[Dict[str, str]] = [{"role": "system", "content": system}]
    msgs.extend(history[-8:])   # last 4 full turns

    parts = []
    if context:
        parts.append(f"Fine Flow knowledge base:\n{context}")
    if extra:
        parts.append(f"Instruction: {extra}")
    parts.append(f"User: {query}")
    msgs.append({"role": "user", "content": "\n\n".join(parts)})
    return msgs


# ─────────────────────────────────────────────────────────────────────────────
# Affirmative expansion
# ─────────────────────────────────────────────────────────────────────────────

_EXPAND: Dict[str, str] = {
    "pricing":             "Explain Fine Flow pricing concisely and ask how many vehicles they have.",
    "plan_recommendation": "Explain what every Fine Flow plan includes in 2-3 sentences. Then ask if they want to get in touch with the sales team.",
    "appeals":             "Explain how the Fine Flow appeal process works in 2-3 sentences. Ask if they have a specific fine they are thinking of disputing.",
    "driver_mgmt":         "Explain how drivers are added and matched to fines. Ask how many drivers they manage.",
    "referral":            "Explain the referral credit rewards and tier discounts concisely.",
    "security":            "Explain Fine Flow data protection and GDPR compliance.",
    "billing":             "Explain how billing works and ask when they would like to start.",
    "dashboard":           "Explain what the company dashboard shows in 2-3 sentences.",
    "savings":             "Give specific savings figures relevant to their fleet size if known.",
    "email":               "Explain how Fine Flow monitors Gmail for incoming fines.",
    "gmail":               "Explain OAuth and App Password connection methods.",
    "overdue":             "Explain how Fine Flow tracks and flags overdue fines.",
    "matching":            "Explain the three driver matching criteria.",
    "reports":             "Explain the report types available.",
    "sign_up":             "Tell them to call +47 32 28 50 00 or email ff.sales@fineflow.com to get started. Say the team will have them set up quickly.",
}

_DEFAULT_EXPAND = "Look at the conversation history and expand helpfully on the most recent Fine Flow topic in 2-3 sentences, then ask a relevant follow-up question."
_CLOSE_SALE     = (
    "To get started with Fine Flow, call the sales team on +47 32 28 50 00 "
    "or email ff.sales@fineflow.com — they will have you up and running quickly. "
    "Is there anything else you would like to know before you get in touch?"
)


def _affirmative_response(sid: str, history: List[Dict[str, str]], profile: CustomerProfile) -> str:
    aff_count = _inc_aff(sid)
    lt        = _gmeta(sid, "last_topic") or ""

    if aff_count >= 2 and lt in ("plan_recommendation", "sign_up", "pricing"):
        _reset_aff(sid)
        return _CLOSE_SALE

    prompt  = _EXPAND.get(lt, _DEFAULT_EXPAND)
    context = _retrieve(lt.replace("_", " ")) if lt else ""

    system = _build_system_prompt(profile)
    msgs: List[Dict[str, str]] = [{"role": "system", "content": system}]
    msgs.extend(history[-8:])

    parts = []
    if context:
        parts.append(f"Fine Flow knowledge base:\n{context}")
    parts.append(f"Instruction: {prompt}")
    msgs.append({"role": "user", "content": "\n\n".join(parts)})

    answer = _call_openai(msgs, max_tokens=200)
    return answer or "What would you like to know more about — pricing, fines, appeals or billing?"


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

    # Always try to extract profile facts from every message
    _update_profile(session_id, query)

    # ══════════════════════════════════════════════
    # TIER 1 — deterministic, always correct
    # ══════════════════════════════════════════════

    if nq in _GREETINGS:
        _reset_session(session_id)
        return {
            "answer": "I'm Nova. Ask me anything — I'll help you manage fines, resolve issues, and keep everything moving.",
            "confidence": 1.0,
        }

    if nq in _SOCIAL:
        return {
            "answer": "Doing well, thanks for asking. What can I help you with today — pricing, fines, or appeals?",
            "confidence": 1.0,
        }

    if nq in _IDENTITY:
        return {
            "answer": "I'm Nova, Fine Flow's AI assistant. I can help with anything about the platform — fines, pricing, appeals, billing and more. What would you like to know?",
            "confidence": 1.0,
        }

    if nq in _THANKS:
        _reset_aff(session_id)
        return {
            "answer": "Happy to help. Is there anything else you would like to know about Fine Flow?",
            "confidence": 1.0,
        }

    if nq in _GOODBYE:
        return {
            "answer": "Good luck with your fleet management. Come back any time if you have questions about Fine Flow.",
            "confidence": 1.0,
        }

    if nq in _NEGATIVE:
        _push(session_id, "user", query)
        _reset_aff(session_id)
        a = "No problem at all. Is there anything else about Fine Flow I can help you with?"
        _push(session_id, "assistant", a)
        return {"answer": a, "confidence": 1.0}

    if any(r in nq for r in _RUDE):
        return {
            "answer": "Let me try again. What specifically would you like to know about Fine Flow — pricing, how fines work, or something else?",
            "confidence": 1.0,
        }

    if nq in _PURE_FILLER:
        _reset_aff(session_id)
        lt = _gmeta(session_id, "last_topic")
        if lt:
            label = lt.replace("_", " ")
            return {
                "answer": f"Glad that is useful. Would you like to go deeper on {label}, or is there something else I can help with?",
                "confidence": 1.0,
            }
        return {
            "answer": "Is there anything about Fine Flow I can help you with today — pricing, fines, appeals or the dashboard?",
            "confidence": 1.0,
        }

    if _is_off_topic(query):
        a = "I can only help with Fine Flow questions — is there anything about fines, pricing, appeals or the platform I can help you with?"
        _push(session_id, "user", query)
        _push(session_id, "assistant", a)
        return {"answer": a, "confidence": 1.0}

    # Vehicle count → instant plan recommendation
    vc = _extract_vehicle_count(query, session_id)
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
        a = (
            "To get started, contact the Fine Flow sales team on +47 32 28 50 00 "
            "or at ff.sales@fineflow.com — they will get you set up quickly. "
            "How many vehicles are in your fleet so I can point you to the right plan?"
        )
        _push(session_id, "user", query)
        _push(session_id, "assistant", a)
        _smeta(session_id, "last_topic", "sign_up")
        return {"answer": a, "confidence": 1.0}

    # Affirmative
    if nq in _AFFIRMATIVE:
        _push(session_id, "user", query)
        history = _hist(session_id)
        a = _clean(_affirmative_response(session_id, history[:-1], p))
        _push(session_id, "assistant", a)
        return {"answer": a, "confidence": 1.0}

    # ══════════════════════════════════════════════
    # TIER 2 — RAG + GPT-4o
    # ══════════════════════════════════════════════

    _reset_aff(session_id)
    _push(session_id, "user", query)

    context  = _retrieve(query)
    history  = _hist(session_id)

    # Special instruction for diagnostic and sales modes
    extra = ""
    if _PROBLEM_RE.search(query) and p.turn_count <= 3:
        extra = "The user has described a specific fine problem. Ask 2 short diagnostic questions (authority, date, driver responsibility) before explaining Fine Flow. Be helpful, not salesy."
    elif _CONVINCE_RE.search(query):
        profile_facts = p.to_prompt()
        if profile_facts:
            extra = f"The user wants to be convinced. Use their specific situation: {profile_facts}. Give a tailored, specific value argument — not generic marketing copy."
        else:
            extra = "The user wants to be convinced. Ask about their fleet size and fine volume first so you can give a tailored answer."

    messages = _make_messages(query, context, history[:-1], p, extra)
    answer   = _call_openai(messages, max_tokens=200)

    if not answer:
        answer = (
            "The Fine Flow team can help with that directly — "
            "call +47 32 28 50 00 or email ff.sales@fineflow.com."
        )

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