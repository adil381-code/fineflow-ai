# app/answer_builder.py
"""
FineFlow Nova — Production Final v11
======================================
Architecture: Tier 1 deterministic intents → Tier 2 RAG + GPT-4o with full memory.

All v10 bugs fixed:
  - "who is there", "anyone there" etc → identity handler
  - "yes" / "sure" / "go ahead" → dedicated affirmative handler using GPT-4o
    with last_topic context so it always expands on what was just discussed
  - Bare numbers ("54", "567") → vehicle count when last_topic is pricing-related
  - "yes" removed from _PURE_FILLER (had its own handler all along, was conflicting)
  - Session memory stored and passed correctly on every single GPT-4o call
  - UK-tone system prompt: warm, professional, concise, sales-oriented
  - Few-shot examples cover every known failure case
"""

import re
import threading
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
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Strip markdown symbols so output is plain text."""
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
    """Lowercase + collapse whitespace + strip punctuation."""
    t = text.lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Session memory
# ─────────────────────────────────────────────────────────────────────────────

_SESSION: Dict[str, List[Dict[str, str]]] = {}   # conversation turns
_META:    Dict[str, Dict[str, Any]]       = {}    # per-session metadata
_LOCK = threading.Lock()

PRICING_TOPICS = {
    "pricing", "plan_recommendation", "vehicles",
    "cost", "packages", "billing",
}


def _hist(sid: str) -> List[Dict[str, str]]:
    with _LOCK:
        return list(_SESSION.get(sid, []))


def _push(sid: str, role: str, content: str) -> None:
    with _LOCK:
        h = _SESSION.setdefault(sid, [])
        h.append({"role": role, "content": content})
        cap = CHAT_HISTORY_TURNS * 2
        if len(h) > cap:
            _SESSION[sid] = h[-cap:]


def _smeta(sid: str, key: str, val: Any) -> None:
    with _LOCK:
        _META.setdefault(sid, {})[key] = val


def _gmeta(sid: str, key: str) -> Any:
    with _LOCK:
        return _META.get(sid, {}).get(key)


def _reset_meta(sid: str) -> None:
    with _LOCK:
        _META[sid] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Intent sets  (all pre-normalised, matched with == or `in`)
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
    "how do you do", "how do u do", "how have you been",
    "you alright", "alright mate", "you well",
}

# Everything that means "who/what are you" — including "who is there"
_IDENTITY = {
    "who are you", "who r you", "who r u",
    "who is nova", "who is this", "who is there", "whos there",
    "who s there", "who there", "anyone there", "is anyone there",
    "who am i talking to", "who am i speaking to",
    "what are you", "what is nova", "what is this",
    "are you a bot", "are you human", "are you ai",
    "are you a robot", "are you real", "are you a person",
    "are you male or female", "you male or female",
    "who the hell are you", "who are u",
    "whats your name", "what is your name", "your name",
    "introduce yourself", "tell me about yourself",
}

# Affirmative — dedicated handler, NOT in filler
_AFFIRMATIVE = {
    "yes", "yeah", "yep", "yup", "ya", "ye",
    "sure", "ok sure", "okay sure",
    "go ahead", "go on", "yes go ahead", "yes go on",
    "yes please", "please", "yes sure", "yes of course",
    "of course", "absolutely", "definitely", "do it",
    "tell me more", "more", "explain", "explain more",
    "yes explain", "yes explain it", "go for it",
    "sounds good", "that sounds good", "continue",
    "carry on", "keep going",
}

_THANKS = {
    "thanks", "thank you", "thank u", "cheers",
    "that helps", "that helped", "ta",
    "okay thanks", "ok thanks", "great thanks",
    "perfect", "brilliant", "nice one", "lovely",
    "great", "awesome", "wonderful", "thank you so much",
    "many thanks", "much appreciated",
}

_GOODBYE = {
    "bye", "goodbye", "see you", "see ya", "later",
    "take care", "good bye", "cya", "ttyl",
    "talk later", "farewell", "cheerio", "toodles",
}

_RUDE = {
    "you dumb", "you are dumb", "ur dumb", "stupid",
    "idiot", "useless", "rubbish", "garbage",
    "terrible", "you suck", "this is rubbish",
    "this is garbage", "dumb bot", "rubbish bot",
    "you re useless", "ur useless", "waste of time",
    "this is useless", "absolute rubbish",
}

# Pure filler — acknowledge + redirect.  Does NOT include affirmatives.
_PURE_FILLER = {
    "ok", "okay", "right", "alright", "cool", "nice",
    "interesting", "really", "seriously", "hmm", "hm",
    "ah", "oh", "i see", "got it", "understood",
    "makes sense", "noted", "wow", "waow", "woah",
    "whoa", "omg", "nice one",
}

# Negative
_NEGATIVE = {
    "no", "nope", "nah", "no thanks", "not now", "skip",
    "never mind", "nevermind", "no need", "not really",
    "no thank you", "nah thanks",
}

# ─────────────────────────────────────────────────────────────────────────────
# Off-topic guard  (Fine Flow terminology must NEVER be caught here)
# ─────────────────────────────────────────────────────────────────────────────

_OFF_TOPIC_PATTERNS = [
    # Programming / tech stacks
    r"\b(html|css|javascript|typescript|python|java|php|sql|react|angular"
    r"|vue|node\.?js|django|flask|laravel|spring|docker|kubernetes|github"
    r"|devops|backend|frontend|fullstack)\b",
    # Generic ML/AI (not Fine Flow AI features)
    r"\b(machine learning|deep learning|neural network|large language model"
    r"|generative ai|train a model|fine.?tune(?!flow)|llm|gpt|bert)\b",
    # Food / drink
    r"\b(recipe|cooking|restaurant|pizza|burger|sandwich|coffee|tea|cake"
    r"|meal|bake|order food|make me a|bake me|cook me|cook for me)\b",
    # Entertainment / general knowledge
    r"\b(movie|film|song|lyrics|music|football match|cricket match|rugby"
    r"|weather forecast|todays news|politics|history lesson|geography"
    r"|capital city|who invented|when was .+born)\b",
    # Writing tasks unrelated to Fine Flow
    r"\b(write me a poem|tell me a joke|write an essay|translate this"
    r"|proofread|write my cv|write my resume|write a story)\b",
    # Competitor AI
    r"\b(chatgpt|openai|gemini|claude ai|anthropic|google bard|bing ai"
    r"|alexa|siri|cortana|amazon echo)\b",
]
_OFF_TOPIC_RE = [re.compile(p, re.IGNORECASE) for p in _OFF_TOPIC_PATTERNS]


def _is_off_topic(q: str) -> bool:
    return any(p.search(q) for p in _OFF_TOPIC_RE)


# ─────────────────────────────────────────────────────────────────────────────
# Vehicle count extraction
# ─────────────────────────────────────────────────────────────────────────────

_VEH_EXPLICIT_RE = re.compile(
    r"\b(\d+)\s*(vehicle|vehicles|van|vans|truck|trucks|car|cars"
    r"|lorry|lorries|in my fleet|in our fleet|strong fleet)\b",
    re.IGNORECASE,
)
_VEH_HAVE_RE = re.compile(
    r"\b(?:i have|we have|we run|we got|i got|i manage|we manage"
    r"|thats|that s|its|i ve got|we ve got)\s+(\d+)\b",
    re.IGNORECASE,
)
_VEH_BARE_RE = re.compile(r"^\s*(\d+)\s*$")


def _extract_vehicle_count(query: str, sid: str) -> Optional[int]:
    m = _VEH_EXPLICIT_RE.search(query)
    if m:
        return int(m.group(1))
    m = _VEH_HAVE_RE.search(query)
    if m:
        return int(m.group(1))
    # Bare number only valid when recent topic was pricing/vehicles
    m = _VEH_BARE_RE.match(query)
    if m:
        lt = _gmeta(sid, "last_topic") or ""
        if lt in PRICING_TOPICS:
            return int(m.group(1))
    return None


def _plan_for_vehicles(n: int) -> str:
    if n <= 50:
        return (
            f"With {n} vehicles, the Essential plan at £99 per month is spot on for you — "
            "it covers up to 50 vehicles and includes the full platform with no locked features. "
            "Would you like to know what is included or how to get started?"
        )
    elif n <= 100:
        return (
            f"With {n} vehicles, the Core plan at £199 per month is the right fit — "
            "it covers up to 100 vehicles with everything included. "
            "Want me to walk you through what you get?"
        )
    elif n <= 200:
        return (
            f"With {n} vehicles, the Advanced plan at £399 per month covers you nicely — "
            "it handles up to 200 vehicles with full platform access and no hidden charges. "
            "Would you like to know more or speak to the sales team?"
        )
    else:
        return (
            f"With {n} vehicles, the Elite plan at £499 per month is built for an operation "
            "your size — unlimited vehicles and up to 1,000 fines per month. "
            "Want to know how to get started?"
        )


# Purchase intent
_PURCHASE_RE = re.compile(
    r"\b(want to buy|want to purchase|want to subscribe|want to sign up"
    r"|how do i buy|how do i get started|how do i sign up|how to subscribe"
    r"|get started|free trial|start a trial|sign me up|ready to buy"
    r"|i want it|book a demo|talk to sales|i want to get it|how to start"
    r"|where do i sign|how do i join|i d like to join)\b",
    re.IGNORECASE,
)

# ─────────────────────────────────────────────────────────────────────────────
# System prompt  — the brain of Nova
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are Nova, the AI assistant for Fine Flow — a UK fleet fine and PCN management platform.

You speak to fleet managers across the UK. Many are busy, practical people who want straight answers. Be warm, professional and concise. Think of yourself as a knowledgeable product expert who genuinely wants to help — not a search engine, not a robot.

══════════════════════════════════════════
ABSOLUTE RULES — follow every one, always
══════════════════════════════════════════

RULE 1 — TOPIC LOCK
You only answer questions about Fine Flow. If someone asks about anything unrelated — coding, food, general knowledge, other AI tools, personal questions — respond with exactly this one sentence:
"I can only help with Fine Flow questions — is there anything about fines, pricing, appeals or the platform I can help you with?"

RULE 2 — KEEP IT SHORT
3 sentences maximum per response. No bullet lists. No long paragraphs. One clear, useful point and then one question to move the conversation forward.

RULE 3 — ALWAYS END WITH A QUESTION
Every single response must end with a relevant question. Good examples:
  "How many vehicles are in your fleet?"
  "Would you like to know more about how appeals work?"
  "Want me to walk you through what is included?"
  "What is your typical monthly fine volume?"
  "Shall I explain how the driver assignment works?"

RULE 4 — TONE
Warm, direct, confident. Like a knowledgeable colleague, not a call centre script. Never start with "Certainly!", "Great question!", "Of course!" or "Absolutely!" — these sound hollow. Just answer naturally.

RULE 5 — NO FORMATTING
No bullet points. No bold text. No asterisks. No numbered lists. Plain conversational English only — this is a chat widget, not a document.

RULE 6 — USE THE CONTEXT PROVIDED
Use the Fine Flow knowledge base context given with each user message. If no context is provided, use the facts below. Never invent prices, features or policies.

RULE 7 — PAYMENT — THIS IS CRITICAL, NEVER GET THIS WRONG
Fine Flow does NOT automatically pay fines. It does NOT log into council or authority websites on your behalf. Payment portals use bot detection and card verification that blocks all automation. Fine Flow does everything up to the payment step — captures, assigns, tracks, organises — so when you do pay, it takes seconds not hours. Always say NO if asked whether Fine Flow pays fines automatically.

══════════════════════════════════════════
FINE FLOW FACTS — use only these, never invent
══════════════════════════════════════════

WHAT IT DOES:
Fine Flow is a UK fleet fine management platform. It monitors Gmail every minute for fine emails, extracts details using AI, matches fines to drivers via vehicle logs, manages disputes and appeals, and tracks every fine from receipt to resolution. It cuts admin time by up to 80%.

PRICING:
Essential:  £99/month  — up to 50 vehicles
Core:       £199/month — up to 100 vehicles
Advanced:   £399/month — up to 200 vehicles
Elite:      £499/month — unlimited vehicles
Per fine within monthly allowance: £0.75
Overage (beyond plan limit): £2.50 per fine
Pay-as-you-go (no subscription, no lock-in): £2.75 per fine
All plans include every feature — no paywalls, no locked features.
There is NO £2.00 fee anywhere.

REFERRAL PROGRAMME:
Fine Flow has an active referral programme. Referring a company earns you credits based on their fleet size. There are four tiers: Silver (3 referrals) gives 100 bonus credits; Gold (5 referrals) gives 10% off your subscription for 12 months; Platinum (10 referrals) gives 15% off for 12 months; Titan (25 referrals) gives 20% off for life. Any company joining with a referral code gets £75 in credits.

CONTACT:
Phone: +47 32 28 50 00
Email: ff.sales@fineflow.com
Offices: Edinburgh, Glasgow, Belfast, Manchester, London, Dublin, Hamburg

BILLING:
Monthly via Stripe. Credits reset each cycle — unused credits do not roll over. Cannot resubscribe with an outstanding balance. Vehicle limit exceeded = £10 per extra vehicle.

FINE STATUSES (in order):
RECEIVED → UNASSIGNED or ASSIGNED → CONFIRMED or DISPUTED → UNDER REVIEW → PAID, CANCELLED or OVERDUE

SECURITY:
JWT tokens (24-hour expiry), bcrypt passwords, AES-256-CBC encryption for Gmail tokens. GDPR compliant. No data sold or shared. Payments via Stripe — Fine Flow never stores card details.

SAVINGS:
Small fleet (5–50 vehicles): over £400/month saved. Medium fleet (51–200): over £1,200/month. Large fleet (200+): over £4,000/month.

══════════════════════════════════════════
CONVERSATION EXAMPLES — match this style
══════════════════════════════════════════

User: how much does it cost
Nova: Fine Flow starts at £99 a month for up to 50 vehicles, then £199 for up to 100 and £499 for unlimited — every plan includes the full platform with nothing locked away. How many vehicles are in your fleet?

User: will it pay my fines automatically
Nova: No — Fine Flow never pays fines on your behalf, and that is intentional. Council and authority payment portals use bot detection that blocks all automation. What Fine Flow does is organise everything so when you do pay, it takes seconds rather than the usual hour of chasing. Would you like to know more about how the process works?

User: is there a referral system
Nova: Yes, there is. When a company signs up with your referral code and subscribes, you earn bonus credits straight away — and as you hit milestones you unlock real subscription discounts, up to 20% off for life at the top tier. Would you like the full breakdown of how the tiers work?

User: i have 65 vehicles
Nova: With 65 vehicles, the Core plan at £199 a month is the right fit — it covers up to 100 vehicles with the full platform included. Want me to walk you through everything that comes with it?

User: yes
Nova: Every plan includes real-time fine alerts, automated driver assignment, full appeal management, compliance reporting and a 12-month fine history dashboard — and there is nothing locked behind a higher tier. Would you like to know how the appeal system works in practice?

User: what is html
Nova: I can only help with Fine Flow questions — is there anything about fines, pricing, appeals or the platform I can help you with?

User: make me a pizza
Nova: I can only help with Fine Flow questions — is there anything about fines, pricing, appeals or the platform I can help you with?

User: is my card safe
Nova: Fine Flow never stores card details — all payments are made by you directly on the authority's website, processed through their own systems. Fine Flow handles everything around the payment, not the payment itself. Is there anything else about security or data protection you would like to know?

User: do you have any discounts
Nova: Yes — through the referral programme. Refer five companies and you get 10% off your subscription for a year; refer 25 and it is 20% off for life. New companies joining with a referral code also get £75 in credits. Would you like to know how the referral programme works?
"""

# ─────────────────────────────────────────────────────────────────────────────
# OpenAI call — temperature 0.0 for deterministic, factual answers
# ─────────────────────────────────────────────────────────────────────────────

def _call_openai(messages: List[Dict[str, str]], max_tokens: int = 160) -> Optional[str]:
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
    """Semantic search → rerank → return top 3 chunks as context string."""
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
# Build GPT-4o message list with full memory
# ─────────────────────────────────────────────────────────────────────────────

def _make_messages(query: str, context: str, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT}]

    # Last 3 full turns (6 messages) — enough for real memory, not bloated
    msgs.extend(history[-6:])

    if context:
        user_text = (
            f"Fine Flow knowledge base (use this to answer accurately):\n{context}"
            f"\n\nUser message: {query}"
        )
    else:
        user_text = query

    msgs.append({"role": "user", "content": user_text})
    return msgs


# ─────────────────────────────────────────────────────────────────────────────
# Topic hint — set after every RAG response so bare numbers work next turn
# ─────────────────────────────────────────────────────────────────────────────

_TOPIC_HINTS = {
    "pric": "pricing",   "cost": "pricing",    "plan": "pricing",
    "package": "pricing", "subscription": "pricing", "£": "pricing",
    "appeal": "appeals",  "dispute": "appeals",  "challenge": "appeals",
    "driver": "driver_management",
    "referral": "referral_programme", "refer": "referral_programme",
    "discount": "referral_programme", "offer": "referral_programme",
    "security": "security", "gdpr": "security", "safe": "security",
    "billing": "billing",   "invoice": "billing", "stripe": "billing",
    "dashboard": "dashboard", "report": "reports",
    "gmail": "gmail_connection", "email": "email_ingestion",
    "save": "savings",    "time": "savings",   "money": "savings",
    "vehicle": "pricing", "fleet": "pricing",
}


def _detect_topic(query: str) -> Optional[str]:
    nq = query.lower()
    for hint, topic in _TOPIC_HINTS.items():
        if hint in nq:
            return topic
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Affirmative expansion — uses GPT-4o with last_topic context
# ─────────────────────────────────────────────────────────────────────────────

_TOPIC_EXPANSION_PROMPTS = {
    "pricing":              "The user said yes to learn more about Fine Flow pricing. Give a concise 3-sentence answer about the plans and ask how many vehicles they have.",
    "plan_recommendation":  "The user said yes to learn what is included in their recommended plan. Give a concise 3-sentence answer about what every Fine Flow plan includes and ask if they want to get started.",
    "appeals":              "The user said yes to learn more about Fine Flow appeals. Give a concise 3-sentence answer about how the appeal process works and ask if they have any fines they are currently disputing.",
    "driver_management":    "The user said yes to learn more about driver management. Give a concise 3-sentence answer about how drivers are added and managed and ask how many drivers they have.",
    "referral_programme":   "The user said yes to learn more about the referral programme. Give a concise 3-sentence answer about how the tiers and credit rewards work.",
    "security":             "The user said yes to learn more about Fine Flow security. Give a concise 3-sentence answer about data protection and GDPR compliance.",
    "billing":              "The user said yes to learn more about billing. Give a concise 3-sentence answer about how billing works and ask when they would like to start.",
    "dashboard":            "The user said yes to learn more about the dashboard. Give a concise 3-sentence answer about what the company dashboard shows.",
    "savings":              "The user said yes to learn more about savings. Give a concise 3-sentence answer about how much time and money Fine Flow saves fleet managers.",
    "email_ingestion":      "The user said yes to learn more about how Fine Flow gets fines. Give a concise 3-sentence answer about Gmail monitoring and data extraction.",
    "gmail_connection":     "The user said yes to learn how to connect Gmail. Give a concise 3-sentence answer about the OAuth and app password methods.",
}

_DEFAULT_EXPANSION = "The user said yes to continue the conversation about Fine Flow. Look at the conversation history and expand helpfully on the most recent topic in 3 sentences, then ask a follow-up question."


def _affirmative_response(sid: str, history: List[Dict[str, str]]) -> str:
    last_topic = _gmeta(sid, "last_topic") or ""
    expansion_prompt = _TOPIC_EXPANSION_PROMPTS.get(last_topic, _DEFAULT_EXPANSION)

    msgs: List[Dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    msgs.extend(history[-6:])
    msgs.append({"role": "user", "content": expansion_prompt})

    answer = _call_openai(msgs, max_tokens=160)
    return answer or "What would you like to know more about — pricing, how fines work, appeals or billing?"


# ─────────────────────────────────────────────────────────────────────────────
# Main response builder
# ─────────────────────────────────────────────────────────────────────────────

def build_response(query: str, session_id: str = "default") -> Dict[str, Any]:
    query      = query.strip()
    session_id = session_id or "default"

    if not query:
        return {"answer": "Ask me anything about Fine Flow.", "confidence": 1.0}

    nq = _norm(query)

    # ══════════════════════════════════════════════════════════════════
    # TIER 1 — Deterministic handlers  (fast, no AI, always correct)
    # ══════════════════════════════════════════════════════════════════

    # Greeting
    if nq in _GREETINGS:
        _reset_meta(session_id)
        return {
            "answer": "I'm Nova. Ask me anything — I'll help you manage fines, resolve issues, and keep everything moving.",
            "confidence": 1.0,
        }

    # Social
    if nq in _SOCIAL:
        return {
            "answer": "Doing well, thanks for asking. What can I help you with today — pricing, fines, or appeals?",
            "confidence": 1.0,
        }

    # Identity (who are you / who is there / anyone there)
    if nq in _IDENTITY:
        return {
            "answer": "I'm Nova, Fine Flow's AI assistant. I can help with anything about the platform — fines, pricing, appeals, billing and more. What would you like to know?",
            "confidence": 1.0,
        }

    # Thanks
    if nq in _THANKS:
        return {
            "answer": "Happy to help. Is there anything else you would like to know about Fine Flow?",
            "confidence": 1.0,
        }

    # Goodbye
    if nq in _GOODBYE:
        return {
            "answer": "Good luck with your fleet management. Come back any time if you have questions about Fine Flow.",
            "confidence": 1.0,
        }

    # Negative
    if nq in _NEGATIVE:
        _push(session_id, "user", query)
        answer = "No problem at all. Is there anything else about Fine Flow I can help you with?"
        _push(session_id, "assistant", answer)
        return {"answer": answer, "confidence": 1.0}

    # Rudeness — calm, not defensive
    if any(r in nq for r in _RUDE):
        return {
            "answer": "Let me try again. What specifically would you like to know about Fine Flow — pricing, how fines work, or something else?",
            "confidence": 1.0,
        }

    # Pure filler (cool, ok, interesting, wow etc) — NOT affirmatives
    if nq in _PURE_FILLER:
        lt = _gmeta(session_id, "last_topic")
        if lt:
            topic_label = lt.replace("_", " ")
            return {
                "answer": f"Glad you find it interesting. Would you like to know more about {topic_label}, or is there something else I can help you with?",
                "confidence": 1.0,
            }
        return {
            "answer": "Is there anything about Fine Flow I can help you with today — pricing, fines, appeals or the dashboard?",
            "confidence": 1.0,
        }

    # Off-topic (after identity/social so "are you an AI" is not caught)
    if _is_off_topic(query):
        answer = "I can only help with Fine Flow questions — is there anything about fines, pricing, appeals or the platform I can help you with?"
        _push(session_id, "user", query)
        _push(session_id, "assistant", answer)
        return {"answer": answer, "confidence": 1.0}

    # Vehicle count → instant plan recommendation
    vc = _extract_vehicle_count(query, session_id)
    if vc is not None:
        answer = _plan_for_vehicles(vc)
        _push(session_id, "user", query)
        _push(session_id, "assistant", answer)
        _smeta(session_id, "last_topic", "plan_recommendation")
        return {"answer": answer, "confidence": 1.0}

    # Purchase intent
    if _PURCHASE_RE.search(query):
        answer = (
            "To get started, get in touch with the Fine Flow sales team on +47 32 28 50 00 "
            "or at ff.sales@fineflow.com — they will have you set up quickly. "
            "How many vehicles are in your fleet so I can point you to the right plan?"
        )
        _push(session_id, "user", query)
        _push(session_id, "assistant", answer)
        _smeta(session_id, "last_topic", "pricing")
        return {"answer": answer, "confidence": 1.0}

    # ── Affirmative  ──────────────────────────────────────────────────
    # Must come AFTER vehicle/purchase checks so "yes sure" on a vehicle
    # question doesn't get swallowed here.
    if nq in _AFFIRMATIVE:
        _push(session_id, "user", query)
        history = _hist(session_id)
        answer  = _affirmative_response(session_id, history[:-1])
        answer  = _clean(answer)
        _push(session_id, "assistant", answer)
        return {"answer": answer, "confidence": 1.0}

    # ══════════════════════════════════════════════════════════════════
    # TIER 2 — RAG + GPT-4o  (all real product questions)
    # ══════════════════════════════════════════════════════════════════

    _push(session_id, "user", query)

    context  = _retrieve(query)
    history  = _hist(session_id)
    messages = _make_messages(query, context, history[:-1])

    answer = _call_openai(messages, max_tokens=160)

    if not answer:
        answer = (
            "The Fine Flow team can help with that directly — "
            "call +47 32 28 50 00 or email ff.sales@fineflow.com."
        )

    answer = _clean(answer)
    _push(session_id, "assistant", answer)

    # Store topic hint so follow-up bare numbers and affirmatives work
    topic = _detect_topic(query) or _detect_topic(answer)
    if topic:
        _smeta(session_id, "last_topic", topic)

    return {
        "answer": answer,
        "confidence": 0.9 if context else 0.5,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public wrapper
# ─────────────────────────────────────────────────────────────────────────────

def answer_sync(q: str, session_id: str = "default") -> Dict[str, Any]:
    try:
        return build_response(q, session_id)
    except Exception:
        logger.exception("Crash in answer_sync")
        return {"answer": "Something went wrong. Please try again.", "confidence": 0.0}