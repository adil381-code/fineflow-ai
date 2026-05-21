# app/answer_builder.py
"""
FineFlow Nova — Production Final v12
Built against every failure seen in testing.
Architecture: deterministic intent layer → RAG → GPT-4o with full memory.
"""

import re
import threading
import uuid
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
# Session memory  (thread-safe)
# ─────────────────────────────────────────────────────────────────────────────

_SESSION : Dict[str, List[Dict[str, str]]] = {}
_META    : Dict[str, Dict[str, Any]]       = {}
_LOCK    = threading.Lock()

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
    "who are you", "who r you", "who r u",
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
    "knock knock",  # common opener — treat as identity
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

# Pure conversational filler — NOT affirmative
_PURE_FILLER = {
    "ok", "okay", "right", "alright", "cool", "nice",
    "interesting", "really", "seriously",
    "hmm", "hm", "ah", "oh", "i see",
    "got it", "understood", "makes sense", "noted",
    "wow", "waow", "woah", "whoa", "omg",
    "anything", "something", "whatever",
}


# ─────────────────────────────────────────────────────────────────────────────
# Off-topic guard
# ─────────────────────────────────────────────────────────────────────────────

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
    r"|capital city|who invented)\b",
    r"\b(write me a poem|tell me a joke|write an essay|translate this"
    r"|proofread|write my cv|write my resume|write a story)\b",
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
_VEH_HAVE_RE = re.compile(
    r"\b(?:i have|we have|we run|we got|i got|i manage|we manage"
    r"|i ve got|we ve got|its|i barked i have)\s+(\d+)\b",
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
    m = _VEH_BARE_RE.match(query)
    if m:
        lt = _gmeta(sid, "last_topic") or ""
        if lt in PRICING_TOPICS:
            return int(m.group(1))
    return None


def _plan_for_vehicles(n: int) -> str:
    if n <= 50:
        return (
            f"With {n} vehicles, the Essential plan at £99 per month is spot on — "
            "it covers up to 50 vehicles and gives you the full platform with nothing locked away. "
            "Would you like to know what is included or how to get started?"
        )
    elif n <= 100:
        return (
            f"With {n} vehicles, the Core plan at £199 per month is the right fit — "
            "covers up to 100 vehicles with everything included from day one. "
            "Want me to walk you through what you get?"
        )
    elif n <= 200:
        return (
            f"With {n} vehicles, the Advanced plan at £399 per month covers you nicely — "
            "handles up to 200 vehicles with full platform access and no hidden charges. "
            "Would you like to know more or speak to the sales team?"
        )
    else:
        return (
            f"With {n} vehicles, the Elite plan at £499 per month is built for an operation "
            "your size — unlimited vehicles, up to 1,000 fines per month, everything included. "
            "Want to know how to get started?"
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
# Topic detection — stored after every response for affirmative follow-ups
# ─────────────────────────────────────────────────────────────────────────────

_TOPIC_HINTS = {
    "pric": "pricing",        "cost": "pricing",      "plan": "pricing",
    "package": "pricing",     "subscription": "pricing",
    "£": "pricing",           "vehicle": "pricing",   "fleet": "pricing",
    "appeal": "appeals",      "dispute": "appeals",   "challenge": "appeals",
    "driver": "driver_mgmt",
    "referral": "referral",   "refer": "referral",
    "discount": "referral",   "offer": "referral",
    "security": "security",   "gdpr": "security",     "safe": "security",
    "billing": "billing",     "invoice": "billing",   "stripe": "billing",
    "dashboard": "dashboard", "report": "reports",
    "gmail": "gmail",         "email": "email",
    "save": "savings",        "time": "savings",      "money": "savings",
    "overdue": "overdue",     "deadline": "overdue",
    "match": "matching",      "assign": "matching",
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

_SYSTEM_PROMPT = """You are Nova, the AI assistant for Fine Flow — a UK fleet fine and PCN management platform.

You speak to fleet managers across the UK. Be warm, professional and concise. You are a knowledgeable product expert who genuinely wants to help — not a search engine.

════════════════════════════════
ABSOLUTE RULES
════════════════════════════════

1. TOPIC LOCK
Only answer Fine Flow questions. For anything unrelated respond with exactly:
"I can only help with Fine Flow questions — is there anything about fines, pricing, appeals or the platform I can help you with?"

2. LENGTH — STRICT
Maximum 3 sentences. No bullet lists. No long paragraphs. One clear point, then one question.

3. ALWAYS END WITH A QUESTION
Every response ends with one short relevant question. Examples:
"How many vehicles are in your fleet?"
"Would you like to know more about how appeals work?"
"Want me to walk you through what is included?"
"What is your typical monthly fine volume?"

4. TONE
Warm, direct, confident. Like a knowledgeable colleague. Never start with "Certainly!", "Great question!", "Of course!", "Absolutely!" — just answer naturally.

5. NO FORMATTING
No bullet points. No bold. No asterisks. No numbered lists. Plain conversational English only.

6. NEVER INVENT
Only use facts from the context provided or the facts listed below. Never make up prices, features or policies.

════════════════════════════════
PAYMENT — NEVER GET THIS WRONG
════════════════════════════════
Fine Flow does NOT automatically pay fines. It does NOT log into council or authority websites.
Payment portals use bot detection and card verification that blocks all automation.
Fine Flow does everything up to the payment step. When you are ready to pay, it takes you to the authority page with everything already prepared — the payment itself is done by you.
If asked whether Fine Flow pays fines automatically, always say NO clearly and explain why.

════════════════════════════════
FINE FLOW FACTS
════════════════════════════════

PRICING:
Essential:  £99/month  — up to 50 vehicles
Core:       £199/month — up to 100 vehicles
Advanced:   £399/month — up to 200 vehicles
Elite:      £499/month — unlimited vehicles
Per fine within monthly allowance: £0.75
Overage (beyond plan limit): £2.50 per fine
Pay-as-you-go (no subscription): £2.75 per fine
All plans include every feature — no paywalls, no locked features.
There is NO £2.00 fee anywhere in Fine Flow.

REFERRAL PROGRAMME:
Fine Flow has an active referral programme.
Earn credits when a referred company subscribes:
  1-25 vehicles: 100 credits
  26-100 vehicles: 250 credits
  101-500 vehicles: 750 credits
  500+ vehicles: 2,000 credits
Tiers: Silver (3 referrals) = 100 bonus credits one-time; Gold (5) = 10% off 12 months; Platinum (10) = 15% off 12 months; Titan (25) = 20% off for life.
New joiners with a referral code get £75 in credits.

CONTACT:
Phone: +47 32 28 50 00
Email: ff.sales@fineflow.com
Offices: Edinburgh, Glasgow, Belfast, Manchester, London, Dublin, Hamburg

BILLING:
Monthly via Stripe. Credits reset each cycle, no rollover.
Cannot resubscribe with outstanding balance.
£10/vehicle if vehicle limit is exceeded.

SECURITY:
JWT 24-hour tokens, bcrypt passwords, AES-256-CBC for Gmail tokens.
GDPR compliant. No data sold or shared. Fine Flow never stores card details.

SAVINGS:
Small fleet (up to 50 vehicles): over £400/month saved
Medium fleet (51-200): over £1,200/month saved
Large fleet (200+): over £4,000/month saved
Admin time cut by up to 80%.

════════════════════════════════
EXAMPLE RESPONSES — match this style exactly
════════════════════════════════

User: how much does it cost
Nova: Fine Flow starts at £99 a month for up to 50 vehicles, rising to £199 for up to 100 and £499 for unlimited — every plan includes the full platform with nothing locked away. How many vehicles are in your fleet?

User: will it pay my fines automatically
Nova: No — and that is intentional. Authority payment portals use bot detection and card verification that makes automation impossible, so payment is always done by you on their site. What Fine Flow does is organise and prepare everything so by the time you pay, there is no thinking left — just click and complete. Would you like to know more about how the process works?

User: is there a referral system
Nova: Yes — Fine Flow has an active referral programme. When a company joins with your referral code and subscribes, you earn bonus credits straight away, and as you hit milestones you unlock real subscription discounts up to 20% off for life. Would you like the full breakdown of how the tiers work?

User: i have 65 vehicles
Nova: With 65 vehicles, the Core plan at £199 a month is the right fit — it covers up to 100 vehicles with the full platform included from day one. Want me to walk you through everything that comes with it?

User: yes
Nova: Every Fine Flow plan includes real-time fine alerts, automated driver assignment, full appeal management and compliance reporting — and nothing is locked behind a higher tier. Would you like to know how the appeal system works in practice?

User: can you appeal a fine for me
Nova: Yes — Fine Flow does not just manage appeals, it learns from them. It generates the appeal using the fine details and historical outcomes, and can send it directly to the issuing authority on your behalf. Over time it refines its approach based on wins and losses, giving you a success probability before you submit. Would you like to know how the process works step by step?

User: do you have any discounts
Nova: Yes — through the referral programme. Refer five companies and you get 10% off your subscription for a year; refer 25 and it is 20% off for life. New companies joining with a referral code also get £75 in credits to start with. Would you like me to explain how the referral system works?

User: what is html
Nova: I can only help with Fine Flow questions — is there anything about fines, pricing, appeals or the platform I can help you with?

User: knock knock
Nova: I am Nova, Fine Flow's AI assistant — happy to help with anything about the platform. What would you like to know about fines, pricing or appeals?
"""


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI call
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
# Build GPT-4o messages with full memory
# ─────────────────────────────────────────────────────────────────────────────

def _make_messages(
    query: str,
    context: str,
    history: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    msgs.extend(history[-6:])   # last 3 full turns

    if context:
        user_text = (
            f"Fine Flow knowledge base — use this to answer accurately:\n{context}"
            f"\n\nUser: {query}"
        )
    else:
        user_text = query

    msgs.append({"role": "user", "content": user_text})
    return msgs


# ─────────────────────────────────────────────────────────────────────────────
# Affirmative expansion — expand on last_topic using GPT-4o + full history
# ─────────────────────────────────────────────────────────────────────────────

_EXPAND: Dict[str, str] = {
    "pricing":           "The user wants to know more about Fine Flow pricing. Explain concisely in 3 sentences and ask how many vehicles they have.",
    "plan_recommendation": "The user said yes to learn what is included in their plan. Explain in 3 sentences what every Fine Flow plan includes then ask if they want to get started.",
    "appeals":           "The user wants more detail on how Fine Flow appeals work. Explain the appeal process in 3 sentences and ask if they have a fine they are thinking of disputing.",
    "driver_mgmt":       "The user wants more detail on driver management. Explain in 3 sentences how drivers are added and matched to fines then ask how many drivers they have.",
    "referral":          "The user wants the full referral breakdown. Explain the credit rewards and tier discounts in 3 sentences.",
    "security":          "The user wants more detail on security. Explain Fine Flow's data protection and GDPR approach in 3 sentences.",
    "billing":           "The user wants more detail on billing. Explain how billing works in 3 sentences and ask when they would like to start.",
    "dashboard":         "The user wants more detail on the dashboard. Explain what the company dashboard shows in 3 sentences.",
    "savings":           "The user wants to know more about time and cost savings. Explain the savings Fine Flow delivers in 3 sentences.",
    "email":             "The user wants to know more about how Fine Flow gets fines. Explain Gmail monitoring and extraction in 3 sentences.",
    "gmail":             "The user wants to know how to connect Gmail. Explain OAuth and app password methods in 3 sentences.",
    "overdue":           "The user wants to know more about overdue fines. Explain how Fine Flow handles overdue fines in 3 sentences.",
    "matching":          "The user wants to know more about driver matching. Explain how Fine Flow matches fines to drivers in 3 sentences.",
    "reports":           "The user wants to know more about reports. Explain the report types available in Fine Flow in 3 sentences.",
}

_DEFAULT_EXPAND = (
    "Look at the conversation history and expand helpfully on the most recent "
    "Fine Flow topic discussed. Keep it to 3 sentences and end with a question."
)


def _affirmative_response(sid: str, history: List[Dict[str, str]]) -> str:
    lt     = _gmeta(sid, "last_topic") or ""
    prompt = _EXPAND.get(lt, _DEFAULT_EXPAND)

    context = _retrieve(lt.replace("_", " ")) if lt else ""

    msgs: List[Dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    msgs.extend(history[-6:])

    if context:
        user_text = (
            f"Fine Flow knowledge base:\n{context}"
            f"\n\nInstruction: {prompt}"
        )
    else:
        user_text = prompt

    msgs.append({"role": "user", "content": user_text})

    answer = _call_openai(msgs, max_tokens=160)
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

    # ══════════════════════════════════════════════
    # TIER 1 — deterministic, no AI, always correct
    # ══════════════════════════════════════════════

    if nq in _GREETINGS:
        _reset_meta(session_id)
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
        a = "No problem at all. Is there anything else about Fine Flow I can help you with?"
        _push(session_id, "assistant", a)
        return {"answer": a, "confidence": 1.0}

    if any(r in nq for r in _RUDE):
        return {
            "answer": "Let me try again. What specifically would you like to know about Fine Flow — pricing, how fines work, or something else?",
            "confidence": 1.0,
        }

    if nq in _PURE_FILLER:
        lt = _gmeta(session_id, "last_topic")
        if lt:
            label = lt.replace("_", " ")
            return {
                "answer": f"Glad that is useful. Would you like to know more about {label}, or is there something else I can help with?",
                "confidence": 1.0,
            }
        return {
            "answer": "Is there anything about Fine Flow I can help you with today — pricing, fines, appeals or the dashboard?",
            "confidence": 1.0,
        }

    # Off-topic (after identity/social so "are you an AI" passes through)
    if _is_off_topic(query):
        a = "I can only help with Fine Flow questions — is there anything about fines, pricing, appeals or the platform I can help you with?"
        _push(session_id, "user", query)
        _push(session_id, "assistant", a)
        return {"answer": a, "confidence": 1.0}

    # Vehicle count → instant plan recommendation
    vc = _extract_vehicle_count(query, session_id)
    if vc is not None:
        a = _plan_for_vehicles(vc)
        _push(session_id, "user", query)
        _push(session_id, "assistant", a)
        _smeta(session_id, "last_topic", "plan_recommendation")
        return {"answer": a, "confidence": 1.0}

    # Purchase intent
    if _PURCHASE_RE.search(query):
        a = (
            "To get started, contact the Fine Flow sales team on +47 32 28 50 00 "
            "or at ff.sales@fineflow.com — they will get you set up quickly. "
            "How many vehicles are in your fleet so I can point you to the right plan?"
        )
        _push(session_id, "user", query)
        _push(session_id, "assistant", a)
        _smeta(session_id, "last_topic", "pricing")
        return {"answer": a, "confidence": 1.0}

    # Affirmative — expand on last topic using GPT-4o + history
    if nq in _AFFIRMATIVE:
        _push(session_id, "user", query)
        history = _hist(session_id)
        a = _clean(_affirmative_response(session_id, history[:-1]))
        _push(session_id, "assistant", a)
        return {"answer": a, "confidence": 1.0}

    # ══════════════════════════════════════════════
    # TIER 2 — RAG + GPT-4o (all product questions)
    # ══════════════════════════════════════════════

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

    # Store topic hint for follow-ups
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