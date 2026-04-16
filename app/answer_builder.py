# app/answer_builder.py

import os
import time
import requests
from typing import List, Dict, Any

from app.logger import logger
from app.retriever import search as rag_search

# ----------------------------
# Config
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
TOP_K = int(os.getenv("TOP_K", "5"))

MIN_SECONDS_BETWEEN_CALLS = float(os.getenv("MIN_SECONDS_BETWEEN_CALLS", "0.25"))
CHAT_HISTORY_TURNS = int(os.getenv("CHAT_HISTORY_TURNS", "10"))

# ----------------------------
# 🔥 MASTER PROMPT (CLIENT READY)
# ----------------------------
SYSTEM_PROMPT = """
You are Nova — the intelligent assistant for FineFlow, a UK-based fleet fine management platform.

IDENTITY:
- You ARE FineFlow. Speak as "we"
- Confident, professional, human tone

RULES:
- Keep answers concise (1–2 sentences)
- Use bullet points only for lists
- Do NOT use markdown
- ONLY use provided context
- NEVER guess or invent data

IF INFO IS MISSING:
Say:
"I don't have that exact information. Please contact support@fineflow.com."

GOAL:
Be accurate, helpful, and natural like a real product expert.
"""

# ----------------------------
# Memory
# ----------------------------
_SESSION_STORE: Dict[str, List[Dict[str, str]]] = {}
_LAST_CALL_TS: Dict[str, float] = {}

def _get_session(session_id: str):
    return _SESSION_STORE.setdefault(session_id, [])

def _add_to_session(session_id: str, role: str, content: str):
    hist = _get_session(session_id)
    hist.append({"role": role, "content": content})

    max_len = CHAT_HISTORY_TURNS * 2
    if len(hist) > max_len:
        _SESSION_STORE[session_id] = hist[-max_len:]

def _rate_limit(session_id: str):
    now = time.time()
    last = _LAST_CALL_TS.get(session_id, 0)

    wait = MIN_SECONDS_BETWEEN_CALLS - (now - last)
    if wait > 0:
        time.sleep(wait)

    _LAST_CALL_TS[session_id] = time.time()

# ----------------------------
# OpenAI Call (robust)
# ----------------------------
def call_openai(messages: List[Dict[str, str]]) -> str:
    if not OPENAI_API_KEY:
        return "Service configuration error."

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 300
    }

    try:
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=25
        )
        r.raise_for_status()

        content = r.json()["choices"][0]["message"]["content"]

        if not content or not content.strip():
            return ""

        return content.strip()

    except Exception:
        logger.exception("OpenAI failed")
        return ""

# ----------------------------
# Build Messages
# ----------------------------
def build_messages(query: str, context: str, history: List[Dict[str, str]]):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # last 6 messages only
    messages += history[-6:]

    messages.append({
        "role": "user",
        "content": f"""
Context:
{context if context else "No relevant documentation found."}

User question:
{query}

Answer as Nova:
"""
    })

    return messages

# ----------------------------
# MAIN
# ----------------------------
def build_response(query: str, session_id: str = "default") -> Dict[str, Any]:
    query = query.strip()
    session_id = session_id or "default"

    _rate_limit(session_id)
    _add_to_session(session_id, "user", query)

    # empty
    if not query:
        return {
            "answer": "What would you like to know about FineFlow?",
            "confidence": 1.0,
            "sources": [],
            "flag": False
        }

    # greeting
    if query.lower() in ["hi", "hello", "hey"]:
        return {
            "answer": "Hi, I'm Nova from FineFlow. How can I help?",
            "confidence": 1.0,
            "sources": [],
            "flag": False
        }

    # ----------------------------
    # RAG
    # ----------------------------
    try:
        docs = rag_search(query, top_k=TOP_K)
    except Exception:
        logger.exception("RAG failed")
        docs = []

    context = "\n\n".join([
        d.get("chunk", "")[:500]
        for d in docs if d.get("chunk")
    ])

    history = _get_session(session_id)
    messages = build_messages(query, context, history)

    # ----------------------------
    # OpenAI
    # ----------------------------
    answer = call_openai(messages)

    # ----------------------------
    # HARD FAILSAFE (IMPORTANT)
    # ----------------------------
    if not answer:
        if docs:
            answer = docs[0].get("chunk", "")[:200]
        else:
            answer = "I couldn't find that information. Please contact support@fineflow.com."

    _add_to_session(session_id, "assistant", answer)

    # ----------------------------
    # Confidence (REAL)
    # ----------------------------
    confidence = float(docs[0]["score"]) if docs else 0.2

    return {
        "answer": answer,
        "confidence": confidence,
        "sources": docs[:3],
        "flag": len(docs) == 0
    }

# ----------------------------
# FastAPI wrapper
# ----------------------------
def answer_sync(q: str, session_id: str = "default") -> Dict[str, Any]:
    try:
        return build_response(q, session_id)
    except Exception:
        logger.exception("answer_sync failed")
        return {
            "answer": "Internal error.",
            "confidence": 0.0,
            "sources": [],
            "flag": True
        }

logger.info("✅ Nova ready (production safe)")