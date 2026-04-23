# app/answer_builder.py

import os
import time
import json
import threading
from typing import List, Dict, Any, Optional
from pathlib import Path

import requests
import numpy as np

from app.logger import logger
from app.retriever import search as rag_search

# ----------------------------
# Config
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

TOP_K = int(os.getenv("TOP_K", "3"))

MIN_SECONDS_BETWEEN_CALLS = 0.2
CHAT_HISTORY_TURNS = 6

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_EMBED_URL = "https://api.openai.com/v1/embeddings"

# ----------------------------
# Load KB
# ----------------------------
KB_PATH = Path(__file__).parent.parent / "data" / "fineflow_kb.json"

_kb_questions = []
_kb_answers = []
_kb_embeddings = None


def _load_kb():
    global _kb_questions, _kb_answers

    if not KB_PATH.exists():
        logger.warning("KB not found")
        return

    with open(KB_PATH, "r", encoding="utf8") as f:
        data = json.load(f)

    _kb_questions = [x["question"] for x in data]
    _kb_answers = [x["answer"] for x in data]

    logger.info(f"Loaded {len(_kb_questions)} KB pairs")


_load_kb()

# ----------------------------
# Memory
# ----------------------------
_SESSION = {}
_LOCK = threading.Lock()


def _get_history(sid):
    with _LOCK:
        return _SESSION.setdefault(sid, [])


def _add_history(sid, role, content):
    with _LOCK:
        hist = _SESSION.setdefault(sid, [])
        hist.append({"role": role, "content": content})
        if len(hist) > CHAT_HISTORY_TURNS * 2:
            _SESSION[sid] = hist[-CHAT_HISTORY_TURNS * 2:]


# ----------------------------
# OpenAI Safe Call
# ----------------------------
def _call_openai(messages):
    if not OPENAI_API_KEY:
        return None

    try:
        r = requests.post(
            OPENAI_CHAT_URL,
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENAI_MODEL,
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 300,
            },
            timeout=15,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    except Exception as e:
        logger.warning(f"OpenAI failed: {e}")
        return None


# ----------------------------
# Simple KB match (NO embeddings → fast + stable)
# ----------------------------
def _simple_kb_match(query: str) -> Optional[str]:
    q = query.lower()

    for i, question in enumerate(_kb_questions):
        if q in question.lower():
            return _kb_answers[i]

    return None


# ----------------------------
# Build Response
# ----------------------------
def build_response(query: str, session_id: str = "default") -> Dict[str, Any]:
    query = query.strip()
    session_id = session_id or "default"

    _add_history(session_id, "user", query)

    # 1. Empty
    if not query:
        return {"answer": "Ask me something about FineFlow.", "confidence": 1.0}

    # 2. Greeting
    if query.lower() in ["hi", "hello", "hey"]:
        return {
            "answer": "I'm Nova. Ask me anything about FineFlow.",
            "confidence": 1.0,
        }

    # 3. KB match (FAST)
    kb = _simple_kb_match(query)
    if kb:
        return {"answer": kb, "confidence": 1.0}

    # 4. RAG (SAFE)
    try:
        docs = rag_search(query, top_k=TOP_K)
    except Exception:
        docs = []

    context = "\n\n".join([d["chunk"][:500] for d in docs]) if docs else ""

    # 5. OpenAI (OPTIONAL)
    messages = [
        {
            "role": "system",
            "content": "You are Nova, assistant for FineFlow. Answer clearly and short.",
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{query}",
        },
    ]

    answer = _call_openai(messages)

    # 6. Fallback if OpenAI fails
    if not answer:
        if context:
            answer = context[:300]  # return best chunk
        else:
            answer = "I don't have that info. Contact support."

    _add_history(session_id, "assistant", answer)

    return {
        "answer": answer,
        "confidence": docs[0]["score"] if docs else 0.3,
        "sources": docs[:2],
    }


def answer_sync(q: str, session_id: str = "default"):
    try:
        return build_response(q, session_id)
    except Exception as e:
        logger.exception("Crash in answer")
        return {"answer": "Error occurred.", "confidence": 0}