# app/answer_builder.py
"""
Nova — FineFlow Answer Builder (Production)
- Golden Q&A knowledge base (client approved)
- Semantic matching with OpenAI embeddings
- RAG fallback via ChromaDB
- Conversation memory
"""

import os
import time
import json
import re
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
SUPPORT_EMAIL = os.getenv("SUPPORT_EMAIL", "support@fineflow.com")
SALES_EMAIL = os.getenv("SALES_EMAIL", "sales@fineflow.com")

MIN_SECONDS_BETWEEN_CALLS = float(os.getenv("MIN_SECONDS_BETWEEN_CALLS", "0.25"))
CHAT_HISTORY_TURNS = int(os.getenv("CHAT_HISTORY_TURNS", "10"))

# ----------------------------
# 🔥 Load Golden Knowledge Base
# ----------------------------
KB_PATH = Path(__file__).parent.parent / "data" / "fineflow_kb.json"
_kb_questions: List[str] = []
_kb_answers: List[str] = []
_kb_embeddings: Optional[np.ndarray] = None

def _load_knowledge_base():
    global _kb_questions, _kb_answers, _kb_embeddings
    if not KB_PATH.exists():
        logger.warning(f"Knowledge base not found at {KB_PATH}")
        return
    try:
        with open(KB_PATH, 'r', encoding='utf8') as f:
            data = json.load(f)
        _kb_questions = [item["question"] for item in data]
        _kb_answers = [item["answer"] for item in data]
        logger.info(f"Loaded {len(_kb_questions)} golden Q&A pairs.")
        if OPENAI_API_KEY:
            _kb_embeddings = _compute_embeddings_batch(_kb_questions)
    except Exception as e:
        logger.exception("Failed to load knowledge base")

def _get_embedding(text: str) -> np.ndarray:
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "text-embedding-3-small", "input": [text]}
    r = requests.post("https://api.openai.com/v1/embeddings", headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    return np.array(r.json()["data"][0]["embedding"], dtype=np.float32)

def _compute_embeddings_batch(texts: List[str]) -> np.ndarray:
    batch_size = 50
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": "text-embedding-3-small", "input": batch}
        r = requests.post("https://api.openai.com/v1/embeddings", headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        batch_embs = [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
        embs.extend(batch_embs)
    emb_array = np.array(embs, dtype=np.float32)
    return emb_array / np.linalg.norm(emb_array, axis=1, keepdims=True)

def _match_kb(query: str, threshold: float = 0.75) -> Optional[str]:
    """Return best matching answer from golden KB if similarity > threshold."""
    if _kb_embeddings is None or not OPENAI_API_KEY:
        return None
    try:
        q_emb = _get_embedding(query)
        q_emb = q_emb / np.linalg.norm(q_emb)
        sim = np.dot(_kb_embeddings, q_emb)
        best_idx = np.argmax(sim)
        if sim[best_idx] > threshold:
            logger.info(f"KB match: score={sim[best_idx]:.3f}, question='{_kb_questions[best_idx][:50]}...'")
            return _kb_answers[best_idx]
    except Exception as e:
        logger.exception("KB matching failed")
    return None

# Load KB on import
_load_knowledge_base()

# ----------------------------
# 🎯 Client-Approved System Prompt
# ----------------------------
SYSTEM_PROMPT = """
You are Nova — the intelligent assistant for FineFlow, a UK fleet fine management platform.

IDENTITY:
- You ARE FineFlow. Use "we", not "they".
- Confident, professional, warm, and concise.

RULES:
- Always answer using the provided context (which includes approved Q&A).
- Be concise: 1–2 sentences unless user asks for detail.
- Use bullet points only when listing plans or features.
- No markdown formatting (no **bold**, `code`, # headers).
- If exact information is not in context, say: "I don't have that exact information. Please contact support@fineflow.com."
- Never guess or invent numbers.

VOICE:
- Opening greeting (when user says hi/hello): "I'm Nova. Ask me anything - I'll help you manage fines, resolve issues, and keep everything moving."
- Lead with value, end with a soft call to action when appropriate.
"""

# ----------------------------
# Memory (In‑Memory Sessions)
# ----------------------------
_SESSION_STORE: Dict[str, List[Dict[str, str]]] = {}
_LAST_CALL_TS: Dict[str, float] = {}
_LOCK = threading.Lock()

def _get_session(session_id: str) -> List[Dict[str, str]]:
    with _LOCK:
        return _SESSION_STORE.setdefault(session_id, [])

def _add_to_session(session_id: str, role: str, content: str):
    with _LOCK:
        hist = _get_session(session_id)
        hist.append({"role": role, "content": content})
        max_len = CHAT_HISTORY_TURNS * 2
        if len(hist) > max_len:
            _SESSION_STORE[session_id] = hist[-max_len:]

def _rate_limit(session_id: str):
    now = time.time()
    with _LOCK:
        last = _LAST_CALL_TS.get(session_id, 0.0)
        wait = MIN_SECONDS_BETWEEN_CALLS - (now - last)
        if wait > 0:
            time.sleep(wait)
        _LAST_CALL_TS[session_id] = time.time()

# ----------------------------
# OpenAI Call
# ----------------------------
def _call_openai(messages: List[Dict[str, str]], max_tokens: int = 350) -> str:
    if not OPENAI_API_KEY:
        return "Service configuration error."
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": OPENAI_MODEL, "messages": messages, "temperature": 0.3, "max_tokens": max_tokens}
    for attempt in range(3):
        try:
            r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=25)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.warning(f"OpenAI attempt {attempt+1} failed: {e}")
            time.sleep(1.5 * (attempt+1))
    return "I'm having trouble responding. Please try again."

# ----------------------------
# Build Messages with Context
# ----------------------------
def _build_messages(query: str, context: str, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-6:])
    user_content = f"""Context from FineFlow documentation:
{context if context else "No relevant documentation found."}

User question: {query}

Answer as Nova (concise, accurate, professional):"""
    messages.append({"role": "user", "content": user_content})
    return messages

# ----------------------------
# Main Response Builder
# ----------------------------
def build_response(query: str, session_id: str = "default") -> Dict[str, Any]:
    query = query.strip()
    session_id = session_id or "default"

    _rate_limit(session_id)
    _add_to_session(session_id, "user", query)

    # 1️⃣ Handle empty / greetings
    if not query:
        reply = "What would you like to know about FineFlow?"
        _add_to_session(session_id, "assistant", reply)
        return {"answer": reply, "confidence": 1.0, "sources": [], "flag": False}

    lower_q = query.lower()
    if lower_q in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]:
        reply = "I'm Nova. Ask me anything - I'll help you manage fines, resolve issues, and keep everything moving."
        _add_to_session(session_id, "assistant", reply)
        return {"answer": reply, "confidence": 1.0, "sources": [], "flag": False}

    # 2️⃣ Golden KB match (priority)
    kb_answer = _match_kb(query)
    if kb_answer:
        _add_to_session(session_id, "assistant", kb_answer)
        return {"answer": kb_answer, "confidence": 1.0, "sources": [], "flag": False}

    # 3️⃣ RAG fallback
    try:
        docs = rag_search(query, top_k=TOP_K)
    except Exception:
        docs = []
    context = "\n\n".join([d.get("chunk", "")[:600] for d in docs if d.get("chunk")])

    history = _get_session(session_id)
    messages = _build_messages(query, context, history)

    answer = _call_openai(messages)
    if not answer:
        answer = "I couldn't find that information. Please contact support@fineflow.com."

    _add_to_session(session_id, "assistant", answer)

    confidence = float(docs[0]["score"]) if docs else 0.2
    sources = [{"title": d.get("meta", {}).get("title", "doc"), "score": d["score"]} for d in docs[:3]]

    return {"answer": answer, "confidence": confidence, "sources": sources, "flag": len(docs) == 0}

def answer_sync(q: str, session_id: str = "default") -> Dict[str, Any]:
    try:
        return build_response(q, session_id)
    except Exception:
        logger.exception("answer_sync failed")
        return {"answer": "Internal error.", "confidence": 0.0, "sources": [], "flag": True}

logger.info(" Nova answer_builder loaded — golden KB + RAG fallback, memory enabled.")