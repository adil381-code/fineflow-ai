"""
FineFlow Nova — Answer Builder (Adaptive AI Edition)
- Personality: Authentic, adaptive, helpful peer (Gemini-style)
- Format: Clean Markdown (Bold/Lists), NO LaTeX for simple text, NO underscores
- Logic: KB First -> OpenAI RAG Fallback
"""

import json
import re
import threading
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

# ---------------------------------------------------------------------------
# KB & SESSION SETUP
# ---------------------------------------------------------------------------
KB_PATH = Path(__file__).parent.parent / "data" / "fineflow_kb.json"
_kb: List[Dict[str, Any]] = []

def _load_kb() -> None:
    if not KB_PATH.exists(): return
    try:
        with open(KB_PATH, "r", encoding="utf-8") as f:
            global _kb
            _kb = json.load(f)
    except Exception: logger.exception("KB load failed")

_load_kb()

_SESSION: Dict[str, List[Dict[str, str]]] = {}
_SESSION_CONTEXT: Dict[str, Optional[Dict[str, Any]]] = {}
_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# REFINED SYSTEM PROMPT (The "Gemini" Personality)
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are Nova, an authentic and adaptive AI collaborator for FineFlow. 
Your goal is to be a helpful peer, not a rigid lecturer. 

GUIDING PRINCIPLES:
1. TONE: Warm, grounded, and slightly witty. Use clear, concise language. 
2. EMPATHY: Validate the user's situation (e.g., "I know fleet fines can be a headache") before diving into the solution.
3. CANDOR: Be direct. If the user asks for something FineFlow doesn't do (like pay fines automatically), correct them gently but clearly.
4. FORMATTING: Use Markdown (bolding, bullet points) to make responses scannable. Avoid dense walls of text.
5. NO REPETITION: Don't start with "Certainly!" or "I understand." Just jump in.

PRODUCT KNOWLEDGE:
- FineFlow automates detection (every minute), extraction (PCN details), and driver assignment (via logs).
- Statuses: RECEIVED -> ASSIGNED -> CONFIRMED/DISPUTED -> PAID/CANCELLED.
- PRICING: Essential (€99/50 vehicles), Core (€199/100 vehicles), Advanced (€399/200 vehicles), Elite (€499/200+).
- FEES: Within allowance is included. Overage is £2.50. NO £2.00 fee exists (correct users on this).

STRICT RULES:
- Never say "I am an AI" or "I don't have that info." Give the best logical answer based on context.
- Use 'becomes' instead of arrows. 
- Do not use LaTeX for simple numbers or units (e.g., use 10%, not $10\%$)."""

# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------
def _clean(text: str) -> str:
    """Ensures output is clean but maintains helpful Markdown."""
    if not text: return ""
    # Remove weird symbols but KEEP bolding and lists for scannability
    text = text.replace("→", "becomes").replace("->", "becomes")
    text = text.replace("±", "plus or minus")
    text = re.sub(r"_{2,}", "", text) # Remove long underscores
    return text.strip()

def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()

def _kb_match(query: str) -> Optional[Dict[str, Any]]:
    nq = _normalise(query)
    for entry in _kb:
        if nq in _normalise(entry.get("question", "")): return entry
    return None

# ---------------------------------------------------------------------------
# CORE LOGIC
# ---------------------------------------------------------------------------
def build_response(query: str, session_id: str = "default") -> Dict[str, Any]:
    query = query.strip()
    if not query: return {"answer": "I'm here! What's on your mind regarding FineFlow?", "confidence": 1.0}

    # 1. Quick Intent Checks (Greetings/Social)
    nq = _normalise(query)
    if any(g in nq for g in ["hi", "hello", "hey"]):
        return {"answer": "Hey! I'm Nova. I help keep your fleet fines organized so you don't have to. What can I dive into for you?", "confidence": 1.0}
    
    if any(s in nq for s in ["how are you", "how r u"]):
        return {"answer": "I'm doing great, thanks for asking! Just ready to crunch some data. How can I help with your fleet today?", "confidence": 1.0}

    # 2. KB Match
    kb_entry = _kb_match(query)
    if kb_entry:
        ans = _clean(kb_entry["answer"])
        if "follow_up" in kb_entry: ans += f"\n\n{kb_entry['follow_up']}"
        return {"answer": ans, "confidence": 1.0}

    # 3. RAG + OpenAI Fallback
    docs = []
    try:
        raw = rag_search(query, top_k=TOP_K)
        docs = rerank_hits(raw, query)
    except: logger.exception("RAG failed")

    context = "\n".join([d["chunk"] for d in docs[:2]]) if docs else ""
    
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
    ]

    answer = _call_openai(messages) or "I'm having a bit of trouble reaching my brain. Check the dashboard or try again in a second!"
    
    return {
        "answer": _clean(answer),
        "confidence": docs[0]["score"] if docs else 0.5
    }

def _call_openai(messages: List[Dict[str, str]]) -> Optional[str]:
    try:
        r = requests.post(OPENAI_API_URL, headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                          json={"model": OPENAI_MODEL, "messages": messages, "temperature": 0.6}, timeout=15)
        return r.json()["choices"][0]["message"]["content"].strip()
    except: return None

def answer_sync(q: str, session_id: str = "default") -> Dict[str, Any]:
    try: return build_response(q, session_id)
    except: return {"answer": "Oops, something went wrong on my end. Try that again?", "confidence": 0.0}