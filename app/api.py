# app/api.py
"""
FineFlow Nova API — Production with MySQL Memory + Ticketing
==============================================================
Endpoints:
  POST /customer          - find or create user by email
  ASK /ask               - send message, get answer (+ ticket popup flag)
  GET  /history/{user_id}  - load full chat history for a user
  POST /ticket              - create a support ticket

Guest users: omit user_id (or send 0) — in-memory session only, nothing saved.
Logged-in users: pass user_id returned from /customer — full MySQL persistence.
"""

import os
import uuid
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.answer_builder import (
    build_response,
    db_find_or_create_user,
    db_load_history,
    db_create_ticket,
)
from app.logger import logger
from app.retriever import build_index
from app.config import CHROMA_DB_DIR

app = FastAPI(title="FineFlow Nova API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Request/response models
# ─────────────────────────────────────────────────────────────────────────────

class CustomerRequest(BaseModel):
    name:       str
    email:      str
    support_id: str = ""


class ChatRequest(BaseModel):
    message:    str
    user_id:    int = 0      # 0 or omitted = guest (no DB persistence)
    session_id: str = ""     # used for guest in-memory sessions


class TicketRequest(BaseModel):
    user_id: int = 0
    subject: str
    message: str


# ─────────────────────────────────────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    if not CHROMA_DB_DIR.exists() or not any(CHROMA_DB_DIR.iterdir()):
        logger.info("ChromaDB index not found — building now...")
        try:
            build_index()
        except Exception as e:
            logger.error("Failed to auto-build index: %s", e)


@app.get("/health")
def health():
    return {"status": "ok"}


# ─────────────────────────────────────────────────────────────────────────────
# POST /customer — find or create user
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/customer")
def customer(body: CustomerRequest):
    """
    Looks up user by email. Creates if not found.
    Returns user_id to be used in all subsequent /ask, /history, /ticket calls.
    """
    name  = body.name.strip()
    email = body.email.strip().lower()
    sid   = body.support_id.strip()

    if not name or not email:
        raise HTTPException(status_code=400, detail="name and email are required")

    user_id, existed = db_find_or_create_user(name, email, sid)

    if user_id == 0:
        # MySQL unavailable — fall back to guest mode gracefully
        return JSONResponse({
            "user_id": 0,
            "exists": False,
            "warning": "Database unavailable — continuing as guest session",
        })

    return JSONResponse({"user_id": user_id, "exists": existed})


# ─────────────────────────────────────────────────────────────────────────────
# GET /history/{user_id} — load chat history
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/history/{user_id}")
def history(user_id: int):
    """
    Returns full chat history for a logged-in user, oldest to newest.
    [{"sender": "user", "message": "..."}, {"sender": "bot", "message": "..."}]
    """
    if user_id <= 0:
        return JSONResponse([])
    rows = db_load_history(user_id, limit=200)
    return JSONResponse(rows)


# ─────────────────────────────────────────────────────────────────────────────
# GET /ask — main chat endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/ask")
def chat(
    q: str = Query(...),
    session_id: str = Query(""),
    user_id: int = Query(0)
):
    """
    Frontend compatibility endpoint.

    Accepts:
    GET /ask?q=hello&session_id=abc123

    Returns:
    {
      "answer": "...",
      "trigger_ticket_popup": false,
      "session_id": "abc123"
    }
    """

    sid = session_id.strip() or str(uuid.uuid4())
    uid = user_id if user_id > 0 else 0

    if uid:
        sid = f"user_{uid}"

    result = build_response(
        q,
        session_id=sid,
        user_id=uid
    )

    return JSONResponse({
        "answer": result.get("answer", ""),
        "trigger_ticket_popup": result.get("trigger_ticket_popup", False),
        "session_id": sid,
    })


# ─────────────────────────────────────────────────────────────────────────────
# POST /ticket — create support ticket
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/ticket")
def ticket(body: TicketRequest):
    """
    Creates a support ticket. Called by frontend when trigger_ticket_popup
    was true and the user fills in the support form.
    """
    subject = body.subject.strip()
    message = body.message.strip()

    if not subject or not message:
        raise HTTPException(status_code=400, detail="subject and message are required")

    ticket_number = db_create_ticket(body.user_id, subject, message)

    if ticket_number == "TKT-ERR":
        raise HTTPException(status_code=503, detail="Ticket system temporarily unavailable")

    return JSONResponse({"success": True, "ticket_id": ticket_number})


# ─────────────────────────────────────────────────────────────────────────────
# Admin
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/admin/build_index")
def admin_build_index(force: bool = Query(False)):
    try:
        build_index(force_rebuild=force)
        return {"status": "ok", "msg": "Index built."}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ─────────────────────────────────────────────────────────────────────────────
# Static / home
# ─────────────────────────────────────────────────────────────────────────────

static_dir = os.path.join("app", "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def home():
    html = os.path.join("app", "static", "chat.html")
    if os.path.exists(html):
        return FileResponse(html)
    return {"status": "FineFlow Nova API is running"}