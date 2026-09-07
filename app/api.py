# app/api.py
"""
FineFlow Nova API
=================
POST /customer                  find-or-create user by email → user_id (migrates guest session if given)
POST /ask   | GET /ask          chat, full answer in one JSON
POST /ask/stream                chat, Server-Sent Events: {"delta": "..."} ... {"done": true, ...}
POST /email-ticket              email form (shown on request_email=true) → row in tickets table
GET  /history/{user_id}         full history for a logged-in user
POST /ticket                    manual support ticket
GET  /health                    liveness + DB + index status
POST /admin/ingest              raw → docs_txt → rebuild index   (X-Admin-Token)
POST /admin/build_index?force=  rebuild index                    (X-Admin-Token)

Guests: send the same session_id every request (frontend keeps it in localStorage).
Logged-in: pass user_id from /customer; session becomes user_{id}.
"""

import json
import os
import re
import threading
import time
import uuid
from collections import defaultdict, deque
from typing import Deque, Dict, Optional

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    import jwt as pyjwt          # PyJWT - pip install PyJWT
except Exception:                # keep the API bootable without it; token auth just disabled
    pyjwt = None


from app.answer_builder import (
    answer_sync, build_response_stream, db, db_all_tickets, db_create_ticket,
    db_find_or_create_user, db_load_history, db_migrate_guest, db_save_email_capture,
    db_user_chat_all_sessions, db_user_has_ticket, db_user_sessions, ensure_tables,
    mark_email_ticket,
)
from app.config import ADMIN_TOKEN, CORS_ORIGINS, RATE_LIMIT_PER_MIN
from app.logger import logger
from app.retriever import build_index, index_size

app = FastAPI(title="FineFlow Nova API", version="3.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=CORS_ORIGINS != ["*"],   # credentials + wildcard is invalid per CORS spec
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ───────────────────────────────────────────────────────────────────

class CustomerRequest(BaseModel):
    name: str
    email: str
    support_id: str = ""
    session_id: str = ""      # guest session to migrate onto the user (optional)


class ChatRequest(BaseModel):
    message: str
    session_id: str = ""
    user_id: int = 0


class TicketRequest(BaseModel):
    user_id: int = 0
    subject: str
    message: str
    email: str = ""
    session_id: str = ""


class EmailTicketRequest(BaseModel):
    """Payload sent by the chat email form (shown when the API returned request_email=true)."""
    email: str
    message: str = ""
    session_id: str = ""
    user_id: int = 0


_EMAIL_FORM_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")


# ── Rate limit (per session, sliding minute) ─────────────────────────────────

_RL: Dict[str, Deque[float]] = defaultdict(deque)
_RL_LOCK = threading.Lock()


def _rate_ok(key: str) -> bool:
    now = time.time()
    with _RL_LOCK:
        q = _RL[key]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE_LIMIT_PER_MIN:
            return False
        q.append(now)
        return True


# ── Startup / health ─────────────────────────────────────────────────────────

@app.on_event("startup")
async def _startup():
    ensure_tables()
    if index_size() == 0:
        logger.info("Index empty — building now")
        try:
            build_index()
        except Exception as e:
            logger.error("Auto-build failed: %s", e)


@app.get("/health")
def health():
    return {"status": "ok", "db": db.healthy(), "index_chunks": index_size()}


# ── Chat ─────────────────────────────────────────────────────────────────────

def _resolve_session(session_id: str, user_id: int) -> str:
    """The client's session_id is always the session key; user_id only tags rows.
    (Previously user_id>0 collapsed everything to user_{id} - that's why chat_hist
    showed 'user_1' instead of the real session ids.)"""
    sid = session_id.strip()
    if sid:
        return sid
    if user_id > 0:
        return f"user_{user_id}"
    return str(uuid.uuid4())


def _guard(message: str, sid: str):
    if not message.strip():
        raise HTTPException(400, "message is required")
    if len(message) > 2000:
        raise HTTPException(413, "message too long")
    if not _rate_ok(sid):
        raise HTTPException(429, "Too many messages — slow down a little.")


JWT_SECRET = os.getenv("JWT_SECRET", "")   # SAME secret as the FineFlow site backend


def _user_from_token(authorization: Optional[str]) -> int:
    """Verify a FineFlow JWT (shared secret, HS256) and return our chatbot user_id.
    Verification = signature check only - no call to the site's database.
    Bad/missing token -> 0 (guest). Never blocks the chat."""
    if not authorization or not JWT_SECRET or pyjwt is None:
        return 0
    token = authorization.replace("Bearer ", "").strip()
    if not token:
        return 0
    try:
        claims = pyjwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except Exception:
        logger.warning("JWT verification failed")
        return 0
    email = (claims.get("email") or "").strip().lower()
    if not email:
        return 0
    name = (claims.get("name") or claims.get("companyName") or email.split("@")[0])
    uid, _ = db_find_or_create_user(str(name)[:100], email)
    return uid


def _auth_and_link(sid: str, body_user_id: int, authorization: Optional[str]) -> int:
    """Resolve the user for this request: verified token wins over the body's claim.
    When a real user is known, stamp this session's earlier guest rows with them."""
    uid = _user_from_token(authorization) or max(body_user_id, 0)
    if uid > 0:
        db_migrate_guest(sid, uid)
    return uid


@app.post("/ask")
def ask_post(body: ChatRequest, authorization: Optional[str] = Header(None)):
    sid = _resolve_session(body.session_id, body.user_id)
    _guard(body.message, sid)
    uid = _auth_and_link(sid, body.user_id, authorization)
    res = answer_sync(body.message, session_id=sid, user_id=uid)
    return JSONResponse({**res, "session_id": sid, "user_id": uid})


@app.get("/ask")
def ask_get(q: str = Query(...), session_id: str = Query(""), user_id: int = Query(0),
            authorization: Optional[str] = Header(None)):
    sid = _resolve_session(session_id, user_id)
    _guard(q, sid)
    uid = _auth_and_link(sid, user_id, authorization)
    res = answer_sync(q, session_id=sid, user_id=uid)
    return JSONResponse({**res, "session_id": sid, "user_id": uid})


@app.post("/ask/stream")
def ask_stream(body: ChatRequest, authorization: Optional[str] = Header(None)):
    sid = _resolve_session(body.session_id, body.user_id)
    _guard(body.message, sid)
    uid = _auth_and_link(sid, body.user_id, authorization)

    def gen():
        try:
            for item in build_response_stream(body.message, sid, uid):
                if isinstance(item, dict):
                    yield "data: " + json.dumps({"done": True, "session_id": sid, **item}) + "\n\n"
                else:
                    yield "data: " + json.dumps({"delta": item}) + "\n\n"
        except Exception:
            logger.exception("stream crashed")
            yield "data: " + json.dumps({"done": True, "session_id": sid, "error": True,
                                         "answer": "Something went wrong on my side — please try that again.",
                                         "request_email": False}) + "\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Email form → ticket ──────────────────────────────────────────────────────

@app.post("/email-ticket")
def email_ticket(body: EmailTicketRequest):
    """
    Called by the chat email form (rendered when a chat response had request_email=true).
    Stores a row in the tickets table with: ticket_number (generated), session_id,
    user_id, email, message, status (defaults OPEN). Returns the ticket number.
    Also updates the chat session state so Nova stops asking for the email.
    """
    email = body.email.strip().lower()
    if not _EMAIL_FORM_RE.match(email):
        raise HTTPException(400, "a valid email is required")
    sid = _resolve_session(body.session_id, body.user_id)
    if not _rate_ok(sid):
        raise HTTPException(429, "Too many requests — slow down a little.")
    uid = max(body.user_id, 0)
    message = body.message.strip() or "Support enquiry submitted via chat email form"

    tkt = db_create_ticket(uid, subject="", message=message, email=email, session_id=sid)
    if tkt == "TKT-ERR":
        raise HTTPException(503, "Ticket system temporarily unavailable")

    db_save_email_capture(email, sid, message)
    # LINK the session to a user record by email so the admin sees whose ticket it is.
    # This is identification only - it never unlocks chat history in the UI
    # (history is revealed by verified token only, see /my-history).
    if uid == 0:
        uid, _ = db_find_or_create_user(email.split("@")[0], email)
    if uid > 0:
        db_migrate_guest(sid, uid)
    try:
        mark_email_ticket(sid, email, uid)   # Nova stops asking for the email in this session
    except Exception:
        logger.exception("mark_email_ticket failed (ticket %s still created)", tkt)

    logger.info("email-ticket session=%s ticket=%s email=%s", sid, tkt, email)
    return JSONResponse({"success": True, "ticket_number": tkt, "status": "OPEN",
                         "session_id": sid,
                         "answer": (f"Thanks - your issue has been logged as {tkt} and the Fine Flow "
                                    f"team will follow up at {email}. The chat stays open if you need "
                                    f"anything else in the meantime.")})


# ── Users / history / tickets ────────────────────────────────────────────────

@app.post("/customer")
def customer(body: CustomerRequest):
    name, email = body.name.strip(), body.email.strip().lower()
    if not name or not email:
        raise HTTPException(400, "name and email are required")
    uid, existed = db_find_or_create_user(name, email, body.support_id.strip())
    if uid == 0:
        return JSONResponse({"user_id": 0, "exists": False,
                             "warning": "Database unavailable — continuing as guest session"})
    if body.session_id.strip():
        db_migrate_guest(body.session_id.strip(), uid)
    return JSONResponse({"user_id": uid, "exists": existed, "session_id": f"user_{uid}"})


@app.get("/my-history")
def my_history(authorization: Optional[str] = Header(None)):
    """Previous sessions for the LOGGED-IN user (chat UI sidebar). Strictly
    token-gated: the user is whoever the verified token says - never an id the
    frontend claims. 401 without a valid token."""
    uid = _user_from_token(authorization)
    if uid <= 0:
        raise HTTPException(401, "valid login token required")
    sessions = db_user_sessions(uid)
    return JSONResponse({"user_id": uid, "session_count": len(sessions),
                         "sessions": sessions})


@app.get("/history/{user_id}")
def history(user_id: int):
    if user_id <= 0:
        return JSONResponse([])
    return JSONResponse(db_load_history(user_id=user_id, limit=200))


@app.post("/ticket")
def ticket(body: TicketRequest):
    if not body.subject.strip() or not body.message.strip():
        raise HTTPException(400, "subject and message are required")
    tkt = db_create_ticket(body.user_id, body.subject.strip(), body.message.strip(),
                           email=body.email.strip(), session_id=body.session_id.strip())
    if tkt == "TKT-ERR":
        raise HTTPException(503, "Ticket system temporarily unavailable")
    return JSONResponse({"success": True, "ticket_id": tkt})


# ── Admin ────────────────────────────────────────────────────────────────────

def _require_admin(token: Optional[str]):
    if ADMIN_TOKEN and token != ADMIN_TOKEN:
        raise HTTPException(401, "invalid admin token")


@app.get("/admin/user-chat/{user_id}")
def admin_user_chat(user_id: int, x_admin_token: Optional[str] = Header(None)):
    """Full chat history for a user across ALL their sessions - only when that user
    has at least one ticket in the tickets table."""
    _require_admin(x_admin_token)
    if user_id <= 0:
        raise HTTPException(400, "valid user_id required")
    if not db_user_has_ticket(user_id):
        raise HTTPException(404, "no ticket found for this user")
    sessions = db_user_chat_all_sessions(user_id)
    return JSONResponse({
        "user_id": user_id,
        "session_count": len(sessions),
        "total_messages": sum(s["message_count"] for s in sessions),
        "sessions": sessions,
    })


@app.get("/admin/tickets")
def admin_tickets(x_admin_token: Optional[str] = Header(None)):
    """All rows, all columns from the tickets table, newest first."""
    _require_admin(x_admin_token)
    tickets = db_all_tickets()
    return JSONResponse({"count": len(tickets), "tickets": tickets})


@app.post("/admin/build_index")
def admin_build_index(force: bool = Query(False), x_admin_token: Optional[str] = Header(None)):
    _require_admin(x_admin_token)
    try:
        return {"status": "ok", "chunks": build_index(force_rebuild=force)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/admin/ingest")
def admin_ingest(x_admin_token: Optional[str] = Header(None)):
    _require_admin(x_admin_token)
    from app.ingest import run as ingest_run
    written, skipped = ingest_run()
    return {"status": "ok", "written": written, "skipped": skipped,
            "chunks": build_index(force_rebuild=True)}


# ── Static ───────────────────────────────────────────────────────────────────

_static = os.path.join("app", "static")
if os.path.isdir(_static):
    app.mount("/static", StaticFiles(directory=_static), name="static")


@app.get("/")
def home():
    html = os.path.join(_static, "chat.html")
    return FileResponse(html) if os.path.exists(html) else {"status": "FineFlow Nova API is running"}