# app/api.py
"""
FineFlow Nova API
=================
POST /customer                  find-or-create user by email → user_id (migrates guest session if given)
POST /ask   | GET /ask          chat, full answer in one JSON
POST /ask/stream                chat, Server-Sent Events: {"delta": "..."} ... {"done": true, ...}
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

from app.answer_builder import (
    answer_sync, build_response_stream, db, db_create_ticket, db_find_or_create_user,
    db_load_history, db_migrate_guest, ensure_tables,
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
    if user_id > 0:
        return f"user_{user_id}"
    return session_id.strip() or str(uuid.uuid4())


def _guard(message: str, sid: str):
    if not message.strip():
        raise HTTPException(400, "message is required")
    if len(message) > 2000:
        raise HTTPException(413, "message too long")
    if not _rate_ok(sid):
        raise HTTPException(429, "Too many messages — slow down a little.")


@app.post("/ask")
def ask_post(body: ChatRequest):
    sid = _resolve_session(body.session_id, body.user_id)
    _guard(body.message, sid)
    res = answer_sync(body.message, session_id=sid, user_id=max(body.user_id, 0))
    return JSONResponse({**res, "session_id": sid})


@app.get("/ask")
def ask_get(q: str = Query(...), session_id: str = Query(""), user_id: int = Query(0)):
    sid = _resolve_session(session_id, user_id)
    _guard(q, sid)
    res = answer_sync(q, session_id=sid, user_id=max(user_id, 0))
    return JSONResponse({**res, "session_id": sid})


@app.post("/ask/stream")
def ask_stream(body: ChatRequest):
    sid = _resolve_session(body.session_id, body.user_id)
    _guard(body.message, sid)
    uid = max(body.user_id, 0)

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
