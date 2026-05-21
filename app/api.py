# app/api.py
"""
FineFlow Nova API
- Session IDs generated on frontend and passed with every request
- Both GET /ask and POST /chat supported
"""

import uuid
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from pathlib import Path

from app.answer_builder import build_response
from app.logger import logger
from app.retriever import build_index
from app.ingest import run as run_ingest
from app.config import CHROMA_DB_DIR

app = FastAPI(title="FineFlow Nova API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: str = ""   # frontend must send this; generated here if missing


@app.on_event("startup")
async def startup_event():
    if not CHROMA_DB_DIR.exists() or not any(CHROMA_DB_DIR.iterdir()):
        logger.info("ChromaDB index not found — building now...")
        try:
            build_index()
        except Exception as e:
            logger.error("Failed to auto-build index: %s", e)
    else:
        logger.info("ChromaDB index found.")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ask")
def ask(q: str = Query(...), session_id: str = Query("")):
    # If frontend sends no session_id, generate one (stateless fallback)
    sid = session_id.strip() or str(uuid.uuid4())
    resp = build_response(q, session_id=sid)
    resp["session_id"] = sid   # return it so frontend can store it
    return JSONResponse(resp)


@app.post("/chat")
def chat(body: ChatRequest):
    sid = body.session_id.strip() or str(uuid.uuid4())
    resp = build_response(body.message, session_id=sid)
    resp["session_id"] = sid
    return JSONResponse(resp)


@app.post("/admin/ingest")
def admin_ingest():
    try:
        written, skipped = run_ingest()
        return {"status": "ok", "written": written, "skipped": skipped}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/admin/build_index")
def admin_build_index(force: bool = Query(False)):
    try:
        build_index(force_rebuild=force)
        return {"status": "ok", "msg": "Index built."}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


static_dir = os.path.join("app", "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def home():
    html = os.path.join("app", "static", "chat.html")
    if os.path.exists(html):
        return FileResponse(html)
    return {"status": "FineFlow Nova API is running"}