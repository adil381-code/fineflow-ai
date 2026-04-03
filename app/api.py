# app/api.py
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

from app.answer_builder import build_response
from app.logger import logger
from app.retriever import build_index
from app.ingest import run as run_ingest

app = FastAPI(title="FineFlow Nova API")

# CORS (open for now — you can lock to your domain later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ask")
def ask(
    q: str = Query(..., description="User question"),
    session_id: str = Query("default", description="Chat session id"),
):
    logger.info("Ask requested: %s (session=%s)", q, session_id)
    resp = build_response(q, session_id=session_id)
    return JSONResponse(resp)

# Optional JSON POST endpoint (if ever needed)
@app.post("/chat")
def chat(body: ChatRequest):
    logger.info("Chat requested: %s (session=%s)", body.message, body.session_id)
    resp = build_response(body.message, session_id=body.session_id)
    return JSONResponse(resp)

# Admin endpoints
@app.post("/admin/ingest")
def admin_ingest():
    try:
        written, skipped = run_ingest()
        return {"status": "ok", "written": written, "skipped": skipped}
    except Exception as e:
        logger.exception("Ingest failed: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/admin/build_index")
def admin_build_index(force: bool = Query(False)):
    try:
        build_index(force_rebuild=force)
        return {"status": "ok", "msg": "Index built."}
    except Exception as e:
        logger.exception("Index build failed: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)

# Static chat UI
static_dir = os.path.join("app", "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def home():
    html = os.path.join("app", "static", "chat.html")
    if os.path.exists(html):
        return FileResponse(html)
    return {"status": "FineFlow Nova API is running"}
