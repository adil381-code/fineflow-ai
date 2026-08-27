# app/retriever.py
"""
Retriever + index builder (ChromaDB + OpenAI embeddings).

Chunking is TOPIC-aware: a knowledge base written as
    TOPIC: <title>
    =========
    <body>
produces one chunk per topic, with the title embedded alongside the body so
short user questions ("what is app password") land on the right section.
Documents without TOPIC headers fall back to paragraph chunking.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import chromadb
import requests

from app.config import (
    DOCS_TXT, CHROMA_DB_DIR, CHUNK_MAX_CHARS,
    OPENAI_API_KEY, OPENAI_EMBED_MODEL, OPENAI_EMBED_API_URL, TOP_K,
)
from app.logger import logger

COLLECTION = "fineflow_docs"

_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
_collection = _client.get_or_create_collection(name=COLLECTION, metadata={"hnsw:space": "cosine"})
logger.info("Retriever ready. embed_model=%s chunks=%d", OPENAI_EMBED_MODEL, _collection.count())


# ── Embeddings ───────────────────────────────────────────────────────────────

def get_openai_embedding(texts: List[str]) -> List[List[float]]:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not configured")
    r = requests.post(
        OPENAI_EMBED_API_URL,
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
        json={"model": OPENAI_EMBED_MODEL, "input": texts},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()["data"]
    return [d["embedding"] for d in sorted(data, key=lambda x: x["index"])]


# ── Chunking ─────────────────────────────────────────────────────────────────

_TOPIC_SPLIT = re.compile(r"\n(?=TOPIC:\s*)", re.I)
_RULE_LINE   = re.compile(r"^\s*=+\s*$", re.M)


def _split_paragraphs(text: str, max_chars: int) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out, cur = [], ""
    for p in paras:
        if len(p) > max_chars:
            if cur:
                out.append(cur); cur = ""
            sents = re.split(r"(?<=[.!?])\s+", p)
            buf = ""
            for s in sents:
                if len(buf) + len(s) + 1 > max_chars and buf:
                    out.append(buf); buf = s
                else:
                    buf = f"{buf} {s}".strip()
            if buf:
                out.append(buf)
        elif len(cur) + len(p) + 2 > max_chars and cur:
            out.append(cur); cur = p
        else:
            cur = f"{cur}\n\n{p}".strip()
    if cur:
        out.append(cur)
    return out


def chunk_document(text: str, max_chars: int = CHUNK_MAX_CHARS) -> List[Tuple[str, str]]:
    """Return [(title, chunk_text)]."""
    text = (text or "").strip()
    if not text:
        return []

    if "TOPIC:" not in text.upper():
        return [("General", c) for c in _split_paragraphs(text, max_chars)]

    chunks: List[Tuple[str, str]] = []
    for part in _TOPIC_SPLIT.split(text):
        part = _RULE_LINE.sub("", part).strip()
        if not part.upper().startswith("TOPIC:"):
            continue  # preamble / header block — not knowledge
        first, _, body = part.partition("\n")
        title = first.split(":", 1)[1].strip() or "Untitled"
        body = body.strip()
        if not body:
            continue
        if len(body) <= max_chars:
            chunks.append((title, f"{title}\n{body}"))
        else:
            for i, sub in enumerate(_split_paragraphs(body, max_chars)):
                chunks.append((title, f"{title} (part {i + 1})\n{sub}"))
    return chunks


# ── Index ────────────────────────────────────────────────────────────────────

def index_size() -> int:
    try:
        return _collection.count()
    except Exception:
        return 0


def build_index(force_rebuild: bool = False) -> int:
    global _collection
    if not force_rebuild and index_size() > 0:
        logger.info("Index already has %d chunks — skipping build.", index_size())
        return index_size()

    if force_rebuild:
        try:
            _client.delete_collection(COLLECTION)
        except Exception:
            pass
        _collection = _client.create_collection(name=COLLECTION, metadata={"hnsw:space": "cosine"})

    files = sorted(Path(DOCS_TXT).glob("*.txt"))
    if not files:
        raise RuntimeError(f"No documents in {DOCS_TXT} — run ingest first.")

    docs, metas, ids = [], [], []
    for f in files:
        text = f.read_text(encoding="utf8", errors="ignore")
        for i, (title, chunk) in enumerate(chunk_document(text)):
            docs.append(chunk)
            metas.append({"source": f.name, "title": title, "chunk_index": i})
            ids.append(f"{f.stem}_{i}")

    if not docs:
        raise RuntimeError("No chunks produced from documents.")

    batch = 100
    for i in range(0, len(docs), batch):
        embs = get_openai_embedding(docs[i:i + batch])
        _collection.upsert(embeddings=embs, documents=docs[i:i + batch],
                           metadatas=metas[i:i + batch], ids=ids[i:i + batch])
        logger.info("Indexed %d/%d", min(i + batch, len(docs)), len(docs))

    logger.info("Index built: %d chunks from %d files", len(docs), len(files))
    return len(docs)


# ── Search ───────────────────────────────────────────────────────────────────

def _tokens(text: str) -> List[str]:
    text = text.lower().replace("fineflow", "fine flow")
    return [t for t in re.split(r"[^0-9a-z£]+", text) if len(t) > 2]


def search(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    if not query or index_size() == 0:
        return []
    try:
        emb = get_openai_embedding([query])[0]
        res = _collection.query(query_embeddings=[emb], n_results=min(top_k, index_size()),
                                include=["documents", "metadatas", "distances"])
    except Exception as e:
        logger.exception("Search failed: %s", e)
        return []
    hits = []
    if res.get("ids") and res["ids"][0]:
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            hits.append({"chunk": doc, "meta": meta or {}, "score": 1.0 - float(dist)})
    return hits


_STOP = {"the", "and", "for", "you", "your", "with", "what", "how", "does", "can", "fine", "flow", "about"}


def rerank_hits(hits: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """Hybrid rerank: embedding score + keyword overlap + title match."""
    if not hits:
        return hits
    q = [t for t in _tokens(query) if t not in _STOP]
    if not q:
        return hits
    out = []
    for h in hits:
        body = h["chunk"].lower()
        title = (h["meta"].get("title") or "").lower()
        overlap = sum(1 for t in q if t in body)
        title_hit = sum(1 for t in q if t in title)
        nh = dict(h)
        nh["score"] = h["score"] + 0.03 * overlap + 0.12 * title_hit
        out.append(nh)
    out.sort(key=lambda x: x["score"], reverse=True)
    return out


def retrieve_context(query: str, top_k: int = TOP_K) -> Tuple[str, float]:
    """Return (formatted context, best score). Never gated — the model decides relevance."""
    hits = rerank_hits(search(query, top_k=top_k + 2), query)[:top_k]
    if not hits:
        return "", 0.0
    blocks = [f"### {h['meta'].get('title', 'Knowledge')}\n{h['chunk']}" for h in hits]
    return "\n\n".join(blocks), hits[0]["score"]
