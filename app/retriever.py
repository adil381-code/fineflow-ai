# app/retriever.py
"""
Retriever + index builder using ChromaDB and OpenAI embeddings.
- Reads cleaned docs from data/docs_txt
- Chunks intelligently (respects paragraphs and lists)
- Embeds with OpenAI text-embedding-3-small
- Stores in ChromaDB (persistent)
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings
import numpy as np
import requests

from app.config import (
    DOCS_TXT, CHROMA_DB_DIR,
    OPENAI_API_KEY, OPENAI_EMBED_MODEL, OPENAI_EMBED_API_URL, TOP_K
)
from app.logger import logger

# --------- ChromaDB client ---------
_chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
_collection = _chroma_client.get_or_create_collection(
    name="fineflow_docs",
    metadata={"hnsw:space": "cosine"}
)

logger.info("Retriever starting. OPENAI_EMBED_MODEL=%s, using ChromaDB", OPENAI_EMBED_MODEL)

# --------- OpenAI embedding ---------
def get_openai_embedding(texts: List[str]) -> List[List[float]]:
    """Get embeddings from OpenAI API for a list of texts."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OpenAI API key not configured")
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": OPENAI_EMBED_MODEL,
        "input": texts
    }
    
    try:
        response = requests.post(OPENAI_EMBED_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        embeddings = [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
        return embeddings
    except Exception as e:
        logger.exception("OpenAI embedding API call failed: %s", e)
        raise

# --------- Helpers ---------
def _extract_title(text: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return "Untitled"
    for ln in lines[:6]:
        if 3 < len(ln) < 120:
            return ln[:120]
    return lines[0][:120]

def chunk_text(text: str, max_chars: int = 2000, overlap: int = 400) -> List[str]:
    """
    Chunk text while preserving paragraph boundaries.
    Splits by double newline first, then by sentences if needed.
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    # Split by double newline to keep logical sections together
    sections = re.split(r'\n\s*\n', text)
    chunks = []
    current = ""

    for section in sections:
        if len(current) + len(section) + 2 <= max_chars:
            current = current + "\n\n" + section if current else section
        else:
            if current:
                chunks.append(current.strip())
            # If the section itself is too long, split further by sentences
            if len(section) > max_chars:
                subchunks = _split_long_section(section, max_chars, overlap)
                chunks.extend(subchunks)
                current = ""
            else:
                current = section
    if current:
        chunks.append(current.strip())
    return chunks

def _split_long_section(section: str, max_chars: int, overlap: int) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', section)
    subchunks = []
    cur = ""
    for sent in sentences:
        if len(cur) + len(sent) + 1 <= max_chars:
            cur = cur + " " + sent if cur else sent
        else:
            if cur:
                subchunks.append(cur.strip())
            cur = sent
    if cur:
        subchunks.append(cur.strip())
    return subchunks

def re_tokenize(text: str) -> List[str]:
    return re.split(r"[^0-9a-zA-Z]+", text)

# --------- Index build ---------
def build_index(force_rebuild: bool = False):
    """
    Build ChromaDB index from DOCS_TXT files.
    """
    logger.info("Building index with OpenAI embeddings (force_rebuild=%s)...", force_rebuild)

    if force_rebuild:
        # Delete existing collection and recreate
        _chroma_client.delete_collection("fineflow_docs")
        global _collection
        _collection = _chroma_client.create_collection(
            name="fineflow_docs",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Cleared existing Chroma collection.")

    # Check if collection already has data
    existing_count = _collection.count()
    if existing_count > 0 and not force_rebuild:
        logger.info("Index already contains %d documents. Skipping build.", existing_count)
        return

    all_chunks = []
    metas = []
    ids = []

    txt_files = sorted(Path(DOCS_TXT).glob("*.txt"))
    if not txt_files:
        raise RuntimeError(f"No documents found in {DOCS_TXT} — run ingest first.")

    for txt_file in txt_files:
        text = txt_file.read_text(encoding="utf8", errors="ignore").strip()
        if not text:
            continue

        pieces = chunk_text(text)
        if not pieces:
            continue

        title = _extract_title(text)
        for i, chunk in enumerate(pieces):
            chunk_id = f"{txt_file.stem}_{i}"
            meta = {
                "source": txt_file.name,
                "title": title,
                "chunk_index": i
            }
            all_chunks.append(chunk)
            metas.append(meta)
            ids.append(chunk_id)

    if not all_chunks:
        raise RuntimeError("No chunks produced from documents.")

    # Generate embeddings in batches
    logger.info("Generating embeddings for %d chunks...", len(all_chunks))
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch_texts = all_chunks[i:i+batch_size]
        batch_metas = metas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]

        embeddings = get_openai_embedding(batch_texts)
        _collection.add(
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=batch_metas,
            ids=batch_ids
        )
        logger.info("Indexed batch %d/%d", i//batch_size + 1, (len(all_chunks)-1)//batch_size + 1)

    logger.info("Index built. Total chunks=%d", len(all_chunks))

# --------- Search ---------
def search(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    Embed query, search ChromaDB, return top_k results.
    """
    if not query:
        return []

    try:
        query_embedding = get_openai_embedding([query])[0]
    except Exception as e:
        logger.exception("Failed to embed query: %s", e)
        return []

    try:
        results = _collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
    except Exception as e:
        logger.exception("Chroma query failed: %s", e)
        return []

    hits = []
    if results["ids"] and results["ids"][0]:
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            # Chroma returns cosine distance; convert to similarity score
            score = 1.0 - dist
            hits.append({
                "chunk": doc,
                "meta": meta,
                "score": score
            })
    return hits

def load_index():
    """Compatibility function – just returns a dummy value."""
    return "openai", _collection, [], []

def rerank_hits(hits: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """Simple token overlap reranker."""
    if not hits or not query:
        return hits
    q_tokens = set(re_tokenize(query.lower()))
    if not q_tokens:
        return hits
    scored = []
    for h in hits:
        chunk = (h.get("chunk") or "").lower()
        overlap = sum(1 for t in q_tokens if t in chunk)
        new_score = h["score"] + 0.05 * overlap
        nh = dict(h)
        nh["score"] = new_score
        scored.append((new_score, nh))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored]