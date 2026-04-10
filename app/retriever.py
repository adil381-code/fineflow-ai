# app/retriever.py
"""
Retriever + index builder (FineFlow, GPT-4o tuned)
--------------------------------------------------
- Reads cleaned docs from data/docs_txt
- Chunks by characters (large, overlapping chunks) so each chunk keeps full context
- Embeds with OpenAI embeddings
- Builds FAISS index (cosine similarity via L2-normalized dot product)
- search(query) -> list of {chunk, meta, score}
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any


import numpy as np
import requests

from app.config import (
    DOCS_TXT, CHUNKS_TEXT, CHUNKS_META, FAISS_INDEX,
    OPENAI_API_KEY, OPENAI_EMBED_MODEL, OPENAI_EMBED_API_URL, TOP_K
)
from app.logger import logger

# --------- Optional deps ---------
try:
    import faiss
except Exception:
    faiss = None

logger.info("Retriever starting. OPENAI_EMBED_MODEL=%s, faiss=%s", OPENAI_EMBED_MODEL, bool(faiss))

# Embedding cache
try:
    EMB_CACHE_NPY = FAISS_INDEX.with_suffix(".npy")
    EMB_CACHE_META = FAISS_INDEX.with_suffix(".cachemeta.json")
except Exception:
    EMB_CACHE_NPY = Path(str(FAISS_INDEX) + ".npy")
    EMB_CACHE_META = Path(str(FAISS_INDEX) + ".cachemeta.json")

from dotenv import load_dotenv
import os

load_dotenv()

def get_openai_embedding(texts: List[str]) -> np.ndarray:
    """
    Get embeddings from OpenAI API for a list of texts.
    Returns numpy array of shape (len(texts), embedding_dim)
    """
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
        
        # Extract embeddings in order
        embeddings = []
        for item in sorted(data['data'], key=lambda x: x['index']):
            embeddings.append(item['embedding'])
        
        return np.array(embeddings, dtype=np.float32)
    except Exception as e:
        logger.exception("OpenAI embedding API call failed: %s", e)
        raise


# ---------- Helpers ----------

def _extract_title(text: str) -> str:
    """
    Pick a short, human-readable title from the first non-empty lines.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return "Untitled"
    for ln in lines[:6]:
        if 3 < len(ln) < 120:
            return ln[:120]
    return lines[0][:120]


def chunk_text(text: str, max_chars: int = 1800, overlap: int = 300) -> List[str]:
    """
    Character-based chunking with overlap.
    This keeps whole sections (like appeal docs, ULEZ explanations, etc.)
    together so GPT-4o sees full context.

    max_chars:   target max chars per chunk
    overlap:     number of chars re-used between chunks
    """
    text = (text or "").strip()
    if not text:
        return []

    n = len(text)
    if n <= max_chars:
        return [text]

    chunks: List[str] = []
    start = 0

    while start < n:
        end = min(n, start + max_chars)
        chunk = text[start:end]

        # Try to cut cleanly at a paragraph or sentence boundary near the end
        rel = chunk
        cut = -1
        # look for double newline or period near the end but not too early
        for sep in ["\n\n", ". ", "\n"]:
            pos = rel.rfind(sep)
            if pos != -1 and pos > int(max_chars * 0.4):
                cut = pos + len(sep)
                break

        if cut != -1:
            end = start + cut
            chunk = text[start:end]

        chunks.append(chunk.strip())

        # move start with overlap
        if end >= n:
            break
        start = max(0, end - overlap)

    return chunks


# ---------- Simple reranker ----------
def rerank_hits(hits: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """
    Optional light-weight reranker: boosts hits that contain query tokens.
    This function is intentionally simple and deterministic.
    """
    if not hits or not query:
        return hits

    q_tokens = set([t for t in re_tokenize(query.lower()) if t])
    if not q_tokens:
        return hits

    scored = []
    for h in hits:
        chunk = (h.get("chunk") or "").lower()
        # count simple token overlap
        overlap = sum(1 for t in q_tokens if t in chunk)
        # base score is the original score
        base = float(h.get("score", 0.0))
        # final score: base + small boost per overlap (tunable)
        final_score = base + (0.05 * overlap)
        scored.append((final_score, h))

    # sort by final_score descending
    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for s, h in scored:
        nh = dict(h)
        nh["score"] = float(s)
        out.append(nh)
    return out


def re_tokenize(text: str) -> List[str]:
    # simple alnum tokenizer
    return [t for t in re_split_non_alnum(text) if t]


def re_split_non_alnum(text: str) -> List[str]:
    return re.split(r"[^0-9a-zA-Z]+", text)


# ---------- Index build ----------

def build_index(force_rebuild: bool = False):
    """
    Build FAISS index and save chunks + meta + embeddings to disk using OpenAI embeddings.
    """
    logger.info("Building index with OpenAI embeddings (force_rebuild=%s)...", force_rebuild)

    all_chunks: List[str] = []
    metas: List[Dict[str, Any]] = []

    # 1) Load all .txt docs from DOCS_TXT
    txt_files = sorted(Path(DOCS_TXT).glob("*.txt")) if isinstance(DOCS_TXT, (str, Path)) else []
    if not txt_files:
        raise RuntimeError(f"No documents found in {DOCS_TXT} — run ingest first.")

    for txt_file in txt_files:
        text = Path(txt_file).read_text(encoding="utf8", errors="ignore").strip()
        if not text:
            logger.warning("Empty text file skipped: %s", Path(txt_file).name)
            continue

        # 2) Chunk this document
        pieces = chunk_text(text, max_chars=1800, overlap=300)
        if not pieces:
            logger.warning("No chunks produced for %s", Path(txt_file).name)
            continue

        for i, c in enumerate(pieces):
            title = _extract_title(text if i == 0 else c)
            metas.append({
                "source": Path(txt_file).name,
                "chunk_id": f"{Path(txt_file).name}__{i}",
                "title": title
            })
            all_chunks.append(c)

    if not all_chunks:
        raise RuntimeError("No chunks built from docs — nothing to index.")

    # 3) Load or compute embeddings with OpenAI
    embeddings = None
    if EMB_CACHE_NPY.exists() and EMB_CACHE_META.exists() and not force_rebuild:
        try:
            logger.info("Loading cached embeddings from %s", EMB_CACHE_NPY)
            embeddings = np.load(EMB_CACHE_NPY)
            cached_meta = json.loads(EMB_CACHE_META.read_text(encoding="utf8"))
            if len(cached_meta) != len(metas):
                logger.warning(
                    "Embedding cache meta size mismatch (%d vs %d) — recomputing embeddings",
                    len(cached_meta), len(metas)
                )
                embeddings = None
            else:
                metas = cached_meta
        except Exception as e:
            logger.warning("Failed loading embedding cache: %s", e)
            embeddings = None

    if embeddings is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OpenAI API key not configured for embeddings.")
        
        logger.info("Encoding %d chunks with OpenAI embeddings...", len(all_chunks))
        
        # Process in batches to avoid rate limits (max 2048 texts per batch for ada-002)
        batch_size = 100
        embs_list = []
        
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            logger.info(f"Encoding batch {i//batch_size + 1}/{(len(all_chunks)-1)//batch_size + 1}")
            emb = get_openai_embedding(batch)
            embs_list.append(emb)
        
        embeddings = np.vstack(embs_list)
        
        # Save cache
        np.save(EMB_CACHE_NPY, embeddings)
        EMB_CACHE_META.write_text(json.dumps(metas, ensure_ascii=False), encoding="utf8")
        logger.info("Saved embedding cache: %s", EMB_CACHE_NPY)

    if faiss is None:
        raise RuntimeError("faiss not installed; cannot build index.")

    # 4) Build FAISS index (cosine via normalized inner product)
    dim = embeddings.shape[1]
    logger.info("Building FAISS index dim=%d", dim)
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    faiss.write_index(index, str(FAISS_INDEX))

    # 5) Save chunks + meta
    CHUNKS_TEXT.write_text(json.dumps(all_chunks, ensure_ascii=False), encoding="utf8")
    CHUNKS_META.write_text(json.dumps(metas, ensure_ascii=False), encoding="utf8")

    logger.info("Index built. chunks=%d docs=%d", len(all_chunks), len(txt_files))


# ---------- Index load + search ----------

def load_index():
    """
    Load FAISS index + chunks + meta.
    Returns (embed_model, index, chunks, metas)
    """
    if faiss is None:
        raise RuntimeError("faiss not installed; cannot load index.")
    if not OPENAI_API_KEY:
        raise RuntimeError("OpenAI API key not configured.")
    if not FAISS_INDEX.exists():
        raise FileNotFoundError("FAISS index missing — run build_index() first.")

    index = faiss.read_index(str(FAISS_INDEX))
    chunks = json.loads(CHUNKS_TEXT.read_text(encoding="utf8"))
    metas = json.loads(CHUNKS_META.read_text(encoding="utf8"))
    if len(chunks) != len(metas):
        logger.warning("chunks/meta length mismatch (%d vs %d)", len(chunks), len(metas))
    logger.info("Loaded index & chunks (count=%d)", len(chunks))
    return "openai", index, chunks, metas


def search(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    Embed query with OpenAI, search FAISS, return top_k results:
    [{ "chunk": str, "meta": {...}, "score": float }, ...]
    """
    try:
        model, index, chunks, metas = load_index()
    except Exception as e:
        logger.exception("load_index failed during search: %s", e)
        return []

    if not query:
        return []

    # Get embedding for query using OpenAI
    try:
        q_emb = get_openai_embedding([query])
        if q_emb.dtype != np.float32:
            q_emb = q_emb.astype("float32")
        faiss.normalize_L2(q_emb)
    except Exception as e:
        logger.exception("Failed to get embedding for query: %s", e)
        return []

    try:
        D, I = index.search(q_emb, top_k)
    except Exception as e:
        logger.exception("faiss search failed: %s", e)
        return []

    results: List[Dict[str, Any]] = []
    scores = D[0].tolist()
    idxs = I[0].tolist()

    for score, idx in zip(scores, idxs):
        if 0 <= idx < len(chunks):
            results.append({
                "chunk": chunks[idx],
                "meta": metas[idx],
                "score": float(score)
            })

    return results