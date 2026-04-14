# app/ingest.py
"""
Robust ingest for FineFlow-RAG:
- Extracts text from PDF/DOCX/TXT
- Cleans whitespace
- Writes to data/docs_txt/<stem>.txt only if content is useful
"""

from pathlib import Path
import re
from typing import Tuple

from app.config import RAW_DIR, DOCS_TXT
from app.logger import logger

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    import docx2txt
except Exception:
    docx2txt = None

DOCS_TXT.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

_MIN_CHARS = 100


def _clean_text(txt: str) -> str:
    if not txt:
        return ""
    txt = txt.replace("\r\n", "\n").replace("\r", "\n")
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    txt = re.sub(r"[\u0000-\u001f\u007f-\u009f]", " ", txt)
    txt = re.sub(r"[ \t]{2,}", " ", txt)
    return txt.strip()


def pdf_to_text(path: Path) -> str:
    if PdfReader is None:
        raise RuntimeError("pypdf not installed")
    try:
        reader = PdfReader(str(path))
        pages = []
        for p in reader.pages:
            try:
                t = p.extract_text() or ""
            except Exception:
                t = ""
            pages.append(t)
        return _clean_text("\n\n".join(pages))
    except Exception as e:
        logger.exception("pdf_to_text failed for %s: %s", path, e)
        return ""


def docx_to_text(path: Path) -> str:
    if docx2txt is None:
        raise RuntimeError("docx2txt not installed")
    try:
        txt = docx2txt.process(str(path)) or ""
        return _clean_text(txt)
    except Exception as e:
        logger.exception("docx_to_text failed for %s: %s", path, e)
        return ""


def txt_to_text(path: Path) -> str:
    try:
        txt = path.read_text(encoding="utf8", errors="ignore") or ""
        return _clean_text(txt)
    except Exception as e:
        logger.exception("txt_to_text failed for %s: %s", path, e)
        return ""


def run() -> Tuple[int, int]:
    """
    Ingest raw files into data/docs_txt/*.txt.
    Returns (written_count, skipped_count)
    """
    written = 0
    skipped = 0
    logger.info("Starting ingest from %s", RAW_DIR)

    for f in sorted(RAW_DIR.iterdir()):
        if not f.is_file():
            continue
        out = DOCS_TXT / f"{f.stem}.txt"
        try:
            sfx = f.suffix.lower()
            if sfx == ".pdf":
                content = pdf_to_text(f)
            elif sfx in (".doc", ".docx"):
                content = docx_to_text(f)
            elif sfx == ".txt":
                content = txt_to_text(f)
            else:
                logger.info("Skipping unsupported file type: %s", f.name)
                skipped += 1
                continue

            if not content or len(content) < _MIN_CHARS:
                logger.warning("Empty/too short output for %s — skipping.", f.name)
                skipped += 1
                continue

            out.write_text(content + "\n", encoding="utf8")
            logger.info("Wrote: %s", out.name)
            written += 1
        except Exception as e:
            logger.exception("Error processing %s: %s", f.name, e)
            skipped += 1

    logger.info("Ingest finished. written=%d skipped=%d", written, skipped)
    return written, skipped


if __name__ == "__main__":
    run()