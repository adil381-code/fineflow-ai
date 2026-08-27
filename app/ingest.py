# app/ingest.py
"""Ingest raw PDF/DOCX/TXT from data/raw into cleaned data/docs_txt/*.txt."""
import re
from pathlib import Path
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

_MIN_CHARS = 100


def _clean_text(txt: str) -> str:
    if not txt:
        return ""
    txt = txt.replace("\r\n", "\n").replace("\r", "\n")
    txt = re.sub(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f]", " ", txt)
    txt = re.sub(r"[ \t]{2,}", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()


def pdf_to_text(path: Path) -> str:
    if PdfReader is None:
        raise RuntimeError("pypdf not installed")
    try:
        return _clean_text("\n\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages))
    except Exception as e:
        logger.exception("pdf_to_text failed for %s: %s", path, e)
        return ""


def docx_to_text(path: Path) -> str:
    if docx2txt is None:
        raise RuntimeError("docx2txt not installed")
    try:
        return _clean_text(docx2txt.process(str(path)) or "")
    except Exception as e:
        logger.exception("docx_to_text failed for %s: %s", path, e)
        return ""


def txt_to_text(path: Path) -> str:
    try:
        return _clean_text(path.read_text(encoding="utf8", errors="ignore"))
    except Exception as e:
        logger.exception("txt_to_text failed for %s: %s", path, e)
        return ""


def run() -> Tuple[int, int]:
    written = skipped = 0
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_TXT.mkdir(parents=True, exist_ok=True)
    for f in sorted(RAW_DIR.iterdir()):
        if not f.is_file():
            continue
        sfx = f.suffix.lower()
        if sfx == ".pdf":
            content = pdf_to_text(f)
        elif sfx in (".doc", ".docx"):
            content = docx_to_text(f)
        elif sfx == ".txt":
            content = txt_to_text(f)
        else:
            logger.info("Skipping unsupported file: %s", f.name)
            skipped += 1
            continue
        if len(content) < _MIN_CHARS:
            logger.warning("Too short, skipping: %s", f.name)
            skipped += 1
            continue
        (DOCS_TXT / f"{f.stem}.txt").write_text(content + "\n", encoding="utf8")
        written += 1
        logger.info("Wrote %s.txt", f.stem)
    logger.info("Ingest done: written=%d skipped=%d", written, skipped)
    return written, skipped


if __name__ == "__main__":
    run()
