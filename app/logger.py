# app/logger.py
import logging
from pathlib import Path

LOG_FILE = Path(__file__).resolve().parent.parent / "data" / "app.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf8"), logging.StreamHandler()],
)
logger = logging.getLogger("fineflow_nova")
