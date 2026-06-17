import json
import logging
from datetime import datetime, timezone
from pathlib import Path

FEEDBACK_DIR = Path("feedback_events")
FEEDBACK_FILE = FEEDBACK_DIR / "feedback.jsonl"

logger = logging.getLogger(__name__)


def append_feedback(payload: dict) -> dict:
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

    record = {
        **payload,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        with FEEDBACK_FILE.open("a", encoding="utf-8") as file_handle:
            file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(
            "Recorded feedback vote=%s session_id=%s",
            record["vote"],
            record.get("session_id"),
        )
    except OSError:
        logger.exception("Failed to persist feedback vote=%s", payload.get("vote"))
        raise

    return record
