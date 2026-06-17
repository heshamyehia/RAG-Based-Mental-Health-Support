import json
from datetime import datetime, timezone
from pathlib import Path


FEEDBACK_DIR = Path("feedback_events")
FEEDBACK_FILE = FEEDBACK_DIR / "feedback.jsonl"


def append_feedback(payload: dict) -> dict:
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

    record = {
        **payload,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }

    with FEEDBACK_FILE.open("a", encoding="utf-8") as file_handle:
        file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record