import os
import json

HISTORY_DIR = "chat_sessions"

def _get_history_path(session_id: str) -> str:
    if not os.path.exists(HISTORY_DIR):
        os.makedirs(HISTORY_DIR)
    
    # Simple sanitization to prevent directory traversal
    safe_session_id = "".join(c for c in session_id if c.isalnum() or c in ('-', '_'))
    return os.path.join(HISTORY_DIR, f"{safe_session_id}.json")

def get_history(session_id: str) -> list:
    """Loads the JSON file for the session and returns previous messages."""
    path = _get_history_path(session_id)
    if not os.path.exists(path):
        return []
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []

def append_history(session_id: str, user_msg: str, ai_msg: str, max_turns: int = 10):
    """Appends the exchange, and enforces a sliding window by keeping only the last max_turns of conversation."""
    history = get_history(session_id)
    history.append({
        "user": user_msg,
        "assistant": ai_msg
    })
    
    # Enforce sliding window (keep last max_turns)
    if len(history) > max_turns:
        history = history[-max_turns:]
        
    path = _get_history_path(session_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
