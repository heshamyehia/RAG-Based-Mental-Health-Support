"""
emotion_classifier/predictor.py
Loads the fine-tuned DistilBERT emotion model and exposes predict_emotion().

Expected folder layout (relative to project root):
    emotion_classifier/
        predictor.py          ← this file
        final_emotion_model/
            config.json
            model.safetensors
            tokenizer.json
            tokenizer_config.json
            tokenizer_args.bin
"""

from pathlib import Path
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Make sure project root is on path so we can import schemas
sys.path.insert(0, str(Path(__file__).parent.parent))
from schemas import Emotion

# ─── Paths ────────────────────────────────────────────────────────────────────

_MODEL_DIR = Path(__file__).parent / "final_emotion_model"

# ─── Load once at import time ─────────────────────────────────────────────────

print(f"[EmotionClassifier] Loading model from {_MODEL_DIR} …")

_tokenizer = AutoTokenizer.from_pretrained(str(_MODEL_DIR))
_model     = AutoModelForSequenceClassification.from_pretrained(str(_MODEL_DIR))
_model.eval()

# id2label is stored in config.json by HuggingFace fine-tuning conventions.
_ID2LABEL: dict[int, str] = _model.config.id2label  # e.g. {0: "joy", 1: "sadness", …}

print(f"[EmotionClassifier] Ready. Labels: {list(_ID2LABEL.values())}")


# ─── Public API ───────────────────────────────────────────────────────────────

def predict_emotion(text: str) -> Emotion:
    """
    Predict the dominant emotion in *text*.

    Returns:
        Emotion enum value. Falls back to Emotion.SADNESS on any error.
    """
    try:
        inputs = _tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        with torch.no_grad():
            inputs.pop("token_type_ids", None)   # DistilBERT doesn't use this
            logits = _model(**inputs).logits

        predicted_id = int(torch.argmax(logits, dim=-1).item())
        raw_label = _ID2LABEL.get(predicted_id, "unknown").lower()
        return Emotion(raw_label)

    except Exception as e:
        print(f"[EmotionClassifier] Prediction error: {e}")
        return Emotion.UNKNOWN
