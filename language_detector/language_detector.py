"""
Module 1 — Language Detection
==============================
Model : TF-IDF (char n-grams) + LinearSVM
        trained on papluca/language-identification (20 languages)

Usage 1- Imported in main.py
      2- can be used as standalone tool to test the model             
"""

import os
import warnings
import joblib
warnings.filterwarnings("ignore")

#  Load model once 
_BASE_DIR = os.path.dirname(__file__)
_model    = joblib.load(os.path.join(_BASE_DIR, "language_detector.joblib"))
_meta     = joblib.load(os.path.join(_BASE_DIR, "language_detector_meta.joblib"))

LANG_NAMES = {
    "ar": "Arabic",     "bg": "Bulgarian",  "de": "German",
    "el": "Greek",      "en": "English",    "es": "Spanish",
    "fr": "French",     "hi": "Hindi",      "it": "Italian",
    "ja": "Japanese",   "nl": "Dutch",      "pl": "Polish",
    "pt": "Portuguese", "ru": "Russian",    "sw": "Swahili",
    "th": "Thai",       "tr": "Turkish",    "ur": "Urdu",
    "vi": "Vietnamese", "zh": "Chinese",
}

# ════════════════════════════════════════════════════════════════════════════
# PUBLIC API — imported by main.py
# ════════════════════════════════════════════════════════════════════════════

def detect_language(text: str) -> str:
    """
    Returns an ISO 639-1 language code e.g. 'en', 'ar', 'fr'.
    Falls back to 'NAN' if confidence < 0.50 or text is too short.
    """
    if not text or len(text.strip()) < 3:
        return "NAN"
    try:
        proba      = _model.predict_proba([text])[0]
        confidence = proba.max()
        lang       = _meta["classes"][proba.argmax()]
        return lang if confidence >= 0.50 else "NAN"
    except Exception:
        return "NAN"


# ════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST — python language_detector/language_detector.py
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("  Module 1 — Language Detector")
    print("  Type your text to detect its language.")
    print("  Type 'quit' to exit.")
    print("=" * 55 + "\n")

    while True:
        try:
            text = input("Enter text (As sentence ) > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if text.lower() in {"quit", "exit", "q"}:
            print("Bye!")
            break
        if not text:
            continue

        proba      = _model.predict_proba([text])[0]
        confidence = proba.max()
        lang       = _meta["classes"][proba.argmax()]

        print(f"\n  ✔ {LANG_NAMES.get(lang, lang)} ({lang})")
