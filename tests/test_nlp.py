import pytest
from types import SimpleNamespace

import sys
import types

# Inject a fake `google.genai` package so importing intent_classifier doesn't require the real SDK
google_mod = types.ModuleType("google")
genai_mod = types.ModuleType("google.genai")
errors_mod = types.ModuleType("google.genai.errors")

class FakeClientError(Exception):
    pass

errors_mod.ClientError = FakeClientError
genai_mod.errors = errors_mod

class FakeClient:
    def __init__(self, api_key=None):
        self.models = types.SimpleNamespace()

genai_mod.Client = FakeClient

google_mod.genai = genai_mod
sys.modules["google"] = google_mod
sys.modules["google.genai"] = genai_mod
sys.modules["google.genai.errors"] = errors_mod

# dotenv shim
dotenv_mod = types.ModuleType("dotenv")
dotenv_mod.load_dotenv = lambda *a, **k: None
sys.modules["dotenv"] = dotenv_mod

import Intent_classifier.intent_classifier as ic
from schemas import Intent, Emotion


def test_classify_intent_llm_label(monkeypatch):
    # Stub Gemini response
    monkeypatch.setattr(ic._client, "models", SimpleNamespace(generate_content=lambda **kwargs: SimpleNamespace(text="greeting")))
    res = ic.classify_intent("Hi there")
    assert res == Intent.GREETING


def test_classify_intent_client_error(monkeypatch):
    class FakeError(Exception):
        pass

    def raise_error(**kwargs):
        raise ic.genai_errors.ClientError("429 Too Many Requests")

    monkeypatch.setattr(ic._client, "models", SimpleNamespace(generate_content=raise_error))
    res = ic.classify_intent("Anything")
    assert res == Intent.CLASSIFICATION_ERROR


def test_get_direct_response_success(monkeypatch):
    monkeypatch.setattr(ic._client, "models", SimpleNamespace(generate_content=lambda **kwargs: SimpleNamespace(text="Hello there")))
    r = ic.get_direct_response(Intent.OUT_OF_SCOPE, Emotion.NEUTRAL, "en")
    assert r == "Hello there"


def test_get_direct_response_on_exception_returns_fallback(monkeypatch):
    def bad_call(**kwargs):
        raise RuntimeError("fail")
    monkeypatch.setattr(ic._client, "models", SimpleNamespace(generate_content=bad_call))
    r = ic.get_direct_response(Intent.OUT_OF_SCOPE, Emotion.SADNESS, "en")
    assert "I'm here for you" in r
