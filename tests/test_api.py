import importlib
from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient

# Patch RAGPipeline before app creation to avoid heavy initialization
import sys
import types

# Inject a lightweight dummy module4_rag.rag_pipeline to avoid importing heavy deps
class DummyPipeline:
    def __init__(self, *args, **kwargs):
        pass

    def answer(self, *args, **kwargs):
        return {"answer": "This is a RAG answer."}

rp_module = types.ModuleType("module4_rag.rag_pipeline")
rp_module.RAGPipeline = DummyPipeline
sys.modules["module4_rag.rag_pipeline"] = rp_module

# Provide a lightweight dotenv shim for tests
dotenv_mod = types.ModuleType("dotenv")
dotenv_mod.load_dotenv = lambda *a, **k: None
sys.modules["dotenv"] = dotenv_mod

# Provide a fake google.genai package used by Intent_classifier
genai_mod = types.ModuleType("google.genai")
errors_mod = types.ModuleType("google.genai.errors")
class FakeClientError(Exception):
    pass
errors_mod.ClientError = FakeClientError

class FakeClient:
    def __init__(self, api_key=None):
        self.models = types.SimpleNamespace()

genai_mod.errors = errors_mod
genai_mod.Client = FakeClient
google_mod = types.ModuleType("google")
google_mod.genai = genai_mod
sys.modules["google"] = google_mod
sys.modules["google.genai"] = genai_mod
sys.modules["google.genai.errors"] = errors_mod

# Provide lightweight shims for emotion and language modules to avoid heavy imports
emotion_mod = types.ModuleType("emotion_classifier.predictor")
emotion_mod.predict_emotion = lambda text: "neutral"
emotion_pkg = types.ModuleType("emotion_classifier")
emotion_pkg.predictor = emotion_mod
sys.modules["emotion_classifier"] = emotion_pkg
sys.modules["emotion_classifier.predictor"] = emotion_mod

lang_pkg = types.ModuleType("language_detector")
lang_mod = types.ModuleType("language_detector.language_detector")
lang_mod.detect_language = lambda text: "en"
lang_pkg.language_detector = lang_mod
sys.modules["language_detector"] = lang_pkg
sys.modules["language_detector.language_detector"] = lang_mod

# Now import and create the app
import importlib
import api.app as api_app
importlib.reload(api_app)
from schemas import Intent, Emotion

# Build a lightweight FastAPI app for tests that reuses the project's router
from fastapi import FastAPI
test_app = FastAPI()
test_app.include_router(api_app.router if hasattr(api_app, 'router') else api_app.app.router)
# attach a dummy pipeline
test_app.state.pipeline = DummyPipeline()
client = TestClient(test_app)


def test_health_check():
    r = client.get("/health")
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "ok"
    assert j["module"] == "chat_pipeline"
    assert "version" in j


def test_chat_routes_to_rag_for_mental_health(monkeypatch):
    # stub module functions
    monkeypatch.setattr("language_detector.language_detector.detect_language", lambda text: "en")
    monkeypatch.setattr("emotion_classifier.predictor.predict_emotion", lambda text: Emotion.FEAR)
    monkeypatch.setattr("api.routes.classify_intent", lambda text: Intent.ASKING_MENTAL_HEALTH)
    monkeypatch.setattr("history_manager.get_history", lambda sid: [])
    monkeypatch.setattr("history_manager.append_history", lambda sid, u, a: None)

    payload = {"message": "I feel anxious and cannot stop worrying."}
    r = client.post("/chat", json=payload)
    assert r.status_code == 200
    j = r.json()
    assert j["response_source"] == "rag"
    assert j["intent"] == Intent.ASKING_MENTAL_HEALTH.value
    assert "session_id" in j


def test_chat_routes_to_direct_for_non_rag(monkeypatch):
    monkeypatch.setattr("language_detector.language_detector.detect_language", lambda text: "en")
    monkeypatch.setattr("emotion_classifier.predictor.predict_emotion", lambda text: Emotion.NEUTRAL)
    # Force direct path
    monkeypatch.setattr("api.routes.classify_intent", lambda text: Intent.OUT_OF_SCOPE)
    monkeypatch.setattr("api.routes.get_direct_response", lambda intent, emotion, language: "I only handle mental health topics." )
    monkeypatch.setattr("history_manager.get_history", lambda sid: [])
    monkeypatch.setattr("history_manager.append_history", lambda sid, u, a: None)

    r = client.post("/chat", json={"message":"Who won the World Cup?"})
    assert r.status_code == 200
    j = r.json()
    assert j["response_source"] == "direct"
    assert j["intent"] == Intent.OUT_OF_SCOPE.value


def test_feedback_endpoint(monkeypatch):
    recorded = {"vote": "thumbs_up", "recorded_at": "2026-01-01T00:00:00Z", "session_id": "session_test"}
    monkeypatch.setattr("feedback_manager.append_feedback", lambda payload: {**payload, "recorded_at": recorded["recorded_at"]})

    r = client.post("/feedback", json={"vote": "thumbs_up", "session_id": "session_test", "comment": "Nice"})
    assert r.status_code == 200
    j = r.json()
    assert j["vote"] == "thumbs_up"
    assert j["recorded_at"] == recorded["recorded_at"]


def test_rag_pipeline_failure_returns_500(monkeypatch):
    # Make classify_intent route to RAG and make pipeline.answer raise
    monkeypatch.setattr("language_detector.language_detector.detect_language", lambda text: "en")
    monkeypatch.setattr("emotion_classifier.predictor.predict_emotion", lambda text: Emotion.FEAR)
    monkeypatch.setattr("api.routes.classify_intent", lambda text: Intent.ASKING_MENTAL_HEALTH)
    monkeypatch.setattr("history_manager.get_history", lambda sid: [])
    monkeypatch.setattr("history_manager.append_history", lambda sid, u, a: None)

    class BrokenPipeline:
        def __init__(self, *a, **k):
            pass
        def answer(self, *a, **k):
            raise RuntimeError("RAG failure")

    # create a test app with a broken pipeline attached
    test_app2 = FastAPI()
    test_app2.include_router(api_app.router if hasattr(api_app, 'router') else api_app.app.router)
    test_app2.state.pipeline = BrokenPipeline()
    client2 = TestClient(test_app2)

    r = client2.post("/chat", json={"message": "I feel anxious"})
    assert r.status_code == 500


def test_direct_response_failure_returns_500(monkeypatch):
    monkeypatch.setattr("language_detector.language_detector.detect_language", lambda text: "en")
    monkeypatch.setattr("emotion_classifier.predictor.predict_emotion", lambda text: Emotion.FEAR)
    monkeypatch.setattr("api.routes.classify_intent", lambda text: Intent.OUT_OF_SCOPE)
    monkeypatch.setattr("api.routes.get_direct_response", lambda intent, emotion, language: (_ for _ in ()).throw(RuntimeError("direct failure")))
    monkeypatch.setattr("history_manager.get_history", lambda sid: [])

    r = client.post("/chat", json={"message": "Tell me a joke"})
    assert r.status_code == 500
