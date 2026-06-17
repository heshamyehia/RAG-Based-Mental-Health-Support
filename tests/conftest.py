import sys
from types import ModuleType, SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# --- Dummy RAG pipeline fixture ------------------------------------------------
@pytest.fixture
def dummy_pipeline():
    class DummyPipeline:
        def answer(self, *args, **kwargs):
            return {"answer": "This is a RAG answer."}

    return DummyPipeline()


# --- Autouse shims for external heavy deps (google.genai, dotenv, torch, joblib) --
@pytest.fixture(autouse=True)
def global_shims():
    # dotenv shim
    dotenv_mod = ModuleType("dotenv")
    dotenv_mod.load_dotenv = lambda *a, **k: None
    sys.modules["dotenv"] = dotenv_mod

    # google.genai shim
    google_mod = ModuleType("google")
    genai_mod = ModuleType("google.genai")
    errors_mod = ModuleType("google.genai.errors")

    class FakeClientError(Exception):
        pass

    errors_mod.ClientError = FakeClientError

    class FakeClient:
        def __init__(self, api_key=None):
            # default behavior: return out_of_scope label
            self.models = SimpleNamespace(
                generate_content=lambda **k: SimpleNamespace(text="out_of_scope")
            )

    genai_mod.errors = errors_mod
    genai_mod.Client = FakeClient
    google_mod.genai = genai_mod

    sys.modules["google"] = google_mod
    sys.modules["google.genai"] = genai_mod
    sys.modules["google.genai.errors"] = errors_mod

    # emotion_classifier shim (predictor)
    emotion_pkg = ModuleType("emotion_classifier")
    emotion_mod = ModuleType("emotion_classifier.predictor")
    emotion_mod.predict_emotion = lambda text: "neutral"
    emotion_pkg.predictor = emotion_mod
    sys.modules["emotion_classifier"] = emotion_pkg
    sys.modules["emotion_classifier.predictor"] = emotion_mod

    # language_detector shim
    lang_pkg = ModuleType("language_detector")
    lang_mod = ModuleType("language_detector.language_detector")
    lang_mod.detect_language = lambda text: "en"
    lang_pkg.language_detector = lang_mod
    sys.modules["language_detector"] = lang_pkg
    sys.modules["language_detector.language_detector"] = lang_mod

    yield


# --- Test FastAPI app builder -------------------------------------------------
@pytest.fixture
def test_app(dummy_pipeline):
    # import router directly to avoid app lifespan side-effects
    import api.routes as routes

    app = FastAPI()
    app.include_router(routes.router)
    app.state.pipeline = dummy_pipeline
    return TestClient(app)
