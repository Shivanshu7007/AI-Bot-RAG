"""Integration tests for POST /ask — Qdrant and LLM are mocked."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Minimal settings stub so importing app.main doesn't need real env vars
# ---------------------------------------------------------------------------
class _Settings:
    ENV = "test"
    SERVICE_API_KEY = "test-key"
    OPENROUTER_API_KEY = "test-openrouter"
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    MODEL_NAME = "test-model"
    QDRANT_URL = "http://localhost:6333"
    QDRANT_API_KEY = None
    SIMILARITY_THRESHOLD = 0.45
    MAX_CONTEXT_LENGTH = 5000
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 150


HEADERS = {"x-api-key": "test-key"}


@pytest.fixture(scope="module")
def client():
    with patch("app.core.config.settings", _Settings()):
        from app.main import app
        return TestClient(app)


def _mock_qdrant_has_docs(has: bool):
    return patch("app.routes.ask.collection_has_documents", return_value=has)


def _mock_embedding():
    return patch("app.routes.ask.get_embedding", return_value=[0.1] * 384)


def _mock_search(results):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"result": results}
    return patch("app.routes.ask.search", return_value=mock_resp)


def _mock_llm(answer: str):
    return patch("app.routes.ask.generate_answer", return_value=answer)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_missing_api_key_returns_401(client):
    resp = client.post("/ask", json={"product_id": 1, "question": "What is this?"})
    assert resp.status_code == 401


def test_wrong_api_key_returns_403(client):
    resp = client.post(
        "/ask",
        json={"product_id": 1, "question": "What is this?"},
        headers={"x-api-key": "wrong"}
    )
    assert resp.status_code == 403


def test_greeting_returns_plain_text(client):
    with _mock_qdrant_has_docs(True):
        resp = client.post("/ask", json={"product_id": 1, "question": "hi"}, headers=HEADERS)
    assert resp.status_code == 200
    assert "help" in resp.text.lower()


def test_no_knowledge_base_returns_json_status(client):
    with _mock_qdrant_has_docs(False):
        resp = client.post(
            "/ask",
            json={"product_id": 999, "question": "What is this?"},
            headers=HEADERS
        )
    assert resp.status_code == 200
    assert resp.json()["status"] == "no_knowledge_base"


def test_low_similarity_returns_fallback_message(client):
    low_score_results = [{"score": 0.1, "payload": {"text": "some text"}}]
    with _mock_qdrant_has_docs(True), \
         _mock_embedding(), \
         _mock_search(low_score_results):
        resp = client.post(
            "/ask",
            json={"product_id": 1, "question": "Completely unrelated question"},
            headers=HEADERS
        )
    assert resp.status_code == 200
    assert "not able to find" in resp.text


def test_successful_answer_returned(client):
    good_results = [{"score": 0.9, "payload": {"text": "Store at 4°C."}}]
    with _mock_qdrant_has_docs(True), \
         _mock_embedding(), \
         _mock_search(good_results), \
         _mock_llm("According to the product manual, store at 4°C."):
        resp = client.post(
            "/ask",
            json={"product_id": 1, "question": "How should I store the kit?"},
            headers=HEADERS
        )
    assert resp.status_code == 200
    assert "According to the product manual" in resp.text


def test_question_too_long_returns_400(client):
    long_question = "x" * 1001
    resp = client.post(
        "/ask",
        json={"product_id": 1, "question": long_question},
        headers=HEADERS
    )
    assert resp.status_code == 400
