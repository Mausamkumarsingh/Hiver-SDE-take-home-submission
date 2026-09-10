import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["brand"] == "AmazonHelp"
    assert data["num_intents"] == 10

def test_intents_endpoint():
    response = client.get("/intents")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 10
    assert len(data["intents"]) == 10

def test_classify_endpoint():
    response = client.post("/classify", json={"text": "Where is my package? It is late."})
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "ORDER_TRACKING_AND_DELIVERY"
    assert 0.0 <= data["confidence"] <= 1.0

def test_retrieve_endpoint():
    response = client.post("/retrieve", json={"query": "How do I return my shoes?", "top_k": 2})
    assert response.status_code == 200
    data = response.json()
    assert "evidence" in data
    assert len(data["evidence"]) <= 2

def test_chat_endpoint():
    response = client.post("/chat", json={"message": "I was charged twice for order 123.", "top_k": 2})
    assert response.status_code == 200
    data = response.json()
    assert "intent" in data
    assert "confidence" in data
    assert "reply" in data
    assert "action" in data
    assert "reason" in data
    assert "evidence" in data
    assert data["action"] == "ESCALATE"
