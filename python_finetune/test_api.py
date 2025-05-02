import pytest
from fastapi.testclient import TestClient
from infer_api import app

client = TestClient(app)

def test_generate():
    response = client.post("/generate", json={"text": "What is the capital of France?"})
    assert response.status_code == 200
    assert "response" in response.json()

def test_upload_pdf():
    with open("../data/test.pdf", "rb") as f:
        response = client.post("/upload", files={"file": ("test.pdf", f, "application/pdf")})
    assert response.status_code == 200
    assert response.json()["status"] == "Indexed"
