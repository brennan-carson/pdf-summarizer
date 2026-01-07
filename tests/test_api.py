from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_rejects_non_pdf():
    response = client.post(
        "/summarize-pdf",
        files={"file": ("test.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400
