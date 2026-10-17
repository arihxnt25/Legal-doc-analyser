import pytest
from fastapi.testclient import TestClient

import api.main as api_main


@pytest.fixture(scope="module")
def client(trained):
    _, path = trained
    api_main.load_state(path)
    with TestClient(api_main.app) as c:
        yield c


def test_health_and_categories(client):
    assert client.get("/health").json()["status"] == "ok"
    cats = client.get("/categories").json()
    assert len(cats) == 41 and {"name", "group", "description"} <= set(cats[0])


def test_analyze_text(client, sample_text):
    r = client.post("/analyze/text", json={"text": sample_text, "title": "msa", "include_highlights": True})
    assert r.status_code == 200
    body = r.json()
    assert {"risk_score", "findings", "clauses", "metadata", "summary", "highlights"} <= set(body)


def test_analyze_file_upload(client, sample_text):
    r = client.post("/analyze/file", files={"file": ("msa.txt", sample_text.encode(), "text/plain")})
    assert r.status_code == 200 and "highlights" not in r.json()


def test_rejects_bad_input(client):
    assert client.post("/analyze/text", json={"text": "too short"}).status_code == 422
    r = client.post("/analyze/file", files={"file": ("x.xlsx", b"PK...", "application/octet-stream")})
    assert r.status_code == 415


def test_search_and_explain(client, sample_text):
    r = client.post("/search", json={"text": sample_text, "query": "limitation of liability", "k": 3})
    assert r.status_code == 200 and len(r.json()) <= 3
    assert client.get("/model/explain/Governing Law").status_code == 200
    assert client.get("/model/explain/Not A Category").status_code == 404
