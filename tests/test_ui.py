"""Tests for the SPA mount and the public /ui/config endpoint."""

from fastapi.testclient import TestClient

from trainpipe.api.main import app


def test_ui_root_serves_html():
    with TestClient(app) as client:
        r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
    assert "<title>trainpipe</title>" in r.text
    assert 'x-data="trainpipe()"' in r.text


def test_ui_config_is_public_and_returns_mlflow_uri():
    with TestClient(app) as client:
        r = client.get("/ui/config")
    assert r.status_code == 200
    body = r.json()
    assert "mlflow_tracking_uri" in body
    assert body["mlflow_tracking_uri"].startswith("http")


def test_ui_config_does_not_leak_api_key():
    with TestClient(app) as client:
        r = client.get("/ui/config")
    body = r.json()
    assert "api_key" not in body
    assert "TRAINPIPE_API_KEY" not in r.text
