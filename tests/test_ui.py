"""Tests for the SPA mount and the public /ui/config endpoint."""

from fastapi.testclient import TestClient

from trainpipe.api.main import _public_mlflow_uri, app


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


def test_public_mlflow_uri_strips_embedded_credentials(monkeypatch):
    monkeypatch.setattr(
        "trainpipe.settings.settings.mlflow_tracking_uri",
        "http://user:s3cret@mlflow.internal:5000/path",
    )
    public = _public_mlflow_uri()
    assert public == "http://mlflow.internal:5000/path"
    assert "s3cret" not in public
    assert "user" not in public


def test_public_mlflow_uri_passthrough_when_no_credentials(monkeypatch):
    monkeypatch.setattr(
        "trainpipe.settings.settings.mlflow_tracking_uri",
        "http://mlflow.internal:5000",
    )
    assert _public_mlflow_uri() == "http://mlflow.internal:5000"
