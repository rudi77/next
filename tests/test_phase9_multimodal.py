"""Tests for Phase 9: multimodal datasets, bundle upload, vision metrics."""

import asyncio
import io
import json
import zipfile

import pytest
from fastapi.testclient import TestClient

from trainpipe.api.deps import get_db, get_gpu_pool, get_scheduler, get_study_manager
from trainpipe.api.main import app
from trainpipe.core import repository
from trainpipe.core.db import Database
from trainpipe.evals.metrics import get_metric_class
from trainpipe.evals.metrics.bbox_iou import _iou
from trainpipe.scheduler.gpu_pool import GpuPool
from trainpipe.training.dataset_formats import detect_and_validate_info

HEADERS = {"X-API-Key": "test-key"}


def _run(coro):
    return asyncio.run(coro)


class _NoopScheduler:
    async def cancel_experiment(self, experiment_id):
        return False


class _StubStudyManager:
    async def create_and_start(self, config):
        return "x"

    async def cancel(self, study_id):
        return True


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setattr("trainpipe.settings.settings.api_key", "test-key")
    monkeypatch.setattr("trainpipe.settings.settings.data_dir", tmp_path)
    db = Database(tmp_path / "test.sqlite3")
    _run(db.init())
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_scheduler] = lambda: _NoopScheduler()
    app.dependency_overrides[get_gpu_pool] = lambda: GpuPool([])
    app.dependency_overrides[get_study_manager] = lambda: _StubStudyManager()
    yield {"db": db, "tmp": tmp_path}
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def test_detect_text_jsonl_has_no_media(tmp_path):
    p = tmp_path / "text.jsonl"
    p.write_text(
        '{"prompt":"hi","response":"there"}\n'
        '{"prompt":"yo","response":"sup"}\n',
        encoding="utf-8",
    )
    info = detect_and_validate_info(p)
    assert info.format == "jsonl"
    assert info.line_count == 2
    assert info.media_kinds == []


def test_detect_image_jsonl_picks_up_images(tmp_path):
    p = tmp_path / "img.jsonl"
    p.write_text(
        '{"images":["images/a.png"],"prompt":"<image>describe","response":"a cat"}\n'
        '{"images":["images/b.png"],"prompt":"<image>describe","response":"a dog"}\n',
        encoding="utf-8",
    )
    info = detect_and_validate_info(p)
    assert info.media_kinds == ["images"]


def test_detect_video_and_image(tmp_path):
    p = tmp_path / "vid.jsonl"
    p.write_text(
        '{"videos":["v/a.mp4"],"images":["i/a.png"],"prompt":"x","response":"y"}\n',
        encoding="utf-8",
    )
    info = detect_and_validate_info(p)
    # Stable order from _MEDIA_FIELDS: images first, then videos.
    assert info.media_kinds == ["images", "videos"]


def test_detect_skips_invalid_media_field(tmp_path):
    """A non-list ``images`` value isn't picked up as multimodal."""
    p = tmp_path / "x.jsonl"
    p.write_text(
        '{"images":null,"prompt":"x","response":"y"}\n'
        '{"images":"just-a-string","prompt":"x","response":"y"}\n',
        encoding="utf-8",
    )
    info = detect_and_validate_info(p)
    assert info.media_kinds == []


# ---------------------------------------------------------------------------
# Bundle upload
# ---------------------------------------------------------------------------


def _make_bundle_zip(image_paths: list[str], with_jsonl: bool = True) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if with_jsonl:
            lines = [
                json.dumps({"images": [p], "prompt": "x", "response": "y"})
                for p in image_paths
            ]
            zf.writestr("train.jsonl", "\n".join(lines))
        for p in image_paths:
            # Minimal PNG (8-byte signature is enough — we don't validate)
            zf.writestr(p, b"\x89PNG\r\n\x1a\n")
    return buf.getvalue()


def test_bundle_upload_creates_multimodal_dataset(state, client):
    zip_bytes = _make_bundle_zip(["images/a.png", "images/b.png"])
    r = client.post(
        "/datasets/bundle",
        headers=HEADERS,
        files={"file": ("bundle.zip", zip_bytes, "application/zip")},
        data={"name": "doc-bundle"},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["format"] == "jsonl"
    assert body["media_kinds"] == ["images"]
    assert body["image_root"] is not None
    assert body["line_count"] == 2


def test_bundle_upload_rejects_text_only(state, client):
    """A bundle without images should be rejected — text uploads go via POST /datasets."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("plain.jsonl", '{"prompt":"x","response":"y"}\n')
    r = client.post(
        "/datasets/bundle",
        headers=HEADERS,
        files={"file": ("bundle.zip", buf.getvalue(), "application/zip")},
        data={"name": "no-img"},
    )
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "bundle_without_media"


def test_bundle_upload_rejects_symlink_entry(state, client):
    """A zip containing a POSIX symlink entry must be rejected before
    ``extractall`` materializes it on disk."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        # Build a ZipInfo with the symlink mode bit set.
        info = zipfile.ZipInfo("images/link.png")
        info.external_attr = (0o120000 | 0o755) << 16
        zf.writestr(info, "../../target")
        zf.writestr("train.jsonl", '{"images":["images/link.png"]}\n')
    r = client.post(
        "/datasets/bundle",
        headers=HEADERS,
        files={"file": ("evil.zip", buf.getvalue(), "application/zip")},
        data={"name": "evil"},
    )
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "unsafe_zip_symlink"


def test_bundle_upload_rejects_zip_slip(state, client):
    """A zip with ``..`` path components must be rejected before extraction."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../escaped.png", b"\x89PNG\r\n\x1a\n")
        zf.writestr("train.jsonl", '{"images":["../escaped.png"]}\n')
    r = client.post(
        "/datasets/bundle",
        headers=HEADERS,
        files={"file": ("evil.zip", buf.getvalue(), "application/zip")},
        data={"name": "evil"},
    )
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "unsafe_zip_path"


def test_bundle_upload_requires_zip_extension(state, client):
    r = client.post(
        "/datasets/bundle",
        headers=HEADERS,
        files={"file": ("bundle.tgz", b"\x1f\x8b", "application/octet-stream")},
        data={"name": "x"},
    )
    assert r.status_code == 422


def test_bundle_invalid_zip(state, client):
    r = client.post(
        "/datasets/bundle",
        headers=HEADERS,
        files={"file": ("bad.zip", b"not a zip", "application/zip")},
        data={"name": "x"},
    )
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "invalid_zip"


def test_bundle_no_jsonl(state, client):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("images/a.png", b"PNG")
    r = client.post(
        "/datasets/bundle",
        headers=HEADERS,
        files={"file": ("b.zip", buf.getvalue(), "application/zip")},
        data={"name": "x"},
    )
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "no_jsonl_in_bundle"


def test_media_route_serves_thumbnail(state, client):
    zip_bytes = _make_bundle_zip(["images/a.png"])
    r = client.post(
        "/datasets/bundle",
        headers=HEADERS,
        files={"file": ("b.zip", zip_bytes, "application/zip")},
        data={"name": "x"},
    )
    ds_id = r.json()["id"]
    media = client.get(
        f"/datasets/{ds_id}/media", params={"path": "images/a.png"}, headers=HEADERS
    )
    assert media.status_code == 200


def test_media_route_blocks_traversal(state, client):
    zip_bytes = _make_bundle_zip(["images/a.png"])
    r = client.post(
        "/datasets/bundle",
        headers=HEADERS,
        files={"file": ("b.zip", zip_bytes, "application/zip")},
        data={"name": "x"},
    )
    ds_id = r.json()["id"]
    media = client.get(
        f"/datasets/{ds_id}/media",
        params={"path": "../../../etc/hosts"},
        headers=HEADERS,
    )
    assert media.status_code == 404


def test_media_route_404_on_text_dataset(state, client):
    """A text-only dataset has no image_root → 404 on /media."""

    async def _make():
        async with state["db"].connect() as conn:
            return await repository.create_dataset(
                conn,
                name="text",
                path="/tmp/text.jsonl",
                fmt="jsonl",
                size_bytes=10,
                sha256="abc",
                line_count=1,
            )

    ds_id = _run(_make())
    r = client.get(
        f"/datasets/{ds_id}/media", params={"path": "x.png"}, headers=HEADERS
    )
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# bbox_iou metric
# ---------------------------------------------------------------------------


def test_bbox_iou_perfect_match():
    m = get_metric_class("bounding_box_iou")()
    pred = json.dumps([[0, 0, 10, 10]])
    sample = {"gold_boxes": [[0, 0, 10, 10]]}
    assert m.score(pred, sample) == 1.0


def test_bbox_iou_disjoint_boxes_zero():
    m = get_metric_class("bounding_box_iou")()
    pred = json.dumps([[0, 0, 10, 10]])
    sample = {"gold_boxes": [[100, 100, 110, 110]]}
    assert m.score(pred, sample) == 0.0


def test_bbox_iou_label_strict_mismatch():
    m = get_metric_class("bounding_box_iou")({"label_strict": True})
    pred = json.dumps([{"box": [0, 0, 10, 10], "label": "header"}])
    sample = {"gold_boxes": [{"box": [0, 0, 10, 10], "label": "footer"}]}
    assert m.score(pred, sample) == 0.0


def test_bbox_iou_label_lenient_match():
    m = get_metric_class("bounding_box_iou")({"label_strict": False})
    pred = json.dumps([{"box": [0, 0, 10, 10], "label": "header"}])
    sample = {"gold_boxes": [{"box": [0, 0, 10, 10], "label": "footer"}]}
    assert m.score(pred, sample) == 1.0


def test_bbox_iou_threshold_below():
    m = get_metric_class("bounding_box_iou")({"iou_threshold": 0.9})
    # 50% overlap area — IoU ~ 1/3
    pred = json.dumps([[0, 0, 10, 10]])
    sample = {"gold_boxes": [[5, 0, 15, 10]]}
    assert m.score(pred, sample) == 0.0


def test_bbox_iou_invalid_prediction_zero():
    m = get_metric_class("bounding_box_iou")()
    assert m.score("not-json", {"gold_boxes": [[0, 0, 1, 1]]}) == 0.0


def test_bbox_iou_both_empty_returns_zero():
    """Both-empty must NOT reward broken empty-predictions models —
    see bbox_iou module docstring for rationale."""
    m = get_metric_class("bounding_box_iou")()
    assert m.score("[]", {"gold_boxes": []}) == 0.0


def test_iou_helper():
    assert _iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert _iou([0, 0, 10, 10], [10, 10, 20, 20]) == 0.0
    # Half-overlap area
    iou = _iou([0, 0, 10, 10], [5, 0, 15, 10])
    assert abs(iou - (50 / 150)) < 1e-9


def test_bbox_iou_partial_recall():
    """Two predictions, one gold — only one is a TP, the other FP."""
    m = get_metric_class("bounding_box_iou")()
    pred = json.dumps([[0, 0, 10, 10], [100, 100, 110, 110]])
    sample = {"gold_boxes": [[0, 0, 10, 10]]}
    # tp=1, fp=1, fn=0 → precision=0.5, recall=1.0 → F1=0.667
    assert abs(m.score(pred, sample) - (2 * 0.5 * 1.0 / 1.5)) < 1e-9


def test_bbox_iou_config_validation():
    cls = get_metric_class("bounding_box_iou")
    with pytest.raises(ValueError):
        cls({"iou_threshold": 1.5})
    with pytest.raises(ValueError):
        cls({"gold_field": ""})


# ---------------------------------------------------------------------------
# structured_extraction_f1 metric
# ---------------------------------------------------------------------------


def test_structured_f1_perfect():
    m = get_metric_class("structured_extraction_f1")(
        {"schema_fields": ["invoice_no", "total"]}
    )
    pred = json.dumps({"invoice_no": "INV-1", "total": 99.0})
    sample = {"gold": {"invoice_no": "INV-1", "total": 99.0}}
    assert m.score(pred, sample) == 1.0


def test_structured_f1_partial_recall():
    m = get_metric_class("structured_extraction_f1")(
        {"schema_fields": ["a", "b", "c"]}
    )
    pred = json.dumps({"a": 1})
    sample = {"gold": {"a": 1, "b": 2, "c": 3}}
    # tp=1, fp=0, fn=2 → P=1, R=1/3, F1=0.5
    assert abs(m.score(pred, sample) - 0.5) < 1e-9


def test_structured_f1_wrong_value_counts_as_both_fp_and_fn():
    m = get_metric_class("structured_extraction_f1")(
        {"schema_fields": ["x"]}
    )
    pred = json.dumps({"x": "wrong"})
    sample = {"gold": {"x": "right"}}
    # tp=0, fp=1, fn=1 → F1=0
    assert m.score(pred, sample) == 0.0


def test_structured_f1_invented_field_is_fp():
    m = get_metric_class("structured_extraction_f1")({"schema_fields": ["a"]})
    pred = json.dumps({"a": 1, "z": "noise"})  # z is out-of-schema FP
    sample = {"gold": {"a": 1}}
    # tp=1, fp=1, fn=0 → P=0.5, R=1, F1=0.667
    assert abs(m.score(pred, sample) - (2 * 0.5 * 1.0 / 1.5)) < 1e-9


def test_structured_f1_numeric_tolerance():
    m = get_metric_class("structured_extraction_f1")(
        {"schema_fields": ["total"], "numeric_tolerance": 0.01}
    )
    pred = json.dumps({"total": 99.005})
    sample = {"gold": {"total": 99.0}}
    assert m.score(pred, sample) == 1.0


def test_structured_f1_case_insensitive_default():
    m = get_metric_class("structured_extraction_f1")(
        {"schema_fields": ["name"]}
    )
    pred = json.dumps({"name": "INV-1"})
    sample = {"gold": {"name": "inv-1"}}
    assert m.score(pred, sample) == 1.0


def test_structured_f1_nested_schema():
    m = get_metric_class("structured_extraction_f1")(
        {"schema_fields": ["addr.city", "addr.zip"]}
    )
    pred = json.dumps({"addr": {"city": "Wien", "zip": "1010"}})
    sample = {"gold": {"addr": {"city": "Wien", "zip": "1010"}}}
    assert m.score(pred, sample) == 1.0


def test_structured_f1_config_requires_schema():
    cls = get_metric_class("structured_extraction_f1")
    with pytest.raises(ValueError):
        cls({})
    with pytest.raises(ValueError):
        cls({"schema_fields": []})


def test_structured_f1_invalid_json_zero():
    m = get_metric_class("structured_extraction_f1")(
        {"schema_fields": ["a"]}
    )
    assert m.score("not-json", {"gold": {"a": 1}}) == 0.0
