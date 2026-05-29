import json

import pytest

from trainpipe.training.dataset_formats import (
    DatasetFormatError,
    detect_and_validate,
)


def test_jsonl_valid(tmp_path):
    p = tmp_path / "train.jsonl"
    p.write_text(
        "\n".join(
            [
                json.dumps({"messages": [{"role": "user", "content": "hi"}]}),
                json.dumps({"messages": [{"role": "user", "content": "bye"}]}),
            ]
        ),
        encoding="utf-8",
    )
    fmt, n = detect_and_validate(p)
    assert fmt == "jsonl"
    assert n == 2


def test_jsonl_with_blank_lines_counted(tmp_path):
    p = tmp_path / "train.jsonl"
    p.write_text('{"a":1}\n\n{"a":2}\n\n\n', encoding="utf-8")
    fmt, n = detect_and_validate(p)
    assert fmt == "jsonl"
    assert n == 2


def test_jsonl_invalid_first_line(tmp_path):
    p = tmp_path / "train.jsonl"
    p.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(DatasetFormatError, match="line 1"):
        detect_and_validate(p)


def test_jsonl_empty(tmp_path):
    p = tmp_path / "train.jsonl"
    p.write_text("", encoding="utf-8")
    with pytest.raises(DatasetFormatError, match="empty"):
        detect_and_validate(p)


def test_json_valid(tmp_path):
    p = tmp_path / "train.json"
    p.write_text(json.dumps([{"a": 1}, {"a": 2}, {"a": 3}]), encoding="utf-8")
    fmt, n = detect_and_validate(p)
    assert fmt == "json"
    assert n == 3


def test_json_must_be_list(tmp_path):
    p = tmp_path / "train.json"
    p.write_text(json.dumps({"a": 1}), encoding="utf-8")
    with pytest.raises(DatasetFormatError, match="top-level list"):
        detect_and_validate(p)


def test_csv_valid(tmp_path):
    p = tmp_path / "train.csv"
    p.write_text("query,response\nfoo,bar\nbaz,qux\n", encoding="utf-8")
    fmt, n = detect_and_validate(p)
    assert fmt == "csv"
    assert n == 2


def test_csv_header_only(tmp_path):
    p = tmp_path / "train.csv"
    p.write_text("query,response\n", encoding="utf-8")
    with pytest.raises(DatasetFormatError, match="no data rows"):
        detect_and_validate(p)


def test_tsv_valid(tmp_path):
    p = tmp_path / "train.tsv"
    p.write_text("a\tb\n1\t2\n3\t4\n", encoding="utf-8")
    fmt, n = detect_and_validate(p)
    assert fmt == "tsv"
    assert n == 2


def test_unsupported_extension(tmp_path):
    p = tmp_path / "train.xyz"
    p.write_text("nope", encoding="utf-8")
    with pytest.raises(DatasetFormatError, match="unsupported"):
        detect_and_validate(p)
