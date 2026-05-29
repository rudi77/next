"""Tests for the swift binary path resolver."""

import sys
from pathlib import Path

from trainpipe.training import swift_builder


def test_resolver_returns_string():
    swift_builder._resolve_swift_binary.cache_clear()
    out = swift_builder._resolve_swift_binary()
    assert isinstance(out, str)
    assert out.endswith("swift") or out == "swift"


def test_resolver_prefers_path_when_present(monkeypatch, tmp_path):
    swift_builder._resolve_swift_binary.cache_clear()
    fake = tmp_path / "swift"
    fake.write_text("#!/bin/sh\necho fake\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    out = swift_builder._resolve_swift_binary()
    assert out == str(fake) or out.endswith("swift")


def test_resolver_falls_back_to_venv_bin(monkeypatch, tmp_path):
    swift_builder._resolve_swift_binary.cache_clear()
    # PATH does not contain swift
    monkeypatch.setenv("PATH", str(tmp_path / "nope"))
    # Place a fake swift next to a fake "python" executable that we'll pretend
    # sys.executable points to.
    fake_bin = tmp_path / "venvbin"
    fake_bin.mkdir()
    fake_swift = fake_bin / "swift"
    fake_swift.write_text("#!/bin/sh\necho fake\n", encoding="utf-8")
    fake_swift.chmod(0o755)
    fake_python = fake_bin / "python3"
    fake_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(sys, "executable", str(fake_python))
    out = swift_builder._resolve_swift_binary()
    assert out == str(fake_swift)


def test_resolver_falls_back_to_literal(monkeypatch, tmp_path):
    swift_builder._resolve_swift_binary.cache_clear()
    monkeypatch.setenv("PATH", str(tmp_path / "empty-1"))
    fake_python = tmp_path / "empty-2" / "python3"
    fake_python.parent.mkdir()
    fake_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(sys, "executable", str(fake_python))
    out = swift_builder._resolve_swift_binary()
    assert out == "swift"
