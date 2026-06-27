"""Tests for the operative ``trainpipe`` CLI.

Handlers are exercised against a fake httpx client that records the request
and returns a canned ``httpx.Response`` — no live server needed. This pins
down the method + path + params each subcommand sends so the CLI stays in
parity with the REST API (and thus the MCP server)."""

import json

import httpx
import pytest

from trainpipe import cli


class FakeClient:
    """Records calls and returns a preset httpx.Response."""

    def __init__(self, response: httpx.Response | None = None):
        self.response = response or httpx.Response(200, json={"ok": True})
        self.calls: list[tuple[str, str, dict]] = []

    def _record(self, method: str, url: str, **kwargs) -> httpx.Response:
        self.calls.append((method, url, kwargs))
        return self.response

    def get(self, url, **kw):
        return self._record("GET", url, **kw)

    def post(self, url, **kw):
        return self._record("POST", url, **kw)

    def delete(self, url, **kw):
        return self._record("DELETE", url, **kw)

    def request(self, method, url, **kw):
        return self._record(method, url, **kw)


def _run(argv: list[str], client: FakeClient):
    """Parse argv, dispatch the handler against the fake client."""
    args = cli.build_parser().parse_args(argv)
    return args.func(client, args), args


def test_no_command_is_serve(monkeypatch):
    called = {}
    monkeypatch.setattr(cli, "serve", lambda: called.setdefault("served", True))
    cli.main([])
    assert called["served"] is True


def test_serve_command(monkeypatch):
    called = {}
    monkeypatch.setattr(cli, "serve", lambda: called.setdefault("served", True))
    cli.main(["serve"])
    assert called["served"] is True


def test_submit_builds_spec_from_flags():
    c = FakeClient()
    _run(["submit", "--model", "m", "--dataset", "ds:a", "--dataset", "ds:b",
          "--train-kind", "grpo", "--gpu-count", "2"], c)
    method, url, kw = c.calls[0]
    assert (method, url) == ("POST", "/experiments")
    assert kw["json"]["model"] == "m"
    assert kw["json"]["dataset"] == ["ds:a", "ds:b"]
    assert kw["json"]["train_kind"] == "grpo"
    assert kw["json"]["gpu_count"] == 2


def test_submit_requires_model_and_dataset():
    c = FakeClient()
    with pytest.raises(ValueError, match="requires --model and --dataset"):
        _run(["submit", "--model", "m"], c)


def test_submit_from_inline_json_spec():
    c = FakeClient()
    _run(["submit", "--spec", '{"model":"x","dataset":["d"]}'], c)
    assert c.calls[0][2]["json"] == {"model": "x", "dataset": ["d"]}


def test_experiments_filters():
    c = FakeClient()
    _run(["experiments", "--status", "running", "--limit", "5"], c)
    method, url, kw = c.calls[0]
    assert (method, url) == ("GET", "/experiments")
    assert kw["params"] == {"limit": 5, "status": "running"}


def test_get_and_cancel_paths():
    c = FakeClient()
    _run(["get", "exp1"], c)
    _run(["cancel", "exp1"], c)
    assert c.calls[0] == ("GET", "/experiments/exp1", {})
    assert c.calls[1] == ("POST", "/experiments/exp1/cancel", {})


def test_run_eval_payload():
    c = FakeClient()
    _run(["run-eval", "--suite", "s1", "--experiment", "e1"], c)
    method, url, kw = c.calls[0]
    assert (method, url) == ("POST", "/evals/runs")
    assert kw["json"] == {"suite_id": "s1", "experiment_id": "e1"}


def test_compare_evals_joins_run_ids():
    c = FakeClient()
    _run(["compare-evals", "r1", "r2", "r3"], c)
    method, url, kw = c.calls[0]
    assert (method, url) == ("GET", "/evals/compare")
    assert kw["params"] == {"run_ids": "r1,r2,r3"}


def test_set_alias_payload():
    c = FakeClient()
    _run(["set-alias", "fam", "production", "3"], c)
    method, url, kw = c.calls[0]
    assert (method, url) == ("POST", "/models/fam/aliases/production")
    assert kw["json"] == {"version": 3}


def test_inference_payload():
    c = FakeClient()
    _run(["inference", "fam@production", "hello", "--max-new-tokens", "64"], c)
    method, url, kw = c.calls[0]
    assert (method, url) == ("POST", "/inferences")
    assert kw["json"]["model_ref"] == "fam@production"
    assert kw["json"]["params"]["max_new_tokens"] == 64


def test_generic_api_command():
    c = FakeClient()
    _run(["api", "delete", "/datasets/abc"], c)
    assert c.calls[0][0] == "DELETE"
    assert c.calls[0][1] == "/datasets/abc"


def test_generic_api_with_json_body():
    c = FakeClient()
    _run(["api", "post", "/studies", "--json", '{"name":"s"}'], c)
    assert c.calls[0][2]["json"] == {"name": "s"}


def test_render_logs_tail():
    c = FakeClient(httpx.Response(200, text="a\nb\nc\nd"))
    result, args = _run(["logs", "exp1", "-n", "2"], c)
    assert cli._render(result, args) == "c\nd"


def test_render_json_pretty():
    c = FakeClient(httpx.Response(200, json={"b": 1, "a": 2}))
    result, args = _run(["experiments"], c)
    out = cli._render(result, args)
    assert json.loads(out) == {"b": 1, "a": 2}


def test_load_json_arg_from_stdin(monkeypatch):
    import io
    monkeypatch.setattr("sys.stdin", io.StringIO('{"k": 1}'))
    assert cli._load_json_arg("@-") == {"k": 1}


def test_load_json_arg_from_file(tmp_path):
    f = tmp_path / "spec.json"
    f.write_text('{"model": "m"}', encoding="utf-8")
    assert cli._load_json_arg(f"@{f}") == {"model": "m"}


class _CtxFake(FakeClient):
    """FakeClient usable as a context manager (build_client returns one)."""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_api_error_exits_nonzero(monkeypatch):
    err_resp = httpx.Response(404, json={"detail": "nope"})
    monkeypatch.setattr(cli, "build_client", lambda: _CtxFake(err_resp))
    with pytest.raises(SystemExit) as exc:
        cli.main(["get", "x"])
    assert exc.value.code == 1
