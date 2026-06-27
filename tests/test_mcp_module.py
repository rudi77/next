"""Smoke tests for trainpipe.mcp: the module imports without TRAINPIPE_API_KEY
set, all expected tools are registered, and each tool has a docstring (the
MCP discovery surface that agents see)."""

import pytest

pytest.importorskip("mcp")


def test_module_imports_without_api_key(monkeypatch):
    monkeypatch.delenv("TRAINPIPE_API_KEY", raising=False)
    # Should not raise — client construction is deferred.
    import importlib

    import trainpipe.mcp as m

    importlib.reload(m)
    assert m.mcp.name == "trainpipe"


@pytest.mark.asyncio
async def test_expected_tools_are_registered():
    import trainpipe.mcp as m

    tools = await m.mcp.list_tools()
    names = {t.name for t in tools}
    expected = {
        "submit_experiment",
        "get_experiment",
        "list_experiments",
        "cancel_experiment",
        "tail_logs",
        "submit_study",
        "list_studies",
        "get_study",
        "cancel_study",
        "gpu_status",
        "upload_dataset",
        "list_datasets",
        "get_dataset",
        "preview_dataset",
        "delete_dataset",
        # Model registry (Phase 7) — the spec lists these as registered tools.
        "register_model",
        "list_models",
        "get_model",
        "set_alias",
        "delete_model",
        # Inference playground (Phase 8) + synth (Phase 14) + compliance (Phase 15).
        "inference",
        "inference_compare",
        "synth_dataset",
        "forget_scan",
        # Evals (Phase 6) — closes the train → eval → improve loop for agents.
        "create_eval_suite",
        "list_eval_suites",
        "get_eval_suite",
        "delete_eval_suite",
        "run_eval",
        "list_eval_runs",
        "get_eval_run",
        "get_eval_results",
        "cancel_eval_run",
        "compare_evals",
    }
    missing = expected - names
    assert not missing, f"missing tools: {missing}"


@pytest.mark.asyncio
async def test_every_tool_has_description():
    import trainpipe.mcp as m

    tools = await m.mcp.list_tools()
    bare = [t.name for t in tools if not (t.description or "").strip()]
    assert not bare, f"tools without description (agents won't know how to call): {bare}"
