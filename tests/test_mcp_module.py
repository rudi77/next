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
    }
    missing = expected - names
    assert not missing, f"missing tools: {missing}"


@pytest.mark.asyncio
async def test_every_tool_has_description():
    import trainpipe.mcp as m

    tools = await m.mcp.list_tools()
    bare = [t.name for t in tools if not (t.description or "").strip()]
    assert not bare, f"tools without description (agents won't know how to call): {bare}"
