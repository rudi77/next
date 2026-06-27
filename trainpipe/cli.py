"""``trainpipe`` CLI — launch the server *and* drive it from the terminal.

With no subcommand (or ``trainpipe serve``) it starts the FastAPI server,
exactly as before. The other subcommands are a thin operative client over
the REST API — the same surface the MCP server exposes to agents — so a
human or a shell script can run the full train → eval → improve loop
without writing ``curl`` by hand::

    trainpipe submit --model Qwen/Qwen2.5-0.5B --dataset ds:ab12 --train-kind sft
    trainpipe experiments --status running
    trainpipe logs <exp-id> -n 50
    trainpipe run-eval --suite <suite-id> --experiment <exp-id>
    trainpipe compare-evals <run-a> <run-b>

Operative subcommands need ``TRAINPIPE_API_KEY`` (and optionally
``TRAINPIPE_BASE_URL``) set, identical to the MCP server. Output is JSON on
stdout so it pipes straight into ``jq``. The generic ``trainpipe api``
command reaches any endpoint, keeping the CLI at full parity with the REST
API even for routes without a dedicated shortcut.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from .client import APIError, MissingAPIKey, build_client


def serve() -> None:
    """Launch the FastAPI server (the default, no-subcommand behaviour)."""
    import uvicorn

    from .settings import settings

    uvicorn.run(
        "trainpipe.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


def _load_json_arg(value: str) -> Any:
    """Parse a JSON value given inline or, with a leading ``@``, from a file
    (``@-`` reads stdin)."""
    if value == "@-":
        return json.loads(sys.stdin.read())
    if value.startswith("@"):
        with open(value[1:], encoding="utf-8") as f:
            return json.load(f)
    return json.loads(value)


# ---------------------------------------------------------------------------
# Command handlers: each takes (client, args) and returns the parsed REST
# response (printed as JSON by main). Kept free of printing/exiting so they
# are unit-testable with a fake client.
# ---------------------------------------------------------------------------


def _cmd_submit(client: httpx.Client, args: argparse.Namespace) -> Any:
    if args.spec is not None:
        spec = _load_json_arg(args.spec)
    else:
        if not args.model or not args.dataset:
            raise ValueError("submit requires --model and --dataset (or --spec)")
        spec = {
            "model": args.model,
            "dataset": args.dataset,
            "train_kind": args.train_kind,
            "sft_type": args.sft_type,
            "gpu_count": args.gpu_count,
        }
        if args.name:
            spec["name"] = args.name
    return client.post("/experiments", json=spec)


def _cmd_experiments(client: httpx.Client, args: argparse.Namespace) -> Any:
    params: dict[str, Any] = {"limit": args.limit}
    if args.status:
        params["status"] = args.status
    if args.study:
        params["study_id"] = args.study
    return client.get("/experiments", params=params)


def _cmd_get(client: httpx.Client, args: argparse.Namespace) -> Any:
    return client.get(f"/experiments/{args.experiment_id}")


def _cmd_cancel(client: httpx.Client, args: argparse.Namespace) -> Any:
    return client.post(f"/experiments/{args.experiment_id}/cancel")


def _cmd_logs(client: httpx.Client, args: argparse.Namespace) -> Any:
    return client.get(f"/experiments/{args.experiment_id}/logs")


def _cmd_datasets(client: httpx.Client, args: argparse.Namespace) -> Any:
    return client.get("/datasets")


def _cmd_upload(client: httpx.Client, args: argparse.Namespace) -> Any:
    with open(args.file, "rb") as f:
        files = {"file": (Path(args.file).name, f.read(), "application/octet-stream")}
    data = {"name": args.name}
    if args.description:
        data["description"] = args.description
    return client.post("/datasets", files=files, data=data)


def _cmd_models(client: httpx.Client, args: argparse.Namespace) -> Any:
    params: dict[str, Any] = {}
    if args.name:
        params["name"] = args.name
    if args.alias:
        params["alias"] = args.alias
    return client.get("/models", params=params)


def _cmd_register_model(client: httpx.Client, args: argparse.Namespace) -> Any:
    payload: dict[str, Any] = {"name": args.name, "experiment_id": args.experiment}
    if args.alias:
        payload["alias"] = args.alias
    if args.description:
        payload["description"] = args.description
    return client.post("/models", json=payload)


def _cmd_set_alias(client: httpx.Client, args: argparse.Namespace) -> Any:
    return client.post(
        f"/models/{args.name}/aliases/{args.alias}", json={"version": args.version}
    )


def _cmd_inference(client: httpx.Client, args: argparse.Namespace) -> Any:
    return client.post(
        "/inferences",
        json={
            "model_ref": args.model_ref,
            "prompt": args.prompt,
            "params": {"max_new_tokens": args.max_new_tokens},
        },
    )


def _cmd_studies(client: httpx.Client, args: argparse.Namespace) -> Any:
    return client.get("/studies")


def _cmd_gpus(client: httpx.Client, args: argparse.Namespace) -> Any:
    return client.get("/gpus")


def _cmd_eval_suites(client: httpx.Client, args: argparse.Namespace) -> Any:
    return client.get("/evals/suites")


def _cmd_create_suite(client: httpx.Client, args: argparse.Namespace) -> Any:
    return client.post("/evals/suites", json=_load_json_arg(args.spec))


def _cmd_run_eval(client: httpx.Client, args: argparse.Namespace) -> Any:
    return client.post(
        "/evals/runs",
        json={"suite_id": args.suite, "experiment_id": args.experiment},
    )


def _cmd_eval_runs(client: httpx.Client, args: argparse.Namespace) -> Any:
    params: dict[str, Any] = {"limit": args.limit}
    if args.suite:
        params["suite_id"] = args.suite
    if args.experiment:
        params["experiment_id"] = args.experiment
    if args.status:
        params["status"] = args.status
    return client.get("/evals/runs", params=params)


def _cmd_eval_results(client: httpx.Client, args: argparse.Namespace) -> Any:
    return client.get(f"/evals/runs/{args.run_id}/results")


def _cmd_compare_evals(client: httpx.Client, args: argparse.Namespace) -> Any:
    return client.get("/evals/compare", params={"run_ids": ",".join(args.run_ids)})


def _cmd_api(client: httpx.Client, args: argparse.Namespace) -> Any:
    kwargs: dict[str, Any] = {}
    if args.json is not None:
        kwargs["json"] = _load_json_arg(args.json)
    return client.request(args.method.upper(), args.path, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="trainpipe",
        description="Run the trainpipe server, or drive it from the terminal.",
    )
    sub = p.add_subparsers(dest="command")

    sub.add_parser("serve", help="Launch the FastAPI server (default).")

    sp = sub.add_parser("submit", help="Queue a training experiment.")
    sp.add_argument("--model")
    sp.add_argument("--dataset", action="append", help="Repeatable dataset ref.")
    sp.add_argument("--train-kind", default="sft",
                    choices=["sft", "pt", "dpo", "kto", "ppo", "grpo"])
    sp.add_argument("--sft-type", default="lora")
    sp.add_argument("--gpu-count", type=int, default=1)
    sp.add_argument("--name")
    sp.add_argument("--spec", help="Full ExperimentSpec as JSON, @file, or @- for stdin.")
    sp.set_defaults(func=_cmd_submit)

    sp = sub.add_parser("experiments", help="List experiments.")
    sp.add_argument("--status")
    sp.add_argument("--study")
    sp.add_argument("--limit", type=int, default=100)
    sp.set_defaults(func=_cmd_experiments)

    sp = sub.add_parser("get", help="Get one experiment.")
    sp.add_argument("experiment_id")
    sp.set_defaults(func=_cmd_get)

    sp = sub.add_parser("cancel", help="Cancel an experiment.")
    sp.add_argument("experiment_id")
    sp.set_defaults(func=_cmd_cancel)

    sp = sub.add_parser("logs", help="Print an experiment's training log.")
    sp.add_argument("experiment_id")
    sp.add_argument("-n", type=int, default=0, help="Only the last N lines (0=all).")
    sp.set_defaults(func=_cmd_logs)

    sub.add_parser("datasets", help="List datasets.").set_defaults(func=_cmd_datasets)

    sp = sub.add_parser("upload", help="Upload a dataset file.")
    sp.add_argument("name")
    sp.add_argument("file")
    sp.add_argument("--description")
    sp.set_defaults(func=_cmd_upload)

    sp = sub.add_parser("models", help="List registered models.")
    sp.add_argument("--name")
    sp.add_argument("--alias")
    sp.set_defaults(func=_cmd_models)

    sp = sub.add_parser("register-model", help="Register a completed experiment.")
    sp.add_argument("--name", required=True)
    sp.add_argument("--experiment", required=True)
    sp.add_argument("--alias")
    sp.add_argument("--description")
    sp.set_defaults(func=_cmd_register_model)

    sp = sub.add_parser("set-alias", help="Move a model alias to a version.")
    sp.add_argument("name")
    sp.add_argument("alias")
    sp.add_argument("version", type=int)
    sp.set_defaults(func=_cmd_set_alias)

    sp = sub.add_parser("inference", help="Run a prompt against a model ref.")
    sp.add_argument("model_ref")
    sp.add_argument("prompt")
    sp.add_argument("--max-new-tokens", type=int, default=512)
    sp.set_defaults(func=_cmd_inference)

    sub.add_parser("studies", help="List studies.").set_defaults(func=_cmd_studies)
    sub.add_parser("gpus", help="Show GPU pool status.").set_defaults(func=_cmd_gpus)
    sub.add_parser("eval-suites", help="List eval suites.").set_defaults(
        func=_cmd_eval_suites)

    sp = sub.add_parser("create-suite", help="Create an eval suite from JSON.")
    sp.add_argument("spec", help="EvalSuiteSpec as JSON, @file, or @- for stdin.")
    sp.set_defaults(func=_cmd_create_suite)

    sp = sub.add_parser("run-eval", help="Enqueue an eval run.")
    sp.add_argument("--suite", required=True)
    sp.add_argument("--experiment", required=True)
    sp.set_defaults(func=_cmd_run_eval)

    sp = sub.add_parser("eval-runs", help="List eval runs.")
    sp.add_argument("--suite")
    sp.add_argument("--experiment")
    sp.add_argument("--status")
    sp.add_argument("--limit", type=int, default=100)
    sp.set_defaults(func=_cmd_eval_runs)

    sp = sub.add_parser("eval-results", help="Per-sample results for an eval run.")
    sp.add_argument("run_id")
    sp.set_defaults(func=_cmd_eval_results)

    sp = sub.add_parser("compare-evals", help="Compare 2+ eval runs.")
    sp.add_argument("run_ids", nargs="+")
    sp.set_defaults(func=_cmd_compare_evals)

    sp = sub.add_parser("api", help="Generic REST call (full API parity).")
    sp.add_argument("method", help="HTTP method, e.g. GET / POST / DELETE.")
    sp.add_argument("path", help="API path, e.g. /experiments.")
    sp.add_argument("--json", help="Request body as JSON, @file, or @- for stdin.")
    sp.set_defaults(func=_cmd_api)

    return p


def _render(result: Any, args: argparse.Namespace) -> str:
    """Turn an httpx.Response into the string to print.

    ``logs`` returns plain text (optionally tail -n); everything else is the
    JSON body pretty-printed.
    """
    from .client import unwrap

    body = unwrap(result)
    if args.command == "logs" and isinstance(body, str):
        n = getattr(args, "n", 0)
        lines = body.splitlines()
        return "\n".join(lines[-n:] if n > 0 else lines)
    if isinstance(body, str):
        return body
    return json.dumps(body, indent=2, ensure_ascii=False)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    command = getattr(args, "command", None)
    if command is None or command == "serve":
        serve()
        return

    func: Callable[[httpx.Client, argparse.Namespace], Any] = args.func
    try:
        with build_client() as client:
            result = func(client, args)
            print(_render(result, args))
    except APIError as e:
        # Server-side rejection.
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    except (MissingAPIKey, ValueError, FileNotFoundError) as e:
        # Local config / input error.
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
