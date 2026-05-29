"""``trainpipe-forget`` CLI — scan registered datasets and impacted models.

Run via::

    trainpipe-forget jane@example.com
    trainpipe-forget --regex 'AT[0-9]{18}' --case-sensitive
    trainpipe-forget --output report.json jane@example.com

The output JSON has the same shape as :meth:`ForgetReport.to_dict` and
is suitable for piping into ``jq`` or attaching to a compliance ticket.
The CLI does NOT redact anything — it only reports. Follow up with
``POST /datasets/{id}/redact`` (Phase 15) or a manual edit before
retraining the listed models.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from ..core.db import Database
from ..settings import settings
from .forget import scan_datasets_for_term


def _print_human(report) -> None:
    """Compact human-readable summary for terminal use."""
    print(f"\nForget scan — term: {report.term!r}")
    print(
        f"  regex={report.is_regex}  case_sensitive={report.case_sensitive}  "
        f"scanned={report.scanned_datasets}  "
        f"skipped={len(report.skipped_datasets)}"
    )
    print(f"\nDataset hits ({len(report.hits)}):")
    if not report.hits:
        print("  (none)")
    for h in report.hits:
        print(
            f"  • ds:{h.dataset_id[:10]}…  {h.dataset_name!r}  "
            f"{h.hit_count} hit(s) on lines {h.sample_line_numbers}"
        )
    print(f"\nImpacted models ({len(report.impacted_models)}):")
    if not report.impacted_models:
        print("  (no registered models trained on hit datasets)")
    for m in report.impacted_models:
        via = ", ".join(d[:8] + "…" for d in m.via_dataset_ids[:3])
        suffix = "…" if len(m.via_dataset_ids) > 3 else ""
        print(
            f"  • {m.name} v{m.version}  (id={m.model_id[:10]}…)  "
            f"via [{via}{suffix}]"
        )
    if report.skipped_datasets:
        print("\nSkipped datasets:")
        for s in report.skipped_datasets:
            print(f"  • {s}")
    print()


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="trainpipe-forget",
        description=(
            "Scan registered datasets for a PII term and list impacted "
            "models. Reports only — does not redact."
        ),
    )
    p.add_argument("term", help="Substring or regex to look for in datasets")
    p.add_argument(
        "--regex",
        action="store_true",
        help="Treat ``term`` as a Python re.search pattern",
    )
    p.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Default is case-insensitive matching",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Write JSON report to this path (default: print human summary)",
    )
    p.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Override the SQLite path (default: settings.sqlite_path)",
    )
    return p


async def _run_async(args: argparse.Namespace) -> int:
    db_path = args.db or settings.sqlite_path
    if not Path(db_path).is_file():
        print(
            f"error: database not found at {db_path} — "
            "is the server initialized?",
            file=sys.stderr,
        )
        return 2
    db = Database(db_path)
    async with db.connect() as conn:
        report = await scan_datasets_for_term(
            conn,
            args.term,
            is_regex=args.regex,
            case_sensitive=args.case_sensitive,
        )
    if args.output:
        args.output.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        _print_human(report)
    # Non-zero exit when hits found — handy for scripts wanting to fail
    # a compliance check.
    return 1 if report.hits else 0


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()
    try:
        rc = asyncio.run(_run_async(args))
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
    sys.exit(rc)


if __name__ == "__main__":
    main()
