"""Seed demo data for trainpipe (paths derived from settings.datasets_dir).

Idempotent-ish: re-running adds duplicates. Run once on a fresh DB.
"""
import asyncio
import json

from trainpipe.api.schemas import (
    EvalRunStatus, ExperimentSpec, InferenceParams,
    MetricAggregate, MetricConfig,
)
from trainpipe.core import repository
from trainpipe.core.db import Database
from trainpipe.settings import settings


async def main():
    db = Database(settings.sqlite_path)
    await db.init()

    settings.datasets_dir.mkdir(parents=True, exist_ok=True)
    ds_file = settings.datasets_dir / "demo-capitals.jsonl"
    ds_file.write_text(
        "\n".join([
            json.dumps({"prompt": "capital of France?", "gold": "Paris"}),
            json.dumps({"prompt": "capital of Germany?", "gold": "Berlin"}),
            json.dumps({"prompt": "capital of Italy?", "gold": "Rome"}),
            json.dumps({"prompt": "capital of Spain?", "gold": "Madrid"}),
            json.dumps({"prompt": "capital of Austria?", "gold": "Vienna"}),
        ]) + "\n",
        encoding="utf-8",
    )

    async with db.connect() as conn:
        await repository.create_dataset(
            conn, name="demo-capitals", path=str(ds_file), fmt="jsonl",
            size_bytes=ds_file.stat().st_size, sha256="demo-sha-1",
            line_count=5, description="Demo: 5 capital-city Q&A samples",
        )
        exp_a = await repository.create_experiment(
            conn, ExperimentSpec(
                name="qwen-baseline", model="Qwen/Qwen2.5-0.5B-Instruct",
                dataset=["demo-capitals"], auto_eval=[],
            ),
        )
        exp_b = await repository.create_experiment(
            conn, ExperimentSpec(
                name="qwen-finetuned", model="Qwen/Qwen2.5-0.5B-Instruct",
                dataset=["demo-capitals"], auto_eval=[],
            ),
        )
        # Fake MLflow linkage so the UI shows the per-experiment "↗" link.
        # Clicking will 404 unless a real MLflow run with this id exists,
        # but the demo just wants to prove the link appears + routes.
        await conn.execute(
            "UPDATE experiments SET status='completed', finished_at=datetime('now'), "
            "mlflow_run_id=?, mlflow_experiment_id=? WHERE id = ?",
            ("demo-run-a-0000000000000000000000000000", "0", exp_a),
        )
        await conn.execute(
            "UPDATE experiments SET status='completed', finished_at=datetime('now'), "
            "mlflow_run_id=?, mlflow_experiment_id=? WHERE id = ?",
            ("demo-run-b-0000000000000000000000000000", "0", exp_b),
        )
        await conn.commit()
        suite_id = await repository.create_eval_suite(
            conn, name="capitals-em",
            description="Exact-match + ROUGE-L over 5 capital-city samples",
            dataset_path=str(ds_file),
            metrics=[MetricConfig(kind="exact_match"), MetricConfig(kind="rouge_l")],
            inference_params=InferenceParams(max_new_tokens=32),
        )

        async def fake_run(exp_id, name, scores):
            rid = await repository.create_eval_run(
                conn, suite_id=suite_id, experiment_id=exp_id,
                model_ref=name, triggered_by="manual",
            )
            await repository.claim_eval_run(conn, rid)
            golds = ["Paris", "Berlin", "Rome", "Madrid", "Vienna"]
            for idx, pred, em in scores:
                await repository.add_eval_result(
                    conn, run_id=rid, sample_index=idx,
                    input={"prompt": f"capital of country {idx}?"},
                    prediction=pred, gold={"gold": golds[idx]},
                    scores={"exact_match": em, "rouge_l": em},
                )
            mean = sum(s for _, _, s in scores) / len(scores)
            await repository.finalize_eval_run(
                conn, rid, status=EvalRunStatus.COMPLETED,
                aggregate={
                    "exact_match": MetricAggregate(mean=mean, std=0.0, count=len(scores)),
                    "rouge_l": MetricAggregate(mean=mean, std=0.0, count=len(scores)),
                },
                sample_count=len(scores),
            )

        await fake_run(exp_a, "qwen-baseline", [
            (0, "Paris", 1.0), (1, "Munich", 0.0), (2, "Rome", 1.0),
            (3, "Barcelona", 0.0), (4, "Vienna", 1.0),
        ])
        await fake_run(exp_b, "qwen-finetuned", [
            (0, "Paris", 1.0), (1, "Berlin", 1.0), (2, "Rome", 1.0),
            (3, "Madrid", 1.0), (4, "Vienna", 1.0),
        ])
    print(f"Seeded into {settings.sqlite_path}")


asyncio.run(main())
