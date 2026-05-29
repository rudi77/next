"""Translate an ExperimentSpec into the argv + env for `swift sft`.

We never use shell=True. CUDA_VISIBLE_DEVICES and NPROC_PER_NODE are passed
via the env dict so multi-GPU launches work on Linux.
"""

from pathlib import Path

from ..api.schemas import ExperimentSpec

_LORA_FAMILY = {"lora", "qlora", "longlora", "adalora"}


def build_swift_command(
    spec: ExperimentSpec,
    gpu_ids: list[int],
    output_dir: Path,
) -> tuple[list[str], dict[str, str]]:
    """Return ``(argv, env)`` for ``asyncio.create_subprocess_exec``."""

    if not gpu_ids:
        raise ValueError("gpu_ids must contain at least one GPU index")

    argv: list[str] = ["swift", "sft"]

    argv += ["--model_id_or_path", spec.model]
    if spec.model_type:
        argv += ["--model_type", spec.model_type]
    argv += ["--sft_type", spec.sft_type]

    for ds in spec.dataset:
        argv += ["--dataset", ds]
    for vds in spec.val_dataset:
        argv += ["--val_dataset", vds]

    hp = spec.hyperparameters
    argv += ["--num_train_epochs", str(hp.num_train_epochs)]
    argv += ["--per_device_train_batch_size", str(hp.per_device_train_batch_size)]
    argv += ["--per_device_eval_batch_size", str(hp.per_device_eval_batch_size)]
    argv += ["--gradient_accumulation_steps", str(hp.gradient_accumulation_steps)]
    argv += ["--learning_rate", str(hp.learning_rate)]
    if hp.max_length is not None:
        argv += ["--max_length", str(hp.max_length)]
    argv += ["--warmup_ratio", str(hp.warmup_ratio)]
    argv += ["--weight_decay", str(hp.weight_decay)]
    argv += ["--lr_scheduler_type", hp.lr_scheduler_type]
    argv += ["--save_steps", str(hp.save_steps)]
    argv += ["--eval_steps", str(hp.eval_steps)]
    argv += ["--logging_steps", str(hp.logging_steps)]
    argv += ["--seed", str(hp.seed)]

    if spec.sft_type in _LORA_FAMILY:
        argv += ["--lora_rank", str(hp.lora_rank)]
        argv += ["--lora_alpha", str(hp.lora_alpha)]
        argv += ["--lora_dropout", str(hp.lora_dropout)]
        for tm in hp.lora_target_modules:
            argv += ["--lora_target_modules", tm]

    argv += ["--output_dir", str(output_dir)]
    argv += ["--report_to", "mlflow"]

    for k, v in spec.extra_args.items():
        flag = f"--{k}" if not k.startswith("--") else k
        if isinstance(v, bool):
            if v:
                argv.append(flag)
        elif isinstance(v, list):
            for item in v:
                argv += [flag, str(item)]
        else:
            argv += [flag, str(v)]

    env: dict[str, str] = {
        "CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in gpu_ids),
        "NPROC_PER_NODE": str(len(gpu_ids)),
    }
    if spec.multimodal is not None:
        env["SIZE_FACTOR"] = str(spec.multimodal.size_factor)
        env["MAX_PIXELS"] = str(spec.multimodal.max_pixels)

    return argv, env
