from pathlib import Path

from trainpipe.api.schemas import (
    ExperimentSpec,
    MultimodalSettings,
    TrainingHyperparameters,
)
from trainpipe.training.swift_builder import build_swift_command


def _argv_pair(argv: list[str], flag: str) -> str | None:
    """Return the value following ``flag`` in argv, or None if missing."""
    try:
        i = argv.index(flag)
    except ValueError:
        return None
    return argv[i + 1] if i + 1 < len(argv) else None


def _all_values(argv: list[str], flag: str) -> list[str]:
    return [argv[i + 1] for i, a in enumerate(argv) if a == flag and i + 1 < len(argv)]


def test_argv_contains_core_flags():
    spec = ExperimentSpec(
        model="qwen/Qwen2-VL-2B-Instruct",
        model_type="qwen2-vl",
        dataset=["d1", "d2"],
        val_dataset=["v1"],
        sft_type="lora",
        hyperparameters=TrainingHyperparameters(
            num_train_epochs=3,
            learning_rate=2e-4,
            max_length=2048,
        ),
    )
    argv, env = build_swift_command(spec, gpu_ids=[0, 1], output_dir=Path("/out"))
    assert argv[0].endswith("swift") or argv[0] == "swift"
    assert argv[1] == "sft"
    assert _argv_pair(argv, "--model") == "qwen/Qwen2-VL-2B-Instruct"
    assert _argv_pair(argv, "--model_type") == "qwen2-vl"
    assert _argv_pair(argv, "--tuner_type") == "lora"
    assert _all_values(argv, "--dataset") == ["d1", "d2"]
    assert _all_values(argv, "--val_dataset") == ["v1"]
    assert _argv_pair(argv, "--num_train_epochs") == "3"
    assert float(_argv_pair(argv, "--learning_rate")) == 2e-4
    assert _argv_pair(argv, "--max_length") == "2048"
    assert _argv_pair(argv, "--output_dir") == str(Path("/out"))
    assert _argv_pair(argv, "--report_to") == "mlflow"


def test_env_contains_cuda_devices_and_nproc():
    spec = ExperimentSpec(model="m", dataset=["d"])
    _, env = build_swift_command(spec, gpu_ids=[2, 3], output_dir=Path("/o"))
    assert env["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert env["NPROC_PER_NODE"] == "2"
    assert "SIZE_FACTOR" not in env


def test_multimodal_settings_emit_env_vars():
    spec = ExperimentSpec(
        model="m",
        dataset=["d"],
        multimodal=MultimodalSettings(size_factor=4, max_pixels=300_000),
    )
    _, env = build_swift_command(spec, gpu_ids=[0], output_dir=Path("/o"))
    assert env["SIZE_FACTOR"] == "4"
    assert env["MAX_PIXELS"] == "300000"


def test_full_finetune_omits_lora_flags():
    spec = ExperimentSpec(model="m", dataset=["d"], sft_type="full")
    argv, _ = build_swift_command(spec, gpu_ids=[0], output_dir=Path("/o"))
    assert "--lora_rank" not in argv
    assert "--lora_alpha" not in argv
    assert "--target_modules" not in argv


def test_target_modules_emitted_per_value():
    spec = ExperimentSpec(model="m", dataset=["d"], sft_type="lora")
    spec.hyperparameters.lora_target_modules = ["q_proj", "k_proj", "v_proj"]
    argv, _ = build_swift_command(spec, gpu_ids=[0], output_dir=Path("/o"))
    assert _all_values(argv, "--target_modules") == ["q_proj", "k_proj", "v_proj"]


def test_extra_args_dispatched_correctly():
    spec = ExperimentSpec(
        model="m",
        dataset=["d"],
        extra_args={
            "custom_flag": True,
            "ignore_flag": False,
            "numeric": 42,
            "multi": ["x", "y"],
        },
    )
    argv, _ = build_swift_command(spec, gpu_ids=[0], output_dir=Path("/o"))
    assert "--custom_flag" in argv
    assert "--ignore_flag" not in argv
    assert _argv_pair(argv, "--numeric") == "42"
    assert _all_values(argv, "--multi") == ["x", "y"]


def test_raises_on_empty_gpu_ids():
    spec = ExperimentSpec(model="m", dataset=["d"])
    try:
        build_swift_command(spec, gpu_ids=[], output_dir=Path("/o"))
    except ValueError:
        return
    raise AssertionError("expected ValueError")
