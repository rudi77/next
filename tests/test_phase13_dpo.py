"""Tests for Phase 13: DPO/RLHF train_kind support."""

import pytest
from pydantic import ValidationError

from trainpipe.api.schemas import ExperimentSpec, RLHFHyperparameters
from trainpipe.training.dataset_formats import detect_and_validate_info
from trainpipe.training.swift_builder import build_swift_command


def test_default_train_kind_is_sft():
    spec = ExperimentSpec(model="m", dataset=["/x"])
    assert spec.train_kind == "sft"


def test_swift_command_sft_uses_sft_subcommand(tmp_path):
    spec = ExperimentSpec(model="m", dataset=["/x"])
    argv, _ = build_swift_command(spec, [0], tmp_path)
    assert argv[1] == "sft"
    assert "rlhf" not in argv
    assert "--rlhf_type" not in argv


def test_swift_command_dpo_uses_rlhf_subcommand(tmp_path):
    spec = ExperimentSpec(model="m", dataset=["/x"], train_kind="dpo")
    argv, _ = build_swift_command(spec, [0], tmp_path)
    assert argv[1] == "rlhf"
    # Right after subcommand: --rlhf_type dpo
    assert argv[2] == "--rlhf_type"
    assert argv[3] == "dpo"
    # Model + dataset still flow through.
    assert "--model" in argv
    assert "--dataset" in argv


def test_swift_command_kto_grpo_ppo_emit_correct_rlhf_type(tmp_path):
    # kto needs no reward; ppo/grpo do, so give them a minimal one.
    cases = [
        ("kto", None),
        ("ppo", RLHFHyperparameters(reward_model="rm")),
        ("grpo", RLHFHyperparameters(reward_funcs=["accuracy"])),
    ]
    for kind, rlhf in cases:
        spec = ExperimentSpec(model="m", dataset=["/x"], train_kind=kind, rlhf=rlhf)
        argv, _ = build_swift_command(spec, [0], tmp_path)
        assert argv[3] == kind


def test_preference_dataset_format_detected(tmp_path):
    p = tmp_path / "dpo.jsonl"
    p.write_text(
        '{"prompt":"Translate hi","chosen":"Hallo","rejected":"Hello1"}\n'
        '{"prompt":"How are you","chosen":"Wie geht es dir","rejected":"how"}\n',
        encoding="utf-8",
    )
    info = detect_and_validate_info(p)
    assert info.is_preference is True


def test_text_dataset_is_not_preference(tmp_path):
    p = tmp_path / "sft.jsonl"
    p.write_text(
        '{"messages":[{"role":"user","content":"hi"}]}\n'
        '{"messages":[{"role":"user","content":"yo"}]}\n',
        encoding="utf-8",
    )
    info = detect_and_validate_info(p)
    assert info.is_preference is False


def test_mixed_dataset_not_marked_preference(tmp_path):
    """One DPO row + one non-DPO row should not be flagged as preference."""
    p = tmp_path / "mixed.jsonl"
    p.write_text(
        '{"prompt":"a","chosen":"b","rejected":"c"}\n'
        '{"messages":[{"role":"user","content":"hi"}]}\n',
        encoding="utf-8",
    )
    info = detect_and_validate_info(p)
    assert info.is_preference is False


def test_preference_requires_non_empty_strings(tmp_path):
    p = tmp_path / "empty-chosen.jsonl"
    p.write_text(
        '{"prompt":"a","chosen":"","rejected":"c"}\n',
        encoding="utf-8",
    )
    info = detect_and_validate_info(p)
    assert info.is_preference is False


def test_rlhf_command_preserves_hyperparameters(tmp_path):
    """The rlhf subcommand must still carry epochs, batch size, lr etc."""
    spec = ExperimentSpec(model="m", dataset=["/x"], train_kind="dpo")
    argv, _ = build_swift_command(spec, [0], tmp_path)
    assert "--num_train_epochs" in argv
    assert "--learning_rate" in argv
    assert "--target_modules" in argv  # LoRA on by default


def test_spec_validation_rejects_unknown_train_kind():
    with pytest.raises(ValidationError):
        ExperimentSpec(model="m", dataset=["/x"], train_kind="orpo")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# RL / GRPO first-class hyperparameters (Phase 13 follow-up)
# ---------------------------------------------------------------------------


def test_grpo_emits_reward_funcs_and_group_size(tmp_path):
    spec = ExperimentSpec(
        model="m",
        dataset=["/x"],
        train_kind="grpo",
        rlhf=RLHFHyperparameters(
            reward_funcs=["accuracy", "format"],
            num_generations=8,
            max_completion_length=256,
            temperature=0.9,
            beta=0.04,
        ),
    )
    argv, _ = build_swift_command(spec, [0], tmp_path)
    # reward_funcs is one flag followed by all values (argparse nargs='+').
    i = argv.index("--reward_funcs")
    assert argv[i + 1 : i + 3] == ["accuracy", "format"]
    assert "--num_generations" in argv and argv[argv.index("--num_generations") + 1] == "8"
    assert "--max_completion_length" in argv
    assert "--temperature" in argv
    assert "--beta" in argv


def test_ppo_emits_reward_model(tmp_path):
    spec = ExperimentSpec(
        model="m",
        dataset=["/x"],
        train_kind="ppo",
        rlhf=RLHFHyperparameters(reward_model="Qwen/Qwen2.5-RM"),
    )
    argv, _ = build_swift_command(spec, [0], tmp_path)
    assert argv[argv.index("--reward_model") + 1] == "Qwen/Qwen2.5-RM"


def test_ppo_without_reward_model_is_rejected():
    with pytest.raises(ValidationError, match="reward_model"):
        ExperimentSpec(model="m", dataset=["/x"], train_kind="ppo")


def test_grpo_without_any_reward_is_rejected():
    with pytest.raises(ValidationError, match=r"reward_model or rlhf\.reward_funcs"):
        ExperimentSpec(model="m", dataset=["/x"], train_kind="grpo")


def test_rlhf_settings_rejected_for_sft():
    with pytest.raises(ValidationError, match="only valid for preference/RL"):
        ExperimentSpec(
            model="m",
            dataset=["/x"],
            train_kind="sft",
            rlhf=RLHFHyperparameters(beta=0.1),
        )


def test_reward_funcs_rejected_for_non_grpo():
    with pytest.raises(ValidationError, match="reward_funcs is only valid"):
        ExperimentSpec(
            model="m",
            dataset=["/x"],
            train_kind="dpo",
            rlhf=RLHFHyperparameters(reward_funcs=["accuracy"]),
        )


def test_dpo_with_beta_emits_beta_no_reward(tmp_path):
    spec = ExperimentSpec(
        model="m",
        dataset=["/x"],
        train_kind="dpo",
        rlhf=RLHFHyperparameters(beta=0.1),
    )
    argv, _ = build_swift_command(spec, [0], tmp_path)
    assert argv[argv.index("--beta") + 1] == "0.1"
    assert "--reward_model" not in argv


def test_sft_emits_no_rlhf_flags(tmp_path):
    spec = ExperimentSpec(model="m", dataset=["/x"])
    argv, _ = build_swift_command(spec, [0], tmp_path)
    for flag in ("--beta", "--reward_model", "--reward_funcs", "--num_generations"):
        assert flag not in argv
