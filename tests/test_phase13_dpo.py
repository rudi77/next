"""Tests for Phase 13: DPO/RLHF train_kind support."""


from trainpipe.api.schemas import ExperimentSpec
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
    for kind in ("kto", "ppo", "grpo"):
        spec = ExperimentSpec(model="m", dataset=["/x"], train_kind=kind)
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
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ExperimentSpec(model="m", dataset=["/x"], train_kind="orpo")  # type: ignore[arg-type]
