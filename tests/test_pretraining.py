"""Tests for the (continued) pretraining path: train_kind='pt' → swift pt."""

import pytest
from pydantic import ValidationError

from trainpipe.api.schemas import ExperimentSpec, RLHFHyperparameters
from trainpipe.training.dataset_formats import detect_and_validate_info
from trainpipe.training.swift_builder import build_swift_command


def test_pt_is_a_valid_train_kind():
    spec = ExperimentSpec(model="m", dataset=["/x"], train_kind="pt")
    assert spec.train_kind == "pt"


def test_swift_command_pt_uses_pt_subcommand(tmp_path):
    spec = ExperimentSpec(model="m", dataset=["/x"], train_kind="pt")
    argv, _ = build_swift_command(spec, [0], tmp_path)
    assert argv[1] == "pt"
    assert "rlhf" not in argv
    assert "--rlhf_type" not in argv
    assert "sft" not in argv[1:2]


def test_pt_preserves_core_hyperparameters(tmp_path):
    """pt flows through the same model/dataset/hyperparameter flags as sft."""
    spec = ExperimentSpec(model="m", dataset=["/x"], train_kind="pt")
    argv, _ = build_swift_command(spec, [0], tmp_path)
    assert "--model" in argv
    assert "--dataset" in argv
    assert "--num_train_epochs" in argv
    assert "--learning_rate" in argv
    assert "--target_modules" in argv  # LoRA on by default for pt too


def test_pt_supports_full_finetuning(tmp_path):
    """Full-parameter continued pretraining emits no LoRA flags."""
    spec = ExperimentSpec(model="m", dataset=["/x"], train_kind="pt", sft_type="full")
    argv, _ = build_swift_command(spec, [0], tmp_path)
    assert argv[1] == "pt"
    assert "--lora_rank" not in argv
    assert "--target_modules" not in argv


def test_pt_rejects_rlhf_settings():
    """pt is not an RLHF kind, so rlhf knobs must be rejected at submit."""
    with pytest.raises(ValidationError, match="only valid for preference/RL"):
        ExperimentSpec(
            model="m",
            dataset=["/x"],
            train_kind="pt",
            rlhf=RLHFHyperparameters(beta=0.1),
        )


def test_raw_text_dataset_detected_as_pretrain(tmp_path):
    p = tmp_path / "corpus.jsonl"
    p.write_text(
        '{"text":"The quick brown fox jumps over the lazy dog."}\n'
        '{"text":"Lorem ipsum dolor sit amet."}\n',
        encoding="utf-8",
    )
    info = detect_and_validate_info(p)
    assert info.is_pretrain is True
    assert info.is_preference is False


def test_instruction_dataset_not_pretrain(tmp_path):
    p = tmp_path / "sft.jsonl"
    p.write_text(
        '{"messages":[{"role":"user","content":"hi"}]}\n'
        '{"messages":[{"role":"user","content":"yo"}]}\n',
        encoding="utf-8",
    )
    info = detect_and_validate_info(p)
    assert info.is_pretrain is False


def test_mixed_text_dataset_not_pretrain(tmp_path):
    p = tmp_path / "mixed.jsonl"
    p.write_text(
        '{"text":"a real document"}\n'
        '{"messages":[{"role":"user","content":"hi"}]}\n',
        encoding="utf-8",
    )
    info = detect_and_validate_info(p)
    assert info.is_pretrain is False


def test_empty_text_not_pretrain(tmp_path):
    p = tmp_path / "empty-text.jsonl"
    p.write_text('{"text":""}\n', encoding="utf-8")
    info = detect_and_validate_info(p)
    assert info.is_pretrain is False
