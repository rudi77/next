"""Tests for Phase 21: tokenizer extension."""

from pathlib import Path

import pytest

from trainpipe.api.schemas import ExperimentSpec
from trainpipe.training.swift_builder import build_swift_command


def test_extra_tokens_defaults_to_empty():
    spec = ExperimentSpec(model="m", dataset=["/x"])
    assert spec.extra_tokens == []


def test_extra_tokens_emitted_one_per_flag():
    spec = ExperimentSpec(
        model="m",
        dataset=["/x"],
        extra_tokens=["[INV_HEAD]", "[INV_FOOT]", "[TAX_ID]"],
    )
    argv, _ = build_swift_command(spec, [0], Path("/tmp/o"))
    # Each token must appear as its own --special_tokens entry.
    count = sum(1 for a in argv if a == "--special_tokens")
    assert count == 3
    assert "[INV_HEAD]" in argv
    assert "[INV_FOOT]" in argv
    assert "[TAX_ID]" in argv


def test_extra_tokens_preserves_order():
    spec = ExperimentSpec(
        model="m",
        dataset=["/x"],
        extra_tokens=["[A]", "[B]", "[C]"],
    )
    argv, _ = build_swift_command(spec, [0], Path("/tmp/o"))
    positions = [i for i, a in enumerate(argv) if a == "--special_tokens"]
    assert argv[positions[0] + 1] == "[A]"
    assert argv[positions[1] + 1] == "[B]"
    assert argv[positions[2] + 1] == "[C]"


def test_no_extra_tokens_emits_no_flag():
    spec = ExperimentSpec(model="m", dataset=["/x"])
    argv, _ = build_swift_command(spec, [0], Path("/tmp/o"))
    assert "--special_tokens" not in argv


def test_extra_tokens_works_with_dpo():
    spec = ExperimentSpec(
        model="m",
        dataset=["/x"],
        train_kind="dpo",
        extra_tokens=["[PREF]"],
    )
    argv, _ = build_swift_command(spec, [0], Path("/tmp/o"))
    assert argv[1] == "rlhf"
    assert "[PREF]" in argv


def test_extra_tokens_max_length_enforced():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ExperimentSpec(
            model="m",
            dataset=["/x"],
            extra_tokens=[f"[T{i}]" for i in range(10001)],
        )


def test_extra_tokens_combine_with_distributed():
    from trainpipe.api.schemas import DistributedConfig

    spec = ExperimentSpec(
        model="m",
        dataset=["/x"],
        extra_tokens=["[X]"],
        distributed=DistributedConfig(deepspeed_zero_stage=2),
    )
    argv, _ = build_swift_command(spec, [0, 1], Path("/tmp/o"))
    assert "--special_tokens" in argv
    assert "--deepspeed_zero2" in argv
