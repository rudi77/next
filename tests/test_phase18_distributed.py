"""Tests for Phase 18: distributed training spec + swift_builder."""

from pathlib import Path

import pytest

from trainpipe.api.schemas import DistributedConfig, ExperimentSpec
from trainpipe.training.swift_builder import build_swift_command


def test_distributed_defaults_to_none():
    spec = ExperimentSpec(model="m", dataset=["/x"])
    assert spec.distributed is None


def test_deepspeed_zero_stage_emitted():
    spec = ExperimentSpec(
        model="m",
        dataset=["/x"],
        distributed=DistributedConfig(deepspeed_zero_stage=3),
    )
    argv, _ = build_swift_command(spec, [0, 1, 2, 3], Path("/tmp/o"))
    assert "--deepspeed_zero3" in argv


def test_deepspeed_stage_0_off_emits_nothing():
    spec = ExperimentSpec(
        model="m",
        dataset=["/x"],
        distributed=DistributedConfig(deepspeed_zero_stage=0),
    )
    argv, _ = build_swift_command(spec, [0, 1], Path("/tmp/o"))
    assert not any(a.startswith("--deepspeed_zero") for a in argv)


def test_multi_node_env_vars_set():
    spec = ExperimentSpec(
        model="m",
        dataset=["/x"],
        distributed=DistributedConfig(
            num_nodes=4,
            master_addr="10.0.0.1",
            master_port=29500,
            host_list=["h1", "h2", "h3", "h4"],
        ),
    )
    _, env = build_swift_command(spec, [0, 1], Path("/tmp/o"))
    assert env["NNODES"] == "4"
    assert env["MASTER_ADDR"] == "10.0.0.1"
    assert env["MASTER_PORT"] == "29500"
    assert env["TRAINPIPE_HOST_LIST"] == "h1,h2,h3,h4"


def test_single_node_does_not_set_multi_env():
    spec = ExperimentSpec(
        model="m",
        dataset=["/x"],
        distributed=DistributedConfig(num_nodes=1),
    )
    _, env = build_swift_command(spec, [0], Path("/tmp/o"))
    assert "NNODES" not in env
    assert "MASTER_ADDR" not in env


def test_deepspeed_stage_out_of_range_rejected():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        DistributedConfig(deepspeed_zero_stage=99)


def test_distributed_combines_with_existing_flags():
    """Distributed config should layer on top of the regular SFT command,
    not replace any of it."""
    spec = ExperimentSpec(
        model="m",
        dataset=["/x"],
        distributed=DistributedConfig(deepspeed_zero_stage=2),
    )
    argv, _ = build_swift_command(spec, [0, 1], Path("/tmp/o"))
    # All the usual flags still there.
    assert "--model" in argv
    assert "--dataset" in argv
    assert "--num_train_epochs" in argv
    assert "--deepspeed_zero2" in argv


def test_distributed_works_with_rlhf():
    spec = ExperimentSpec(
        model="m",
        dataset=["/x"],
        train_kind="dpo",
        distributed=DistributedConfig(deepspeed_zero_stage=3),
    )
    argv, _ = build_swift_command(spec, [0, 1, 2, 3], Path("/tmp/o"))
    assert argv[1] == "rlhf"
    assert "--rlhf_type" in argv
    assert "--deepspeed_zero3" in argv
