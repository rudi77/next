from trainpipe.api.schemas import (
    ExperimentSpec,
    MultimodalSettings,
    SearchSpaceEntry,
    StudyConfig,
)


def _minimal_spec() -> ExperimentSpec:
    return ExperimentSpec(model="qwen/Qwen2-VL-2B-Instruct", dataset=["alpaca"])


def test_experiment_spec_defaults():
    spec = _minimal_spec()
    assert spec.sft_type == "lora"
    assert spec.gpu_count == 1
    assert spec.hyperparameters.num_train_epochs == 1
    assert spec.multimodal is None
    assert spec.tags == {}
    assert spec.val_dataset == []


def test_experiment_spec_roundtrip_json():
    spec = ExperimentSpec(
        name="run-1",
        model="qwen/Qwen2-VL-2B-Instruct",
        sft_type="lora",
        dataset=["d1", "d2"],
        val_dataset=["v1"],
        gpu_count=2,
        multimodal=MultimodalSettings(size_factor=4),
        extra_args={"foo": 1, "flag": True, "list_arg": ["a", "b"]},
        tags={"team": "rd"},
    )
    blob = spec.model_dump_json()
    restored = ExperimentSpec.model_validate_json(blob)
    assert restored == spec


def test_experiment_spec_rejects_unknown_keys():
    try:
        ExperimentSpec.model_validate({"model": "m", "dataset": ["d"], "bogus": 1})
    except Exception as e:
        assert "bogus" in str(e)
    else:
        raise AssertionError("expected ValidationError")


def test_study_config_roundtrip():
    cfg = StudyConfig(
        name="sweep-1",
        base_spec=_minimal_spec(),
        search_space={
            "hyperparameters.learning_rate": SearchSpaceEntry(
                kind="loguniform", low=1e-5, high=1e-3
            ),
            "hyperparameters.lora_rank": SearchSpaceEntry(
                kind="categorical", choices=[4, 8, 16]
            ),
        },
        target_metric="eval/loss",
        n_trials=20,
    )
    blob = cfg.model_dump_json()
    restored = StudyConfig.model_validate_json(blob)
    assert restored == cfg
    assert restored.direction == "minimize"
    assert restored.sampler == "tpe"
