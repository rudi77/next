from trainpipe.api.schemas import ExperimentSpec
from trainpipe.training.dataset_paths import (
    looks_like_local_path,
    missing_for_specs,
)


def _spec(dataset, val_dataset=None) -> ExperimentSpec:
    return ExperimentSpec(
        model="m",
        dataset=dataset,
        val_dataset=val_dataset or [],
    )


def test_looks_local_unix_absolute():
    assert looks_like_local_path("/data/train.jsonl") is True


def test_looks_local_relative():
    assert looks_like_local_path("./train.jsonl") is True
    assert looks_like_local_path("../data/train.jsonl") is True


def test_looks_local_home():
    assert looks_like_local_path("~/datasets/foo.jsonl") is True


def test_looks_local_windows_absolute():
    assert looks_like_local_path("C:\\data\\train.jsonl") is True
    assert looks_like_local_path("D:/data/train.jsonl") is True


def test_looks_local_by_extension_only():
    assert looks_like_local_path("train.jsonl") is True
    assert looks_like_local_path("data.parquet") is True
    assert looks_like_local_path("set.csv") is True


def test_hf_or_registry_name_is_not_local():
    assert looks_like_local_path("AI-ModelScope/alpaca-gpt4-data-en") is False
    assert looks_like_local_path("meta-llama/Llama-3.1-8B") is False
    assert looks_like_local_path("tatsu-lab/alpaca") is False


def test_subsample_suffix_is_stripped():
    assert looks_like_local_path("/data/train.jsonl#500") is True
    assert looks_like_local_path("AI-ModelScope/alpaca#1000") is False
    assert looks_like_local_path("train.jsonl#100") is True


def test_missing_for_specs_existing_file_ok(tmp_path):
    f = tmp_path / "train.jsonl"
    f.write_text("{}\n", encoding="utf-8")
    assert missing_for_specs([_spec([str(f)])]) == []


def test_missing_for_specs_missing_file_reported(tmp_path):
    missing = tmp_path / "nope.jsonl"
    bad = missing_for_specs([_spec([str(missing)])])
    assert len(bad) == 1
    assert bad[0].field == "dataset"
    assert bad[0].path == str(missing)
    assert bad[0].spec_index == 0


def test_missing_for_specs_both_fields(tmp_path):
    spec = _spec(
        dataset=[str(tmp_path / "bad-train.jsonl")],
        val_dataset=[str(tmp_path / "bad-val.jsonl")],
    )
    bad = missing_for_specs([spec])
    assert {(m.field, m.path) for m in bad} == {
        ("dataset", str(tmp_path / "bad-train.jsonl")),
        ("val_dataset", str(tmp_path / "bad-val.jsonl")),
    }


def test_missing_for_specs_mixed_remote_and_local(tmp_path):
    spec = _spec(["AI-ModelScope/alpaca", str(tmp_path / "missing.jsonl")])
    bad = missing_for_specs([spec])
    assert len(bad) == 1
    assert bad[0].path == str(tmp_path / "missing.jsonl")


def test_missing_for_specs_subsample_strip(tmp_path):
    f = tmp_path / "ok.jsonl"
    f.write_text("{}\n", encoding="utf-8")
    assert missing_for_specs([_spec([f"{f}#100"])]) == []


def test_missing_for_specs_batch_indexing(tmp_path):
    good = tmp_path / "good.jsonl"
    good.write_text("{}\n", encoding="utf-8")
    bad = tmp_path / "bad.jsonl"
    specs = [_spec([str(good)]), _spec([str(bad)])]
    out = missing_for_specs(specs)
    assert len(out) == 1
    assert out[0].spec_index == 1


def test_missing_for_specs_directory_existing(tmp_path):
    d = tmp_path / "dataset_dir"
    d.mkdir()
    spec = _spec([str(d)])
    assert missing_for_specs([spec]) == []
