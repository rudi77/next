import json

import pytest

from trainpipe.evals.metrics import get_metric_class
from trainpipe.evals.metrics.field_level_f1 import FieldLevelF1Metric
from trainpipe.evals.metrics.llm_as_judge import (
    LLMAsJudgeMetric,
    _parse_score,
    _render_prompt,
)
from trainpipe.evals.metrics.rouge_l import RougeLMetric, _lcs_length

# ---------------------------------------------------------------------------
# field_level_f1
# ---------------------------------------------------------------------------


def test_field_level_f1_discovered():
    assert get_metric_class("field_level_f1") is FieldLevelF1Metric


def test_field_level_f1_perfect():
    m = FieldLevelF1Metric()
    sample = {"gold": {"name": "Alice", "amount": 100, "currency": "USD"}}
    pred = json.dumps({"name": "Alice", "amount": 100, "currency": "USD"})
    assert m.score(pred, sample) == 1.0


def test_field_level_f1_partial_credit():
    m = FieldLevelF1Metric()
    sample = {"gold": {"a": 1, "b": 2, "c": 3}}
    pred = json.dumps({"a": 1, "b": 2})  # missing c, no spurious fields
    s = m.score(pred, sample)
    # P = 2/2 = 1.0, R = 2/3, F1 = 2*1*0.667/(1+0.667) = 0.8
    assert 0.79 < s < 0.81


def test_field_level_f1_spurious_field_penalizes():
    m = FieldLevelF1Metric()
    sample = {"gold": {"a": 1}}
    pred = json.dumps({"a": 1, "extra": "noise"})
    s = m.score(pred, sample)
    # P = 1/2 = 0.5, R = 1/1, F1 = 2*0.5*1/(0.5+1) = 0.667
    assert 0.66 < s < 0.68


def test_field_level_f1_invalid_json_returns_zero():
    m = FieldLevelF1Metric()
    sample = {"gold": {"a": 1}}
    assert m.score("definitely not json", sample) == 0.0


def test_field_level_f1_non_dict_prediction_returns_zero():
    m = FieldLevelF1Metric()
    sample = {"gold": {"a": 1}}
    assert m.score(json.dumps([1, 2, 3]), sample) == 0.0


def test_field_level_f1_missing_gold_returns_zero():
    m = FieldLevelF1Metric()
    assert m.score(json.dumps({"a": 1}), {}) == 0.0


def test_field_level_f1_nested_objects_flatten():
    m = FieldLevelF1Metric()
    sample = {"gold": {"customer": {"name": "Alice", "id": 7}, "total": 99}}
    pred = json.dumps({"customer": {"name": "alice", "id": 7}, "total": 99})
    # case-insensitive default → all three flat fields match
    assert m.score(pred, sample) == 1.0


def test_field_level_f1_ignore_keys():
    m = FieldLevelF1Metric({"ignore_keys": ["meta.timestamp"]})
    sample = {"gold": {"meta": {"timestamp": "x"}, "value": 1}}
    pred = json.dumps({"meta": {"timestamp": "DIFFERENT"}, "value": 1})
    assert m.score(pred, sample) == 1.0


def test_field_level_f1_case_sensitive():
    m = FieldLevelF1Metric({"case_insensitive": False})
    sample = {"gold": {"name": "Alice"}}
    pred = json.dumps({"name": "alice"})
    assert m.score(pred, sample) == 0.0


# ---------------------------------------------------------------------------
# rouge_l
# ---------------------------------------------------------------------------


def test_rouge_l_discovered():
    assert get_metric_class("rouge_l") is RougeLMetric


def test_lcs_length_basic():
    assert _lcs_length(["a", "b", "c"], ["a", "x", "b", "y", "c"]) == 3
    assert _lcs_length([], ["a"]) == 0
    assert _lcs_length(["a", "b"], ["c", "d"]) == 0


def test_rouge_l_identical():
    m = RougeLMetric()
    s = m.score("the quick brown fox", {"gold": "the quick brown fox"})
    assert s == 1.0


def test_rouge_l_partial_overlap():
    m = RougeLMetric()
    s = m.score(
        "the quick brown fox jumped",
        {"gold": "the quick brown fox sat down"},
    )
    # LCS = 4 (the quick brown fox), len(pred)=5, len(gold)=6
    # P = 4/5 = 0.8, R = 4/6 = 0.667, F1 = 0.727
    assert 0.72 < s < 0.74


def test_rouge_l_no_overlap():
    m = RougeLMetric()
    s = m.score("alpha beta", {"gold": "gamma delta"})
    assert s == 0.0


def test_rouge_l_case_insensitive_default():
    m = RougeLMetric()
    s = m.score("THE FOX", {"gold": "the fox"})
    assert s == 1.0


def test_rouge_l_invalid_beta():
    with pytest.raises(ValueError):
        RougeLMetric({"beta": 0})


def test_rouge_l_missing_gold():
    m = RougeLMetric()
    assert m.score("anything", {}) == 0.0


# ---------------------------------------------------------------------------
# llm_as_judge
# ---------------------------------------------------------------------------


def test_llm_as_judge_discovered():
    assert get_metric_class("llm_as_judge") is LLMAsJudgeMetric


def _rubric(min_=1, max_=5):
    return {
        "criteria": "Score how well the candidate matches the reference.",
        "scale": {"min": min_, "max": max_},
    }


def test_llm_as_judge_requires_model_in_prod():
    with pytest.raises(ValueError, match="model is required"):
        LLMAsJudgeMetric({"rubric": _rubric()})


def test_llm_as_judge_requires_rubric():
    with pytest.raises(ValueError, match="rubric must be a dict"):
        LLMAsJudgeMetric({"model": "x"})


def test_llm_as_judge_rejects_bad_scale():
    with pytest.raises(ValueError, match="scale must have"):
        LLMAsJudgeMetric(
            {"model": "x", "rubric": {"criteria": "c", "scale": {"min": 5, "max": 1}}}
        )


def test_llm_as_judge_rejects_bad_provider():
    with pytest.raises(ValueError, match="provider must be"):
        LLMAsJudgeMetric(
            {"provider": "mistral", "model": "x", "rubric": _rubric()},
        )


def test_llm_as_judge_score_normalizes_to_unit_interval():
    def judge(_prompt: str) -> str:
        return '{"score": 4}'

    m = LLMAsJudgeMetric(
        {"rubric": _rubric(1, 5)}, judge_callable=judge,
    )
    s = m.score("any prediction", {"gold": "any reference"})
    # (4 - 1) / (5 - 1) = 0.75
    assert s == 0.75


def test_llm_as_judge_clamps_out_of_range():
    def judge(_prompt: str) -> str:
        return '{"score": 99}'

    m = LLMAsJudgeMetric(
        {"rubric": _rubric(1, 5)}, judge_callable=judge,
    )
    assert m.score("p", {"gold": "g"}) == 1.0


def test_llm_as_judge_handles_unparseable_reply():
    def judge(_prompt: str) -> str:
        return "I refuse to comply"

    m = LLMAsJudgeMetric(
        {"rubric": _rubric(), "max_retries": 0},
        judge_callable=judge,
    )
    # No JSON object → ValueError caught → returns 0.0
    assert m.score("p", {"gold": "g"}) == 0.0


def test_llm_as_judge_retries():
    calls = {"n": 0}

    def flaky(_prompt: str) -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient API failure")
        return '{"score": 3}'

    m = LLMAsJudgeMetric(
        {"rubric": _rubric(1, 5), "max_retries": 3}, judge_callable=flaky,
    )
    s = m.score("p", {"gold": "g"})
    assert calls["n"] == 3
    assert s == 0.5  # (3-1)/(5-1)


def test_llm_as_judge_prompt_includes_reference_and_candidate():
    captured: dict[str, str] = {}

    def capture(prompt: str) -> str:
        captured["prompt"] = prompt
        return '{"score": 3}'

    m = LLMAsJudgeMetric(
        {"rubric": _rubric()}, judge_callable=capture,
    )
    m.score("THE CANDIDATE", {"gold": "THE REFERENCE"})
    assert "THE CANDIDATE" in captured["prompt"]
    assert "THE REFERENCE" in captured["prompt"]


def test_parse_score_handles_leading_text():
    raw = 'Here is my answer:\n{"score": 4, "comment": "ok"}'
    assert _parse_score(raw, "score") == 4.0


def test_parse_score_missing_field_raises():
    with pytest.raises(ValueError, match="missing field"):
        _parse_score('{"other": 1}', "score")


def test_render_prompt_includes_examples_when_provided():
    rubric = {
        "criteria": "x",
        "scale": {"min": 1, "max": 3},
        "examples": [{"prediction": "p1", "gold": "g1", "score": 2}],
    }
    prompt = _render_prompt(rubric, "candidate", {"gold": "ref"}, "gold", "score")
    assert "Examples:" in prompt
    assert "'p1'" in prompt
