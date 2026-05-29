import pytest

from trainpipe.evals.metrics import (
    UnknownMetricKind,
    get_metric_class,
    list_metric_kinds,
)
from trainpipe.evals.metrics.exact_match import ExactMatchMetric


def test_exact_match_is_discovered_by_scan():
    kinds = list_metric_kinds()
    assert "exact_match" in kinds


def test_get_metric_class_returns_exact_match():
    cls = get_metric_class("exact_match")
    assert cls is ExactMatchMetric


def test_unknown_kind_raises():
    with pytest.raises(UnknownMetricKind):
        get_metric_class("totally-made-up")


def test_exact_match_perfect():
    m = ExactMatchMetric({"gold_field": "gold"})
    assert m.score("Paris", {"gold": "Paris"}) == 1.0


def test_exact_match_normalize_default_true():
    m = ExactMatchMetric()
    assert m.score("  paris  ", {"gold": "PARIS"}) == 1.0


def test_exact_match_normalize_off():
    m = ExactMatchMetric({"normalize": False})
    assert m.score("Paris", {"gold": "paris"}) == 0.0


def test_exact_match_missing_gold_returns_zero():
    m = ExactMatchMetric()
    assert m.score("anything", {}) == 0.0


def test_exact_match_strip_punctuation():
    m = ExactMatchMetric({"strip_punctuation": True})
    assert m.score("Paris.", {"gold": "paris"}) == 1.0
    # without the flag, the punctuation would mismatch
    m2 = ExactMatchMetric({"strip_punctuation": False})
    assert m2.score("Paris.", {"gold": "Paris"}) == 0.0


def test_exact_match_custom_gold_field():
    m = ExactMatchMetric({"gold_field": "answer"})
    assert m.score("blue", {"answer": "Blue"}) == 1.0


def test_exact_match_invalid_gold_field_config():
    with pytest.raises(ValueError):
        ExactMatchMetric({"gold_field": ""})


def test_aggregate_mean_std_count():
    m = ExactMatchMetric()
    agg = m.aggregate([1.0, 1.0, 0.0, 1.0])
    assert agg.count == 4
    assert agg.mean == 0.75
    assert agg.std is not None
    assert agg.std > 0


def test_aggregate_empty_list():
    m = ExactMatchMetric()
    agg = m.aggregate([])
    assert agg.count == 0
    assert agg.mean == 0.0


def test_aggregate_single_score():
    m = ExactMatchMetric()
    agg = m.aggregate([0.8])
    assert agg.count == 1
    assert agg.mean == 0.8
    assert agg.std == 0.0


def test_register_rejects_kindless_class():
    from trainpipe.evals.metrics.base import Metric, register

    class NoKind(Metric):
        kind = ""

        def score(self, prediction, sample):
            return 0.0

    with pytest.raises(ValueError, match="must set a non-empty 'kind'"):
        register(NoKind)


def test_register_rejects_kind_collision():
    from trainpipe.evals.metrics.base import Metric, register

    class FakeExactMatch(Metric):
        kind = "exact_match"

        def score(self, prediction, sample):
            return 0.0

    with pytest.raises(ValueError, match="already registered"):
        register(FakeExactMatch)
