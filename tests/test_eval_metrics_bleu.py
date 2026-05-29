import pytest

from trainpipe.evals.metrics import get_metric_class
from trainpipe.evals.metrics.bleu import BleuMetric, _ngram_counts, sentence_bleu


def test_bleu_discovered():
    assert get_metric_class("bleu") is BleuMetric


def test_ngram_counts_basic():
    assert _ngram_counts(["a", "b", "c"], 1) == {("a",): 1, ("b",): 1, ("c",): 1}
    assert _ngram_counts(["a", "b", "c"], 2) == {("a", "b"): 1, ("b", "c"): 1}
    assert _ngram_counts(["a"], 2) == {}


def test_bleu_identical():
    m = BleuMetric()
    s = m.score("the quick brown fox jumps", {"gold": "the quick brown fox jumps"})
    assert s == pytest.approx(1.0)


def test_bleu_zero_when_no_overlap():
    m = BleuMetric()
    s = m.score("alpha beta gamma delta", {"gold": "zulu yankee xray whiskey"})
    # All n-gram orders have 0 matches; smoothing-1 still keeps it nonzero
    # but extremely small. Without smoothing it would be exactly 0.
    assert 0.0 <= s < 0.05


def test_bleu_no_smoothing_zero():
    m = BleuMetric({"smoothing": False})
    s = m.score("alpha beta gamma delta", {"gold": "zulu yankee xray whiskey"})
    assert s == 0.0


def test_bleu_partial_overlap_higher_than_no_overlap():
    m = BleuMetric()
    s_partial = m.score(
        "the quick brown fox jumps over the lazy dog",
        {"gold": "the quick brown fox sat down quietly"},
    )
    s_none = m.score(
        "abc def ghi jkl",
        {"gold": "the quick brown fox sat down quietly"},
    )
    assert s_partial > s_none


def test_bleu_brevity_penalty_short_prediction():
    m = BleuMetric()
    long_ref = "the quick brown fox jumps over the lazy dog today"
    # short but correct prediction → BP penalizes it
    short_pred = "the quick brown fox"
    long_pred = "the quick brown fox jumps over the lazy dog today"
    s_short = m.score(short_pred, {"gold": long_ref})
    s_long = m.score(long_pred, {"gold": long_ref})
    assert s_long > s_short


def test_bleu_case_insensitive_default():
    m = BleuMetric()
    s_ci = m.score("THE QUICK BROWN FOX", {"gold": "the quick brown fox"})
    assert s_ci == pytest.approx(1.0)


def test_bleu_case_sensitive_config():
    m = BleuMetric(
        {"case_insensitive": False, "max_n": 1, "smoothing": False},
    )
    # max_n=1 to avoid 0 from higher-order n-grams when no overlap;
    # smoothing off so true 0 overlap gives exactly 0
    s = m.score("HELLO WORLD", {"gold": "hello world"})
    assert s == 0.0  # no overlap at any case-sensitive token


def test_bleu_unigram_only():
    m = BleuMetric({"max_n": 1})
    s = m.score("the cat", {"gold": "the cat"})
    assert s == pytest.approx(1.0)


def test_bleu_weights_must_match_max_n():
    with pytest.raises(ValueError, match="weights length"):
        BleuMetric({"max_n": 4, "weights": [1.0]})


def test_bleu_weights_must_be_positive():
    with pytest.raises(ValueError, match="must sum to a positive"):
        BleuMetric({"max_n": 2, "weights": [0.0, 0.0]})


def test_bleu_max_n_out_of_range():
    with pytest.raises(ValueError, match="max_n must be between"):
        BleuMetric({"max_n": 0})


def test_bleu_missing_gold_returns_zero():
    m = BleuMetric()
    assert m.score("anything", {}) == 0.0


def test_bleu_empty_prediction():
    m = BleuMetric()
    assert m.score("", {"gold": "the cat"}) == 0.0


def test_sentence_bleu_helper_handles_short_prediction_with_match():
    # pred shorter than max_n shouldn't crash; orders > pred_len return 0 matches
    s = sentence_bleu(
        ["the"], ["the", "cat"], max_n=4, smoothing=True, weights=[0.25] * 4,
    )
    # smoothing keeps it positive
    assert 0.0 < s < 1.0


def test_bleu_clipping_caps_at_reference_count():
    """A prediction that repeats a word more than the reference does
    shouldn't get full credit for those duplicates (clipping)."""
    m = BleuMetric({"max_n": 1, "smoothing": False})
    # ref has "the" once; prediction has it three times. Clipped count = 1,
    # total = 3, so unigram precision = 1/3.
    s = m.score("the the the", {"gold": "the cat"})
    # plus brevity penalty: pred_len=3 > ref_len=2 → BP=1
    # so BLEU ≈ 1/3
    assert 0.30 < s < 0.36
