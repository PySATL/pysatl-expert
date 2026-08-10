import pytest

from pysatl_expert.models.feature_vector import FeatureVector


@pytest.fixture
def mock_data():
    sample_stats = {
        "min": 0.0,
        "max": 10.0,
        "sample_size": 100,
        "skew": 0.5,
        "kurtosis": 3.0,
        "coef_of_variation": 0.2,
        "relative_iqr": 1.1,
        "entropy": 2.5,
        "extra_key": 999,
    }
    candidates_scores = {
        "Normal": {"shapiro_wilk": 0.95, "ks_test": 0.01},
        "Exponential": {"gini_index": 0.48},
    }
    return sample_stats, candidates_scores


def test_feature_vector_init_filtering(mock_data):
    stats, scores = mock_data
    fv = FeatureVector(stats, scores)

    assert "extra_key" not in fv.sample_stats
    assert len(fv.sample_stats) == len(FeatureVector.STAT_KEYS)
    assert "normal" in fv.candidates_scores
    assert "exponential" in fv.candidates_scores


def test_feature_vector_as_flat_list_length(mock_data):
    stats, scores = mock_data
    fv = FeatureVector(stats, scores)
    flat = fv.as_flat_list()

    num_stats = len(FeatureVector.STAT_KEYS)
    num_criteria = len(FeatureVector.CRITERIA_SCHEMA)

    expected_length = num_stats + num_criteria
    assert len(flat) == expected_length


def test_feature_vector_as_flat_list_order(mock_data):
    stats, scores = mock_data
    fv = FeatureVector(stats, scores)
    flat = fv.as_flat_list()

    assert flat[0] == 0.0  # min
    assert flat[1] == 10.0  # max
    assert flat[2] == 100.0  # sample_size


def test_feature_vector_as_flat_list_missing_values():
    fv = FeatureVector({}, {"Normal": {}})
    flat = fv.as_flat_list()

    assert all(val == -1.0 for val in flat)


def test_feature_vector_as_dict(mock_data):
    stats, scores = mock_data
    fv = FeatureVector(stats, scores)
    d = fv.as_dict()

    assert "stats" in d
    assert "scores" in d
    assert d["stats"]["sample_size"] == 100
    assert d["scores"]["normal"]["shapiro_wilk"] == 0.95
