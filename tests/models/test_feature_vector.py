import math

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
    assert fv.descriptive_stats == stats
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


def test_feature_vector_uses_stable_unique_feature_names():
    assert len(FeatureVector.CRITERIA_SCHEMA) == 146
    assert len(FeatureVector.FEATURE_NAMES) == 151
    assert len(FeatureVector.FEATURE_NAMES) == len(set(FeatureVector.FEATURE_NAMES))
    assert "normal__ks" in FeatureVector.FEATURE_NAMES
    assert "beta__ks" in FeatureVector.FEATURE_NAMES
    assert "normal__dap" in FeatureVector.FEATURE_NAMES
    assert "lognormal__kl_int" in FeatureVector.FEATURE_NAMES


def test_feature_vector_excludes_unstable_criteria_from_the_raw_schema():
    unstable_criteria = {
        f"{distribution}__{criterion}"
        for distribution in ("normal", "lognormal")
        for criterion in ("glb", "sh", "zwa", "zwc")
    }

    assert not unstable_criteria.intersection(FeatureVector.FEATURE_NAMES)
    assert unstable_criteria <= FeatureVector.EXCLUDED_TRAINING_FEATURES
    assert not FeatureVector.EXCLUDED_TRAINING_FEATURES.intersection(FeatureVector.FEATURE_NAMES)
    assert FeatureVector.TRAINING_FEATURE_NAMES == FeatureVector.FEATURE_NAMES


def test_feature_vector_excludes_invalid_and_exact_duplicate_raw_features():
    excluded = {
        "beta__mode",
        "beta__lillie",
        "student__lillie",
        "uniform__lillie",
        "weibull__lillie",
        "normal__rj",
        "lognormal__rj",
        "uniform__censored_stein_u",
    }

    assert not excluded.intersection(FeatureVector.FEATURE_NAMES)
    assert excluded <= FeatureVector.EXCLUDED_TRAINING_FEATURES


def test_feature_vector_omits_location_bounds_from_model_schema():
    excluded = {"min", "max", "coef_of_variation"}
    assert not excluded.intersection(FeatureVector.STAT_KEYS)
    assert not excluded.intersection(FeatureVector.FEATURE_NAMES)


def test_feature_vector_maps_lognormal_scores_to_lognormal_schema():
    fv = FeatureVector({}, {"LogNormal": {"ks": 0.25}})
    feature_values = dict(zip(FeatureVector.FEATURE_NAMES, fv.as_flat_list(), strict=True))

    assert feature_values["lognormal__ks"] == 0.25


def test_feature_vector_as_flat_list_order(mock_data):
    stats, scores = mock_data
    fv = FeatureVector(stats, scores)
    flat = fv.as_flat_list()

    assert flat[0] == 100.0  # sample_size


def test_feature_vector_as_flat_list_missing_values():
    fv = FeatureVector({}, {"Normal": {}})
    flat = fv.as_flat_list()

    assert all(math.isnan(val) for val in flat)


def test_feature_vector_preserves_positional_missing_value():
    fv = FeatureVector({}, {})

    assert all(value == -1.0 for value in fv.as_flat_list(-1.0))


def test_feature_vector_can_follow_an_existing_model_schema():
    fv = FeatureVector(
        {"sample_size": 100, "skew": 0.25},
        {"Normal": {"ks": 0.5}},
    )

    values = fv.as_flat_list(feature_names=["normal__ks", "sample_size", "legacy__missing", "skew"])

    assert values[:2] == [0.5, 100.0]
    assert math.isnan(values[2])
    assert values[3] == 0.25


def test_feature_vector_as_dict(mock_data):
    stats, scores = mock_data
    fv = FeatureVector(stats, scores)
    d = fv.as_dict()

    assert "stats" in d
    assert "scores" in d
    assert d["stats"]["sample_size"] == 100
    assert d["scores"]["normal"]["shapiro_wilk"] == 0.95
