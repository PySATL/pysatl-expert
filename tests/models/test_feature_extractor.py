import numpy as np
import pytest

from pysatl_expert.models.feature_extractor import FeatureExtractor


def test_calculate_sample_stats_standard():
    extractor = FeatureExtractor()
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    stats = extractor.calculate_sample_stats(data)

    assert stats["sample_size"] == 5
    assert stats["min"] == 1.0
    assert stats["max"] == 5.0
    assert isinstance(stats["skew"], float)
    assert isinstance(stats["kurtosis"], float)
    assert "coef_of_variation" not in stats
    assert stats["relative_iqr"] > 0
    assert isinstance(stats["entropy"], float)


def test_calculate_sample_stats_are_shift_and_scale_invariant():
    extractor = FeatureExtractor()
    data = np.array([-3.0, -1.0, 0.5, 2.0, 7.0])

    base = extractor.calculate_sample_stats(data)
    transformed = extractor.calculate_sample_stats(data * 7.0 + 100.0)

    for feature in ("skew", "kurtosis", "relative_iqr", "entropy"):
        assert np.isclose(base[feature], transformed[feature])


def test_relative_iqr_is_invariant_below_previous_absolute_scale_threshold():
    extractor = FeatureExtractor()
    data = np.array([-3.0, -1.0, 0.5, 2.0, 7.0])

    base = extractor.calculate_sample_stats(data)
    scaled = extractor.calculate_sample_stats(data * 1e-12)

    assert scaled["relative_iqr"] == pytest.approx(base["relative_iqr"])


def test_calculate_sample_stats_zero_median():
    extractor = FeatureExtractor()
    data = np.array([-5.0, 0.0, 5.0])
    stats = extractor.calculate_sample_stats(data)

    q50 = np.percentile(data, 50)
    assert q50 == 0.0
    assert stats["relative_iqr"] > 0.0
    assert np.isclose(stats["relative_iqr"], 5.0 / np.std(data))


def test_calculate_sample_stats_types():
    extractor = FeatureExtractor()
    data = np.random.normal(0, 1, 100)
    stats = extractor.calculate_sample_stats(data)

    assert isinstance(stats["sample_size"], int)
    assert isinstance(stats["min"], float)
    assert isinstance(stats["max"], float)
    assert isinstance(stats["relative_iqr"], float)


def test_calculate_sample_stats_constant_data():
    extractor = FeatureExtractor()
    data = np.array([1.0, 1.0, 1.0, 1.0])

    with pytest.warns(RuntimeWarning, match="Precision loss occurred"):
        stats = extractor.calculate_sample_stats(data)

    assert stats["min"] == 1.0
    assert stats["relative_iqr"] == 0.0
