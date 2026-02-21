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
    assert stats["coef_of_variation"] > 0
    assert stats["relative_iqr"] > 0
    assert isinstance(stats["entropy"], float)


def test_calculate_sample_stats_zero_mean():
    extractor = FeatureExtractor()
    data = np.array([-1.0, 0.0, 1.0])
    stats = extractor.calculate_sample_stats(data)

    assert np.mean(data) == 0.0
    assert stats["coef_of_variation"] == 0.0


def test_calculate_sample_stats_zero_median():
    extractor = FeatureExtractor()
    data = np.array([-5.0, 0.0, 5.0])
    stats = extractor.calculate_sample_stats(data)

    q50 = np.percentile(data, 50)
    assert q50 == 0.0
    assert stats["relative_iqr"] == 0.0


def test_calculate_sample_stats_types():
    extractor = FeatureExtractor()
    data = np.random.normal(0, 1, 100)
    stats = extractor.calculate_sample_stats(data)

    assert isinstance(stats["sample_size"], int)
    assert isinstance(stats["min"], float)
    assert isinstance(stats["max"], float)
    assert isinstance(stats["coef_of_variation"], float)


def test_calculate_sample_stats_constant_data():
    extractor = FeatureExtractor()
    data = np.array([1.0, 1.0, 1.0, 1.0])

    with pytest.warns(RuntimeWarning, match="Precision loss occurred"):
        stats = extractor.calculate_sample_stats(data)

    assert stats["min"] == 1.0
