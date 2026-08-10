import numpy as np

from pysatl_expert.distributions.beta import BetaDistribution
from pysatl_expert.distributions.exponential import ExponentialDistribution
from pysatl_expert.distributions.gamma import GammaDistribution
from pysatl_expert.distributions.log_normal import LogNormalDistribution
from pysatl_expert.distributions.weibull import WeibullDistribution
from pysatl_expert.pipeline import DistributionPipeline


def test_exponential_shifted_sample():
    dist = ExponentialDistribution()
    # Generate shifted exponential sample: exp(scale=2.0) - 5.0
    rng = np.random.default_rng(42)
    sample = rng.exponential(scale=2.0, size=500) - 5.0

    params = dist.fit(sample)
    assert "loc" in params
    assert "lambda" in params
    assert np.isclose(params["loc"], np.min(sample), atol=1e-3)

    cdf_vals = dist.cdf(sample, params)
    assert np.all(cdf_vals >= 0.0) and np.all(cdf_vals <= 1.0)


def test_weibull_shifted_sample():
    dist = WeibullDistribution()
    rng = np.random.default_rng(42)
    sample = rng.weibull(a=1.5, size=500) * 2.0 - 10.0

    params = dist.fit(sample)
    assert "loc" in params
    assert "shape" in params
    assert "scale" in params

    cdf_vals = dist.cdf(sample, params)
    assert np.all(cdf_vals >= 0.0) and np.all(cdf_vals <= 1.0)


def test_pipeline_pre_validate_preserves_shifted_distributions():
    pipeline = DistributionPipeline(components=None)
    distributions = [
        ExponentialDistribution(),
        WeibullDistribution(),
        GammaDistribution(),
        LogNormalDistribution(),
        BetaDistribution(),
    ]

    # Sample with negative values (data_min = -3.5, data_max = 10.2)
    valid_dists = pipeline._pre_validate(data_min=-3.5, data_max=10.2, distributions=distributions)
    valid_names = [d.name for d in valid_dists]

    assert "Exponential" in valid_names
    assert "Weibull" in valid_names
    assert "Gamma" in valid_names
    assert "LogNormal" in valid_names
    # Beta distribution is strictly bounded on (0, 1) and must be excluded
    assert "Beta" not in valid_names


def test_pipeline_pre_validate_strictly_bounded_beta():
    pipeline = DistributionPipeline(components=None)
    beta = BetaDistribution()

    # Valid Beta sample in (0, 1)
    valid_dists = pipeline._pre_validate(data_min=0.1, data_max=0.9, distributions=[beta])
    assert len(valid_dists) == 1

    # Invalid Beta sample (data_max > 1)
    invalid_dists = pipeline._pre_validate(data_min=0.1, data_max=1.2, distributions=[beta])
    assert len(invalid_dists) == 0
