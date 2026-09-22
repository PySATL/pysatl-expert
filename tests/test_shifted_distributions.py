import numpy as np
import pytest

from pysatl_expert.core.pipeline_components import PipelineComponents
from pysatl_expert.criteria.selectors.selector import CriterionSelector
from pysatl_expert.distributions.beta import BetaDistribution
from pysatl_expert.distributions.exponential import ExponentialDistribution
from pysatl_expert.distributions.gamma import GammaDistribution
from pysatl_expert.distributions.log_normal import LogNormalDistribution
from pysatl_expert.distributions.normal import NormalDistribution
from pysatl_expert.distributions.student import StudentDistribution
from pysatl_expert.distributions.uniform import UniformDistribution
from pysatl_expert.distributions.weibull import WeibullDistribution
from pysatl_expert.models.feature_extractor import FeatureExtractor
from pysatl_expert.models.feature_vector import FeatureVector
from pysatl_expert.pipeline import DistributionPipeline


def test_exponential_fit_uses_fixed_origin():
    dist = ExponentialDistribution()
    rng = np.random.default_rng(42)
    sample = rng.exponential(scale=2.0, size=500)

    params = dist.fit(sample)
    assert "loc" in params
    assert "lambda" in params
    assert params["loc"] == 0.0

    cdf_vals = dist.cdf(sample, params)
    assert np.all(cdf_vals >= 0.0) and np.all(cdf_vals <= 1.0)


def test_weibull_fit_uses_fixed_origin():
    dist = WeibullDistribution()
    rng = np.random.default_rng(42)
    sample = rng.weibull(a=1.5, size=500) * 2.0

    params = dist.fit(sample)
    assert "loc" in params
    assert "shape" in params
    assert "scale" in params
    assert params["loc"] == 0.0

    cdf_vals = dist.cdf(sample, params)
    assert np.all(cdf_vals >= 0.0) and np.all(cdf_vals <= 1.0)


def test_pipeline_pre_validate_rejects_negative_data_for_fixed_origin_distributions():
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

    assert valid_names == []


def test_pipeline_pre_validate_strictly_bounded_beta():
    pipeline = DistributionPipeline(components=None)
    beta = BetaDistribution()

    # Valid Beta sample in (0, 1)
    valid_dists = pipeline._pre_validate(data_min=0.1, data_max=0.9, distributions=[beta])
    assert len(valid_dists) == 1

    # Invalid Beta sample (data_max > 1)
    invalid_dists = pipeline._pre_validate(data_min=0.1, data_max=1.2, distributions=[beta])
    assert len(invalid_dists) == 0


@pytest.mark.parametrize(
    "distribution",
    [
        ExponentialDistribution(),
        WeibullDistribution(),
        GammaDistribution(),
        LogNormalDistribution(),
    ],
)
def test_fixed_origin_distribution_fit_keeps_positive_observations_inside_support(
    distribution,
):
    sample = np.array([0.2, 0.5, 0.9, 1.7, 2.5, 4.0])

    params = distribution.fit(sample)
    cdf_values = distribution.cdf(sample, params)

    assert params["loc"] == 0.0
    assert params["scale"] > 0
    assert np.all(np.isfinite(cdf_values))
    assert np.all((0.0 < cdf_values) & (cdf_values <= 1.0))


def test_complete_raw_feature_vector_is_scale_invariant_for_fixed_origin_contract():
    components = PipelineComponents(
        distributions=[
            NormalDistribution(),
            ExponentialDistribution(),
            WeibullDistribution(),
            UniformDistribution(),
            StudentDistribution(),
            GammaDistribution(),
            BetaDistribution(),
            LogNormalDistribution(),
        ],
        criterion_selector=CriterionSelector(),
        strategy=None,
        feature_extractor=FeatureExtractor(),
    )
    pipeline = DistributionPipeline(components)
    data = np.random.default_rng(20260914).lognormal(0.5, 0.75, size=200) + 2.0

    base, _ = pipeline._evaluate_sample(data)
    transformed, _ = pipeline._evaluate_sample(7.0 * data)
    base_values = np.asarray(base.as_flat_list())
    transformed_values = np.asarray(transformed.as_flat_list())
    finite = np.isfinite(base_values)

    assert len(FeatureVector.CRITERIA_SCHEMA) == 146
    assert np.array_equal(finite, np.isfinite(transformed_values))
    assert np.count_nonzero(finite) >= 140
    np.testing.assert_allclose(
        transformed_values[finite],
        base_values[finite],
        rtol=1e-7,
        atol=1e-7,
    )
