import numpy as np
import pytest

from pysatl_expert.distributions.exponential import ExponentialDistribution
from pysatl_expert.distributions.gamma import GammaDistribution
from pysatl_expert.distributions.log_normal import LogNormalDistribution
from pysatl_expert.distributions.normal import NormalDistribution
from pysatl_expert.distributions.student import StudentDistribution
from pysatl_expert.distributions.uniform import UniformDistribution
from pysatl_expert.distributions.weibull import WeibullDistribution


@pytest.mark.parametrize(
    "distribution",
    [
        ExponentialDistribution(),
        WeibullDistribution(),
        GammaDistribution(),
        LogNormalDistribution(),
    ],
)
def test_fixed_origin_distribution_fit_keeps_loc_at_zero(distribution):
    data = np.array([0.5, 1.0, 2.0, 4.0, 8.0])

    params = distribution.fit(data)

    assert params["loc"] == 0.0


@pytest.mark.parametrize(
    "distribution",
    [
        NormalDistribution(),
        StudentDistribution(),
        UniformDistribution(),
    ],
)
def test_location_scale_fit_is_affine_equivariant(distribution):
    rng = np.random.default_rng(20260914)
    data = rng.normal(loc=1.25, scale=2.75, size=200)
    factor = 7.0
    offset = 50.0

    base = distribution.fit(data)
    transformed = distribution.fit(factor * data + offset)

    if distribution.name == "Normal":
        assert transformed["mu"] == pytest.approx(factor * base["mu"] + offset)
        assert transformed["std"] == pytest.approx(factor * base["std"])
    elif distribution.name == "Uniform":
        assert transformed["a"] == pytest.approx(factor * base["a"] + offset, rel=1e-12, abs=1e-12)
        assert transformed["b"] == pytest.approx(factor * base["b"] + offset, rel=1e-12, abs=1e-12)
    else:
        assert transformed["loc"] == pytest.approx(factor * base["loc"] + offset)
        assert transformed["scale"] == pytest.approx(factor * base["scale"], rel=1e-7)

        shape_keys = set(base) - {"loc", "scale", "lambda"}
        for key in shape_keys:
            assert transformed[key] == pytest.approx(base[key], rel=1e-7)


@pytest.mark.parametrize(
    "distribution",
    [
        ExponentialDistribution(),
        WeibullDistribution(),
        GammaDistribution(),
        LogNormalDistribution(),
    ],
)
def test_fixed_origin_distribution_fit_is_scale_equivariant(distribution):
    data = np.random.default_rng(20260914).lognormal(0.5, 0.75, size=200)
    factor = 7.0

    base = distribution.fit(data)
    transformed = distribution.fit(factor * data)

    assert transformed["loc"] == 0.0
    assert transformed["scale"] == pytest.approx(factor * base["scale"], rel=1e-7)
    shape_keys = set(base) - {"loc", "scale", "lambda"}
    for key in shape_keys:
        assert transformed[key] == pytest.approx(base[key], rel=1e-7)
    if "lambda" in base:
        assert transformed["lambda"] == pytest.approx(base["lambda"] / factor)


@pytest.mark.parametrize(
    "distribution",
    [
        ExponentialDistribution(),
        WeibullDistribution(),
        GammaDistribution(),
        LogNormalDistribution(),
    ],
)
@pytest.mark.parametrize("missing_parameter", ["loc", "scale"])
def test_fixed_origin_distribution_requires_fitted_location_and_scale(
    distribution, missing_parameter
):
    data = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    params = distribution.fit(data)
    params.pop(missing_parameter)

    with pytest.raises(KeyError, match=missing_parameter):
        distribution.pdf(data, params)
    with pytest.raises(KeyError, match=missing_parameter):
        distribution.cdf(data, params)
    with pytest.raises(KeyError, match=missing_parameter):
        distribution.prepare_criterion_input(data, params)
