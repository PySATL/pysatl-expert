from unittest.mock import MagicMock

import numpy as np
import pytest

from pysatl_expert.criteria.calculate.generic import GenericCriterion
from pysatl_expert.criteria.catalog import CRITERIA_SPEC_BY_KEY, CRITERIA_SPECS
from pysatl_expert.distributions.beta import BetaDistribution
from pysatl_expert.distributions.exponential import ExponentialDistribution
from pysatl_expert.distributions.gamma import GammaDistribution
from pysatl_expert.distributions.log_normal import LogNormalDistribution
from pysatl_expert.distributions.normal import NormalDistribution
from pysatl_expert.distributions.student import StudentDistribution
from pysatl_expert.distributions.uniform import UniformDistribution
from pysatl_expert.distributions.weibull import WeibullDistribution


def test_generic_criterion_init_auto_name():
    mock_engine = MagicMock()
    mock_engine.code.return_value = "KS_ENGINE"

    criterion = GenericCriterion(mock_engine)
    assert criterion.name == "KS_ENGINE"
    mock_engine.code.assert_called_once()


def test_generic_criterion_init_display_name():
    mock_engine = MagicMock()
    criterion = GenericCriterion(mock_engine, display_name="manual_name")
    assert criterion.name == "manual_name"


def test_generic_criterion_calculate(mocker):
    mock_engine = MagicMock()
    mock_engine.execute_statistic.return_value = 0.123
    mock_engine.code.return_value = "MOCK"

    mock_dist = MagicMock()
    mock_dist.cdf.return_value = np.array([0.1, 0.5, 0.9])

    criterion = GenericCriterion(mock_engine)

    data = np.array([1, 2, 3])
    params = {"mu": 0, "sigma": 1}
    mock_dist.prepare_criterion_input.return_value = (data, params)

    result = criterion.calculate(data, mock_dist, params)

    assert result == 0.123

    mock_dist.cdf.assert_called_once_with(data, params)

    mock_engine.execute_statistic.assert_called_once()
    args, kwargs = mock_engine.execute_statistic.call_args
    assert np.array_equal(kwargs["rvs"], data)
    assert np.array_equal(kwargs["cdf_vals"], mock_dist.cdf.return_value)


def test_generic_criterion_delegates_input_preparation_to_distribution():
    class Statistic:
        mean = None

        @staticmethod
        def code():
            return "MOCK"

        def execute_statistic(self, rvs):
            self.received = rvs
            return 0.25

    engine = Statistic()
    distribution = MagicMock()
    distribution.prepare_criterion_input.return_value = (
        np.array([-1.0, 1.0]),
        {"mean": 0.0},
    )
    criterion = GenericCriterion(engine)
    data = np.array([2.0, 4.0])
    params = {"mu": 3.0, "std": 1.0}

    result = criterion.calculate(data, distribution, params)

    assert result == 0.25
    distribution.prepare_criterion_input.assert_called_once_with(data, params)
    assert np.array_equal(engine.received, np.array([-1.0, 1.0]))
    assert engine.mean == 0.0


@pytest.mark.parametrize(
    ("distribution_key", "criterion_key", "distribution", "base_data", "params"),
    [
        (
            "exponential",
            "ks",
            ExponentialDistribution(),
            np.array([0.2, 0.5, 0.9, 1.4, 2.1]),
            {"loc": 0.0, "scale": 2.0, "lambda": 0.5},
        ),
        (
            "weibull",
            "ks",
            WeibullDistribution(),
            np.array([0.4, 0.8, 1.3, 2.0, 3.1]),
            {"shape": 1.7, "loc": 0.0, "scale": 2.5},
        ),
        (
            "gamma",
            "ks",
            GammaDistribution(),
            np.array([0.3, 0.7, 1.1, 1.8, 2.9]),
            {"shape": 2.0, "loc": 0.0, "scale": 1.5},
        ),
        (
            "lognormal",
            "ks",
            LogNormalDistribution(),
            np.array([0.5, 0.9, 1.4, 2.2, 3.6]),
            {"s": 0.8, "loc": 0.0, "scale": 1.3},
        ),
    ],
)
def test_positive_family_criterion_is_invariant_to_location_shift(
    distribution_key, criterion_key, distribution, base_data, params
):
    statistic_class = CRITERIA_SPEC_BY_KEY[(distribution_key, criterion_key)].statistic_class
    base_criterion = GenericCriterion(statistic_class())
    shifted_criterion = GenericCriterion(statistic_class())
    shift = -7.0
    shifted_params = {**params, "loc": params["loc"] + shift}

    base_value = base_criterion.calculate(base_data, distribution, params)
    shifted_value = shifted_criterion.calculate(base_data + shift, distribution, shifted_params)

    assert shifted_value == pytest.approx(base_value)


def test_weibull_parameters_are_mapped_to_ordinary_weibull_engine():
    statistic_class = CRITERIA_SPEC_BY_KEY[("weibull", "ks")].statistic_class
    criterion = GenericCriterion(statistic_class())
    distribution = WeibullDistribution()
    data = np.array([-3.0, -2.0, 0.0, 2.0])

    criterion.calculate(
        data,
        distribution,
        {"shape": 1.7, "loc": -4.0, "scale": 2.5},
    )

    assert criterion.engine.a == 1.0
    assert criterion.engine.k == pytest.approx(1.7)


@pytest.mark.parametrize(
    ("distribution_key", "criterion_key", "distribution", "base_data", "params"),
    [
        (
            "normal",
            "ks",
            NormalDistribution(),
            np.array([-2.1, -0.4, 0.3, 1.2, 2.7]),
            {"mu": 0.5, "std": 1.7},
        ),
        (
            "normal",
            "cvm",
            NormalDistribution(),
            np.array([-2.1, -0.4, 0.3, 1.2, 2.7]),
            {"mu": 0.5, "std": 1.7},
        ),
        (
            "exponential",
            "ks",
            ExponentialDistribution(),
            np.array([-1.8, -1.0, 0.2, 1.5, 4.0]),
            {"loc": -2.0, "scale": 2.5, "lambda": 0.4},
        ),
        (
            "exponential",
            "hg1",
            ExponentialDistribution(),
            np.array([-1.8, -1.0, 0.2, 1.5, 4.0]),
            {"loc": -2.0, "scale": 2.5, "lambda": 0.4},
        ),
        (
            "exponential",
            "hg2",
            ExponentialDistribution(),
            np.array([-1.8, -1.0, 0.2, 1.5, 4.0]),
            {"loc": -2.0, "scale": 2.5, "lambda": 0.4},
        ),
        (
            "student",
            "ks",
            StudentDistribution(),
            np.array([-3.0, -0.5, 1.0, 2.2, 5.5]),
            {"df": 4.5, "loc": 1.0, "scale": 2.0},
        ),
        (
            "uniform",
            "sherman",
            UniformDistribution(),
            np.array([-1.5, -0.3, 1.1, 2.0, 3.5]),
            {"a": -2.0, "b": 4.0},
        ),
        (
            "uniform",
            "quesenberry_miller",
            UniformDistribution(),
            np.array([-1.5, -0.3, 1.1, 2.0, 3.5]),
            {"a": -2.0, "b": 4.0},
        ),
    ],
)
def test_fitted_location_scale_criterion_is_affine_invariant(
    distribution_key,
    criterion_key,
    distribution,
    base_data,
    params,
):
    statistic_class = CRITERIA_SPEC_BY_KEY[(distribution_key, criterion_key)].statistic_class
    base_criterion = GenericCriterion(statistic_class())
    transformed_criterion = GenericCriterion(statistic_class())
    factor = 3.0
    offset = 7.0

    if distribution_key == "normal":
        transformed_params = {
            "mu": factor * params["mu"] + offset,
            "std": factor * params["std"],
        }
    elif distribution_key == "exponential":
        transformed_params = {
            "loc": factor * params["loc"] + offset,
            "scale": factor * params["scale"],
            "lambda": params["lambda"] / factor,
        }
    elif distribution_key == "student":
        transformed_params = {
            "df": params["df"],
            "loc": factor * params["loc"] + offset,
            "scale": factor * params["scale"],
        }
    else:
        transformed_params = {
            "a": factor * params["a"] + offset,
            "b": factor * params["b"] + offset,
        }

    base_value = base_criterion.calculate(base_data, distribution, params)
    transformed_value = transformed_criterion.calculate(
        factor * base_data + offset,
        distribution,
        transformed_params,
    )

    assert transformed_value == pytest.approx(base_value, rel=1e-12, abs=1e-12)


def test_every_location_scale_raw_criterion_is_affine_invariant():
    rng = np.random.default_rng(123)
    sample_size = 200
    cases = {
        "normal": (
            NormalDistribution(),
            rng.normal(2.0, 3.0, sample_size),
            {"mu": 2.0, "std": 3.0},
        ),
        "exponential": (
            ExponentialDistribution(),
            -2.0 + 2.0 * rng.exponential(size=sample_size),
            {"loc": -2.0, "scale": 2.0, "lambda": 0.5},
        ),
        "weibull": (
            WeibullDistribution(),
            -1.0 + 2.5 * rng.weibull(1.7, sample_size),
            {"shape": 1.7, "loc": -1.0, "scale": 2.5},
        ),
        "gamma": (
            GammaDistribution(),
            -3.0 + 1.5 * rng.gamma(2.2, size=sample_size),
            {"shape": 2.2, "loc": -3.0, "scale": 1.5},
        ),
        "lognormal": (
            LogNormalDistribution(),
            -4.0 + 1.3 * rng.lognormal(0.0, 0.8, sample_size),
            {"s": 0.8, "loc": -4.0, "scale": 1.3},
        ),
        "student": (
            StudentDistribution(),
            1.0 + 2.0 * rng.standard_t(5.0, sample_size),
            {"df": 5.0, "loc": 1.0, "scale": 2.0},
        ),
        "uniform": (
            UniformDistribution(),
            rng.uniform(-2.0, 4.0, sample_size),
            {"a": -2.0, "b": 4.0},
        ),
    }
    factor = 3.0
    offset = 7.0
    checked = 0

    for spec in CRITERIA_SPECS:
        if spec.distribution == "beta":
            continue
        distribution, data, params = cases[spec.distribution]
        if spec.distribution == "normal":
            transformed_params = {
                "mu": factor * params["mu"] + offset,
                "std": factor * params["std"],
            }
        elif spec.distribution == "uniform":
            transformed_params = {
                "a": factor * params["a"] + offset,
                "b": factor * params["b"] + offset,
            }
        else:
            transformed_params = {
                **params,
                "loc": factor * params["loc"] + offset,
                "scale": factor * params["scale"],
            }
            if "lambda" in params:
                transformed_params["lambda"] = params["lambda"] / factor

        base_value = GenericCriterion(spec.statistic_class()).calculate(
            data, distribution, params
        )
        transformed_value = GenericCriterion(spec.statistic_class()).calculate(
            factor * data + offset,
            distribution,
            transformed_params,
        )

        assert transformed_value == pytest.approx(
            base_value,
            rel=1e-10,
            abs=1e-10,
        ), spec.feature_name
        checked += 1

    beta_feature_count = sum(spec.distribution == "beta" for spec in CRITERIA_SPECS)
    assert checked == len(CRITERIA_SPECS) - beta_feature_count == 137


def test_every_fixed_support_beta_raw_criterion_is_finite():
    rng = np.random.default_rng(123)
    data = rng.beta(2.0, 5.0, size=200)
    distribution = BetaDistribution()
    params = distribution.fit(data)
    checked = 0

    for spec in CRITERIA_SPECS:
        if spec.distribution != "beta":
            continue
        value = GenericCriterion(spec.statistic_class()).calculate(data, distribution, params)
        assert np.isfinite(value), spec.feature_name
        checked += 1

    assert checked == 9


def test_beta_roundoff_at_closed_storage_boundaries_remains_calculable():
    data = np.array([0.0, 0.02, 0.1, 0.4, 0.75, 0.98, 1.0])
    distribution = BetaDistribution()
    params = distribution.fit(data)

    values = [
        GenericCriterion(spec.statistic_class()).calculate(data, distribution, params)
        for spec in CRITERIA_SPECS
        if spec.distribution == "beta"
    ]

    assert len(values) == 9
    assert np.all(np.isfinite(values))


@pytest.mark.parametrize("outside", [-1e-12, 1.0 + 1e-12])
def test_beta_fit_rejects_values_outside_fixed_support(outside):
    data = np.array([0.1, 0.4, 0.8, outside])

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        BetaDistribution().fit(data)
