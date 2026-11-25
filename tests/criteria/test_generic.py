import numpy as np
import pytest
from pysatl_criterion.distribution import WeibullDistributionDescriptor

from pysatl_expert.criteria.adapters.criterion_adapter import CriterionAdapter
from pysatl_expert.criteria.calculate.generic import GenericCriterion
from pysatl_expert.criteria.catalog import CRITERIA_SPEC_BY_KEY, CRITERIA_SPECS
from pysatl_expert.distributions import (
    BetaDistribution,
    ExponentialDistribution,
    ExponentiatedWeibullDistribution,
    GammaDistribution,
    HyperbolicDistribution,
    InverseGammaDistribution,
    LaplaceDistribution,
    LogLogisticDistribution,
    LogNormalDistribution,
    NormalDistribution,
    ParetoDistribution,
    StudentDistribution,
    UniformDistribution,
    WeibullDistribution,
)


def test_catalog_uses_the_public_continuous_criterion_api_for_all_fourteen_families():
    expected = {
        "pysatl_criterion.statistics.goodness_of_fit.continuous.beta",
        "pysatl_criterion.statistics.goodness_of_fit.continuous.exponent",
        "pysatl_criterion.statistics.goodness_of_fit.continuous.exponentiated_weibull",
        "pysatl_criterion.statistics.goodness_of_fit.continuous.gamma",
        "pysatl_criterion.statistics.goodness_of_fit.continuous.hyperbolic",
        "pysatl_criterion.statistics.goodness_of_fit.continuous.inverse_gamma",
        "pysatl_criterion.statistics.goodness_of_fit.continuous.laplace",
        "pysatl_criterion.statistics.goodness_of_fit.continuous.log_logistic",
        "pysatl_criterion.statistics.goodness_of_fit.continuous.log_normal",
        "pysatl_criterion.statistics.goodness_of_fit.continuous.normal",
        "pysatl_criterion.statistics.goodness_of_fit.continuous.pareto",
        "pysatl_criterion.statistics.goodness_of_fit.continuous.student",
        "pysatl_criterion.statistics.goodness_of_fit.continuous.uniform",
        "pysatl_criterion.statistics.goodness_of_fit.continuous.weibull",
    }
    assert {spec.statistic_class.__module__ for spec in CRITERIA_SPECS} == expected
    assert all(spec.statistic_class.supported_hypotheses() for spec in CRITERIA_SPECS)


@pytest.mark.parametrize(
    ("family", "distribution", "data"),
    [
        ("normal", NormalDistribution(), [-1.2, -0.4, 0.1, 0.5, 1.1, 1.8]),
        ("exponential", ExponentialDistribution(), [0.2, 0.5, 0.9, 1.4, 2.1, 2.7]),
        ("weibull", WeibullDistribution(), [0.3, 0.7, 1.2, 1.9, 2.8, 3.2]),
        ("uniform", UniformDistribution(), [0.1, 0.2, 0.35, 0.52, 0.75, 0.9]),
        ("student", StudentDistribution(), [-2.2, -0.6, 0.1, 0.8, 1.5, 3.0]),
        ("gamma", GammaDistribution(), [0.3, 0.7, 1.1, 1.8, 2.9, 3.7]),
        ("beta", BetaDistribution(), [0.1, 0.2, 0.35, 0.52, 0.75, 0.9]),
        ("lognormal", LogNormalDistribution(), [0.3, 0.5, 0.8, 1.3, 2.1, 3.0]),
        ("laplace", LaplaceDistribution(), [-2.0, -0.8, -0.1, 0.3, 1.0, 2.5]),
        ("pareto", ParetoDistribution(), [1.1, 1.3, 1.7, 2.4, 3.8, 6.2]),
        ("loglogistic", LogLogisticDistribution(), [0.3, 0.6, 1.0, 1.7, 2.8, 4.5]),
        ("inversegamma", InverseGammaDistribution(), [0.25, 0.5, 0.9, 1.6, 2.7, 4.1]),
        (
            "exponentiatedweibull",
            ExponentiatedWeibullDistribution(),
            [0.3, 0.7, 1.2, 1.9, 2.8, 3.5],
        ),
        ("hyperbolic", HyperbolicDistribution(), [-1.8, -0.9, -0.2, 0.4, 1.1, 2.2]),
    ],
)
def test_ks_criterion_executes_for_each_supported_expert_family(family, distribution, data):
    sample = np.asarray(data, dtype=float)
    params = distribution.fit(sample)
    criterion = GenericCriterion(CRITERIA_SPEC_BY_KEY[(family, "ks")].statistic_class)

    value = criterion.calculate(sample, distribution, params)

    assert np.isfinite(value)


def test_native_empty_hypothesis_uses_original_sample_without_fit_coordinates():
    sample = np.array([0.3, 0.7, 1.2, 1.9, 2.8, 3.2])
    statistic_class = CRITERIA_SPEC_BY_KEY[("weibull", "ks")].statistic_class
    engine = statistic_class(WeibullDistributionDescriptor.default_parameterization().parse({}))
    expected = CriterionAdapter(engine).execute_prepared(sample).statistic

    actual = GenericCriterion(statistic_class).calculate(
        sample, WeibullDistribution(), WeibullDistribution().fit(sample)
    )

    assert actual == pytest.approx(expected)


def test_selector_catalog_contains_each_feature_once():
    names = [spec.feature_name for spec in CRITERIA_SPECS]
    assert len(names) == len(set(names))


@pytest.mark.parametrize(("code", "requires_fit"), [("ad", False), ("ks", True)])
def test_evaluate_retains_real_procedure_identity_and_scalar_wrapper(code, requires_fit):
    sample = np.array([-1.2, -0.4, 0.1, 0.5, 1.1, 1.8])
    distribution = NormalDistribution()
    params = distribution.fit(sample)
    statistic_class = CRITERIA_SPEC_BY_KEY[("normal", code)].statistic_class
    criterion = GenericCriterion(statistic_class)

    result = criterion.evaluate(sample, distribution, params)

    assert criterion.requires_fit is requires_fit
    assert result.parameter_mode == ("fitted" if requires_fit else "family")
    assert result.source == f"{code.upper()}_NORMALITY_GOODNESS_OF_FIT"
    assert result.alternative == "greater"
    assert result.parameterization == "normal.mean_var"
    assert result.configuration == {"schema_version": 1, "statistic_version": 1}
    assert bool(result.hypothesis) is requires_fit
    assert criterion.calculate(sample, distribution, params) == result.statistic
