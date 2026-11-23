import numpy as np
import pytest
import scipy.stats as st

from pysatl_expert.app import analyze_sample
from pysatl_expert.criteria.catalog import DISTRIBUTIONS
from pysatl_expert.distributions import DISTRIBUTION_REGISTRY, create_all_distributions
from pysatl_expert.reporting.common import get_scipy_dist


def test_central_distribution_registry_matches_catalog_keys():
    instances = create_all_distributions()
    assert len(instances) == 14
    assert {dist.name.lower() for dist in instances} == set(DISTRIBUTIONS)
    assert set(DISTRIBUTION_REGISTRY) == {dist.name for dist in instances}


@pytest.mark.parametrize(
    ("name", "sample", "statistics"),
    [
        (
            "Laplace",
            st.laplace.rvs(loc=1.2, scale=1.8, size=60, random_state=np.random.default_rng(1)),
            ["laplace__ks", "laplace__ad"],
        ),
        (
            "Pareto",
            st.pareto.rvs(
                b=2.4, loc=0.0, scale=1.5, size=60, random_state=np.random.default_rng(2)
            ),
            ["pareto__ks", "pareto__ad", "pareto__zhang_za"],
        ),
        (
            "LogLogistic",
            st.fisk.rvs(c=2.2, loc=0.0, scale=1.4, size=60, random_state=np.random.default_rng(3)),
            ["loglogistic__ks", "loglogistic__ad"],
        ),
        (
            "InverseGamma",
            st.invgamma.rvs(
                a=3.1, loc=0.0, scale=1.9, size=60, random_state=np.random.default_rng(4)
            ),
            ["inversegamma__ks", "inversegamma__ad"],
        ),
        (
            "ExponentiatedWeibull",
            st.exponweib.rvs(
                a=1.5, c=2.1, loc=0.0, scale=1.3, size=60, random_state=np.random.default_rng(5)
            ),
            ["exponentiatedweibull__ks", "exponentiatedweibull__ad"],
        ),
        (
            "Hyperbolic",
            st.genhyperbolic.rvs(
                p=1.0,
                a=2.5,
                b=0.4,
                loc=-0.3,
                scale=1.1,
                size=60,
                random_state=np.random.default_rng(6),
            ),
            ["hyperbolic__ks", "hyperbolic__ad"],
        ),
    ],
)
def test_new_distribution_fits_evaluates_cdf_and_runs_catalog_criteria(name, sample, statistics):
    dist = DISTRIBUTION_REGISTRY[name]()
    params = dist.fit(sample)
    assert all(np.isfinite(value) for value in params.values())

    pdf_values = dist.pdf(sample, params)
    cdf_values = dist.cdf(sample, params)
    assert np.all(np.isfinite(pdf_values)) and np.all(pdf_values >= 0.0)
    assert np.all(np.isfinite(cdf_values)) and np.all((cdf_values >= 0.0) & (cdf_values <= 1.0))

    scipy_dist = get_scipy_dist(name, params)
    assert scipy_dist is not None
    np.testing.assert_allclose(cdf_values, scipy_dist.cdf(sample), atol=1e-12)

    result = analyze_sample(sample, distributions=[name], statistics=statistics)
    assert result.failures == {}
    for stat_id in statistics:
        assert stat_id in result.procedure_results
        assert np.isfinite(result.procedure_results[stat_id].statistic)


@pytest.mark.parametrize(
    "name",
    ["Pareto", "LogLogistic", "InverseGamma", "ExponentiatedWeibull"],
)
def test_positive_new_distributions_reject_non_positive_observations(name):
    dist = DISTRIBUTION_REGISTRY[name]()
    with pytest.raises(ValueError, match="strictly positive"):
        dist.fit(np.array([0.0, 1.0, 2.0, 3.0]))

    stat_id = f"{name.lower()}__ks"
    result = analyze_sample(
        np.array([-1.0, 0.5, 1.5, 2.5]), distributions=[name], statistics=[stat_id]
    )
    assert "outside distribution support" in result.failures[f"fit:{name}"]
    assert "outside distribution support" in result.failures[stat_id]
