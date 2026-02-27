import numpy as np

from pysatl_expert.distributions.weibull import WeibullDistribution


def test_weibull_distribution_logic():
    dist = WeibullDistribution()
    data = np.array([0.5, 1.5, 2.5])

    params = dist.fit(data)
    assert "shape" in params
    assert "scale" in params

    pdf = dist.pdf(data, params)
    cdf = dist.cdf(data, params)

    assert pdf.shape == (3,)
    assert cdf.shape == (3,)
    assert np.all(cdf >= 0) and np.all(cdf <= 1)
    assert dist.support == (0, np.inf)
