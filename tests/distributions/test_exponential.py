import numpy as np

from pysatl_expert.distributions.exponential import ExponentialDistribution


def test_exponential_distribution_logic():
    dist = ExponentialDistribution()
    data = np.array([1.0, 2.0, 5.0])

    params = dist.fit(data)
    assert "lambda" in params
    assert params["lambda"] > 0

    pdf = dist.pdf(data, params)
    cdf = dist.cdf(data, params)

    assert pdf.shape == (3,)
    assert cdf.shape == (3,)
    assert np.all(cdf >= 0) and np.all(cdf <= 1)
    assert dist.support == (0, np.inf)
