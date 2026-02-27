import numpy as np

from pysatl_expert.distributions.normal import NormalDistribution


def test_normal_distribution_logic():
    dist = NormalDistribution()
    data = np.array([1.0, 2.0, 3.0])

    params = dist.fit(data)
    assert "mu" in params
    assert "std" in params
    assert isinstance(params["mu"], float)

    pdf = dist.pdf(data, params)
    cdf = dist.cdf(data, params)

    assert pdf.shape == (3,)
    assert cdf.shape == (3,)
    assert np.all(cdf >= 0) and np.all(cdf <= 1)
    assert dist.support == (-np.inf, np.inf)
