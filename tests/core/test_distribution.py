import numpy as np

from pysatl_expert.core.distribution import AbstractDistribution


class DistributionStub(AbstractDistribution):
    def fit(self, data: np.ndarray) -> dict:
        return {"param": 1.0}

    def pdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        return np.ones_like(data)

    def cdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        return np.zeros_like(data)


def test_abstract_distribution_properties():
    name = "Stub"
    support = (0, 100)
    dist = DistributionStub(name, support)

    assert dist.name == name
    assert dist.support == support


def test_abstract_distribution_methods():
    dist = DistributionStub("Stub", (0, 1))
    data = np.array([0.5])
    params = dist.fit(data)

    assert params == {"param": 1.0}
    assert np.array_equal(dist.pdf(data, params), [1.0])
    assert np.array_equal(dist.cdf(data, params), [0.0])


def test_abstract_distribution_calls():
    dist = DistributionStub("Test", (0, 1))
    data = np.array([0.5])
    dist.fit(data)
    dist.pdf(data, {})
    dist.cdf(data, {})
    assert dist.name == "Test"
    assert dist.support == (0, 1)


def test_abstract_distribution_coverage():
    class FullStub(AbstractDistribution):
        def fit(self, data):
            return super().fit(data)

        def pdf(self, data, params):
            return super().pdf(data, params)

        def cdf(self, data, params):
            return super().cdf(data, params)

    stub = FullStub("Test", (0, 1))
    stub.fit(np.array([1]))
    stub.pdf(np.array([1]), {})
    stub.cdf(np.array([1]), {})
    assert stub.name == "Test"
