import numpy as np
import scipy.stats as st

from pysatl_expert.core.distribution import AbstractDistribution


class LogNormalDistribution(AbstractDistribution):
    """
    Three-parameter implementation of the Log-Normal probability distribution.

    Defined by a shape parameter (s), location (loc), and scale. A variable X is
    log-normally distributed if X - loc has a log-normal distribution.
    Support is (-inf, inf) to allow fitting shifted samples with negative values.

    Mapping to SciPy: 's' maps to shape, 'scale' is exp(mean), location estimated via MLE.
    """

    def __init__(self):
        super().__init__(name="LogNormal", support=(-np.inf, np.inf))

    def fit(self, data: np.ndarray) -> dict:
        shape, loc, scale = st.lognorm.fit(data)
        return {"s": shape, "loc": loc, "scale": scale}

    def pdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        loc = params.get("loc", 0)
        return st.lognorm.pdf(data, s=params["s"], loc=loc, scale=params["scale"])

    def cdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        loc = params.get("loc", 0)
        return st.lognorm.cdf(data, s=params["s"], loc=loc, scale=params["scale"])
