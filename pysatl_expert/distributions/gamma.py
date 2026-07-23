import numpy as np
import scipy.stats as st

from pysatl_expert.core.distribution import AbstractDistribution


class GammaDistribution(AbstractDistribution):
    """
    Three-parameter implementation of the Gamma probability distribution.

    Defined by a shape parameter (a), location (loc), and a scale parameter.
    Support is (-inf, inf) to allow fitting shifted samples with negative values.

    Mapping to SciPy: 'shape' maps to 'a', location is estimated via MLE.
    """

    def __init__(self):
        super().__init__(name="Gamma", support=(-np.inf, np.inf))

    def fit(self, data: np.ndarray) -> dict:
        shape, loc, scale = st.gamma.fit(data)
        return {"shape": shape, "loc": loc, "scale": scale}

    def pdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        loc = params.get("loc", 0)
        return st.gamma.pdf(data, a=params["shape"], loc=loc, scale=params["scale"])

    def cdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        loc = params.get("loc", 0)
        return st.gamma.cdf(data, a=params["shape"], loc=loc, scale=params["scale"])
