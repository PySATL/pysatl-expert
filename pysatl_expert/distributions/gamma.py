import numpy as np
import scipy.stats as st

from pysatl_expert.core.distribution import AbstractDistribution


class GammaDistribution(AbstractDistribution):
    """
    Two-parameter implementation of the Gamma probability distribution.

    Defined by a shape parameter (a) and a scale parameter. Features[0, inf)
    support, which allows for early-fail validation of samples containing
    negative values. Frequently used to model waiting times or positively
    skewed continuous variables.

    Mapping to SciPy: 'shape' maps to 'a', location is fixed to zero (floc=0).
    """
    def __init__(self):
        super().__init__(name="Gamma", support=(0, np.inf))

    def fit(self, data: np.ndarray) -> dict:
        shape, loc, scale = st.gamma.fit(data, floc=0)
        return {"shape": shape, "scale": scale}

    def pdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        return st.gamma.pdf(data, a=params["shape"], scale=params["scale"])

    def cdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        return st.gamma.cdf(data, a=params["shape"], scale=params["scale"])
