import numpy as np
import scipy.stats as st

from pysatl_expert.core.distribution import AbstractDistribution


class LogNormalDistribution(AbstractDistribution):
    """
    Two-parameter implementation of the Log-Normal probability distribution.

    Defined by a shape parameter (s) and scale. A variable X is log-normally
    distributed if its natural logarithm is normally distributed.
    Features strictly positive theoretical support (0, inf).

    Mapping to SciPy: 's' maps to shape, 'scale' is exp(mean) with
    location fixed to zero.
    """
    def __init__(self):
        super().__init__(name="LogNormal", support=(0, np.inf))

    def fit(self, data: np.ndarray) -> dict:
        shape, loc, scale = st.lognorm.fit(data, floc=0)
        return {"s": shape, "scale": scale}

    def pdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        return st.lognorm.pdf(data, s=params["s"], scale=params["scale"])

    def cdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        return st.lognorm.cdf(data, s=params["s"], scale=params["scale"])
