import numpy as np
import scipy.stats as st

from pysatl_expert.core.distribution import AbstractDistribution


class WeibullDistribution(AbstractDistribution):
    """
    Two-parameter implementation of the Weibull probability distribution (minimum).

    Defined by shape (c) and scale parameters. Features [0, inf) support,
    allowing early-fail validation for non-positive data samples.

    Mapping to SciPy: uses 'weibull_min' with 'shape' mapped to 'c'
    and location fixed to zero.
    """

    def __init__(self):
        """
        Initializes the distribution with a theoretical support of [0, inf).
        """
        super().__init__(name="Weibull", support=(0, np.inf))

    def fit(self, data: np.ndarray) -> dict:
        """
        Estimates 'shape' and 'scale' parameters via MLE with fixed location (floc=0).
        """
        shape, _, scale = st.weibull_min.fit(data, floc=0)
        return {"shape": shape, "scale": scale}

    def pdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        """
        Evaluates the Weibull probability density function (PDF).
        """
        return st.weibull_min.pdf(data, c=params["shape"], scale=params["scale"])

    def cdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        """
        Evaluates the cumulative distribution function (CDF) for GoF analysis.
        """
        return st.weibull_min.cdf(data, c=params["shape"], scale=params["scale"])
