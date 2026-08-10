import numpy as np
import scipy.stats as st

from pysatl_expert.core.distribution import AbstractDistribution


class WeibullDistribution(AbstractDistribution):
    """
    Three-parameter implementation of the Weibull probability distribution (minimum).

    Defined by shape (c), location (loc), and scale parameters.
    Support is (-inf, inf) to allow fitting shifted samples with negative values.

    Mapping to SciPy: uses 'weibull_min' with 'shape' mapped to 'c'.
    """

    def __init__(self):
        """
        Initializes the distribution with universal theoretical support (-inf, inf).
        """
        super().__init__(name="Weibull", support=(-np.inf, np.inf))

    def fit(self, data: np.ndarray) -> dict:
        """
        Estimates 'shape', 'loc', and 'scale' parameters via MLE.
        """
        shape, loc, scale = st.weibull_min.fit(data)
        return {"shape": shape, "loc": loc, "scale": scale}

    def pdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        """
        Evaluates the Weibull probability density function (PDF).
        """
        loc = params.get("loc", 0)
        return st.weibull_min.pdf(data, c=params["shape"], loc=loc, scale=params["scale"])

    def cdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        """
        Evaluates the cumulative distribution function (CDF) for GoF analysis.
        """
        loc = params.get("loc", 0)
        return st.weibull_min.cdf(data, c=params["shape"], loc=loc, scale=params["scale"])
