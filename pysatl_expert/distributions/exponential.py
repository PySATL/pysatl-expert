import numpy as np
import scipy.stats as st

from pysatl_expert.core.distribution import AbstractDistribution


class ExponentialDistribution(AbstractDistribution):
    """
    One-parameter implementation of the Exponential probability distribution.

    Characterized by a rate parameter (λ). Fixed at [0, inf) support,
    enabling early-fail validation for samples containing non-positive values.

    Mapping to SciPy: uses 'scale = 1/lambda' with location fixed to zero.
    """

    def __init__(self):
        """
        Initializes the distribution with a theoretical support of [0, inf).
        """
        super().__init__(name="Exponential", support=(0, np.inf))

    def fit(self, data: np.ndarray) -> dict:
        """
        Estimates the rate parameter (λ) via MLE with fixed location (floc=0).
        """
        _, scale = st.expon.fit(data, floc=0)
        return {"lambda": 1 / scale}

    def pdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        """
        Evaluates the probability density function (PDF).
        """
        return st.expon.pdf(data, scale=1 / params["lambda"])

    def cdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        """
        Evaluates the cumulative distribution function (CDF) for GoF assessment.
        """
        return st.expon.cdf(data, scale=1 / params["lambda"])
