import numpy as np
import scipy.stats as st

from pysatl_expert.core.distribution import AbstractDistribution


class ExponentialDistribution(AbstractDistribution):
    """
    Two-parameter implementation of the Exponential probability distribution with location shift.

    Characterized by rate parameter (λ = 1/scale) and location parameter (loc).
    Support is (-inf, inf) to allow fitting shifted samples with negative values.

    Mapping to SciPy: 'scale = 1/lambda', with location estimated via MLE.
    """

    def __init__(self):
        """
        Initializes the distribution with universal theoretical support (-inf, inf).
        """
        super().__init__(name="Exponential", support=(-np.inf, np.inf))

    def fit(self, data: np.ndarray) -> dict:
        """
        Estimates location (loc) and rate parameter (λ) via MLE.
        """
        loc, scale = st.expon.fit(data)
        return {"loc": loc, "scale": scale, "lambda": 1 / scale if scale > 0 else 1.0}

    def pdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        """
        Evaluates the probability density function (PDF).
        """
        loc = params.get("loc", 0)
        scale = params.get("scale", 1 / params.get("lambda", 1))
        return st.expon.pdf(data, loc=loc, scale=scale)

    def cdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        """
        Evaluates the cumulative distribution function (CDF) for GoF assessment.
        """
        loc = params.get("loc", 0)
        scale = params.get("scale", 1 / params.get("lambda", 1))
        return st.expon.cdf(data, loc=loc, scale=scale)
