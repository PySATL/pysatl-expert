import numpy as np
import scipy.stats as st

from pysatl_expert.core.distribution import AbstractDistribution


class ExponentialDistribution(AbstractDistribution):
    """
    Fixed-origin implementation of the Exponential probability distribution.

    Characterized by rate parameter (λ = 1/scale) with loc fixed at zero.
    Support is [0, inf), matching the experiment generator used for training data.

    Mapping to SciPy: 'scale = 1/lambda', with ``floc=0``.
    """

    def __init__(self):
        """
        Initialize the distribution with fixed-origin support [0, inf).
        """
        super().__init__(name="Exponential", support=(0.0, np.inf))

    def fit(self, data: np.ndarray) -> dict:
        """
        Estimate rate parameter (λ) via MLE with location fixed at zero.
        """
        _, scale = st.expon.fit(data, floc=0.0)
        return {"loc": 0.0, "scale": scale, "lambda": 1 / scale}

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

    def prepare_criterion_input(
        self, data: np.ndarray, params: dict
    ) -> tuple[np.ndarray, dict]:
        """Standardize observations for canonical Exponential criteria."""
        observations = self._standardize_criterion_input(
            data,
            params.get("loc", 0.0),
            params.get("scale", 1.0),
            positive_support=True,
        )
        return observations, {"lam": 1.0}
