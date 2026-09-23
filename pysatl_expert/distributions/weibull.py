import numpy as np
import scipy.stats as st

from pysatl_expert.core.distribution import AbstractDistribution


class WeibullDistribution(AbstractDistribution):
    """
    Fixed-origin implementation of the Weibull probability distribution (minimum).

    Defined by shape (c) and scale, with loc fixed at zero.
    Support is [0, inf), matching the experiment generator used for training data.

    Mapping to SciPy: uses 'weibull_min' with 'shape' mapped to 'c'.
    """

    def __init__(self):
        """
        Initialize the distribution with fixed-origin support [0, inf).
        """
        super().__init__(name="Weibull", support=(0.0, np.inf))

    def fit(self, data: np.ndarray) -> dict:
        """
        Estimate shape and scale via MLE with location fixed at zero.
        """
        values = np.asarray(data, dtype=float)
        reference_scale = float(np.ptp(values))
        standardized = values / reference_scale
        shape, _, scale = st.weibull_min.fit(standardized, floc=0)
        return {
            "shape": shape,
            "loc": 0.0,
            "scale": scale * reference_scale,
        }

    def pdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        """
        Evaluates the Weibull probability density function (PDF).
        """
        loc = params["loc"]
        return st.weibull_min.pdf(data, c=params["shape"], loc=loc, scale=params["scale"])

    def cdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        """
        Evaluates the cumulative distribution function (CDF) for GoF analysis.
        """
        loc = params["loc"]
        return st.weibull_min.cdf(data, c=params["shape"], loc=loc, scale=params["scale"])

    def prepare_criterion_input(self, data: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
        """Standardize observations for canonical Weibull criteria."""
        observations = self._standardize_criterion_input(
            data,
            params["loc"],
            params["scale"],
            positive_support=True,
        )
        return observations, {"a": 1.0, "k": float(params["shape"])}
