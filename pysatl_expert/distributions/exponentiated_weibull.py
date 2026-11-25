import numpy as np
import scipy.stats as st

from pysatl_expert.core.distribution import AbstractDistribution


class ExponentiatedWeibullDistribution(AbstractDistribution):
    """Fixed-origin Exponentiated Weibull probability distribution.

    Defined by exponent (a > 0), shape (k > 0), and scale (lambda > 0),
    with loc fixed at zero. Support is [0, inf).
    """

    def __init__(self):
        super().__init__(name="ExponentiatedWeibull", support=(0.0, np.inf))

    def fit(self, data: np.ndarray) -> dict:
        values = np.asarray(data, dtype=float)
        if np.any(values <= 0.0):
            raise ValueError("ExponentiatedWeibull fitting requires strictly positive observations")
        reference_scale = float(np.ptp(values))
        standardized = values / reference_scale
        exponent, shape, _, scale = st.exponweib.fit(standardized, floc=0.0)
        return {
            "exponent": float(exponent),
            "shape": float(shape),
            "loc": 0.0,
            "scale": float(scale * reference_scale),
        }

    def pdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        return st.exponweib.pdf(
            data,
            a=params["exponent"],
            c=params["shape"],
            loc=params["loc"],
            scale=params["scale"],
        )

    def cdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        return st.exponweib.cdf(
            data,
            a=params["exponent"],
            c=params["shape"],
            loc=params["loc"],
            scale=params["scale"],
        )

    def prepare_criterion_input(self, data: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
        """Standardize observations for canonical Exponentiated Weibull criteria."""
        observations = self._standardize_criterion_input(
            data,
            params["loc"],
            params["scale"],
            positive_support=True,
        )
        return observations, {
            "exponent": float(params["exponent"]),
            "shape": float(params["shape"]),
            "scale": 1.0,
        }
