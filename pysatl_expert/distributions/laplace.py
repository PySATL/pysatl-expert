import numpy as np
import scipy.stats as st

from pysatl_expert.core.distribution import AbstractDistribution


class LaplaceDistribution(AbstractDistribution):
    """Two-parameter Laplace (double exponential) probability distribution.

    Defined by location (loc) and scale (> 0) with support (-inf, inf).
    """

    def __init__(self):
        super().__init__(name="Laplace", support=(-np.inf, np.inf))

    def fit(self, data: np.ndarray) -> dict:
        loc, scale = st.laplace.fit(data)
        return {"loc": float(loc), "scale": float(scale)}

    def pdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        return st.laplace.pdf(data, loc=params["loc"], scale=params["scale"])

    def cdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        return st.laplace.cdf(data, loc=params["loc"], scale=params["scale"])

    def prepare_criterion_input(self, data: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
        """Standardize observations for canonical Laplace criteria."""
        observations = self._standardize_criterion_input(data, params["loc"], params["scale"])
        return observations, {"t": 0.0, "s": 1.0}
